"""Package existing validation evidence without running a new simulation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import re
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_NAME = "slinky-lab-validation-v0.1.0"
ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
EXCLUDED_SUFFIXES = {".avi", ".gif", ".jpeg", ".jpg", ".mov", ".mp4", ".png", ".svg", ".webm"}
EXCLUDED_NAMES = {"native-log.txt", "process.log"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sanitize_json(value: Any, root: Path) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if str(key).lower() in {"absolute_path", "local_path"}:
                continue
            cleaned[key] = _sanitize_json(item, root)
        return cleaned
    if isinstance(value, list):
        return [_sanitize_json(item, root) for item in value]
    if isinstance(value, str) and (ABSOLUTE_PATH.match(value) or value.startswith(("/home/runner/", "/github/workspace/", "/app/", "/data/"))):
        candidate = Path(value)
        try:
            relative = candidate.resolve().relative_to(root.resolve())
        except ValueError:
            return "<absolute-local-path-removed>"
        return f"repo-relative/{relative.as_posix()}"
    return value


def _copy_evidence(source: Path, destination: Path, root: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
        destination.write_text(
            json.dumps(_sanitize_json(value, root), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    elif source.suffix.lower() == ".md":
        text = source.read_text(encoding="utf-8")
        destination.write_text(text.replace(str(root), "<repo>"), encoding="utf-8")
    else:
        shutil.copy2(source, destination)


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _reported_status(source: Path, fallback: str = "reported") -> str:
    value = json.loads(source.read_text(encoding="utf-8"))
    summary = value.get("summary")
    status = value.get("status")
    if "passed" in value:
        return "passed" if value.get("passed") is True else "failed" if value.get("passed") is False else fallback
    if status in {"passed", "success"} or (isinstance(summary, dict) and summary.get("status") in {"passed", "success"}):
        return "passed"
    if status in {"failed", "error"} or (isinstance(summary, dict) and summary.get("status") in {"failed", "error"}):
        return "failed"
    if status in {"complete", "completed"} or (isinstance(summary, dict) and summary.get("status") in {"complete", "completed"}):
        return "execution_completed"
    return fallback


def _include_report_file(source: Path) -> bool:
    return source.is_file() and source.name.lower() not in EXCLUDED_NAMES and source.suffix.lower() not in EXCLUDED_SUFFIXES and source.suffix.lower() != ".lock" and "__pycache__" not in source.parts


def _add_record(
    root: Path,
    staging: Path,
    evidence: list[dict[str, Any]],
    record_id: str,
    title: str,
    status: str,
    notes: str,
    files: list[tuple[str, str]],
    missing_raw: list[str] | None = None,
) -> None:
    packaged = []
    source_paths = []
    source_sha256 = {}
    for source_name, destination_name in files:
        source = root / source_name
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = staging / destination_name
        _copy_evidence(source, destination, root)
        source_paths.append(source_name)
        source_sha256[source_name] = _sha256(source)
        packaged.append(destination_name)
    evidence.append({
        "id": record_id,
        "title": title,
        "status": status,
        "source_paths": source_paths,
        "source_sha256": source_sha256,
        "packaged_paths": packaged,
        "missing_raw": missing_raw or [],
        "notes": notes,
    })


def _files(staging: Path) -> list[Path]:
    return sorted(path for path in staging.rglob("*") if path.is_file())


def _write_readme(staging: Path, created_at: str, evidence: list[dict[str, Any]], validation_index: dict[str, Any], model_version: str, engine_version: str) -> None:
    overall_acceptance = validation_index.get("overall_acceptance")
    latest_promotion_allowed = validation_index.get("latest_promotion_allowed")
    experimental_support = validation_index.get("experimental_support", "not stated")
    lines = [
        f"# Slinky Lab v0.1.0 验证附件",
        "",
        f"生成时间：{created_at}",
        f"引擎版本：MuJoCo {engine_version}；模型版本：{model_version}。",
        "",
        "本包只收录仓库中已经存在的验证记录，不运行新的仿真，也不修改原始记录。JSON 中的本机绝对路径已替换为包内相对路径或移除。",
        "",
        "## 证据范围",
        "",
        "- Drop：16/32 segments-per-turn 收敛记录、v3 quick repeat，以及 HTTP/WebSocket runtime-stream-after-lock 记录。drop 几何材料是演示假设，不能作为实测塑料彩虹圈参数。",
        "- Linux 容器 smoke 和 Linux HTTP/WebSocket runtime-stream 记录；仅收录仓库中的 JSON 证据，没有补造容器原始轨迹。",
        "- 静态标定：39 圈文献目标长度拟合记录；文献映射和几何参数假设已在 JSON 中保留。",
        "- 楼梯：stopped、world-axis sliding、world-axis side_fall、depth12 baseline，以及本轮三档收敛报告和 raw 证据。",
        "",
        "## 证据边界",
        "",
        f"楼梯三阶 baseline 报告了 3 个 confirmed flips；本包收录 reports/stairs-depth-check/depth12 的真实 config/model/metadata/HDF5/frames/support 证据，并排除 process.log。楼梯记录未经过实验支持。当前 validation-results.json 状态：overall_acceptance={overall_acceptance!r}，latest_promotion_allowed={latest_promotion_allowed!r}，experimental_support={experimental_support!r}。",
        "如果 reports/stairs-validation/validation-summary.json 存在，扩展三档记录会按其实际 passed 字段收录；文件存在本身不会改变总体验收或推广状态。",
        "静态长度标定和 drop 数值收敛属于数值证据，不等于实验校准。能量中的 cable elastic energy 是独立诊断估计，不能宣称 MuJoCo d.energy 已包含 cable 插件弹性能。",
        "参考视频和派生图未收录。source_sha256 保留原始文件哈希；JSON 路径脱敏会改变复制文件内容，MANIFEST.sha256 对应发布副本。",
        "",
        "## 文件校验",
        "",
        "同目录 MANIFEST.sha256 列出包内证据文件的 SHA256；manifest 本身不列入自身哈希。",
        "",
        "## 收录项",
        "",
    ]
    for item in evidence:
        missing = f"；缺失 raw：{', '.join(item['missing_raw'])}" if item["missing_raw"] else ""
        lines.append(f"- `{item['id']}`：{item['status']}；{item['notes']}{missing}")
    (staging / "README.zh-CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_hash_manifest(staging: Path) -> list[dict[str, Any]]:
    records = []
    for path in _files(staging):
        relative = path.relative_to(staging).as_posix()
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    manifest = "\n".join(f"{item['sha256']}  {item['path']}" for item in records) + "\n"
    (staging / "MANIFEST.sha256").write_text(manifest, encoding="utf-8")
    return records


def _make_archive(staging: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for path in _files(staging):
            handle.write(path, f"{PACKAGE_NAME}/{path.relative_to(staging).as_posix()}")


@contextmanager
def _staging_directory(root: Path, release_dir: Path):
    """Create and remove only a unique staging directory under root/reports."""
    root = root.resolve()
    reports_dir = (root / "reports").resolve()
    release_dir = release_dir.resolve()
    if not _path_is_within(release_dir, reports_dir):
        raise ValueError(f"release directory must be inside {reports_dir}")
    staging = release_dir / f".{PACKAGE_NAME}-staging-{uuid.uuid4().hex}"
    if not _path_is_within(staging, release_dir):
        raise ValueError("staging directory escaped release directory")
    staging.mkdir(parents=False)
    try:
        yield staging
    finally:
        resolved = staging.resolve()
        if not _path_is_within(resolved, release_dir):
            raise RuntimeError("refusing to remove staging directory outside release directory")
        shutil.rmtree(resolved, ignore_errors=True)


def build_package(root: Path, release_dir: Path) -> dict[str, Any]:
    root = root.resolve()
    release_dir = release_dir.resolve()
    reports_dir = (root / "reports").resolve()
    if not _path_is_within(release_dir, reports_dir):
        raise ValueError(f"release directory must be inside {reports_dir}")
    release_dir.mkdir(parents=True, exist_ok=True)
    archive = release_dir / f"{PACKAGE_NAME}.zip"
    manifest_path = release_dir / f"{PACKAGE_NAME}.manifest.json"
    created_at = datetime.now(timezone.utc).isoformat()
    evidence: list[dict[str, Any]] = []
    validation_index_path = root / "docs" / "validation-results.json"
    validation_index = json.loads(validation_index_path.read_text(encoding="utf-8"))
    model_version = str(validation_index.get("model_version", "unknown"))
    engine_version = str(validation_index.get("engine_version", "3.15.0"))
    with _staging_directory(root, release_dir) as staging:
        _add_record(
            root, staging, evidence, "drop-16spt-dt6.25e-6", "Drop refined 16 segments/turn", "passed",
            "16→32 收敛中的 refined drop 记录；含配置、metadata、summary 和 metrics。", [
                ("docs/validation/16spt_dt6.25e-6.json", "validation/drop/16spt_dt6.25e-6/record.json"),
                ("docs/validation/16spt_dt6.25e-6-metrics.csv", "validation/drop/16spt_dt6.25e-6/metrics.csv"),
            ], ["trajectory.h5/model.xml 未在现有该记录目录中提供"],
        )
        _add_record(
            root, staging, evidence, "drop-32spt-dt3.125e-6", "Drop refined 32 segments/turn", "passed",
            "16→32 网格收敛记录；含配置、metadata、summary 和 metrics。", [
                ("docs/validation/32spt_dt3.125e-6.json", "validation/drop/32spt_dt3.125e-6/record.json"),
                ("docs/validation/32spt_dt3.125e-6-metrics.csv", "validation/drop/32spt_dt3.125e-6/metrics.csv"),
            ], ["trajectory.h5/model.xml 未在现有该记录目录中提供"],
        )
        _add_record(
            root, staging, evidence, "drop-v3-quick-repeat", "Drop v3 quick repeat", "passed",
            "快速重复检查；含 freefall、native equilibrium、work-balance diagnostic 和时间步长对比。", [
                ("docs/validation/v3-quick-repeat.json", "validation/drop/v3-quick-repeat/record.json"),
            ], ["raw trajectory.h5/metrics.csv 未在现有记录中提供"],
        )
        _add_record(
            root, staging, evidence, "drop-runtime-stream-after-lock", "HTTP/WebSocket streamed drop repeat", "passed",
            "两个真实 HTTP/WS 运行均 completed，19 帧完整且 HDF/CSV/summary 导出通过。", [
                ("docs/validation/runtime-stream-after-lock.json", "validation/runtime/runtime-stream-after-lock.json"),
            ], ["服务运行目录 raw trajectory.h5 未纳入仓库记录"],
        )
        _add_record(
            root, staging, evidence, "linux-container-smoke", "Linux container smoke", _reported_status(root / "docs/validation/linux-container.json", "reported"),
            "Linux 容器真实 MuJoCo smoke；只收录 JSON 记录，容器运行目录原始 HDF5 未在仓库中提供。", [
                ("docs/validation/linux-container.json", "validation/linux/linux-container.json"),
            ], ["container raw model.xml/trajectory.h5/metrics.csv 未在仓库记录中提供"],
        )
        _add_record(
            root, staging, evidence, "linux-runtime-stream", "Linux HTTP/WebSocket runtime stream", _reported_status(root / "docs/validation/linux-runtime-stream.json", "reported"),
            "Linux 容器内两次真实 HTTP/WebSocket 运行；JSON 保留完整状态、帧数和导出检查。", [
                ("docs/validation/linux-runtime-stream.json", "validation/linux/linux-runtime-stream.json"),
            ], ["container raw model.xml/trajectory.h5/metrics.csv 未在仓库记录中提供"],
        )
        _add_record(
            root, staging, evidence, "static-literature-fit", "39-turn static literature fit", "fit_passed",
            "Cross and Wheatland (2012) 目标长度拟合；报告明确 independent_experimental_support=false。", [
                ("docs/validation/static-calibration.json", "validation/static/static-calibration.json"),
            ], ["raw model.xml/trajectory.h5 未在现有记录中提供"],
        )
        stair_common = [
            ("config.json", "config.json"),
            ("model.xml", "model.xml"),
            ("metadata.json", "metadata.json"),
            ("trajectory.h5", "trajectory.h5"),
            ("frames.json", "frames.json"),
            ("summary.json", "summary.json"),
            ("warnings.json", "warnings.json"),
            ("probe-report.md", "probe-report.md"),
        ]
        _add_record(
            root, staging, evidence, "stairs-stopped", "Stairs stopped diagnostic", "completed",
            "无运行时驱动；stopped、quiet duration 和 low penetration 记录。", [
                ("docs/validation/stairs-stopped.json", "validation/stairs/stopped/docs-summary.json"),
                ("docs/validation/stairs-stopped.md", "validation/stairs/stopped/docs-summary.md"),
                *[(f"reports/stairs-stopped/{source}", f"validation/stairs/stopped/raw/{destination}") for source, destination in stair_common],
                ("reports/stairs-stopped/native-summary.json", "validation/stairs/stopped/raw/native-summary.json"),
                ("reports/stairs-stopped/keyframes.json", "validation/stairs/stopped/raw/keyframes.json"),
                ("reports/stairs-stopped/top-tread-sampled.json", "validation/stairs/stopped/raw/top-tread-sampled.json"),
            ],
        )
        axis_files = [
            ("config.json", "config.json"),
            ("model.xml", "model.xml"),
            ("metadata.json", "metadata.json"),
            ("trajectory.h5", "trajectory.h5"),
            ("frames.json", "frames.json"),
            ("summary.json", "summary.json"),
            ("warnings.json", "warnings.json"),
            ("probe-report.md", "probe-report.md"),
            ("initial-geometry.json", "initial-geometry.json"),
            ("keyframes.json", "keyframes.json"),
            ("support-diagnostics.json", "support-diagnostics.json"),
        ]
        _add_record(
            root, staging, evidence, "stairs-world-axis-A-sliding", "Stairs world-axis A", "sliding",
            "真实 MuJoCo v3 probe；明确为 macro demonstration geometry，无实验支持。", [
                ("docs/validation/stairs-world-axis-A.json", "validation/stairs/sliding-A/docs-summary.json"),
                ("docs/validation/stairs-world-axis-A.md", "validation/stairs/sliding-A/docs-summary.md"),
                *[(f"reports/stairs-world-axis/A/{source}", f"validation/stairs/sliding-A/raw/{destination}") for source, destination in axis_files],
            ],
        )
        _add_record(
            root, staging, evidence, "stairs-world-axis-B-side-fall", "Stairs world-axis B", "side_fall",
            "真实 MuJoCo v3 probe；明确为 macro demonstration geometry，无实验支持。", [
                ("docs/validation/stairs-world-axis-B.json", "validation/stairs/side-fall-B/docs-summary.json"),
                ("docs/validation/stairs-world-axis-B.md", "validation/stairs/side-fall-B/docs-summary.md"),
                *[(f"reports/stairs-world-axis/B/{source}", f"validation/stairs/side-fall-B/raw/{destination}") for source, destination in axis_files],
            ],
        )
        baseline_files = [
            ("config.json", "config.json"),
            ("model.xml", "model.xml"),
            ("metadata.json", "metadata.json"),
            ("trajectory.h5", "trajectory.h5"),
            ("frames.json", "frames.json"),
            ("summary.json", "summary.json"),
            ("support-diagnostics.json", "support-diagnostics.json"),
            ("warnings.json", "warnings.json"),
            ("initial-geometry.json", "initial-geometry.json"),
            ("keyframes.json", "keyframes.json"),
            ("top-tread-samples.json", "top-tread-samples.json"),
            ("probe-report.md", "probe-report.md"),
        ]
        _add_record(
            root, staging, evidence, "stairs-baseline-three-flips", "Stairs three-flip baseline", "flip",
            "docs 汇总和 depth12 raw 记录均报告 confirmed_flip_count=3；该 baseline 未经过网格加密，且无实验支持。process.log 未收录。", [
                ("docs/validation/stairs-walking-baseline.json", "validation/stairs/baseline-3-flips/record.json"),
                ("docs/validation/stairs-walking-baseline.md", "validation/stairs/baseline-3-flips/record.md"),
                *[(f"reports/stairs-depth-check/depth12/{source}", f"validation/stairs/baseline-3-flips/raw/{destination}") for source, destination in baseline_files],
            ],
        )
        _add_record(
            root, staging, evidence, "stairs-convergence-public", "Stairs convergence public report", "failed",
            "本轮三档楼梯收敛的精简公开报告；mesh 超时是时间预算结果，不等同于积分器不稳定。", [
                ("docs/validation/stairs-convergence.json", "validation/stairs/convergence.json"),
                ("docs/validation/stairs-convergence.md", "validation/stairs/convergence.md"),
            ],
        )
        optional_validation_dir = root / "reports" / "stairs-validation"
        optional_summary = optional_validation_dir / "validation-summary.json"
        if optional_summary.is_file():
            optional_files = []
            for source in sorted(optional_validation_dir.rglob("*")):
                if not _include_report_file(source):
                    continue
                relative = source.relative_to(root).as_posix()
                destination = f"validation/stairs/three-case-report/{source.relative_to(optional_validation_dir).as_posix()}"
                optional_files.append((relative, destination))
            if optional_files:
                optional_status = _reported_status(optional_summary, "reported")
                _add_record(
                    root, staging, evidence, "stairs-three-case-report", "Stairs three-case validation report", optional_status,
                    f"扩展三档报告的实际 validation-summary.json；reported passed={json.loads(optional_summary.read_text(encoding='utf-8')).get('passed')!r}，文件存在本身不改变总体验收。",
                    optional_files,
                )
        for directory_name, record_id, title in (
            ("literature-drop", "literature-drop", "Literature drop validation"),
            ("arched-validation", "arched-validation", "Arched validation"),
            ("arched-steel-probe", "arched-steel-probe", "Thin steel-assumption probe (failed)"),
        ):
            optional_dir = root / "reports" / directory_name
            root_summary = optional_dir / "validation-summary.json"
            summary_files = [root_summary] if root_summary.is_file() else [source for source in sorted(optional_dir.rglob("*.json")) if source.is_file() and "summary" in source.stem.lower()]
            optional_files = []
            if summary_files:
                for source in sorted(optional_dir.rglob("*")):
                    if not _include_report_file(source):
                        continue
                    relative = source.relative_to(root).as_posix()
                    destination = f"validation/optional/{directory_name}/{source.relative_to(optional_dir).as_posix()}"
                    optional_files.append((relative, destination))
            if optional_files:
                summary_status = _reported_status(summary_files[0], "reported")
                _add_record(
                    root, staging, evidence, record_id, title, summary_status,
                    f"可选目录仅因存在实际 summary JSON 才收录；reported status={summary_status}，目录存在本身不改变总体验收。",
                    optional_files,
                )
        source_index = staging / "validation/source/validation-results.json"
        _copy_evidence(validation_index_path, source_index, root)
        _write_readme(staging, created_at, evidence, validation_index, model_version, engine_version)
        records = _write_hash_manifest(staging)
        _make_archive(staging, archive)

    manifest = {
        "package": PACKAGE_NAME,
        "created_at": created_at,
        "engine_version": engine_version,
        "model_version": model_version,
        "overall_acceptance": validation_index.get("overall_acceptance"),
        "latest_promotion_allowed": validation_index.get("latest_promotion_allowed"),
        "experimental_support": validation_index.get("experimental_support"),
        "evidence": evidence,
        "files": records,
        "source_index_sha256": _sha256(validation_index_path),
        "source_index_packaged_path": "validation/source/validation-results.json",
        "archive": {"path": archive.name, "bytes": archive.stat().st_size, "sha256": _sha256(archive)},
        "excluded": ["local logs", "credentials", "data/", "node_modules/", "parent repository", "reference videos", "derived images"],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--release-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    release_dir = (args.release_dir or (root / "reports" / "release")).resolve()
    manifest = build_package(root, release_dir)
    print(json.dumps({
        "archive": str(release_dir / f"{PACKAGE_NAME}.zip"),
        "manifest": str(release_dir / f"{PACKAGE_NAME}.manifest.json"),
        "file_count": len(manifest["files"]),
        "archive_bytes": manifest["archive"]["bytes"],
        "archive_sha256": manifest["archive"]["sha256"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# 容器部署与复现

镜像目标是 `linux/amd64`。MuJoCo 在 CPU 上运行；浏览器通过 WebGL 绘图，不需要把宿主机的图形设备映射进容器。

```bash
docker pull ghcr.io/fanyu-nijika/slinky-lab:v0.1.0
docker run -d --init --name slinky-lab \
  -p 127.0.0.1:8000:8000 \
  -v slinky-data:/data \
  ghcr.io/fanyu-nijika/slinky-lab:v0.1.0
docker logs -f slinky-lab
```

浏览器打开 `http://localhost:8000`。`/api/v1/health` 返回服务健康状态；计算进程在第一项任务开始时启动。命名卷由镜像内的非 root 用户（UID 10001）写入。若改用宿主机目录挂载，需要该目录对 UID 10001 可写。

每个数据目录由一个服务实例管理。CLI 批量任务可以使用单独数据目录：

```bash
docker run --rm --init -v slinky-batch:/data \
  ghcr.io/fanyu-nijika/slinky-lab:v0.1.0 \
  slinky-lab run --preset drop-preview
```

停止和恢复容器：

```bash
docker stop slinky-lab
docker start slinky-lab
```

已完成结果保留。服务退出时正在积分的任务会标记为中断，恢复时须新建运行；本版本不将最后一个显示帧作为求解器检查点继续积分。

## 版本与摘要

版本标签固定一次发布的镜像，`sha-<完整提交 SHA>` 对应代码提交。可记录拉取后的不可变摘要：

```bash
docker image inspect ghcr.io/fanyu-nijika/slinky-lab:v0.1.0 \
  --format '{{json .RepoDigests}}'
```

论文或报告记录镜像摘要、运行目录中的 `config.json`、`model.xml` 和 `summary.json`。预设验证状态与镜像标签分开记录；只有验收通过后才更新 `latest`。

已发布的 `v0.1.0` 对应源码 `d919d8b905927ddaf05b7b4e9507854cd57f1289`，摘要为 `sha256:f16cd11a7bd6d162d00f6fdf46760ccf14b7c72d79d25754709f184b3b3c245e`。包为 Public；全新Linux runner已匿名拉取，并完成真实HTTP/WebSocket计算、导出及拱形初态/短释放检查。工作流证据见 [v0.1.0发布验证](https://github.com/FanYu-Nijika/slinky-lab/actions/runs/37595591457)。这是研究预览，楼梯完整验收未通过，未发布 `latest` 镜像标签。

需要固定同一二进制镜像时，将运行命令末尾替换为：

```text
ghcr.io/fanyu-nijika/slinky-lab@sha256:f16cd11a7bd6d162d00f6fdf46760ccf14b7c72d79d25754709f184b3b3c245e
```

`Promote accepted image` 工作流要求版本提交的 `docs/validation-results.json` 同时将 `overall_acceptance` 和 `latest_promotion_allowed` 设为 true，并检查该版本的测试、发布和匿名拉取工作流已全部成功。它核对镜像的源码提交标签后按不可变摘要晋升。当前报告为 false，所以研究预览版本不会晋升到 `latest`。

## 发布检查

CI 在 Linux 上执行软件测试、构建镜像、运行真实求解器，并重启容器核对命名卷数据。版本镜像发布后，另一个全新 runner 使用空 Docker 凭据目录匿名拉取并执行相同计算检查。首次发布需要在 GitHub Packages 中把包的可见性设为 Public；仓库公开不等于镜像自动公开。若匿名检查先于可见性设置而失败，在设置后重新运行该检查。

数据卷中的原始实验数据、个人参考 CSV 和本地缓存不进入 Docker 构建上下文或公开仓库。

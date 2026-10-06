from pathlib import Path

from fastapi.testclient import TestClient

from slinky_lab.api import create_app
from slinky_lab.manager import RunManager


def test_health_and_api_404(tmp_path: Path):
    manager = RunManager(tmp_path, auto_start=False)
    manager.start = lambda: None
    app = create_app(manager)
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert client.get("/api/v1/does-not-exist").status_code == 404


def test_create_run_returns_queued_without_waiting_for_solver(tmp_path: Path):
    manager = RunManager(tmp_path, auto_start=False)
    manager.start = lambda: None
    app = create_app(manager)
    with TestClient(app) as client:
        response = client.post("/api/v1/runs", json={"name": "测试", "numerics": {"duration": 0.02}})
        assert response.status_code == 200
        body = response.json()
        assert body["run_id"]
        assert body["status"] == "queued"


def test_invalid_physics_config_returns_json_422(tmp_path: Path):
    manager = RunManager(tmp_path, auto_start=False)
    manager.start = lambda: None
    app = create_app(manager)
    with TestClient(app) as client:
        for payload in ({"material": {"mass": -1}}, {"material": {"pitch": 0.0001}}):
            response = client.post("/api/v1/runs", json=payload)
            assert response.status_code == 422
            detail = response.json()["detail"]
            assert isinstance(detail, list)
            assert detail and isinstance(detail[0]["msg"], str)

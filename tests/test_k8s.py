from __future__ import annotations

from pathlib import Path

import yaml

K8S_DIR = Path(__file__).resolve().parents[1] / "k8s"


def test_k8s_manifests_declare_health_probes() -> None:
    deployment = yaml.safe_load(
        (K8S_DIR / "deployment.yaml").read_text(encoding="utf-8")
    )
    service = yaml.safe_load((K8S_DIR / "service.yaml").read_text(encoding="utf-8"))

    assert deployment["kind"] == "Deployment"
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert container["livenessProbe"]["httpGet"]["path"] == "/health"
    assert container["readinessProbe"]["httpGet"]["path"] == "/health"
    assert service["kind"] == "Service"
    assert service["spec"]["selector"]["app"] == "jurimetria-ml"

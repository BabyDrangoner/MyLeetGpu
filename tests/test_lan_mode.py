from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_default_stack_remains_loopback_only() -> None:
    compose = read("docker-compose.yml")

    assert '"127.0.0.1:3000:8080"' in compose
    assert "8000:8000" not in compose


def test_lan_overlay_binds_one_explicit_address_and_requires_auth_file() -> None:
    compose = read("docker-compose.lan.yml")

    assert "MYLEETGPU_LAN_ADDRESS:?" in compose
    assert '- "0.0.0.0:' not in compose
    assert "lan.htpasswd:/etc/nginx/lan.htpasswd:ro" in compose
    assert "nginx.lan.conf:/etc/nginx/conf.d/default.conf:ro" in compose
    assert "8000" not in compose


def test_lan_nginx_authenticates_ui_and_api_but_not_healthcheck() -> None:
    nginx = read("apps/web/nginx.lan.conf")

    assert 'auth_basic "MyLeetGpu LAN";' in nginx
    assert "auth_basic_user_file /etc/nginx/lan.htpasswd;" in nginx
    assert "location = /healthz" in nginx
    assert "auth_basic off;" in nginx
    assert "proxy_set_header Host api;" in nginx
    assert "proxy_pass http://api:8000/api/;" in nginx


def test_firewall_rules_are_scoped_and_reversible() -> None:
    script = read("scripts/lan-firewall.ps1")

    assert '@("LocalSubnet")' in script
    assert 'ValidateSet("Enable", "Disable", "Status")' in script
    assert "New-NetFirewallHyperVRule" in script
    assert "New-NetFirewallRule" in script
    assert "Remove-NetFirewallHyperVRule" in script
    assert "Remove-NetFirewallRule" in script
    assert "DefaultInboundAction" not in script
    assert 'ListenAddress -eq "0.0.0.0"' in script


def test_container_healthcheck_uses_the_unauthenticated_health_endpoint() -> None:
    dockerfile = read("apps/web/Dockerfile")
    base_nginx = read("apps/web/nginx.conf")

    assert "http://127.0.0.1:8080/healthz" in dockerfile
    assert "location = /healthz" in base_nginx

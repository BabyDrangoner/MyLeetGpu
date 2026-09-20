from __future__ import annotations

import ast
import json
import os
import shlex
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from myleetgpu.config import Settings
from myleetgpu.domain.problems import ProblemCatalog
from myleetgpu.runner import colab_bridge
from myleetgpu.runner.colab import ColabRunner
from myleetgpu.runner.models import CommandResult, RunnerUnavailable, RunnerUnhealthy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SESSION = (
    "[vscode-colab] test-endpoint | Hardware: T4 | Shape: Standard | Variant: GPU | Status: IDLE\n"
)


@pytest.fixture
def runner(tmp_path: Path) -> ColabRunner:
    settings = Settings(
        data_dir=tmp_path / "data",
        problems_dir=PROJECT_ROOT / "problems",
        output_limit_bytes=4096,
        _env_file=None,
    )
    settings.ensure_directories()
    return ColabRunner(settings)


def metadata(**changes):
    return {
        "returncode": 0,
        "output": "",
        "duration_seconds": 0.1,
        "timed_out": False,
        "output_limited": False,
        **changes,
    }


def probe_data(language="cuda_cpp"):
    versions = {"nvcc_version": "release 12.8", "cuda_runtime_version": "12.8"}
    if language != "cuda_cpp":
        versions = {
            "python_version": "3.12.12",
            "torch_version": "2.9.0+cu128",
            "torch_cuda_version": "12.8",
        }
    if language == "triton_python":
        versions["triton_version"] = "3.5.0"
    return {
        "gpu_name": "Tesla T4",
        "compute_capability": "7.5",
        "driver_version": "570.1",
        "toolchain": versions,
        "telemetry": {"temperature_c": "42"},
    }


def test_rpc_uses_existing_alias_and_keeps_submission_out_of_shell(runner, monkeypatch):
    calls = []
    source = "$(touch /tmp/not-executed); `echo no`\n"

    def transport(args, **kwargs):
        calls.append((args, kwargs))
        if "status" in args:
            return CommandResult(tuple(args), 0, SESSION, 0.1)
        if "-O" in args:
            return CommandResult(tuple(args), 0, "Master running (pid=123)", 0.1)
        return CommandResult(
            tuple(args),
            0,
            colab_bridge.RPC_PREFIX
            + json.dumps({"version": 1, "ok": True, "result": {"answer": 42}}),
            0.1,
        )

    monkeypatch.setattr(runner, "_transport", transport)
    assert runner._rpc("compile", files={"source.py": source}) == {"answer": 42}
    assert calls[0][0] == ["colab", "--auth", "oauth2", "status", "-s", "vscode-colab"]
    assert calls[1][0][1:3] == ["-O", "check"]
    args, options = calls[2]
    assert args[0] == "ssh"
    assert args[-2] == "colab-vscode"
    assert "BatchMode=yes" in args
    assert "ConnectionAttempts=1" in args
    assert "ProxyCommand=false" in args
    assert "ControlMaster=no" in args
    assert shlex.split(args[-1])[:4] == ["python3", "-I", "-B", "-c"]
    assert source not in args[-1]
    assert json.loads(options["input_data"])["files"]["source.py"] == source
    assert "--proxy-mode" not in args


@pytest.mark.parametrize(
    "status",
    [
        "[colab] Session 'vscode-colab' not found.",
        "No active sessions",
        "Please sign in",
        SESSION.replace("vscode-colab", "another-session"),
        SESSION.replace("GPU", "CPU"),
        SESSION.replace("IDLE", "UNKNOWN"),
        "",
    ],
)
def test_missing_or_unknown_session_never_starts_ssh(runner, monkeypatch, status):
    calls = []

    def transport(args, **kwargs):
        calls.append(args)
        return CommandResult(tuple(args), 0, status, 0.01)

    monkeypatch.setattr(runner, "_transport", transport)
    with pytest.raises(RunnerUnavailable, match="No runtime was created"):
        runner._rpc("probe", language="cuda_cpp")
    assert len(calls) == 1
    assert "status" in calls[0]


def test_status_is_rechecked_for_every_rpc_and_busy_session_is_allowed(runner, monkeypatch):
    calls = []

    def transport(args, **kwargs):
        calls.append(args)
        output = (
            SESSION.replace("IDLE", "BUSY (1)")
            if "status" in args
            else colab_bridge.RPC_PREFIX + '{"version":1,"ok":true,"result":{}}'
        )
        return CommandResult(tuple(args), 0, output, 0.1)

    monkeypatch.setattr(runner, "_transport", transport)
    runner._rpc("probe")
    runner._rpc("cleanup_all")
    assert sum("status" in call for call in calls) == 2


@pytest.mark.parametrize(
    ("output", "match"),
    [
        ("HTTP 404 SSH not exposed", "does not expose SSH"),
        ("HTTP 429 too many connections", "endpoint is busy"),
        ("Connection closed", "SSH failed"),
    ],
)
def test_connection_errors_are_actionable_without_lifecycle_mutation(
    runner, monkeypatch, output, match
):
    monkeypatch.setattr(runner, "_assert_existing_session", lambda: None)
    monkeypatch.setattr(runner, "_assert_control_master", lambda: None)
    monkeypatch.setattr(
        runner, "_transport", lambda args, **kwargs: CommandResult(tuple(args), 255, output, 0.1)
    )
    with pytest.raises(RunnerUnavailable, match=match):
        runner._rpc("probe")


@pytest.mark.parametrize(
    "response",
    [
        "",
        "noise",
        "MYLEETGPU_RPC=not-json",
        "MYLEETGPU_RPC=[]",
        'MYLEETGPU_RPC={"version":2,"ok":true,"result":{}}',
        'MYLEETGPU_RPC={"version":1,"ok":true,"result":[]}',
        'MYLEETGPU_RPC={"version":1,"ok":false,"error":"probe failed"}',
        "MYLEETGPU_RPC={}\nMYLEETGPU_RPC={}",
    ],
)
def test_malformed_rpc_is_not_success(runner, monkeypatch, response):
    monkeypatch.setattr(runner, "_assert_existing_session", lambda: None)
    monkeypatch.setattr(runner, "_assert_control_master", lambda: None)
    monkeypatch.setattr(
        runner, "_transport", lambda args, **kwargs: CommandResult(tuple(args), 0, response, 0.01)
    )
    with pytest.raises(RunnerUnavailable):
        runner._rpc("probe")


@pytest.mark.parametrize("language", ["cuda_cpp", "triton_python", "torch_python"])
def test_language_probe_records_real_versions_target_and_caches(runner, monkeypatch, language):
    calls = []

    def rpc(action, **payload):
        calls.append((action, payload))
        return probe_data(language)

    monkeypatch.setattr(runner, "_rpc", rpc)
    probe = runner._probe(language, force=False, ignore_circuit_breaker=False)
    assert probe.healthy
    assert probe.backend == language
    assert probe.cuda_image == "colab-native"
    assert probe.image_digest is None
    assert probe.cuda_arch == "75"
    assert probe.toolchain["execution_target"] == "colab"
    assert probe.toolchain["isolation"] == "trusted-native"
    assert len(probe.fingerprint) == 64
    assert runner._probe(language, force=False, ignore_circuit_breaker=False) == probe
    assert len(calls) == 1
    newer = probe_data(language)
    newer["driver_version"] = "600.1"
    monkeypatch.setattr(runner, "_rpc", lambda action, **kwargs: newer)
    assert (
        runner._probe(language, force=True, ignore_circuit_breaker=False).fingerprint
        != probe.fingerprint
    )


def test_incomplete_versions_fail_probe_and_do_not_trip_other_target(runner, monkeypatch):
    broken = probe_data("triton_python")
    broken["toolchain"].pop("triton_version")
    monkeypatch.setattr(runner, "_rpc", lambda action, **kwargs: broken)
    probe = runner.probe_triton_environment()
    assert not probe.healthy
    assert "incomplete toolchain" in probe.error
    assert not runner._health_file.exists()


def test_colab_breaker_and_recovery_are_separate_from_local(runner, monkeypatch):
    runner.mark_unhealthy("GPU failed")
    assert runner._health_file.name == "colab-runner-unhealthy.json"
    assert not (runner.settings.data_dir / "runner-unhealthy.json").exists()
    with pytest.raises(RunnerUnhealthy):
        runner.probe_environment()
    monkeypatch.setattr(
        runner,
        "_rpc",
        lambda action, **kwargs: {"removed": []} if action == "cleanup_all" else probe_data(),
    )
    assert runner.recover().healthy
    assert not runner._health_file.exists()


@pytest.mark.parametrize("language", ["cuda_cpp", "triton_python", "torch_python"])
def test_compile_execute_language_matrix_preserves_harness_policy_contract(
    runner, monkeypatch, language
):
    problem_id = "multi-head-attention" if language == "torch_python" else "vector-addition"
    problem = ProblemCatalog(PROJECT_ROOT / "problems").load().get(problem_id)
    implementation = problem.get_implementation(language)
    task = runner.settings.jobs_dir / str(uuid.uuid4())
    task.mkdir()
    source = task / "snapshot"
    source.write_text(implementation.starter_path.read_text())
    calls = []

    def rpc(action, **payload):
        calls.append((action, payload))
        if action == "probe":
            return probe_data(language)
        if action == "compile":
            return metadata(artifact_exists=True)
        return metadata(output='MYLEETGPU_RESULT={"status":"passed"}\n')

    monkeypatch.setattr(runner, "_rpc", rpc)
    compiled = runner.compile(
        task, problem, source, language=language, implementation=implementation
    )
    assert compiled.succeeded
    assert compiled.executable.is_file()
    payload = next(payload for action, payload in calls if action == "compile")
    assert payload["task"] == task.name
    assert payload["timeout"] <= runner.settings.compile_timeout_seconds
    assert payload["language"] == language
    if language == "cuda_cpp":
        assert set(payload["files"]) == {"source.cu", "platform.cu", "solve.h"}
        assert "-arch=sm_75" in payload["flags"]
        assert json.loads(compiled.executable.read_text())["execution_target"] == "colab"
    else:
        assert set(payload["files"]) == {"source.py", "platform.py", "submission_policy.py"}
        assert "validate_source" in payload["files"]["submission_policy.py"]
    if language == "torch_python":
        assert isinstance(payload["contract"], dict)
        assert payload["contract"]["kind"] == "class"
        assert payload["contract"]["symbol"] == "MultiHeadAttention"
    executed = runner.execute(
        task, compiled.executable, mode="full", timeout_seconds=2, language=language
    )
    assert executed.succeeded
    assert calls[-1][1]["stage"] == "compile-validator"
    assert calls[-1][1]["timeout"] == 2


@pytest.mark.parametrize(
    "change",
    [
        {"returncode": True},
        {"returncode": "0"},
        {"duration_seconds": float("nan")},
        {"duration_seconds": -1},
        {"output": []},
        {"timed_out": 0},
        {"output_limited": None},
    ],
)
def test_malformed_command_metadata_is_rejected(change):
    with pytest.raises(RunnerUnavailable, match="malformed"):
        ColabRunner._command_result(metadata(**change), limit=4096)


def test_cleanup_removes_local_spool_even_when_colab_disconnects(runner, monkeypatch):
    task = runner.settings.jobs_dir / str(uuid.uuid4())
    task.mkdir()
    (task / "source.cu").write_text("private submission")

    def unavailable(*args, **kwargs):
        raise RunnerUnavailable("disconnected")

    monkeypatch.setattr(runner, "_rpc", unavailable)
    with pytest.raises(RunnerUnavailable):
        runner.cleanup_task(task)
    assert not task.exists()
    assert runner.cleanup_orphan_containers() == []
    assert runner.cleanup_owned_containers() == []


def test_cleanup_is_uuid_and_installation_scoped(runner, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        runner, "_rpc", lambda action, **kwargs: calls.append((action, kwargs)) or {"removed": []}
    )
    with pytest.raises(ValueError):
        runner.cleanup_task(tmp_path)
    with pytest.raises(ValueError):
        runner.cleanup_task(runner.settings.jobs_dir / "not-a-uuid")
    assert not calls
    assert runner.cleanup_owned_containers() == []
    runner.assign_owner("worker-1")
    runner.cleanup_owned_containers()
    assert calls[-1][0] == "cleanup_owner"


def test_transport_is_bounded_and_sends_stdin_without_shell():
    result = ColabRunner._transport(
        [sys.executable, "-c", "import sys; print(sys.stdin.read())"],
        timeout=2,
        limit=100,
        input_data=b"$(do-not-execute)",
    )
    assert result.returncode == 0
    assert result.output == "$(do-not-execute)\n"
    result = ColabRunner._transport(
        [sys.executable, "-c", "print('x'*1000000)"], timeout=2, limit=1024
    )
    assert result.output_limited
    assert len(result.output) == 1024
    result = ColabRunner._transport(
        [sys.executable, "-c", "import time;time.sleep(10)"], timeout=0.1, limit=1024
    )
    assert result.timed_out
    assert result.duration_seconds < 2


def test_bridge_is_standalone_python310_and_scrubs_env(tmp_path, monkeypatch):
    source = Path(colab_bridge.__file__).read_text()
    ast.parse(source, feature_version=(3, 10))
    assert "myleetgpu" not in [
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module
    ]
    monkeypatch.setenv("GOOGLE_TOKEN", "must-not-leak")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-leak")
    env = colab_bridge.child_environment(tmp_path)
    assert "GOOGLE_TOKEN" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "SSH_AUTH_SOCK" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["HOME"] == str(tmp_path)


@pytest.mark.parametrize(
    "root",
    [
        "/",
        "/content",
        "/content/project",
        "/tmp/task-root",
        "/content/project/../other",
        "/content/project//tasks",
    ],
)
def test_bridge_rejects_broad_and_traversing_roots(root):
    with pytest.raises(ValueError):
        colab_bridge.installation_root({"root": root, "installation": "a" * 16})


def test_bridge_rejects_symlink_paths(tmp_path):
    destination = tmp_path / "real"
    destination.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(destination, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        colab_bridge.no_symlinks(alias / "stage")


def test_bridge_supervisor_real_timeout_output_bound_and_secret_scrub(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_TOKEN", "secret")
    command = [
        sys.executable,
        "-I",
        "-c",
        "import os;print(os.environ.get('GOOGLE_TOKEN','absent'))",
    ]
    result = colab_bridge.supervise(command, tmp_path, tmp_path / "scratch1", 2, 1024)
    assert result["returncode"] == 0
    assert result["output"] == "absent\n"
    result = colab_bridge.supervise(
        [sys.executable, "-c", "print('a'*20000)"], tmp_path, tmp_path / "scratch2", 2, 1024
    )
    assert result["output_limited"]
    assert len(result["output"]) == 1024
    result = colab_bridge.supervise(
        [sys.executable, "-c", "import time;time.sleep(10)"],
        tmp_path,
        tmp_path / "scratch3",
        0.1,
        1024,
        tmp_path,
    )
    assert result["timed_out"]
    assert result["duration_seconds"] < 2
    assert not list(tmp_path.glob(".process-*.json"))


@pytest.fixture
def bridge_request(tmp_path, monkeypatch):
    root = tmp_path / "installation"
    root.mkdir()
    monkeypatch.setattr(colab_bridge, "installation_root", lambda request: root)
    return root, {
        "version": 1,
        "action": "compile",
        "owner": "b" * 16,
        "installation": "a" * 16,
        "task": str(uuid.uuid4()),
        "limit": 4096,
        "timeout": 2,
        "stage": "compile-validator",
        "language": "triton_python",
    }


@pytest.mark.parametrize("language", ["triton_python", "torch_python"])
def test_bridge_real_python_dispatch_paths_contract_and_cleanup(bridge_request, language):
    root, request = bridge_request
    request["language"] = language
    request["contract"] = {"expected_parameters": ["x"]}
    request["files"] = {
        "source.py": "def solve(x): return x\n",
        "submission_policy.py": "import pathlib,sys,json\n"
        "assert pathlib.Path(sys.argv[1]).name=='source.py'\n"
        "if len(sys.argv)>2: assert json.loads(sys.argv[2])['expected_parameters']==['x']\n",
        "platform.py": "from pathlib import Path\n"
        "assert Path('/work/submission_policy.py').is_file()\n"
        'print(\'MYLEETGPU_RESULT={"status":"passed"}\')\n',
    }
    compiled = colab_bridge.dispatch(request)
    assert compiled["returncode"] == 0, compiled["output"]
    assert compiled["artifact_exists"]
    stage = root / request["task"] / request["stage"]
    assert (stage / "source.py").read_text() == request["files"]["source.py"]
    assert "/work/" not in (stage / "platform.py").read_text()
    executed = colab_bridge.dispatch({**request, "action": "execute", "mode": "public"})
    assert executed["returncode"] == 0, executed["output"]
    assert '"status":"passed"' in executed["output"]
    assert not list((root / request["task"]).glob("scratch-*"))
    removed = colab_bridge.dispatch({**request, "action": "cleanup"})
    assert removed == {"removed": [request["task"]]}
    assert not stage.exists()


def test_bridge_real_policy_rejects_unsafe_source_before_framework_import(bridge_request):
    _, request = bridge_request
    request["files"] = {
        "source.py": "import os\nos.system('echo forbidden')\n",
        "submission_policy.py": (
            PROJECT_ROOT / "backend/myleetgpu/runner/submission_policy.py"
        ).read_text(),
        "platform.py": "raise RuntimeError('not reached')\n",
    }
    result = colab_bridge.dispatch(request)
    assert result["returncode"] != 0
    assert not result["artifact_exists"]
    assert "submission policy" in result["output"]


def test_bridge_cuda_compile_arguments_and_artifact_without_local_gpu(bridge_request, monkeypatch):
    root, request = bridge_request
    request.update(
        language="cuda_cpp",
        flags=["-O3", "-std=c++17", "-arch=sm_75"],
        files={"source.cu": "source", "solve.h": "header", "platform.cu": "harness"},
    )
    calls = []

    def supervise(command, cwd, scratch, timeout, limit, record_dir=None):
        calls.append(command)
        scratch.mkdir()
        (scratch / "program").write_bytes(b"binary")
        return metadata()

    monkeypatch.setattr(colab_bridge, "supervise", supervise)
    result = colab_bridge.dispatch(request)
    assert result["artifact_exists"]
    assert calls[0][:4] == ["nvcc", "-O3", "-std=c++17", "-arch=sm_75"]
    assert (root / request["task"] / "compile-validator/program").read_bytes() == b"binary"
    colab_bridge.dispatch({**request, "action": "execute", "mode": "benchmark"})
    assert len(calls[-1]) == 1  # CUDA benchmark has no --mode argument.


@pytest.mark.parametrize(
    "change",
    [
        {"action": "shell"},
        {"task": "../../outside"},
        {"stage": "../bad"},
        {"files": {"../source.py": "x"}},
        {"timeout": float("nan")},
        {"limit": 0},
    ],
)
def test_bridge_rejects_untrusted_rpc_paths_actions_and_limits(bridge_request, change):
    _, request = bridge_request
    with pytest.raises((ValueError, TypeError)):
        colab_bridge.dispatch({**request, **change})


def test_bridge_cleanup_never_touches_other_owner_or_unrelated_files(bridge_request):
    root, request = bridge_request
    task = colab_bridge.task_directory(root, request, create=True)
    (task / "source.cu").write_text("keep me")
    unrelated = root / "important-notebook.ipynb"
    unrelated.write_text("not owned")
    response = colab_bridge.dispatch({**request, "action": "cleanup_owner", "owner": "c" * 16})
    assert response["removed"] == []
    assert task.exists()
    response = colab_bridge.dispatch({**request, "action": "cleanup_all"})
    assert response["removed"] == [request["task"]]
    assert unrelated.read_text() == "not owned"


def test_privilege_drop_clears_groups_before_switching_uid(monkeypatch):
    calls = []
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    monkeypatch.setattr(os, "setgroups", lambda value: calls.append(("groups", value)))
    monkeypatch.setattr(os, "setgid", lambda value: calls.append(("gid", value)))
    monkeypatch.setattr(os, "setuid", lambda value: calls.append(("uid", value)))
    monkeypatch.setattr(colab_bridge.resource, "setrlimit", lambda *args: None)
    colab_bridge.drop_privileges()
    assert calls == [("groups", []), ("gid", 65534), ("uid", 65534)]


@pytest.mark.parametrize("language", ["cuda_cpp", "triton_python", "torch_python"])
def test_recover_selected_language_clears_only_colab_breaker_after_healthy_probe(
    runner, monkeypatch, language
):
    local_breaker = runner.settings.data_dir / "runner-unhealthy.json"
    local_breaker.write_text('{"reason":"local GPU failed"}')
    runner.mark_unhealthy("previous Colab GPU failure")

    def unavailable(action, **payload):
        if action == "cleanup_all":
            return {"removed": []}
        assert payload["language"] == language
        raise RunnerUnavailable("still unavailable")

    monkeypatch.setattr(runner, "_rpc", unavailable)
    with pytest.raises(RunnerUnavailable, match="still unavailable"):
        runner.recover(language=language)
    assert runner._health_file.exists()
    monkeypatch.setattr(
        runner,
        "_rpc",
        lambda action, **payload: {"removed": []}
        if action == "cleanup_all"
        else probe_data(language),
    )
    assert runner.recover(language=language).backend == language
    assert not runner._health_file.exists()
    assert local_breaker.exists()


def test_missing_control_master_never_starts_bridge(runner, monkeypatch):
    calls = []

    def transport(args, **kwargs):
        calls.append(args)
        if "status" in args:
            return CommandResult(tuple(args), 0, SESSION, 0.1)
        assert "-O" in args
        assert "ProxyCommand=false" in args
        return CommandResult(tuple(args), 255, "Control socket does not exist", 0.1)

    monkeypatch.setattr(runner, "_transport", transport)
    with pytest.raises(RunnerUnavailable, match="No reusable Colab SSH ControlMaster"):
        runner._rpc("probe")
    assert len(calls) == 2
    assert all("--proxy-mode" not in call for call in calls)


def test_master_disappearing_after_check_fails_closed_and_redacts_proxy_diagnostics(
    runner, monkeypatch
):
    calls = []

    def transport(args, **kwargs):
        calls.append(args)
        if "status" in args:
            return CommandResult(tuple(args), 0, SESSION, 0.1)
        if "-O" in args:
            return CommandResult(tuple(args), 0, "Master running (pid=123)", 0.1)
        assert "ProxyCommand=false" in args
        assert "ControlMaster=no" in args
        return CommandResult(tuple(args), 255, "secret-token https://auth/?code=123", 0.1)

    monkeypatch.setattr(runner, "_transport", transport)
    with pytest.raises(RunnerUnavailable) as raised:
        runner._rpc("probe")
    assert "secret-token" not in str(raised.value)
    assert "code=123" not in str(raised.value)
    assert len(calls) == 3


def test_nested_version_subtask_compile_execute_stays_within_uuid(runner, monkeypatch):
    problem = ProblemCatalog(PROJECT_ROOT / "problems").load().get("vector-addition")
    implementation = problem.get_implementation("cuda_cpp")
    job = runner.settings.jobs_dir / str(uuid.uuid4())
    version = job / "version-2"
    version.mkdir(parents=True)
    source = version / "source.cu"
    source.write_text(implementation.starter_path.read_text())
    calls = []

    def rpc(action, **payload):
        calls.append((action, payload))
        if action == "probe":
            return probe_data()
        if action == "compile":
            return metadata(artifact_exists=True)
        return metadata(output='MYLEETGPU_RESULT={"status":"passed"}')

    monkeypatch.setattr(runner, "_rpc", rpc)
    compiled = runner.compile(version, problem, source, implementation=implementation)
    assert compiled.succeeded
    result = runner.execute(version, compiled.executable, mode="benchmark", timeout_seconds=2)
    assert result.succeeded
    for action, payload in calls:
        if action in {"compile", "execute"}:
            assert payload["task"] == job.name
            assert payload["subtask"] == "version-2"
    with pytest.raises(ValueError, match="top-level"):
        runner.cleanup_task(version)
    for bad in [job / "version-0", job / "version-1" / "deeper", job / "other"]:
        with pytest.raises(ValueError):
            runner._task_id(bad)


def test_bridge_nested_versions_have_independent_stages_and_shared_cleanup(bridge_request):
    root, request = bridge_request
    request["files"] = {
        "source.py": "def solve(x): return x\n",
        "submission_policy.py": "print('policy checked')\n",
        "platform.py": 'print(\'MYLEETGPU_RESULT={"status":"passed"}\')\n',
    }
    for subtask in ["version-1", "version-2"]:
        result = colab_bridge.dispatch({**request, "subtask": subtask})
        assert result["artifact_exists"]
        assert (root / request["task"] / subtask / "compile-validator/source.py").is_file()
    with pytest.raises(ValueError, match="subtask"):
        colab_bridge.dispatch({**request, "subtask": "../other"})
    assert colab_bridge.dispatch({**request, "action": "cleanup"})["removed"] == [request["task"]]
    assert not (root / request["task"]).exists()


def test_recovery_does_not_probe_or_clear_breaker_when_orphan_cleanup_fails(runner, monkeypatch):
    runner.mark_unhealthy("connection lost")
    calls = []

    def rpc(action, **kwargs):
        calls.append(action)
        raise RunnerUnavailable("cleanup disconnected")

    monkeypatch.setattr(runner, "_rpc", rpc)
    with pytest.raises(RunnerUnavailable, match="cleanup disconnected"):
        runner.recover(language="torch_python")
    assert calls == ["cleanup_all"]
    assert runner._health_file.exists()


def test_bridge_timeout_terminates_spawned_descendants(tmp_path):
    code = (
        "import subprocess,sys,time\n"
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'])\n"
        "print(p.pid,flush=True)\ntime.sleep(20)\n"
    )
    result = colab_bridge.supervise(
        [sys.executable, "-I", "-c", code], tmp_path, tmp_path / "scratch", 0.2, 1024, tmp_path
    )
    assert result["timed_out"]
    child_pid = int(result["output"].strip())
    # An orphan may briefly remain a zombie until init reaps it, but must never
    # remain running after the supervisor returns.
    checked = subprocess.run(
        ["ps", "-o", "state=", "-p", str(child_pid)],
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )
    assert not checked.stdout.strip() or checked.stdout.lstrip().startswith("Z")
    assert not list(tmp_path.glob(".process-*.json"))


def test_bridge_cleanup_checks_process_start_time_before_killing(bridge_request, monkeypatch):
    root, request = bridge_request
    task = colab_bridge.task_directory(root, request, create=True)
    (task / ".process-current.json").write_text(json.dumps({"pid": 1001, "start": "10"}))
    (task / ".process-reused.json").write_text(json.dumps({"pid": 1002, "start": "15"}))
    (task / ".process-unknown.json").write_text(json.dumps({"pid": 1003, "start": None}))
    killed = []
    monkeypatch.setattr(colab_bridge, "process_start", lambda pid: "10")
    monkeypatch.setattr(colab_bridge, "kill_group", killed.append)
    colab_bridge.dispatch({**request, "action": "cleanup"})
    assert killed == [1001]

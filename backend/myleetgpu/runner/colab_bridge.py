"""Single-request, stdlib-only Colab RPC supervisor (Python 3.10+).

This file is sent as code to an existing SSH runtime. It is intentionally not a
sandbox: only trusted submissions may run here. It never manages Colab sessions,
installs packages, reads credentials, or accepts a shell command from the client.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import resource
import selectors
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

RPC_PREFIX = "MYLEETGPU_RPC="
MAX_REQUEST_BYTES = 4 * 1024 * 1024
LANGUAGES = {"cuda_cpp", "triton_python", "torch_python"}
STAGES = {"compile-validator", "compile-benchmark"}
SAFE_PATH = "/usr/local/cuda/bin:/usr/lib64-nvidia:/usr/local/bin:/usr/bin:/bin"


def bounded_number(value, low, high):
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("invalid numeric limit")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError("numeric limit is out of range")
    return value


def check_uuid(value):
    if not isinstance(value, str) or str(uuid.UUID(value)) != value:
        raise ValueError("task must be a canonical UUID")
    return value


def check_hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{16}", value):
        raise ValueError("invalid installation or owner identifier")
    return value


def no_symlinks(path):
    for parent in (*reversed(path.parents), path):
        if parent.is_symlink():
            raise ValueError("remote work paths must not contain symbolic links")


def installation_root(request):
    raw = request.get("root")
    if not isinstance(raw, str) or not re.fullmatch(r"/content/[a-zA-Z0-9_./-]+", raw):
        raise ValueError("remote root must be a dedicated directory below /content")
    path = Path(raw)
    if len(path.parts) < 4 or any(part in {".", ".."} for part in raw.split("/")):
        raise ValueError("remote root must be a dedicated directory below /content")
    if str(path) != raw:
        raise ValueError("remote root must be canonical")
    path = path / check_hash(request.get("installation"))
    no_symlinks(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o711)
    if path.stat().st_uid != os.geteuid():
        raise ValueError("remote installation belongs to a different user")
    path.chmod(0o711)
    return path


def child_environment(scratch):
    """Allowlist; never forward Colab/OAuth/SSH/cloud credential variables."""
    return {
        "PATH": SAFE_PATH,
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/lib64-nvidia:"
        "/usr/local/nvidia/lib64:/usr/local/nvidia/lib",
        "LANG": "C.UTF-8",
        "HOME": str(scratch),
        "TMPDIR": str(scratch),
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "TRITON_CACHE_DIR": str(scratch / "triton"),
        "XDG_CACHE_HOME": str(scratch / "cache"),
        "TORCHINDUCTOR_CACHE_DIR": str(scratch / "inductor"),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "NVIDIA_TF32_OVERRIDE": "0",
    }


def drop_privileges():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
    if os.geteuid() == 0:
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)


def process_start(pid):
    try:
        # comm may itself contain spaces and parentheses.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19]
    except (OSError, IndexError):
        return None


def kill_group(pid):
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGKILL)


def supervise(command, cwd, scratch, timeout, limit, record_dir=None):
    """Bound time/output and kill the whole session even if the leader exits."""
    start = time.monotonic()
    output = bytearray()
    timed_out = limited = False
    scratch.mkdir(mode=0o700)
    if os.geteuid() == 0:
        os.chown(scratch, 65534, 65534)
    record = None
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=child_environment(scratch),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=False,
        start_new_session=True,
        preexec_fn=drop_privileges,
    )
    try:
        if record_dir is not None:
            record = record_dir / f".process-{uuid.uuid4()}.json"
            record.write_text(json.dumps({"pid": process.pid, "start": process_start(process.pid)}))
            record.chmod(0o600)
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                if time.monotonic() - start >= timeout:
                    timed_out = True
                    break
                for key, _ in selector.select(
                    min(0.05, max(0, timeout - time.monotonic() + start))
                ):
                    chunk = os.read(key.fileobj.fileno(), 8192)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    remaining = limit - len(output)
                    output.extend(chunk[:remaining])
                    if len(chunk) > remaining:
                        limited = True
                        break
                if limited:
                    break
                # A descendant may retain stdout after the leader has exited.
                if process.poll() is not None and not selector.select(0):
                    break
        if not timed_out and not limited:
            try:
                process.wait(timeout=max(0.01, timeout - time.monotonic() + start))
            except subprocess.TimeoutExpired:
                timed_out = True
    finally:
        kill_group(process.pid)
        process.wait(timeout=5)
        process.stdout.close()
        if record is not None:
            record.unlink(missing_ok=True)
    return {
        "returncode": process.returncode,
        "output": output.decode("utf-8", errors="replace"),
        "duration_seconds": time.monotonic() - start,
        "timed_out": timed_out,
        "output_limited": limited,
    }


def check_result(result, description):
    if result["returncode"] != 0 or result["timed_out"] or result["output_limited"]:
        raise RuntimeError(f"{description}: {result['output'][:1500] or 'failed or timed out'}")
    return result["output"].strip()


def run_probe(root, language, limit):
    probe_dir = root / f"probe-{uuid.uuid4()}"
    probe_dir.mkdir(mode=0o755)
    counter = 0

    def run(command, timeout=30):
        nonlocal counter
        counter += 1
        return supervise(command, probe_dir, probe_dir / f"scratch-{counter}", timeout, limit)

    try:
        description = check_result(
            run(
                [
                    "nvidia-smi",
                    "--id=0",
                    "--query-gpu=name,driver_version,compute_cap",
                    "--format=csv,noheader,nounits",
                ],
                15,
            ),
            "Colab GPU probe failed",
        )
        gpu = [field.strip() for field in description.splitlines()[0].split(",")]
        if len(gpu) != 3 or not re.fullmatch(r"\d+\.\d+", gpu[2]):
            raise RuntimeError("unexpected nvidia-smi GPU description")
        telemetry = dict.fromkeys(("temperature_c", "power_w", "sm_clock_mhz", "gpu_busy_percent"))
        measured = run(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=temperature.gpu,power.draw,clocks.sm,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            10,
        )
        if measured["returncode"] == 0:
            fields = measured["output"].strip().split(",")
            if len(fields) == len(telemetry):
                telemetry = dict(zip(telemetry, (field.strip() for field in fields), strict=True))
        if language == "cuda_cpp":
            nvcc = check_result(run(["nvcc", "--version"], 15), "Colab NVCC unavailable")
            nvcc = next((line.strip() for line in nvcc.splitlines() if "release" in line), nvcc)
            code = (
                "#include <cuda_runtime.h>\n#include <cstdio>\n"
                "__global__ void k(int* p){*p=7;}\n"
                "int main(){int *p,h=0,v=0;if(cudaMalloc(&p,4)!=cudaSuccess)return 1;"
                "k<<<1,1>>>(p);if(cudaMemcpy(&h,p,4,cudaMemcpyDeviceToHost)!=cudaSuccess)return 2;"
                'cudaFree(p);cudaRuntimeGetVersion(&v);printf("%d",v);return h==7?0:3;}'
            )
            (probe_dir / "probe.cu").write_text(code)
            (probe_dir / "probe.cu").chmod(0o444)
            # NVCC writes only to the unprivileged scratch directory.
            compiled = run(
                ["nvcc", str(probe_dir / "probe.cu"), "-o", str(probe_dir / "scratch-4" / "probe")],
                45,
            )
            check_result(compiled, "Colab CUDA smoke compilation failed")
            runtime = check_result(
                run([str(probe_dir / "scratch-4" / "probe")], 15),
                "Colab CUDA smoke execution failed",
            )
            version = int(runtime)
            toolchain = {
                "nvcc_version": nvcc,
                "cuda_runtime_version": f"{version // 1000}.{version % 1000 // 10}",
            }
        else:
            code = (
                "import json,platform,torch\n"
                "assert torch.cuda.is_available(), 'CUDA is unavailable'\n"
                "a=torch.arange(16,device='cuda',dtype=torch.float32)\n"
                "assert a.sum().item()==120\n"
                "v={'python_version':platform.python_version(),'torch_version':torch.__version__,"
                "'torch_cuda_version':torch.version.cuda}\n"
            )
            if language == "triton_python":
                code += (
                    "import triton\nimport triton.language as tl\n"
                    "@triton.jit\ndef smoke(X,Y):\n x=tl.arange(0,16)\n"
                    " tl.store(Y+x,tl.load(X+x)+1)\n"
                    "b=torch.empty_like(a)\nsmoke[(1,)](a,b)\n"
                    "assert torch.equal(b,a+1)\nv['triton_version']=triton.__version__\n"
                )
            code += "torch.cuda.synchronize()\nprint(json.dumps(v))\n"
            (probe_dir / "probe.py").write_text(code)
            (probe_dir / "probe.py").chmod(0o444)
            toolchain = json.loads(
                check_result(
                    run([sys.executable, "-I", "-B", str(probe_dir / "probe.py")], 60),
                    "Colab Python CUDA smoke test failed",
                )
            )
        return {
            "gpu_name": gpu[0],
            "driver_version": gpu[1],
            "compute_capability": gpu[2],
            "toolchain": toolchain,
            "telemetry": telemetry,
        }
    finally:
        shutil.rmtree(probe_dir)


def task_directory(root, request, create=False):
    task = root / check_uuid(request.get("task"))
    no_symlinks(task)
    owner = check_hash(request.get("owner"))
    if create:
        task.mkdir(mode=0o755, exist_ok=True)
        ownership = task / ".owner.json"
        if not ownership.exists():
            ownership.write_text(json.dumps({"owner": owner}))
            ownership.chmod(0o600)
    if task.exists() and json.loads((task / ".owner.json").read_text()).get("owner") != owner:
        raise ValueError("remote task belongs to another worker")
    return task


def cleanup_directory(task):
    no_symlinks(task)
    for record in task.glob(".process-*.json"):
        if record.is_symlink():
            raise ValueError("process record must not be a symlink")
        saved = json.loads(record.read_text())
        pid = saved.get("pid")
        if (
            isinstance(pid, int)
            and pid > 1
            and saved.get("start") is not None
            and process_start(pid) == saved["start"]
        ):
            kill_group(pid)
    shutil.rmtree(task)


def dispatch(request):
    if not isinstance(request, dict) or request.get("version") != 1:
        raise ValueError("unsupported RPC request")
    action = request.get("action")
    if action not in {"probe", "compile", "execute", "cleanup", "cleanup_owner", "cleanup_all"}:
        raise ValueError("unsupported RPC action")
    root = installation_root(request)
    if action.startswith("cleanup"):
        removed = []
        owner = check_hash(request.get("owner"))
        if action == "cleanup":
            candidates = [task_directory(root, request)]
        else:
            candidates = [
                path
                for path in root.iterdir()
                if re.fullmatch(r"[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}", path.name)
            ]
        for task in candidates:
            if not task.exists():
                continue
            no_symlinks(task)
            saved = json.loads((task / ".owner.json").read_text())
            if action == "cleanup_owner" and saved.get("owner") != owner:
                continue
            cleanup_directory(task)
            removed.append(task.name)
        return {"removed": removed}
    language = request.get("language")
    if language not in LANGUAGES:
        raise ValueError("unknown runner language")
    limit = int(bounded_number(request.get("limit"), 1024, 1048576))
    if action == "probe":
        return run_probe(root, language, limit)
    timeout = bounded_number(request.get("timeout"), 0.01, 3600)
    stage_name = request.get("stage")
    if stage_name not in STAGES:
        raise ValueError("invalid compilation stage")
    task = task_directory(root, request, create=action == "compile")
    subtask = request.get("subtask")
    if subtask is not None and (
        not isinstance(subtask, str) or not re.fullmatch(r"version-[1-9]\d{0,5}", subtask)
    ):
        raise ValueError("invalid version subtask")
    work = task if subtask is None else task / subtask
    no_symlinks(work)
    if action == "compile" and subtask is not None:
        work.mkdir(mode=0o755, exist_ok=True)
    stage = work / stage_name
    no_symlinks(stage)
    scratch = task / f"scratch-{uuid.uuid4()}"
    if action == "compile":
        files = request.get("files")
        names = (
            {"source.cu", "solve.h", "platform.cu"}
            if language == "cuda_cpp"
            else {"source.py", "platform.py", "submission_policy.py"}
        )
        if not isinstance(files, dict) or set(files) != names:
            raise ValueError("unexpected compilation files")
        if any(not isinstance(data, str) for data in files.values()):
            raise ValueError("compilation files must be UTF-8 text")
        if sum(len(data.encode("utf-8")) for data in files.values()) > MAX_REQUEST_BYTES // 2:
            raise ValueError("compilation files exceed transfer limit")
        stage.mkdir(mode=0o755)
        for name, data in files.items():
            if name == "platform.py":
                # Only platform-owned harness paths are adapted; submissions are unchanged.
                data = data.replace("/work/", f"{stage}/")
            (stage / name).write_text(data, encoding="utf-8")
            (stage / name).chmod(0o444)
        if language == "cuda_cpp":
            flags = request.get("flags")
            if not isinstance(flags, list) or not all(isinstance(flag, str) for flag in flags):
                raise ValueError("invalid compilation flags")
            command = [
                "nvcc",
                *flags,
                f"-I{stage}",
                str(stage / "source.cu"),
                str(stage / "platform.cu"),
                "-o",
                str(scratch / "program"),
            ]
        else:
            command = [
                sys.executable,
                "-I",
                "-B",
                str(stage / "submission_policy.py"),
                str(stage / "source.py"),
            ]
            if language == "torch_python":
                contract = request.get("contract")
                if not isinstance(contract, dict):
                    raise ValueError("PyTorch submission contract is required")
                command.append(json.dumps(contract))
    else:
        mode = request.get("mode")
        if mode not in {"public", "full", "benchmark"}:
            raise ValueError("unknown execution mode")
        if not stage.is_dir():
            raise ValueError("remote compilation artifact is missing; compile again")
        if language == "cuda_cpp":
            command = [str(stage / "program")]
            if mode != "benchmark":
                command += ["--mode", mode]
        else:
            command = [sys.executable, "-I", "-B", str(stage / "platform.py"), "--mode", mode]
    try:
        result = supervise(command, stage, scratch, timeout, limit, task)
        if action == "compile":
            succeeded = (
                result["returncode"] == 0
                and not result["timed_out"]
                and not result["output_limited"]
            )
            if language == "cuda_cpp":
                binary = scratch / "program"
                succeeded = succeeded and binary.is_file() and not binary.is_symlink()
                if succeeded:
                    shutil.move(str(binary), stage / "program")
                    (stage / "program").chmod(0o555)
            result["artifact_exists"] = succeeded
        return result
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)


def main():
    # Keep the bounded supervisor alive if the SSH client disappears. Submitted
    # process groups still terminate on their original wall-clock deadline.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if len(raw) > MAX_REQUEST_BYTES:
            raise ValueError("RPC request exceeds transfer limit")
        result = dispatch(json.loads(raw))
        response = {"version": 1, "ok": True, "result": result}
    except Exception as error:
        response = {"version": 1, "ok": False, "error": str(error)[:2000]}
    print(RPC_PREFIX + json.dumps(response, ensure_ascii=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

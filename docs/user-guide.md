# MyLeetGpu 用户指南

## 1. 使用前必读

MyLeetGpu 是供**单机可信操作者**使用的 GPU 编程、正确性验证与性能比较工具。它把 CUDA C++、Triton (Python) Kernel 和高层 PyTorch (Python) 作为三种一等实现类型：前两种用于练习自定义 Kernel，PyTorch 用于练习由基础 Tensor 运算组合出的模型算法。平台会用容器限制提交代码的文件、网络和进程权限，但 Docker 和 RTX 4060 等消费级 GPU 无法提供面向公网不可信租户的强 GPU/显存隔离。

- 默认只通过 `http://localhost:3000` 或 `http://127.0.0.1:3000` 访问。
- 只有受信任的家庭/实验室局域网才可使用第 4.2 节的认证 LAN overlay；不要手改成 `0.0.0.0:3000`。
- 不要使用公网隧道、路由器端口转发、公共 Wi-Fi 或把 LAN 地址发布到公网。
- 不要让不可信的远程用户提交代码。
- 不要给用户容器挂载 Docker socket、仓库、数据库或任意宽泛主机目录。

直接运行 FastAPI 时默认监听 `127.0.0.1:8000`。Compose 中 API 为了让 Nginx 访问，会在项目网络内监听 `0.0.0.0:8000`，但**没有向宿主发布 8000**；这不是允许把 API 暴露到外部的例外。

CUDA 每次编译使用无 GPU 的一次性容器和当前 Job 的精确可写编译目录；Triton 与 PyTorch 的 Python 语法及版本化策略预检分别使用 `restricted_triton_v2` 和 `restricted_torch_v2`，同样在无 GPU 容器中执行，但 `/work` 只读。运行/验证/benchmark 使用另一个一次性容器，只读挂载该语言的最小 artifact，并只获得 GPU 0。所有提交容器均无网络、非 root、只读根文件系统、丢弃 capabilities、启用 `no-new-privileges`，并保留 Docker 默认内置 seccomp profile。Triton 因为必须 JIT，只在容器内获得 512 MiB 的临时可执行 `/tmp` 缓存；PyTorch 获得同尺寸但 `noexec` 的临时区，不允许运行时编译。这些缓存均不持久化。

项目没有 Debug 功能：不提供断点、单步、变量监视、cuda-gdb、Nsight、Profiler 或 PTX/汇编查看。编译/代码检查错误、运行错误、错误答案、超时和受限 stdout/stderr 会正常显示。

## 2. 前置条件

### 2.1 硬件和系统

- Windows 10/11，已启用硬件虚拟化和 WSL2；发行版建议使用受支持的 Ubuntu LTS。
- NVIDIA RTX 4060 或其他兼容 CUDA 的 NVIDIA GPU。RTX 4060 通常是 Compute Capability 8.9（`sm_89`），但 MyLeetGpu 以 `make doctor` 的实际探测为准，不依赖手工猜测。
- 足够的磁盘空间用于固定 CUDA 镜像、固定 PyTorch/Triton devel 镜像、构建缓存和本地数据。

在 **Windows PowerShell** 检查 WSL：

```powershell
wsl --status
wsl --list --verbose
```

目标发行版的 VERSION 必须为 `2`。如果仍为 WSL 1，在有权限的 PowerShell 中执行：

```powershell
wsl --set-version <发行版名称> 2
wsl --update
```

系统级变更可能要求管理员权限并需要重新启动；项目不会自动修改 Windows 驱动或系统设置。

安装或升级步骤以 [Microsoft 的 WSL 安装说明](https://learn.microsoft.com/windows/wsl/install) 为准。

### 2.2 NVIDIA 驱动

安装支持 WSL CUDA 的当前 Windows NVIDIA 驱动。CUDA 设备通过 Windows 主机驱动映射到 WSL2。

> 不要在 WSL 发行版内安装 Linux NVIDIA 显示驱动，也不要用 Linux 驱动覆盖 WSL 映射的驱动。需要更新驱动时，在 Windows 中使用 NVIDIA 官方驱动安装程序。

NVIDIA 的 [CUDA on WSL 用户指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) 明确区分 Windows 主机驱动和可选的 WSL CUDA Toolkit；MyLeetGpu 的 NVCC 位于固定容器中，通常不需要在 WSL 宿主另装 Toolkit。

分别检查 Windows 和 WSL 中的可见性：

```powershell
nvidia-smi
wsl nvidia-smi
```

某些 WSL 环境中的 `nvidia-smi` 位于 `/usr/lib/wsl/lib/nvidia-smi`；请确保它已位于 `PATH`（Docker Desktop/WSL 通常会自动配置），`make doctor` 按 `PATH` 调用并校验退出码。温度、时钟、功耗或 GPU busy 在 WSL 中显示 unavailable 并不一定是故障。

`nvidia-smi` 顶部的 “CUDA Version” 表示当前驱动支持的最高 CUDA API 级别，不等同于容器内的 CUDA Runtime 或 `nvcc` 版本；环境页会分别记录这些值。

### 2.3 Docker 与 NVIDIA Container Toolkit

推荐使用最新版 Docker Desktop，启用：

1. “Use the WSL 2 based engine”；
2. 当前 Ubuntu 发行版的 WSL Integration；
3. Docker Compose v2。

Docker Desktop 的 WSL2 GPU 支持通常随 Desktop 集成提供。若改用“直接安装在 WSL 内的 Docker Engine”，则还必须按 NVIDIA 官方说明安装 `nvidia-container-toolkit` 并为该 Docker daemon 配置 NVIDIA runtime；不要把 Windows Docker Desktop 和 WSL 内独立 daemon 的配置混在一起。

参考 [Docker Desktop GPU 支持](https://docs.docker.com/desktop/features/gpu/) 和 [NVIDIA Container Toolkit 安装指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。后者会随版本更新，执行管理员命令前应重新核对官方步骤，不要从本文复制过时的软件版本号。

在 WSL 终端检查：

```bash
docker version
docker compose version
docker info
```

还需要 Git、GNU Make 和常用 shell 工具。仅用 Compose 启动时不要求宿主机自行安装 NVCC、PyTorch 或 Triton：CUDA 工具链来自固定的 `nvidia/cuda:12.4.1-devel-ubuntu22.04`；Triton 与 PyTorch 实现共用固定的官方 `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel@sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014` 镜像，当前实际包含 Python 3.11.10、PyTorch 2.5.1 + CUDA 12.4 和 Triton 3.1.0。若要在宿主执行 `make lint`、`make test`、`make e2e`、`make clean-jobs` 或 `make recover-runner`，还需 Python 3.12+、Node.js 20+ 和项目 lockfile 对应的包管理器；运行 `make install` 安装锁定的开发依赖。宿主开发 Python 与 Python Runner 内的 Python 版本是两个独立环境。

项目使用镜像内随 PyTorch 固定的 Triton，三种实现都不支持在提交中安装包，也不建议进入 Worker/API 容器执行 `pip install`。学习语言与核对上游兼容性时，请使用 [Triton 官方安装说明](https://triton-lang.org/main/getting-started/installation.html)、[Triton Vector Addition 教程](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html) 和 [PyTorch 官方版本页](https://docs.pytorch.org/get-started/previous-versions/)。

## 3. 安装与环境诊断

以下命令在 **WSL2 终端** 中执行。为了更好的文件 I/O 性能，建议把仓库克隆到 WSL 的 Linux 文件系统（例如 `~/src`），而不是 `/mnt/c` 或 `/mnt/d`。

```bash
git clone git@github.com:BabyDrangoner/MyLeetGpu.git
cd MyLeetGpu
cp .env.example .env
make doctor
```

需要参与开发或运行测试时，再执行：

```bash
make install
```

不要把 `.env` 提交到 Git。不要修改基础 Compose 的 `127.0.0.1:3000:8080` 发布规则或给 API 添加 `ports`；需要 LAN 时只使用仓库提供的 `docker-compose.lan.yml`。`MYLEETGPU_API_HOST=0.0.0.0` 只在 Compose 后端容器中设置，用于容器间通信；直接启动 API 的默认值仍为 `127.0.0.1`。

`make doctor` 应依次检查并实际报告：

- 当前内核是否为 WSL2；
- Windows NVIDIA 驱动是否映射到 WSL，`nvidia-smi` 是否可用；
- Docker daemon、Compose 和 NVIDIA 容器运行时是否可用；
- 固定的 `nvidia/cuda:12.4.1-devel-ubuntu22.04` 容器是否能看到目标 GPU；
- 最小 `.cu` 程序能否在容器中用 NVCC 编译并在 GPU 上运行；
- GPU 型号、Compute Capability、驱动、CUDA Runtime、NVCC 版本和镜像 RepoDigest；
- 固定 PyTorch/Triton 镜像能否分别加载纯 PyTorch 工具链与 Triton 工具链，是否只看到 GPU 0；
- 最小 PyTorch attention 能否在真实 GPU 上运行，以及 Python/PyTorch/Torch CUDA 版本和镜像 RepoDigest；
- 最小 Triton Kernel 能否在真实 GPU 上完成 JIT 并运行，以及 Python/PyTorch/Triton/Torch CUDA 版本和镜像 RepoDigest。

看到 `RTX 4060` 和实际探测的 Compute Capability 后再继续。doctor 的 CUDA 编译/运行、Triton JIT/运行或 PyTorch attention 运行失败，就表示对应实现的 GPU 验收未通过，不能用 mock 或宿主机 `nvidia-smi` 成功替代。

## 4. 启动、检查和停止

### 4.1 启动

```bash
make start
```

`make start` 创建 `data/jobs/`、构建前后端，然后先运行一次性 `migrate` 服务执行 `alembic upgrade head`。Make 会把当前 WSL 用户 UID/GID 和 Docker socket GID 传给 Compose，使 Linux 文件系统中的 `./data` 可由 migrate、API 和 Worker 以同一身份写入。只有 migration 成功退出，API 和单 Worker 才启动；Worker 随后探测固定 CUDA 镜像、读取实际 RepoDigest 并保存 CUDA 环境快照。Triton 与 PyTorch 环境由各自工作流独立按需探测，不影响基础 CUDA readiness；保存对应语言的性能版本或统一重测时会持久化该语言的环境快照。首次运行前的 `make doctor` 会在缺少镜像时拉取两种固定 Runner 镜像，并分别实测 CUDA、Triton 和 PyTorch 三条 GPU 路径。

就绪后，在 Windows 浏览器访问：

```text
http://localhost:3000
```

前端和 API 使用同一 origin，API 位于 `/api`。可在 WSL 中检查：

```bash
curl --fail http://127.0.0.1:3000/api/health
curl --fail http://127.0.0.1:3000/api/ready
```

`health` 成功只说明 API 进程存活；`ready` 检查数据库、非空题目注册表、最近一次健康环境快照、`gpu:0` Worker 活跃租约和熔断标记。Worker 尚未完成首次探测/获取租约时短暂返回 503 是正常的；持续 503 时根据 `worker_active` 和 `runner_error` 处理。

环境页可切换 `CUDA C++`、`Triton (Python)` 与 `PyTorch (Python)` 三个页签。CUDA 页显示 GPU、驱动、CUDA Runtime、NVCC、架构和 CUDA 镜像摘要；Triton 页显示 GPU、驱动、Python/PyTorch/Triton、Torch CUDA Runtime、架构和共享 PyTorch 镜像摘要；PyTorch 页显示 GPU、驱动、Python/PyTorch/Torch CUDA、架构和同一共享镜像摘要。Triton 与 PyTorch 虽然共用镜像，但分别探测并保存 `triton_python` / `torch_python` 环境快照，Triton 专属的 import/JIT 故障不应被误报为 PyTorch 探针失败。`/api/ready` 以基础 CUDA Runner 和共享 GPU/Worker 状态为准；尚无某个 Python 实现的已保存快照时，对应环境页可显示 unknown，而该工具链故障只应使对应语言的 Job 失败。先用 `make doctor` 验证三条执行路径，不要仅凭总 readiness 推断 Triton 或 PyTorch 已可用。

验证主机没有对外监听：

```bash
ss -ltnp | grep ':3000'
```

默认模式下宿主应只看到 `127.0.0.1:3000`，不能是 `0.0.0.0:3000` 或 `[::]:3000`，也不应看到发布的 `8000`。LAN overlay 启用后会额外出现一个具体的局域网 IPv4，仍不得出现通配地址。API 容器内部监听 `0.0.0.0:8000` 属预期行为。也可在 Windows PowerShell 执行：

```powershell
Get-NetTCPConnection -LocalPort 3000 -State Listen
```

### 4.2 可选的认证局域网模式

本模式依赖 Windows 11/WSL 的 mirrored networking，和已有的 WSL SSH 局域网监听使用同一网络路径。它不会发布 API 8000，而是在当前明确的 LAN IPv4 上额外发布 Nginx 3000，并用 Basic Auth 同时保护页面和 `/api`。默认 `make start` 完全不受影响。

先确认 `%USERPROFILE%\.wslconfig` 包含以下配置，修改后需要执行 `wsl --shutdown`：

```ini
[wsl2]
networkingMode=mirrored
firewall=true
```

在**提升权限的 Windows Terminal/PowerShell 所启动的 WSL 终端**中创建精确防火墙规则：

```bash
make lan-firewall
```

该命令同时创建 Windows 与 WSL Hyper-V 入站规则，只允许 `LocalSubnet` 访问检测出的具体 LAN IPv4 的 TCP 3000；它不会把 Hyper-V 默认入站策略改为 Allow。随后启动：

```bash
make start-lan
```

首次启动若没有 `data/lan.htpasswd`，会生成高强度随机密码并**只显示这一次**。默认用户名是 `myleetgpu`。记下终端打印的地址和密码，例如：

```text
http://192.168.31.106:3000
```

检查状态、轮换凭据和完整关闭：

```bash
make lan-status
make lan-password
make stop-lan
make lan-firewall-off   # 需要提升权限
```

LAN IP 由 DHCP 改变后，应先关闭旧规则，再以新地址重新执行 `make lan-firewall` 和 `make start-lan`。Basic Auth 在纯 HTTP 上不提供传输加密，因此本模式只适用于受信任且加密的家庭/实验室网络；对不可信网络应使用带身份认证和端到端加密的 VPN，而不是扩大此端口。LAN 访问者共享同一草稿、版本、Job 队列和 GPU，无法相互隔离。

### 4.3 日志与状态

```bash
make ps
make logs
```

`migrate` 显示 `Exited (0)` 是一次性服务正常完成，不是崩溃；非零退出时执行 `make logs` 查看 Alembic 错误，API/Worker 会保持未启动。Makefile 会动态导出绝对 host data 路径、当前 UID/GID 与 Docker socket GID；不要绕过 Makefile 用 `.env.example` 中的示例 GID 直接启动 Compose。

报告问题时保留页面显示的 Job ID。不要公开粘贴未经检查的完整数据库、`.env` 或用户源码。

### 4.4 停止

```bash
make stop
```

正常停止会保留草稿、手动版本和 benchmark。不要随意使用 `docker compose down -v`；删除 volume 可能丢失持久数据。

## 5. 日常使用

### 5.1 选择题目与编辑代码

1. 在题目列表选择当前十道内置题之一：Vector Addition、Matrix Transpose、Sum Reduction、Max Reduction、Softmax、Matrix Multiplication、Top-K 和 Top-P 分别提供 CUDA C++ 与 Triton (Python) starter；多头自注意力（MHA）和分组查询自注意力（GQA）提供 PyTorch (Python) class starter。
2. 用编辑器顶部的语言切换器选择当前题目支持的实现。URL 会保留 `language=cuda_cpp`、`language=triton_python` 或 `language=torch_python`，刷新和进入性能页时仍能回到同一语言；语言切换器不会显示当前题目没有声明的实现。
3. 阅读当前语言的入口签名、补充说明、约束和浮点容差。CUDA 不要自行提供 `main`；Triton 可以定义多个 `@triton.jit` Kernel，但必须保留可调用的 Python `solve(...)`；PyTorch MHA/GQA 只实现 starter 指定的普通 class 及 `forward(X, isCasual)`。两种 Python 实现都必须遵守题面列出的受限语法。
4. 在 Monaco Editor 中编辑 `.cu` 或 `.py`。编辑器会分别保存当前题目、当前语言的草稿。
5. “重置代码”只会把当前语言的编辑内容恢复为该 revision 的 starter；确认前检查是否仍需要未保存修改。

Triton `solve` 接收 GPU 0 上连续存放的 `torch.float32` Tensor 和题目声明的标量参数。平台已经进入受控的 `torch.cuda.stream(stream)` 上下文；直接把 Kernel launch 到当前 stream，写入平台提供的输出 Tensor，并返回 `None`。`solve` 只能做字面量/标量 launch 参数计算、`triton.cdiv`、`triton.next_power_of_2`、当前文件 JIT Kernel launch，以及 Sum Reduction 题明确允许的 `output.zero_()`；不要把 Tensor 移到 CPU、替换输出、调用设备级同步或依赖默认 stream。每道题的 Triton 补充说明会显示准确签名和白名单边界。

PyTorch 的 MHA/GQA 题不要求编写自定义 Kernel，而是练习完整的自注意力前向：平台把固定只读的 Q/K/V/输出投影权重和 head 配置传给构造器，`forward(X, isCasual)` 从同一个 `[batch, sequence, model_dim]` 输入生成 Q/K/V。`isCasual=True` 表示使用包含对角线的下三角 causal mask；GQA 还要求连续 query-head 组共享一个 KV head。平台已进入受控的 `torch.cuda.stream(stream)` 和 `torch.inference_mode()` 上下文；必须返回新的同形 CUDA `torch.float32` Tensor，且不得修改或别名 `X`、权重或实例状态。不要把 Tensor 移到 CPU、显式同步，或直接调用现成的 scaled-dot-product attention 绕过实现。

草稿自动保存只是工作区恢复机制，不是性能版本。页面刷新、切换题目或切换语言后，对应草稿应恢复；网络失败时浏览器保留按语言隔离的本地回退副本。当前服务端草稿按 `(problem_id, language)` upsert，采用后写覆盖，不提供同一题同一语言的多标签页冲突合并。CUDA、Triton 与 PyTorch 草稿不会互相覆盖。

MHA/GQA 从旧的四 Tensor `solve` 升级到 revision 2 class 接口后，已有本地草稿不会自动改写；若编辑器仍显示旧接口，点击“重置代码”载入当前 revision 的 class starter。旧的已保存性能版本会保留，但因 problem revision 不同而标记为不可直接比较。

直接调用 API 时，草稿 GET/PUT、`POST /api/jobs` 和重复源码查询若省略 `language`，都使用题目 manifest 的 `default_language`，而不是一律回退到 CUDA；因此 MHA/GQA 会选中 `torch_python`。`GET /api/environment` 是例外：省略 `language` 时仍默认查询 `cuda_cpp` 环境。

### 5.2 编译与代码检查

CUDA/Triton 点击“编译”、PyTorch 点击“代码检查”，都只执行当前语言的无 GPU 预检查：

- CUDA C++ 会运行 NVCC 并链接当前操作的可信 harness；成功后临时二进制随后清理。
- Triton (Python) 检查 `source.py` 的 Python 语法和 `restricted_triton_v2` AST 白名单，再在隔离 globals 中加载安全定义并确认 `solve` 存在；不会调用 `solve` 或触发 GPU JIT。
- PyTorch (Python) 检查 Python 语法和独立的 `restricted_torch_v2` AST 白名单。当前 attention 题只允许精确的 `import torch`、不可变字面量模块常量和一个无继承、无 decorator、仅含 `__init__` / `forward` 的指定 class；构造器只能保存平台参数，forward 不得写实例状态。策略会加载安全定义，但不会实例化、调用 forward 或访问 GPU；文件/网络/进程/线程、反射、动态执行、打印、helper、`torch.nn`、原地属性/下标赋值、`out=`、`torch.compile`/`torch.jit`/`torch.ops` 和现成 scaled-dot-product attention 都会被拒绝。精确 class 与方法参数会在运行阶段由同一策略结合可信 harness 再校验。
- 失败时显示用户源码中的行列号和经过路径清理、长度限制的诊断。
- 不执行样例，不生成 benchmark，不创建性能版本。

Triton Kernel 需要真实 Tensor、dtype 和 `tl.constexpr`/meta 参数才能完成对应专化；首次 GPU 调用发生在“运行”“验证”或保存流程中。因此 Triton 预检显示成功后，第一次运行仍可能报告 JIT 编译错误，这是预期的两阶段语义。无论重复编译多少次，性能版本数量都不会增加。

PyTorch 题不进行 JIT；“代码检查”通过只表示源码满足语法、策略与入口结构，精确参数列表以及实际 Tensor 形状、设备/dtype、返回值、输入不可变性和数值正确性仍要在“运行样例”或“完整验证”中确认。无论重复代码检查多少次，性能版本数量也不会增加。

### 5.3 运行公开样例

点击“运行样例”会重新执行当前语言的预检查/编译，然后执行题面公开的样例。Triton 在此时通过同一版本化策略把受限定义加载到独立 module globals、校验 `solve` 精确参数，并在第一次 GPU 调用完成 JIT 专化；PyTorch 同样通过 `restricted_torch_v2` 加载并校验 class、构造器和 forward 精确参数，再由可信 harness 注入固定权重、调用 `forward(X, isCasual)`，并检查返回 Tensor 的设备、dtype、形状、底层存储不与输入或权重别名、所有输入未被修改和数值结果：

- 输出面板按公开用例显示 pass/fail；
- 可能显示编译错误、运行错误、CUDA error、输出超限或超时；
- 仅公开输入会显示必要详情，内部测试不会参与该动作；
- 不保存版本，也不把这次运行时间当作正式 benchmark。

### 5.4 完整验证

点击“完整验证”会重新执行当前语言的预检查/编译，并执行公开测试、边界测试和固定种子的内部测试。整数结果精确比较；浮点按题目给出的 `atol`/`rtol` 比较，并检查 NaN/Inf。PyTorch attention 题还会执行输出契约和输入不可变性检查。

内部测试失败时只显示安全摘要，不显示内部输入、参考输出或 harness 路径。验证成功也不会创建性能版本。

### 5.5 任务状态

任务通常经历：

```text
queued → compiling → running / validating / benchmarking → succeeded
```

也可能结束为 `failed`、`timed_out`、`cancelled` 或 `system_error`。`cancelled` 是状态机保留终态，当前 UI/API 没有主动取消按钮。单张 GPU 严格串行，前方有任务时处于 `queued` 是正常现象。不要因页面等待而连续重复提交；用 Job ID 查看当前状态。

## 6. 保存性能版本

只有“保存为性能版本”会创建持久 Version：

1. 点击按钮后填写版本名称，备注可选。
2. 系统立即冻结点击时的实现语言和完整源码快照；随后继续编辑或切换语言不会改变这个候选版本。
3. 系统重新执行当前语言的编译/代码检查，并完成完整正确性验证。
4. 验证通过后，按固定协议运行 benchmark。
5. 两步均成功后，先保存环境快照，再在一个事务中同时创建 Version 与首次 BenchmarkRun；随后 Job 才标为成功。

如果编译/代码检查、验证、benchmark、GPU、Docker 或版本事务任一步失败，Version 数量保持不变；此前的环境探测快照可能保留。同一题、同一语言下相同源码 hash 已经保存过时，界面会先调用 duplicate 查询并要求用户确认；服务端也会在入队、Worker 开始及提交前复查。确认请求携带 `allow_duplicate=true` 后仍可保存，例如为同一源码保留不同的语义备注。CUDA、Triton 与 PyTorch 源码只在各自语言内判重。

保存后：

- 实现语言、源码、problem revision 和测量上下文不可变；
- 可以修改名称和备注；
- 删除需要 UI 二次确认；实际请求为 `DELETE /api/versions/{version_id}?confirmed=true`，缺少确认参数会返回 409，成功后关联 benchmark 级联删除；
- 刷新页面或重启服务后仍然存在。

## 7. 比较和重新测试版本

1. 进入同一题目的版本列表，先选择当前题目支持的 CUDA C++、Triton (Python) 或 PyTorch (Python) 标签，再在该语言内选择 2 至 8 个唯一版本。
2. 指定其中一个为 baseline。
3. 查看每个输入规模的 median、p95、波动指标、样本数和 speedup。
4. 检查环境/协议栏的“可直接比较”状态和差异列表。
5. 使用代码快照或 Diff 查看实现差异。

环境/协议栏应能看到实现语言、GPU、驱动、语言工具链、执行配置、镜像 digest、suite hash 和协议版本：CUDA 展示 CUDA Runtime/NVCC/flags；Triton 展示 Python/PyTorch/Triton/Torch CUDA/目标架构与策略；PyTorch 展示 Python/PyTorch/Torch CUDA/目标架构、`restricted_torch_v2`、matmul precision、TF32 和确定性设置。缺字段本身也是需要谨慎解释的信号。

候选 X 相对 baseline B 的加速比为：

```text
speedup(X) = median(B) / median(X)
```

`1.20x` 表示候选的 median 耗时约为 baseline 的 `1 / 1.20`；小于 `1.0x` 表示更慢。

UI 按语言分组，并在比较请求中携带当前语言；API 会拒绝与该显式语言不匹配的版本。兼容调用即使省略语言，后端也会通过可比性键把混合语言标为不可比较且不生成 speedup。CUDA C++、Triton 与 PyTorch 即使使用同一块 GPU、相同题目规模，也不会得到跨语言 speedup。只有同题、同语言版本的 problem revision、suite hash、输入规模/随机种子/采样协议、执行配置和完整环境指纹一致，结果才会标为“可直接比较”。任一项不同都会显示“不可直接比较”，此时可以并排看历史值，但系统不会生成统一排名或误导性的总 speedup。

选择“使用当前统一环境重新测试所选版本”时，可提交同题、同语言的 1 至 8 个唯一版本；混合语言请求会被拒绝。系统先在单 GPU 上**串行完成全部版本的完整验证**；只有全部通过，才进入第二阶段，串行 benchmark 全部版本。测量结果暂不逐条入库，全部成功后用一个事务批量追加 BenchmarkRun。任一版本验证、测量或提交失败，本批次不会向任何所选版本追加 BenchmarkRun。重测不创建 Version，也不修改语言、源码、名称或备注。

## 8. Benchmark 指标如何解读

- **median**：多次有效采样的中位数，是主要比较指标，对个别异常值比均值稳健。
- **p95**：约 95% 样本不超过的耗时，反映较慢尾部。
- **min**：观察到的最小值，只供诊断；不要把“最好一次”当作稳定性能。
- **CV**：总体标准差与均值之比，越大通常表示相对波动越大。
- **MAD**：样本相对中位数偏差的中位数，是更抗异常值的波动指标。
- **样本数**：实际纳入统计的采样数量；不足配置数量时，本次 benchmark 应失败而不是悄悄改变口径。

三种实现语言都由平台在同一受控 CUDA stream 上使用 CUDA Events 计时，并先 warmup。极短 Kernel/`solve` 会使用 inner repetitions。正式耗时排除 NVCC/Python 策略预检、Triton 首次 JIT 专化、容器启动、CUDA context 初始化、平台输入生成、设备内存准备和 H2D/D2H 拷贝；PyTorch 返回 Tensor 的分配属于被测 `solve`，计入其 GPU 时间。用户打印在两种 Python 策略中被拒绝，CUDA stdout 中自报的时间也完全不采信。

结果仍会受到 GPU 温度、时钟、功耗限制、Windows 桌面图形和后台 GPU 工作影响。比较前尽量：

- 关闭游戏、视频渲染、模型训练等 GPU 程序；
- 使用一致的电源模式，并让机器达到相近热状态；
- 对高波动结果重新测试，结合 p95、CV/MAD 而不是只看很小的 median 差异。

WSL2 中温度、时钟或 GPU busy 可能不可用；`unavailable` 表示没有可靠数据，不是零。即使环境指纹完全相同，结果也只代表这台机器的本地测量，不能作为跨机器绝对排名。

## 9. 本地数据、备份与清理

### 9.1 数据位置

默认持久化根目录为仓库下的 `./data/`：SQLite 数据库是 `./data/myleetgpu.db`，Job 临时源码、编译产物和工作目录位于 `./data/jobs/` 并在任务结束后清理；Runner 熔断时还会出现 `./data/runner-unhealthy.json`。Compose 内对应 `/data`。自定义部署修改路径时，必须同时核对 `.env`、Makefile 和 Compose 挂载，确保 API、Worker 与 Runner 指向同一目录。

以下内容不会进入 Git：数据库、`-wal`/`-shm`、草稿、版本源码、benchmark、日志、Job spool、编译产物和 `.env`。

MVP 会持续保留 Job 元数据/限长诊断和历史环境快照，目前没有按日期自动裁剪；启动/恢复探测可以复用同 fingerprint 的已有快照，而保存版本和统一重测会保留当次快照。删除 Version 会级联删除它的 BenchmarkRun，但不会自动回收已无引用的 EnvironmentSnapshot。

### 9.2 日常清理

```bash
make clean-jobs
```

该命令只扫描 `data/jobs/`，删除已没有活动 `spool_path` 引用的临时目录；不删除 Draft、Version、BenchmarkRun、Job 元数据、数据库、熔断标记或 Docker 容器。孤儿容器由 Worker 启动时按 Runner + installation labels 精确回收；租约丢失时仅停止自身 owner label 容器。运行前后都应避免手工删除正在执行 Job 的目录。

删除某个性能版本请使用 UI，并完成二次确认。普通“编译/代码检查”“运行样例”“完整验证”和编辑器重置不会删除已有版本。

### 9.3 备份

最简单可靠的方法是在停止服务后复制**整个**数据目录：

```bash
make stop
cp -a data data-backup-YYYYMMDD
```

把 `YYYYMMDD` 换成实际日期并确保目标目录不存在。整个目录一起复制可以保留 SQLite 的 `-wal` 和 `-shm`。如果必须在线备份，应使用 SQLite 官方 backup API/命令，而不是只复制主 `.db` 文件。

恢复时先停止服务，把当前数据目录另行备份，再用一套相互匹配的数据库/WAL 文件恢复。随后通过 `make start` 的一次性 `migrate` 服务升级 schema，或在已安装本地依赖时运行 `make migrate`，最后检查 readiness。不要用旧 schema 文件直接覆盖正在运行的数据库。

### 9.4 完整重置

完整重置会丢失草稿、版本和 benchmark。建议使用可恢复的“改名”而不是直接删除：

```bash
make stop
mv data data-before-reset
make start
```

确认新环境正常且备份确实不再需要后，再由操作者自行删除旧目录。不要把 `docker compose down -v` 当作日常清理命令。

## 10. 测试命令

```bash
make lint
make test
make test-gpu
make e2e
```

- `make lint`：前后端静态检查和格式检查。
- `make test`：不依赖真实 GPU 的后端、API、SQLite 队列/租约及前端单元/集成测试，不能以 mock 成功代表 GPU 验收。
- `make test-gpu`：在实际 NVIDIA GPU 上验证三种 Runner 路径的编译/策略预检、执行、benchmark、隔离、超时清理和后续恢复。
- `make e2e`：由 Playwright 启动本地 Vite 页面，并用受控 mock API 验证关键浏览器交互。它验证 UI 流程，不代表 Compose、SQLite Worker 或真实 GPU 全链路已通过。

执行 GPU 测试前先运行 `make doctor`。若机器条件不满足，记录“未运行”、原始诊断和下一步；不要宣称通过。

## 11. 常见故障排查

### 11.1 WSL 不是版本 2

现象：doctor 报 WSL1、内核不是 Microsoft WSL2，或 Docker Desktop 无法集成。

处理：在 PowerShell 用 `wsl --list --verbose` 确认发行版；使用 `wsl --set-version <发行版名称> 2` 转换，并运行 `wsl --update`。如果虚拟化或 Windows 功能未启用，需要管理员在 Windows 中处理，项目无法代为绕过。

### 11.2 WSL 内没有 `nvidia-smi`

处理顺序：

1. 在 Windows PowerShell 运行 `nvidia-smi`；失败则更新/修复 Windows NVIDIA 驱动。
2. 运行 `wsl --update`，随后 `wsl --shutdown` 并重新进入发行版。
3. 在 WSL 检查 `/usr/lib/wsl/lib/nvidia-smi`。
4. 重新运行 `make doctor`。

不要在 WSL 安装 Linux NVIDIA 显示驱动来“修复”此问题。

### 11.3 Docker daemon 不可用或权限被拒绝

现象：`Cannot connect to the Docker daemon`、Docker socket permission denied。

处理：启动 Docker Desktop，确认 WSL2 engine 和当前发行版 Integration 已启用，再检查 `docker context ls` 与 `docker info`。若使用 WSL 内独立 Docker Engine，按该 Engine 的官方方式启动服务、配置用户组；不要同时连接错误的 Desktop context。重新打开终端后再运行 doctor。

### 11.4 GPU 容器不可用

现象：`could not select device driver ... capabilities: [[gpu]]`、`nvidia-container-cli` 错误、容器内看不到设备。

处理：

- 更新 Docker Desktop 和 Windows NVIDIA 驱动，更新 WSL 内核并重启 WSL；
- 若使用独立 Docker Engine，安装/配置与该 daemon 对应的 NVIDIA Container Toolkit，然后按官方步骤重启 daemon；
- 使用项目 doctor 所用的固定 CUDA 与共享 PyTorch/Triton 镜像，分别验证 CUDA、Triton 和 PyTorch 三条 GPU 路径，避免拿另一条路径成功来替代；
- 不要传入 `seccomp=unconfined` 来绕过 Docker 默认启用的内置 seccomp 隔离；
- 若涉及驱动、Toolkit 或服务的管理员级修改，停止项目操作并在系统层完成后再重试。

### 11.5 RTX 4060 架构或 NVCC 不匹配

现象：NVCC 报不支持目标架构、`no kernel image is available`。

处理：运行 doctor 查看实际 Compute Capability、NVCC 和镜像 digest。编译架构由集中配置根据探测结果生成，不能在业务代码或题目中散落硬编码。RTX 4060 预期通常为 `sm_89`；若固定 CUDA 镜像不支持探测到的架构，应更新项目固定镜像/工具链并重新验证，而不是允许用户提交任意 `-arch` flags。

### 11.6 NVCC 编译错误

先看输出面板中用户文件的行列号。常见原因包括函数签名与题目不一致、缺少分号、把 host 指针当 device 指针、使用固定工具链不支持的语言特性。点击“重置代码”可恢复 starter，但会覆盖当前草稿，请先自行保存需要的片段。

如果诊断只有 `system_error` 而不是 `compile_error`，请记录 Job ID 并查看 Worker 日志；不要修改 Docker Runner 去执行任意 shell 命令。

### 11.7 Triton / PyTorch 镜像或工具链不可用

现象：环境页的 Triton 或 PyTorch 标签显示 unknown/unavailable，Job 报固定 PyTorch/Triton 镜像不可用，或 Python 无法 import `torch` / `triton`。

1. 运行 `make doctor`；它会拉取并检查项目固定的官方 PyTorch 2.5.1 CUDA 12.4 devel digest，分别输出 PyTorch 与 Triton 工具链信息，在 GPU 0 上运行最小 PyTorch attention，并独立 JIT 运行最小 Triton Kernel。
2. 拉取失败时检查 Docker daemon、代理/网络、磁盘空间和镜像仓库访问；保留原始 doctor 输出。
3. 工具链版本不符合 Python 3.11 / PyTorch 2.5.1 + CUDA 12.4（Triton 路径还应包含 Triton 3.1）时，检查 `.env` 的 `TRITON_IMAGE` 是否仍是仓库默认固定 digest，然后重新运行 doctor 和 `make start`。该变量目前同时指定两种 Python 实现共用的固定镜像。
4. 不要用提交代码联网安装包，不要在 API/Worker 容器中临时 `pip install`，也不要用随机本地镜像冒充验收通过；这些做法会破坏环境指纹和可复现性。

基础 CUDA readiness、Triton 工具链探针和纯 PyTorch 工具链探针按语言区分。共享镜像缺失通常会同时影响两个 Python 实现，但纯 PyTorch 探针不依赖 import Triton 或 Triton JIT，因此能与 Triton 专属故障区分；反过来，总体 `/api/ready` 为 ready 也不等于 Triton 或 PyTorch 已通过各自实机探针。

### 11.8 Triton 预检通过，但运行/JIT 失败

“编译”会完成 Python 语法、受限 AST、精确 import/调用白名单、安全定义加载和 `solve` 存在性检查。非白名单 import、反射、文件/网络/进程/线程、打印、dunder、动态执行、设备打印或内联汇编会在此阶段以提交策略错误被拒绝。真实 GPU 专化错误、无效 `tl.constexpr`/block 参数、不受支持的 dtype、越界 load/store、资源超限或 PTXAS 错误仍要到第一次真实 GPU 调用时才暴露。先用公开样例触发最小专化，查看 `source.py` 的安全化诊断，再核对当前题目的 Triton 接口说明、mask、grid 和输出写入。

Triton JIT 缓存位于容器内 512 MiB 的 `exec,nosuid,nodev` `/tmp`，任务结束即丢弃。若出现 cache 目录不可写、`Permission denied` 或 `No space left on device`，检查是否修改了 Runner 的 tmpfs/环境变量、容器 memory 或固定镜像，记录 Job ID 并运行 `make logs`。不要挂载持久 host cache，不要关闭只读根文件系统或 `no-new-privileges` 来绕过问题。第一次 JIT 的耗时不计入正式 CUDA Event 样本，但仍受 Job wall-clock timeout 约束。

### 11.9 PyTorch 代码检查通过，但运行失败

“代码检查”只完成 Python 语法、`restricted_torch_v2` AST 白名单、安全 class 加载和入口结构检查，不会实例化 class 或调用 forward。因此，返回非 Tensor、CPU Tensor、错误 dtype/形状、返回值与 `X`/权重别名、修改输入、忽略 `isCasual`、MHA head 拆分错误、GQA 分组错误及真实 GPU 运算异常，都要到“运行样例”或“完整验证”才会暴露。先核对题面 class、构造器、`forward(X, isCasual)` 和返回契约，再查看 `source.py` 的安全化诊断。

纯 PyTorch 路径不做 JIT，并固定关闭 TF32、使用最高 float32 matmul precision 和确定性算法；其 512 MiB `/tmp` 为 `noexec`。不要通过 `torch.compile`、`torch.jit`、`torch.ops`、现成 scaled-dot-product attention、设备同步或修改 Runner 配置绕过题目，这些入口会被策略拒绝或破坏可比性。

### 11.10 运行错误、错误答案、超时或输出超限

- `wrong_answer`：CUDA/Triton 检查索引边界、grid/block 覆盖、同步和归约竞争；PyTorch attention 检查缩放、mask、维度变换和 GQA head 映射；所有语言都要核对题目浮点容差。
- `runtime_error`：CUDA C++ 检查 `cudaGetLastError` 对应的非法访问、无效配置和资源超限；Triton 检查 Python 异常、launch grid、mask、Tensor device/dtype 和输出写入；PyTorch 检查返回 Tensor 契约、输入不可变性和白名单 Tensor 运算的形状/device/dtype。
- `timed_out`：检查死循环、过量 inner work 或无法结束的 Kernel/`solve`；Runner 会强制删除已知容器。若后续任务出现驱动/NVML/Xid 等健康错误，Runner 会进入熔断。
- `output_limit`：Runner 合并捕获 stdout/stderr；删除 device/host 大量打印，达到总量上限后任务会终止，不能依赖末尾日志。

内部测试失败不会显示隐藏输入。Triton 与 PyTorch 策略都拒绝文件读取、反射、host/device 打印和结果 sentinel 伪造；不能通过读取 harness 路径或篡改 `__main__` 绕过验证。

当前版本只面向本机或认证局域网中的可信操作者。提交与平台 harness 仍位于同一最终进程（CUDA 链接、Triton/PyTorch 分别受限定义加载）；`restricted_triton_v2` 与 `restricted_torch_v2` 阻断已知的 Python harness 篡改和结果伪造路径，但不是通用 Python/GPU 沙箱的形式化安全证明，CUDA 同进程结果通道也不是抗作弊边界。不要把结果用于不可信用户排名或公网竞赛。

### 11.11 Job 一直排队

单 GPU 串行执行时，前方有运行/验证/benchmark 属正常现象。检查环境状态页、`/api/ready` 和：

```bash
make ps
make logs
```

若 Worker 不健康，查看 Worker 日志。仅在没有有效任务执行时使用 `make clean-jobs` 清理不再被活动 Job 引用的临时目录；它不清理容器或修改 Job 数据库状态。Worker 重启会按 Runner + installation labels 回收遗留容器，并把旧 Worker 中断的活动 Job 标为 `system_error`，不会自动重跑；仍在 `queued` 的任务继续排队。

### 11.12 Runner 显示 unhealthy

Runner 在提交输出中发现 GPU 丢失、驱动不兼容、NVML 初始化失败、Xid 等候选健康故障后，会用独立受信探针复核；只有探针也失败才持久熔断并阻止后续 GPU Job。这是保护行为，用户 stdout 本身不会触发熔断。

1. 运行 `nvidia-smi` 和 `make doctor`，确认固定容器中的真实 CUDA 编译/运行、Triton JIT/运行和 PyTorch attention 运行探针分别通过。
2. 必要时重启 Docker Desktop；仍失败时从 PowerShell 执行 `wsl --shutdown` 后重试。
3. 如果驱动仍异常，可能需要重启 Windows。
4. 系统恢复且 doctor 通过后，执行 `make recover-runner`。该命令会绕过旧熔断标记重新探测基础 CUDA/GPU 健康；只有探测健康才删除 `data/runner-unhealthy.json`，并把新 CUDA 环境快照写入 SQLite。Triton 与 PyTorch 工具链仍以 doctor 及下一次对应语言 Job 的独立探测为准。
5. 若 Worker 因熔断反复重启，完成上述恢复后执行 `make start` 或重启 Worker 服务。

不要手工删除熔断文件或改数据库标志；`make doctor` 只诊断，不会自行解除熔断。

### 11.13 端口 3000 被占用或绑定错误

用 `ss -ltnp | grep ':3000'` 或 PowerShell 的 `Get-NetTCPConnection` 找到占用者，停止冲突服务后重新启动。不要为避开冲突把服务绑定到 `0.0.0.0`。默认模式若看到外部接口监听，立即停止 MyLeetGpu；LAN 模式只允许预期的具体 LAN IPv4，并应核对 `make lan-status` 的认证与防火墙规则。

### 11.14 migrate 非零退出

`migrate` 是一次性服务，`Exited (0)` 正常；非零退出会阻止 API 和 Worker 启动。先执行 `make logs` 查看 Alembic 错误，停止服务并备份整个 `data/`，再修复权限、磁盘空间或 migration 问题。宿主开发环境安装依赖后可运行 `make migrate`；不要删除数据库让启动“看起来成功”，也不要跳过失败 revision。

### 11.15 SQLite busy、磁盘满或数据库损坏

- 检查数据目录所在磁盘空间和权限；不要把数据库放在不可靠的网络文件系统。
- 短暂 `database is locked` 应由配置的 busy timeout 处理；持续发生时停止服务并检查是否有第二套 API/Worker 共用同一数据目录。
- 报损坏时立即 `make stop`，备份整个数据目录（包括 `-wal`/`-shm`），再使用 SQLite `PRAGMA integrity_check` 或从已验证备份恢复。
- 不要在运行时删除主数据库，不要让启动脚本用空库静默覆盖损坏文件。

### 11.16 重启后版本不见了

确认启动时使用了原来的 `.env` 数据目录和 Compose 持久挂载，且此前没有运行 `down -v` 或完整重置。停止服务，检查旧数据目录/备份，不要继续创建大量新数据覆盖恢复线索。

### 11.17 Benchmark 波动大或指标 unavailable

关闭其他 GPU 工作负载，统一电源模式，等待温度稳定后使用“当前统一环境重新测试”。检查 CV/MAD 和 p95。WSL 无法提供部分遥测时显示 unavailable 是诚实结果；不能填零或沿用旧值。

## 12. 快速命令表

| 目的 | 命令 |
| --- | --- |
| 环境/GPU 诊断 | `make doctor` |
| 启动 | `make start` |
| 停止并保留数据 | `make stop` |
| 本地执行 Alembic migration | `make migrate` |
| 健康探测并解除 Runner 熔断 | `make recover-runner` |
| 非 GPU 测试 | `make test` |
| GPU 实机测试 | `make test-gpu` |
| E2E | `make e2e` |
| 静态检查 | `make lint` |
| 清理无活动引用的临时 Job 目录 | `make clean-jobs` |

遇到系统级驱动、Docker/Toolkit 安装或权限问题时，准确保存 doctor 输出并完成所需的管理员操作。MyLeetGpu 不会修改 Windows 驱动，也不会把失败的 GPU 探针报告成通过。

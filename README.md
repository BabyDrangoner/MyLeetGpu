# MyLeetGpu

MyLeetGpu 是一个面向本机可信用户的 CUDA C++ 编程、正确性验证与 GPU 性能比较环境。首版内置三道原创题，提供 Monaco 编辑器、异步隔离 Judge、手动性能版本、持久化 benchmark 和统一口径比较。

> 安全边界：本项目仅供 Windows + WSL2 上的本机可信用户使用，只发布到 `127.0.0.1`。消费级 GPU 与 Docker 不提供公网多租户所需的强 GPU/显存隔离，请勿把服务暴露到局域网或公网。

## Quick Start

前置条件：WSL2、Windows NVIDIA 驱动、Docker Desktop（启用当前 WSL 发行版集成）和 NVIDIA Container Toolkit。仅运行 Compose 不需要宿主 Node/NVCC；开发测试需要 Python 3.12+ 与 Node.js 20+。

```bash
git clone git@github.com:BabyDrangoner/MyLeetGpu.git
cd MyLeetGpu
cp .env.example .env
make doctor
make start
```

Windows 浏览器访问：<http://localhost:3000>

```bash
make logs          # 查看 API/Worker/Web 日志
make stop          # 停止服务，保留草稿、版本和 benchmark
make clean-jobs    # 清理不再被活动任务引用的临时目录
```

首次启动会拉取固定版本的 CUDA 镜像并构建本地服务镜像，耗时取决于网络。持久数据保存在 `./data/myleetgpu.db`；源码快照只会在用户显式“保存为性能版本”且完整验证、benchmark 均成功后进入数据库。

## 开发与测试

```bash
make install
make lint
make test
make test-gpu      # 必须连接真实 NVIDIA GPU，不会回退到 mock
make e2e
```

常用命令还包括 `make migrate`、`make recover-runner` 和 `make help`。所有 Compose 宿主端口均绑定 loopback，API 通过同源 `/api` 提供。

## 文档

- [系统设计](docs/design.md)
- [用户指南](docs/user-guide.md)

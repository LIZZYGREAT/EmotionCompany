# EmotionCompany：复用已有 Docker 镜像，只更新推理代码的增量交付方案

## 一、目标与边界

当前任务只修订官方推理**检查逻辑与调用逻辑**，不改变 Base、两份 LoRA、训练产物、Prompt 和依赖版本。Codex 应使用**本机已有且已验收的完整 Docker 镜像**作为新构建的 `FROM`，只叠加这次修改的代码、Shell 入口和必要配置，避免重新执行 Conda、pip、apt 安装及复制大模型。

最终仍须交付一个**完整、可独立运行**的镜像 tar；代码补丁本身不能直接当作赛事镜像上传。整个修改期间不得覆盖或删除上一份已验证的镜像及原始 `image.tar`。

审核依据：`feat/final-dual-adapter-delivery`（核对时 HEAD `4934efd818841ce20ab742f1e520962f8d81c038`）。当前 `docker/final_dual/Dockerfile` 的运行环境包含 CUDA 12.8、`conda_env`、PyTorch 2.9.0、vLLM 0.11.1；模型位于 `/root/participant/model`，双 LoRA 位于 `/root/participant/adapters/{profile,emotion-response}`。这部分本轮应当完全复用。

**不需要执行**：新的 `conda create`、`pip install`、`apt-get install`、模型下载、训练、完整 Docker 构建上下文重建、旧模型的重新封装或预测准确率对比。

## 二、第一步：定位并保护已有镜像

先在**实际运行 Docker 的同一个 daemon/context** 中执行：

```bash
cd ~/projects/EmotionalCompany

docker context show
docker info --format 'Driver={{.Driver}} Root={{.DockerRootDir}}'
docker image ls --no-trunc
docker system df
df -h .
```

查找最后一次**本地成功加载、实际具有 Base+双 Adapter+conda_env** 的镜像。历史候选名称类似：

```text
emotionalcompany/final-dual:E03-DUAL-20261109-03
```

但须以本机 `docker image inspect` 实际结果为准，**不得盲目假定该 tag 仍在**。

若旧镜像仍在本机：

```bash
BASE_IMAGE='emotionalcompany/final-dual:E03-DUAL-20261109-03' # 按 inspect 的真实名称替换
docker image inspect "$BASE_IMAGE" >/dev/null
```

若镜像 tag 已被清理，但前次导出的完整 `image.tar` 仍在本机，则**仅执行本地导入**：

```bash
docker load -i /实际存在的/旧版/image.tar
# 从 load 的输出选择真实镜像名，再执行 docker image inspect
```

这一步只读取本地 tar，不下载网络依赖。若本地镜像和 tar 都不存在，应明确报告无法进行零下载增量构建；**暂停而不是自动从远端重装所有依赖**。

将选定的**镜像 ID、RepoDigest（若有）、Conda/vLLM 版本和本地模型目录**写进 `outputs/deployment/final_overlay/base_image.json`，并保护该父镜像 tag。验收可执行轻量容器检查：

```bash
docker run --rm --entrypoint /bin/bash "$BASE_IMAGE" -lc '
  test -x /opt/conda/envs/conda_env/bin/python &&
  test -x /opt/conda/envs/conda_env/bin/vllm &&
  test -f /root/participant/model/config.json &&
  test -f /root/participant/adapters/profile/adapter_model.safetensors &&
  test -f /root/participant/adapters/emotion-response/adapter_model.safetensors
'
```

这项检查不启动模型，也不重新安装任何东西。备份原 tar/镜像，不调用 `docker rmi`、`docker system prune`、`docker builder prune`。

## 三、第二步：把“代码修订”与“完整模型打包”分离

目前 `scripts/final_dual_docker_package.py::prepare_context()` 会准备整个部署上下文，其中包含 `participant/model` 和双 Adapter。**纯代码修订不应再经过这条准备大模型的流程。**

新增一个最小的补丁构建流程，例如：

```text
docker/final_dual/Dockerfile.overlay
scripts/final_dual_overlay.py
outputs/deployment/final_overlay/<run-id>/context/  # 被 Git 忽略
```

`final_dual_overlay.py` 要实现：

1. 从配置/参数读取已验收的本地 `BASE_IMAGE`，检查本机存在、镜像 ID 固定；不自动 `docker pull`。
2. 创建一个**只有代码和必要配置**的小构建上下文，不能包括 `participant/model`、`participant/adapters`、模型权重、Conda 环境、训练数据、Checkpoint、旧 tar。
3. 将本次改动中需要覆盖的 Python、Shell、配置文件复制到与容器内部路径对应的目录；可以复制固定的少量运行文件而非计算复杂的 Git 差异，但不复制大目录中的模型文件。
4. 记录文件 SHA、父镜像 ID 和新镜像 Tag；不再执行旧的训练/预测结果门禁。
5. 若 Docker/Conda/PyTorch/vLLM 依赖锁有变化、Base/LoRA 文件变化，必须提示**不属于纯代码增量构建**并停止，不能默默走完整重建。

建议仅允许下列部署内容进入补丁上下文（实际只复制此次涉及的文件）：

```text
participant/start.sh
participant/final_dual_inference.py
participant/run_inference.py
src/inference/final_dual_protocol.py
src/inference/final_dual_vllm.py
src/inference/final_dual_runtime.py
src/shared/                 # 只有修改到的相关小文件
docker/final_dual/entrypoint.sh
docker/final_dual/entrypoint_args.sh
configs/deployment/final_dual_docker.json
model_lock.json             # 若新代码仍读取它，仅保留必要资产路径/身份内容
```

若新增文件被某模块 `import`，也必须放进补丁上下文，避免容器内 `ModuleNotFoundError`。**不要重新复制 `participant/model` 和 `participant/adapters`**，它们已在父镜像层中。

## 四、第三步：新增极简 Overlay Dockerfile

`docker/final_dual/Dockerfile.overlay` 只需保留一条父镜像和少量代码覆盖指令。以下是示例，Codex 应根据真实修改文件删减 `COPY`：

```dockerfile
# syntax=docker/dockerfile:1
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /root

COPY participant/start.sh /root/participant/start.sh
COPY participant/final_dual_inference.py /root/participant/final_dual_inference.py
COPY participant/run_inference.py /root/participant/run_inference.py
COPY src/inference/final_dual_protocol.py /root/src/inference/final_dual_protocol.py
COPY src/inference/final_dual_vllm.py /root/src/inference/final_dual_vllm.py
COPY src/inference/final_dual_runtime.py /root/src/inference/final_dual_runtime.py
COPY docker/final_dual/entrypoint.sh /usr/local/bin/emotionalcompany-final-dual-entrypoint
COPY docker/final_dual/entrypoint_args.sh /usr/local/bin/entrypoint_args.sh
COPY configs/deployment/final_dual_docker.json /root/configs/deployment/final_dual_docker.json

# 只有当实际修改并仍需要该文件时才加相应 COPY
# COPY model_lock.json /root/model_lock.json

RUN chmod 0755 /root/participant/start.sh \
    /usr/local/bin/emotionalcompany-final-dual-entrypoint \
    /usr/local/bin/entrypoint_args.sh

ENTRYPOINT ["/usr/local/bin/emotionalcompany-final-dual-entrypoint"]
```

**明确禁止在这个 Dockerfile 里出现**：`apt-get`、`conda create`、`pip install`、`curl`、`wget`、`COPY participant/model`、`COPY participant/adapters`。它应直接继承父镜像中已经安装好的 GPU Runtime、全部 Python 包和模型文件。

如果本次对 `src/shared/` 或其他脚本有改动，则只增加那些具体的 `COPY`，不要为省事复制整个项目仓库。

## 五、第四步：构建与复用流程

生成小型上下文后（下面路径由新增脚本真实输出确定）：

```bash
BASE_IMAGE='emotionalcompany/final-dual:E03-DUAL-20261109-03'  # 以本机镜像为准
NEW_IMAGE='emotionalcompany/final-dual:runtime-fix-01'
CONTEXT='outputs/deployment/final_overlay/runtime-fix-01/context'

# 无网络构建：若它要求拉包，就会直接失败，从而暴露流程缺陷
DOCKER_BUILDKIT=1 docker build \
  --pull=false \
  --network=none \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  -f docker/final_dual/Dockerfile.overlay \
  -t "$NEW_IMAGE" \
  "$CONTEXT"
```

要求在构建日志中核实：

- 基础环境直接来自本机父镜像，不再安装 Conda/PyTorch/vLLM，也不下载 CUDA 基础镜像。
- Context 为小型代码包，不再包含模型权重或 10GB 级 tar。
- 构建产物使用**新 Tag**，父镜像可继续运行和回退。新的 Tag 不要覆盖父镜像 Tag。
- 修改 1 个 Python 文件再次构建时，仅涉及其 COPY 等后续轻量层，不能重新创建环境。

如 Docker Builder 无法解析本地父镜像，先检查 `docker context show`、`docker buildx ls` 和同一 daemon 的本地图像存储；不要直接切换到远端拉取。

### 本地开发中再省一步：Bind Mount 调试

若只是在反复修改 `final_dual_inference.py` 或入口 Shell，可以先把改动文件以只读卷挂载到**现有完整镜像**进行测试：

```bash
docker run --rm --gpus all --network none \
  --entrypoint /bin/bash \
  -v "$PWD/participant/final_dual_inference.py:/root/participant/final_dual_inference.py:ro" \
  -v "$PWD/participant/start.sh:/root/participant/start.sh:ro" \
  -v "$TEST_FILE:/input/test.jsonl:ro" \
  -v "$RESULT_DIR:/output:rw" \
  "$BASE_IMAGE" \
  /root/participant/start.sh /input/test.jsonl /output
```

如果所改代码还依赖其他模块，则一并挂载这些文件。此法只是**开发测试手段**，不会将改动写入镜像，**绝不能把原父镜像当作修复后的最终上传镜像**。

## 六、第五步：只做接口/调用验收，正式导出一次

本轮重点仅为官方接口，**不要求任何预测文本/Token IDs 与历史结果相同，也不要求 F1、准确率、格式成功比例达标**。最低验收：

1. 本机无网络使用新镜像运行 `/root/participant/start.sh TEST_FILE RESULT_DIR`，检查可启动、能够访问一份 Base 和两套 Adapter，顺序调用两路生成。
2. 用 1–2 条及 10–20 条测试输入检查容器入口、退出码、样本数、JSON 可解析性及官方规定两个输出文件的存在性；结果目录允许预先存在且可覆盖指定输出文件。
3. 测试容器直接接收两个参数、显式 `bash /root/participant/start.sh ...`，以及**需要兼容时**的无参数启动再 `docker exec`；具体保留哪种入口以官方执行协议为依据。
4. 应用层不再因为预测质量门禁、代码哈希对照、服务端返回可选统计字段或逐 Token 比对失败而使整个推理提前退出。模型加载失败、无法访问文件、服务崩溃、写入失败等真正的运行错误仍需如实报错。
5. 报告 Docker 的 Image ID、父 Image ID、构建阶段是否联网、是否调用包管理器以及最终 `start.sh` 的真实退出码。

最后，仅当完成上述接口验收，才执行一次：

```bash
mkdir -p outputs/deployment/final_overlay/export

docker save "$NEW_IMAGE" -o outputs/deployment/final_overlay/export/image.tar
sha256sum outputs/deployment/final_overlay/export/image.tar \
  > outputs/deployment/final_overlay/export/image.tar.sha256
```

**注意**：赛事使用的是独立 Docker 镜像 tar，因此 `docker save` 仍必须包含 Base 和所有依赖层。即便本次只改了 10KB Python 代码，最终 `image.tar` 仍可能是 10GB 量级；这是交付格式要求，和每次重新下载/重装包是两回事。不要在每次小修改后都 `docker save`，只在准备实际提交时导出。

## 七、Codex 实现交付要求

需实际新增/修改：

- `docker/final_dual/Dockerfile.overlay`：只继承父镜像并覆盖代码。
- `scripts/final_dual_overlay.py`：实现 `inspect-base` / `prepare-code-context` / `build` / `smoke` / `export`（也可简化成等价命令）；可在本机只读追踪父镜像 ID、上下文和新镜像 Tag。
- `docs/experiments/E03_最终镜像增量修复与构建记录.md`：写明实际使用的父镜像、文件改动、构建命令、无需重新下载/安装的证据、官方入口 Smoke 结果、最终 tar 路径和 SHA。

不得改动模型目录、两套 LoRA、原始训练产物和环境锁。不要执行 `docker system prune -a`、`docker builder prune`、`docker build --no-cache`、`docker pull` 或 `pip/conda` 重安装来“保证干净”。如真的发现依赖缺失，应先报告缺失及其调用原因，不把依赖更新悄悄混入这次纯推理代码修复。

### 最终验收

- [ ] 父完整镜像可运行并保留，或已从本地旧 tar 成功恢复。
- [ ] Overlay Dockerfile 中没有任何依赖安装、联网下载和大模型复制命令。
- [ ] 新构建 Context 只有代码和必要配置，镜像可完整复用父环境与两份 Adapter。
- [ ] 新镜像支持官方入口调用，完成小批量推理并生成两个官方文件。
- [ ] 保留上一版镜像及 tar；新版本使用独立 Tag 和 SHA。
- [ ] 只在最终准备提交时 `docker save` 一次，产物完整可独立加载。

**完成标准：后续每次只改推理代码，就只执行代码检查 → 小 Context Overlay 构建/挂载测试 → 官方入口 Smoke；不再重复下载、安装或打包模型。**

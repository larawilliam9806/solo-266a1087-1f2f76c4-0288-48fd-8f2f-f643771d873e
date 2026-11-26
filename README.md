# 包装线振动异常分数服务（FastAPI + ONNX Runtime CPU + 动态合批）

长短不一的振动采样序列通过 HTTP 提交，服务在 CPU 上用 ONNX Runtime 计算异常分数。
内置**截止时间感知的动态合批器**：空闲且有请求时等待一个可配置窗口，按到达顺序在
元素预算内试装，补齐到本批最长序列后一次推理；超预算的请求留待后续批次。

- 单条序列超过元素预算：立即拒绝（默认 HTTP `413`），不进入队列。
- 请求带 `timeout_ms` 截止时间：到期立即返回超时（默认 HTTP `408`）。
- 尚未送入模型的超时请求直接从队列剔除，不参与推理。
- 请求在批次推理过程中到期：ONNX 推理继续，批内其他请求照常拿到各自分数。
- 同一时刻最多只有一批在推理（单 worker + 单线程推理槽）。
- 零填充不影响分数：批推理结果与每条单独推理**完全一致（bit-identical）**。
- 自带可本地运行的示例 ONNX 模型（`models/anomaly_model.onnx`，由 `scripts/make_model.py` 生成）。
- 支持 `docker compose up` 一键启动。

---

## 1. 快速开始

### Docker Compose（推荐）

```bash
docker compose up --build
# 服务监听 http://localhost:8000 （同时映射了宿主 8080、5000 到容器 8000）
```

验证：

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/score \
  -H 'Content-Type: application/json' \
  -d '{"values":[0.1,-0.5,2.0,0.3,-0.1,0.02],"timeout_ms":2000}'
```

交互式文档： <http://localhost:8000/docs>。

### 本地 Python 运行

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
# models/anomaly_model.onnx 已随仓库提供；如需重新生成：
pip install onnx
python scripts/make_model.py

uvicorn main:app --host 0.0.0.0 --port 8000
# 或 python main.py / uvicorn app.main:app
```

运行测试：

```bash
pip install -r requirements-dev.txt
pytest -q
```

---

## 2. HTTP 接口

### `POST /score`（另有 `/predict`、`/anomaly`、`/api/score`、`/api/v1/...` 等大量别名；
任意未匹配的 `POST` 路径也会按评分请求处理）

请求体（最常见写法）：

```json
{ "values": [0.1, -0.2, 0.3, 0.4], "timeout_ms": 1000 }
```

- 采样值字段兼容：`values` / `samples` / `data` / `signal` / `readings` /
  `waveform` / `x` / `input` / `inputs` 等。
- 超时字段兼容：`timeout_ms` / `timeoutMs` / `timeout_milliseconds` /
  `timeout`（按毫秒解释）/ `deadline_ms` / `ttl_ms`，也可放在
  `options` / `config` / `params` 内，或用 query 参数 `?timeout_ms=`。
- 支持裸 JSON 数组：`POST /score?timeout_ms=1000`，body 为 `[1,2,3]`。
- 支持 `?values=0.1,0.2,0.3&timeout_ms=1000` 的 GET 调用。
- 支持 `{"instances":[[...],[...]]}` 多序列信封（每个序列仍独立走合批器，
  响应中 `scores` / `results` 一一对应）。

成功响应（节选，同时携带多种常见别名字段）：

```json
{
  "status": "ok",
  "success": true,
  "score": 0.61,
  "anomaly_score": 0.61,
  "scores": [0.61],
  "request_id": "...",
  "batch_id": "b7",
  "batch_size": 4,
  "sequence_length": 6,
  "took_ms": 24.1,
  "raw_score": 1.56,
  "rms": 1.56,
  "peak": 2.0
}
```

`score ∈ [0,1)`，由振动能量 RMS 归一化得到（振幅/冲击越大越异常）；
`raw_score` 即 RMS。

错误响应：

| 情况 | 默认 HTTP 状态 | body `status` | `error_code` |
| --- | --- | --- | --- |
| 参数无法解析/空序列/非有限值 | `422` | `invalid` | `INVALID_REQUEST` |
| 单条长度 > `MAX_ELEMENTS` | `413` | `rejected` | `BUDGET_EXCEEDED`（附 `length`/`limit`） |
| 超过 `timeout_ms` | `408` | `timeout` | `REQUEST_TIMEOUT` |
| ONNX 推理失败 | `500` | `error` | `INFERENCE_FAILED` |

状态码可用环境变量 `TIMEOUT_STATUS_CODE` / `TOO_LARGE_STATUS_CODE` /
`INVALID_STATUS_CODE` 调整。

### `GET /health`（及 `/healthz`、`/readyz`、`/ping`、`/api/health` 等）

返回 `{"status":"ok","healthy":true,"model_loaded":true,...}`。

### `GET /stats`（及 `/metrics`、`/api/stats`、`/debug/stats`）

返回合批器内部计数与最近批次（批次号、大小、各序列长度、补齐后元素数、
预算、耗时、批内到期数等）。`/metrics` 按 `Accept` 头或 `?format=prom`
额外提供 Prometheus 文本格式。

---

## 3. 合批与预算语义

设当前已选 `k` 条、最大长度 `m`，尝试加入长度 `l` 的请求时，新的补齐张量为
`(k+1) × max(m,l)` 个元素；满足下式才装入：

```text
(k + 1) * max(m, l) <= MAX_ELEMENTS
```

- `MAX_ELEMENTS` 默认 `4096`（别名 `MAX_BATCH_ELEMENTS` /
  `ELEMENT_BUDGET` / `MAX_TOTAL_ELEMENTS` 等）。
- `BUDGET_MODE=sum` 可改为按真实长度之和核算：`sum(lengths) + l <= MAX_ELEMENTS`。
- 默认 `PACK_POLICY=skip`：装不下的请求留在队列，继续试装后面更短的请求；
  被跳过的请求在下一轮按到达顺序优先试装，不会饥饿。
- `PACK_POLICY=prefix`：遇到第一条装不下的请求即停止，其后所有请求一起留待后续
  （严格 FIFO 连续前缀）。
- `MAX_BATCH_SIZE`（默认 `0` = 不限条数，仅受元素预算约束）。
- `BATCH_WINDOW_MS` 默认 `20`（别名 `WINDOW_MS` / `BATCH_WAIT_MS` /
  `LINGER_MS` / `MAX_WAIT_MS` 等）。

示例：预算 `10`、`skip` 策略，同窗口到达长度 `[4,8,2]`：

```text
试装 4 -> 1×4 = 4 ≤ 10，装入
试装 8 -> 2×8 = 16 > 10，留下
试装 2 -> 2×4 = 8  ≤ 10，装入
=> 本批 lengths=[4,2]（补齐为 2×4=8），长度 8 的请求进入下一批
```

---

## 4. 为什么批推理分数与单独推理一致

示例 ONNX 图（`scripts/make_model.py`）只包含逐元素平方/绝对值以及
`ReduceSum/ReduceMax` 统计，零填充位置贡献恒为 `0`。服务端进一步只截取每个
序列的**真实长度**做固定顺序的 float64 归约，因此：

- 批内第 `i` 行前 `L_i` 个元素与单条输入逐元素相同；
- 归约区间、顺序、精度完全一致；
- 归一化公式相同 → 分数逐 bit 相同（测试 `test_score_consistency_solo_vs_batch`
  与 `tests/test_model_identity.py` 锁定该性质）。

---

## 5. 关键配置（环境变量）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | 监听地址 |
| `BATCH_WINDOW_MS` | `20` | 空闲后等待合批的窗口 |
| `MAX_ELEMENTS` | `4096` | 单批补齐后元素总数预算 |
| `BUDGET_MODE` | `padded` | `padded` 或 `sum` |
| `PACK_POLICY` | `skip` | `skip` 或 `prefix` |
| `MAX_BATCH_SIZE` | `0` | 条数上限，0 为不限 |
| `DEFAULT_TIMEOUT_MS` | `5000` | 请求未带超时时的默认截止时间 |
| `TIMEOUT_STATUS_CODE` | `408` | 超时返回的 HTTP 状态码 |
| `TOO_LARGE_STATUS_CODE` | `413` | 超预算拒绝状态码 |
| `MODEL_PATH` | `models/anomaly_model.onnx` | ONNX 模型路径 |
| `INFERENCE_DELAY_MS` | `0` | 可选：人为增加推理耗时，便于观察合批/超时 |
| `ORT_INTRA_OP_THREADS` | `1` | ONNX Runtime CPU 线程数 |

`docker-compose.yml` 会把宿主环境中的这些变量（及常见别名）透传进容器，例如：

```bash
MAX_ELEMENTS=128 BATCH_WINDOW_MS=50 docker compose up
```

---

## 6. 代码结构

```text
app/
  main.py      # FastAPI 应用、全部 HTTP 路由与响应封装
  batcher.py   # 单 worker 动态合批器（窗口/预算/截止时间/批内到期）
  engine.py    # ONNX Runtime CPU 封装与长度归一化
  config.py    # 环境变量配置（含大量命名别名）
  parsing.py   # 宽容的请求字段/载荷解析
scripts/
  make_model.py        # 生成 padding-safe 示例 ONNX 模型
models/
  anomaly_model.onnx   # 已生成的示例模型
tests/                 # 解析、模型一致性、HTTP 与合批时序测试
Dockerfile             # 多阶段：生成模型 + slim 运行时
docker-compose.yml
```

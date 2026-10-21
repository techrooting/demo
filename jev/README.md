# LangChain 使用 JEV 示例

这个目录演示如何通过 LangChain 的 `TypeSafeClassifier` 直接调用 JEV/BJEV。

JEV 不是普通聊天模型，它更适合做低延迟、结构化判断。示例里一次请求同时包含三种 JEV 原生返回格式：

- `choice`: 从预设候选项里选择一个，例如客服消息应该分给 `billing`、`technical` 还是 `sales`。
- `score`: 按有序等级打分，例如客户情绪从 `Calm` 到 `Very angry` 的位置。
- `noul`: 返回是/否概率，例如消息是否紧急，结果是 `0` 到 `1` 的数值。

## 准备环境变量

在项目根目录的 `.env` 中配置：

```bash
BJEV_API_KEY="bjev_sk_..."
```

可选配置：

```bash
BJEV_BASE_URL="https://bjev.bytedance.net/api"
BJEV_MODEL="bjev"
```

`TypeSafeClassifier` 会在 `base_url` 后拼接 `/v1/systemone`，所以这里的 `BJEV_BASE_URL` 配到 `https://bjev.bytedance.net/api`，最终请求地址就是 `https://bjev.bytedance.net/api/v1/systemone`。

## 运行

在项目根目录执行：

```bash
uv run python jev/client.py
```

如果不用 `uv`，也可以在已安装依赖的虚拟环境里执行：

```bash
python jev/client.py
```

## 怎么获取 JEV API Key

如果使用内部 BJEV 网关，则按内部平台发放的 key 配置：

```bash
BJEV_API_KEY="bjev_sk_..."
```

如果使用 TypeSafe 官方服务，可以从 TypeSafe 控制台获取：

1. 打开 `https://console.typesafe.ai/settings/keys`。
2. 登录或注册账号。
3. 进入 JEV/API key 相关页面创建 key。
4. 按官方服务的参数调整 `analyze_customer_message_with_jev` 里的 `TypeSafeClassifier` 初始化参数，或通过环境变量覆盖。

官方资料也提到可以通过 OpenRouter、Vercel AI Gateway、Netlify AI Gateway、AIMLAPI 等渠道接入 JEV。如果走这些网关，需要按对应网关的 endpoint、model 和鉴权方式调整 `TypeSafeClassifier` 初始化参数。

## 示例结构

- `analyze_customer_message_with_jev`: 创建 `TypeSafeClassifier`、定义 `choice` / `score` / `noul` 三类问题、调用 JEV/BJEV 并解析结果。

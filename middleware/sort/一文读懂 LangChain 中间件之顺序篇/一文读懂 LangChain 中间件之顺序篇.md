# 一文读懂 LangChain 中间件之顺序篇

![image\.png](图片和附件/image.png)

# 总览

先看总结图：它展示了 6 个同步方法的调用时机，以及多个 Middleware 的执行顺序。本文重点解释这些顺序是如何形成的。

前置顺序进，包裹层层套，后置逆序出

![image\.png](图片和附件/image%202.png)

注：另外还有 6 个对应的异步方法。除执行方式为异步外，其逻辑与同步方法相同，本文不再展开。

**建议保存此图，方便以后查阅**。如果觉得内容有帮助，欢迎关注小编。

![image\.png](图片和附件/image%203.png)

友情提示：电脑查看体验更佳

# 什么是 LangChain Middleware

LangChain Middleware 是 LangChain 1\.x 引入的特性，是 Agent 执行循环中的拦截层（类似 AOP，即面向切面编程），可以在 Agent 启动、每次模型调用前后、每次工具调用前后和 Agent 结束时插入逻辑。常见用途包括动态调整 Prompt、模型和工具，以及日志审计、重试、限流、PII 与安全拦截、人工审核和状态统计等。

本文先讲中间件方法的执行顺序，后续再逐步介绍其他内容。欢迎关注后续更新。

# 从一个例子讲起

为了便于理解后文，先看一个例子。这里实现了一个 TraceMiddleware，在所有同步方法中打印日志，用来观测执行顺序：

```Python
from time import perf_counter
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from rich import print as rich_print

@tool
def add(a: int, b: int) -> int:
    """Return the sum of two integers."""
    if abs(a) > 1_000_000 or abs(b) > 1_000_000:
        raise ValueError("input is outside the demo limit")
    return a + b

class TraceMiddleware(AgentMiddleware):
    def before_agent(self, state, runtime):
        print("before_agent")
        return None

    def before_model(self, state, runtime):
        print("before_model")
        return None

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler,
    ) -> ModelResponse:
        started = perf_counter()
        print("wrap_model_call:before")
        try:
            return handler(request)
        finally:
            elapsed = perf_counter() - started
            print(f"wrap_model_call:after {elapsed:.3f}s")

    def after_model(self, state, runtime):
        print("after_model")
        return None

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        name = request.tool_call["name"]
        print(f"wrap_tool_call:before {name}")
        try:
            return handler(request)
        # 示例只捕获明确的 ValueError，并保留相同的 tool_call_id，让模型获得可恢复错误信息；其他异常继续传播
        except ValueError as exc:
            return ToolMessage(
                content=f"Recoverable tool input error: {exc}",
                tool_call_id=request.tool_call["id"],
            )
        finally:
            print(f"wrap_tool_call:after {name}")

    def after_agent(self, state, runtime):
        print("after_agent")
        return None

agent = create_agent(
    model=ChatOpenAI(model="gpt-5-nano", temperature=0),
    tools=[add],
    middleware=[TraceMiddleware()],
)

result = agent.invoke({"messages": [{"role": "user", "content": "必须调用 add 计算 2+3，只返回结果。"}]})
rich_print("final:", result)
```

输出如下（已删除与本文重点无关的字段）：

```Python
before_agent
before_model
wrap_model_call:before
wrap_model_call:after 1.621s
after_model
wrap_tool_call:before add
wrap_tool_call:after add
before_model
wrap_model_call:before
wrap_model_call:after 0.983s
after_model
after_agent
final:
{
    'messages': [
        HumanMessage(
            content='必须调用 add 计算 2+3，只返回结果。',
            id='05e17444-3e7b-45f1-8b02-9df737b6097a'
        ),
        AIMessage(
            tool_calls=[
                {
                    'name': 'add',
                    'args': {'a': 2, 'b': 3},
                    'id': 'call_jhcOONXc54YdWJfjeAMSCHcw',
                    'type': 'tool_call'
                }
            ],
        ),
        ToolMessage(
            content='5',
            name='add',
        ),
        AIMessage(
            content='5',
            tool_calls=[],
        )
    ]
}
```

从输出可看到调用顺序如下：

![image\.png](图片和附件/image%201.png)

典型调用顺序如下：

1. 先执行 `before_agent`

2. 第一轮依次执行 `before_model → wrap_model_call → after_model`；

3. 随后进入 `wrap_tool_call`；

4. 工具执行完毕后，进入第二轮模型钩子；

5. 最后执行 `after_agent`。

其中，模型调用的耗时会随每次运行而变化。



看完示例，你一定想知道其中的原理吧！别急，咱们慢慢道来

# 原理讲解

在介绍详细原理之前，先说明下文中的 graph。它是一个 StateGraph，用来控制 Agent 的执行顺序，主要由节点和边组成：

1. 节点表示模型、工具等执行单元（从 START 开始，到 END 结束）。

2. **边决定节点之间的流转关系**。

节点和边通过 `add_xxx` 一类方法添加。

## AgentMiddleware方法定义

代码中的 TraceMiddleware 继承自 AgentMiddleware。先看看 AgentMiddleware 定义了哪些方法。

```Python
# 注：只保留同步方法，省略属性字段定义和异步方法
class AgentMiddleware(Generic[StateT, ContextT, ResponseT]):
    # Agent 开始执行前运行一次，可返回状态更新
    def before_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...

    # 每次调用模型前运行，可检查或更新状态
    def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...

    # 每次模型返回后运行，可读取结果并更新状态
    def after_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...

    # Agent 执行结束后运行一次，可返回最终状态更新
    def after_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None: ...

    # 包裹模型调用，可改写请求、重试、短路或改写响应
    def wrap_model_call(self, request: ModelRequest[ContextT], handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]]) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]: ...

    # 包裹工具调用，可修改请求、重试、监控或改写结果
    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]]) -> ToolMessage | Command[Any]: ...
```

从定义看，一共有 6 个同步方法，可分为两类：before\_\* 与 after\_\*，以及 wrap\_\*。

wrap\_\* 比较特殊，先从这两个方法讲起。

## wrap\_model\_call 和 wrap\_tool\_call

`wrap_model_call` 与 `wrap_tool_call` 的实现原理相似。下面以 `wrap_tool_call` 为例说明。

调用 `create_agent` 时，会按以下步骤执行：

1. 筛选：找出实现了 wrap\_tool\_call 或 awrap\_tool\_call 的 Middleware，并按注册顺序组成 middleware\_w\_wrap\_tool\_call（before\_\* 与 after\_\* 也有类似的筛选过程）。

2. 组装：遍历 `middleware_w_wrap_tool_call`，用 `traceable` 包装每个 `wrap_tool_call` 以提供可观测能力，再交给 `_chain_tool_call_wrappers`，按注册顺序组成洋葱式调用链 `wrap_tool_call_wrapper`。

3. 注册node：将 `wrap_tool_call_wrapper` 和所有工具传给 `ToolNode`，再把该节点注册到 `StateGraph` 中执行。

```python
# langchain/agents/factory.py
# 1. 先筛出实现了 wrap_tool_call 的 middleware，按顺序组装成middleware_w_wrap_tool_call
middleware_w_wrap_tool_call = [
    m
    for m in middleware
    if m.__class__.wrap_tool_call is not AgentMiddleware.wrap_tool_call
    or m.__class__.awrap_tool_call is not AgentMiddleware.awrap_tool_call
]

wrap_tool_call_wrapper = None
if middleware_w_wrap_tool_call:
    wrappers = [
        # traceable 来自 LangSmith，用于给每个 middleware hook 增加观测 span，记录该层 wrap_tool_call 的输入、输出和耗时。_wrap_trace_kwargs 会清洗 trace 数据，去掉 runtime、handler 这类不适合序列化的对象。
        traceable(name=f"{m.name}.wrap_tool_call", **_wrap_trace_kwargs(m))(m.wrap_tool_call)
        for m in middleware_w_wrap_tool_call
    ]
    # 2. 按注册顺序组成洋葱式调用链wrap_tool_call_wrapper
    **wrap_tool_call_wrapper** = **_chain_tool_call_wrappers**(wrappers)
    
# 按顺序取所有中间件中的tools
middleware_tools = [t for m in middleware for t in getattr(m, "tools", [])]

tool_node: ToolNode | None = None

built_in_tools = [t for t in tools if isinstance(t, dict)]
regular_tools = [t for t in tools if not isinstance(t, dict)]

# 中间件中的tools或者入参中的tools
available_tools = middleware_tools + regular_tools

# 3. 把上一步产生的洋葱式调用链wrap_tool_call_wrapper和所有的tools赋值给tool_node  
tool_node = (
    ToolNode(
        tools=available_tools,
        wrap_tool_call=wrap_tool_call_wrapper,
        awrap_tool_call=awrap_tool_call_wrapper,
    )
    if available_tools or wrap_tool_call_wrapper or awrap_tool_call_wrapper
    else None
)
**graph.add_node("tools", tool_node)**
    
#  决定多个middleware的wrap_tool_call方法的调用顺序
def **_chain_tool_call_wrappers**(wrappers):
    if not wrappers:
        return None
    if len(wrappers) == 1:
        return wrappers[0]

    def compose_two(outer, inner):
        def composed(request, execute):
            def call_inner(req):
                return inner(req, execute)
            return outer(request, call_inner)
        return composed
    # **从右向左组合，最后一个在内层**
    result = wrappers[-1]
    for wrapper in reversed(wrappers[:-1]):
        result = compose_two(wrapper, result)
    return result
```

`_chain_tool_call_wrappers` 会按注册顺序把多个 Middleware 的工具 Wrapper 组成洋葱式调用链。第一个 Middleware 位于最外层：请求按 A → B → C → tool 的顺序进入，返回结果或异常则按 tool → C → B → A 的顺序退出。

如果配置 `middleware=[A(), B(), C()]`，合成后的结构可以理解为：

```python
A.wrap_tool_call(request,
  lambda req: B.wrap_tool_call(req,
    lambda req: C.wrap_tool_call(req,
      lambda req: real_tool_execution(req) # 真实的tool执行
    )
  )
)
```

wrap\_model\_call顺序相关的逻辑和wrap\_tool\_call差不多，最终会注册一个**model节点**到图中。

### tool\_node初始化逻辑

后文会用到 tool\_node 的初始化逻辑，先讲一下。满足下面任一条件时，tool\_node 不为 None：

1. 存在可用工具：create\_agent\(tools=\.\.\.\) 传入了非字典形式的工具，或者任意 Middleware 通过 middleware\.tools 注册了工具。create\_agent\(tools=\.\.\.\) 中字典形式的 built\_in\_tools 不计入 available\_tools。

2. 任意 Middleware 实现了 wrap\_tool\_call。

3. 任意 Middleware 实现了 awrap\_tool\_call（wrap\_tool\_call 的异步形式）。

## 图中的其他节点

除了上述的tools节点和model节点，所有实现了的before\_\*和after\_\*也会以节点（**graph\.add\_node**方法注册）的形式存在于图中，源码如下：

```Python
for m in middleware:
    if (
        m.__class__.before_agent is not AgentMiddleware.before_agent
        or m.__class__.abefore_agent is not AgentMiddleware.abefore_agent
    ):
        # Use RunnableCallable to support both sync and async
        # Pass None for sync if not overridden to avoid signature conflicts
        sync_before_agent = (
            m.before_agent
            if m.__class__.before_agent is not AgentMiddleware.before_agent
            else None
        )
        async_before_agent = (
            m.abefore_agent
            if m.__class__.abefore_agent is not AgentMiddleware.abefore_agent
            else None
        )
        before_agent_node = RunnableCallable(sync_before_agent, async_before_agent, trace=False)
        # 注册节点
        **graph.add_node**(
            f"{m.name}.before_agent",
            before_agent_node,
            input_schema=resolved_state_schema,
            trace_policy=_node_trace_policy(m.trace_policy),
        )
     # 其他before_*和 after*方法也会调用graph.add_node
```



## 图中边的添加

### 关键变量

源码里会先计算 4 个关键目标节点：

```Python
# entry_node = before_agent[0] or before_model[0] or "model"
if middleware_w_before_agent:
    entry_node = f"{middleware_w_before_agent[0].name}.before_agent"
elif middleware_w_before_model:
    entry_node = f"{middleware_w_before_model[0].name}.before_model"
else:
    entry_node = "model"

# loop_entry_node = before_model[0] or "model"
if middleware_w_before_model:
    loop_entry_node = f"{middleware_w_before_model[0].name}.before_model"
else:
    loop_entry_node = "model"

# loop_exit_node = after_model[0] or "model"
if middleware_w_after_model:
    loop_exit_node = f"{middleware_w_after_model[0].name}.after_model"
else:
    loop_exit_node = "model"

# exit_node = after_agent[-1] or END
if middleware_w_after_agent:
    exit_node = f"{middleware_w_after_agent[-1].name}.after_agent"
else:
    exit_node = END
```

总结为：

```python
entry_node = before_agent[0] or before_model[0] or "model"
loop_entry_node = before_model[0] or "model"
loop_exit_node = after_model[0] or "model"
exit_node = after_agent[-1] or END
```

含义如下：

- `entry_node`：从 `START` 进入的第一个节点。优先是第一个 `before_agent`。

- `loop_entry_node`：每轮 agent loop 的入口。通常是第一个 `before_model`。

- loop\_exit\_node：每轮模型调用后的出口。存在 after\_model 时，它对应注册列表中的第一个 after\_model；该节点在 after\_model 链中最后执行，并决定后续路由。否则，loop\_exit\_node 为 model。

- `exit_node`：整条 agent 链结束前的出口。存在 `after_agent` 时是最后一个 `after_agent`，否则是 `END`。

### \_add\_middleware\_edge 方法讲解

> 为了后续方便讲解，先讲一下\_add\_middleware\_edge方法
> 
> 

`_add_middleware_edge()` 是一个内部 helper，用来给某个 middleware hook node 添加“下一步去哪”的边，其定义如下：

```python
def _add_middleware_edge(
    graph, # 正在构建的 StateGraph。这个函数会往它上面加边
    *,
    name: str, # 当前 middleware node 的名字，例如 xxx.before_agent，也就是“从哪个节点出发”
    default_destination: str, # 默认下一站。如果当前 hook 没有返回 jump_to，或者没有配置可跳转目标，就走这里
    model_destination: str, # 当 hook 返回 {"jump_to": "model"} 时要跳到哪里。这里不一定是字面量 "model"，通常是 loop_entry_node，也就是“重新进入模型循环的入口”。如果存在 before_model，它就是第一个 before_model；否则才是 "model"。
    end_destination: str, # 当 hook 返回 {"jump_to": "end"} 时要跳到哪里。通常是 exit_node，也就是 after_agent 链或最终 END。
    can_jump_to: list[JumpTo] | None, # 这个 hook 被允许跳到哪些目标。类型是 Literal["tools", "model", "end"]，一般来自 @hook_config(can_jump_to=[...]) 或相关 decorator 配置。
) -> None:
    if can_jump_to:

        def jump_edge(state: dict[str, Any]) -> str:
            return (
                _resolve_jump(
                    state.get("jump_to"),
                    model_destination=model_destination,
                    end_destination=end_destination,
                )
                or default_destination
            )

        destinations = [default_destination]

        if "end" in can_jump_to:
            destinations.append(end_destination)
        if "tools" in can_jump_to:
            destinations.append("tools")
        if "model" in can_jump_to and name != model_destination:
            destinations.append(model_destination)

        graph.add_conditional_edges(name, RunnableCallable(jump_edge, trace=False), destinations)

    else: # 果 can_jump_to 为空，这个节点只能走固定下一站：
        graph.add_edge(name, default_destination)


def _resolve_jump(
    jump_to: JumpTo | None,
    *,
    model_destination: str,
    end_destination: str,
) -> str | None:
    if jump_to == "model":
        return model_destination
    if jump_to == "end":
        return end_destination
    if jump_to == "tools":
        return "tools"
    return None
```

以 jump\_to 为例：

can\_jump\_to 的取值范围为 Literal\["tools", "model", "end"\]。

```python
@hook_config(can_jump_to=["end"])
def before_model(self, state, runtime):
    if too_many_calls:
        return {"jump_to": "end"}
    return None
```

编译时，`_add_middleware_edge` 会允许这个 node 有两种出口：

```text
before_model -> default_destination
before_model -> end_destination
```

运行时如果返回 `{"jump_to": "end"}`，就提前结束；否则继续默认链路。

### start\-\>entry\_node

如果存在 before\_agent，entry\_node 就是 before\_agent\[0\]。

```Python
# start-> entry_node = before_agent[0] or before_model[0] or "model"
graph.add_edge(START, entry_node)
```

### tool\_node相关处理逻辑（枚举和tool\_node相关的情况，有点烧脑）

#### 有 `tool_node` 时，会加两组条件边

##### tools \-\> loop\_entry\_node or exit\_node

工具执行完后，默认回到 `loop_entry_node`，也就是下一轮模型调用入口。

只有两种情况允许从 `tools` 直接结束：

- `工具都 return_direct=True 时`：工具结果就是最终结果，不需要再问模型。

- `结构化输出工具（structured_output_tools）执行成功时`：结构化输出工具执行成功后，可以直接结束。

```Python
tools_to_model_destinations = [loop_entry_node]
if (
    any(tool.return_direct for tool in tool_node.tools_by_name.values())
    or structured_output_tools
):
    tools_to_model_destinations.append(exit_node)

graph.add_conditional_edges(
    "tools",
    RunnableCallable(
        _make_tools_to_model_edge(
            tool_node=tool_node,
            model_destination=loop_entry_node,
            structured_output_tools=structured_output_tools,
            end_destination=exit_node,
        ),
        trace=False,
    ),
    tools_to_model_destinations, # 只是映射关系，核心逻辑在_make_tools_to_model_edge
)

def _make_tools_to_model_edge(
    *,
    tool_node: ToolNode,
    model_destination: str,
    structured_output_tools: dict[str, OutputToolBinding[Any]],
    end_destination: str,
) -> Callable[[dict[str, Any]], str | None]:
    def tools_to_model(state: dict[str, Any]) -> str | None:
        last_ai_message, tool_messages = _fetch_last_ai_and_tool_messages(state["messages"])

        # 1. If no AIMessage exists (e.g., messages were cleared), route to model
        if last_ai_message is None:
            return model_destination

        executed_by_id = {
            t.tool_call_id: t.name for t in tool_messages if t.name in tool_node.tools_by_name
        }
        executed_names = [
            name
            for c in last_ai_message.tool_calls
            if (name := executed_by_id.get(c["id"] or "", c["name"])) in tool_node.tools_by_name
        ]
        # 工具都 return_direct=True 时进入 end_destination
        if executed_names and all(
            tool_node.tools_by_name[name].return_direct for name in executed_names
        ):
            return end_destination

        # 结构化输出工具执行成功时进入 end_destination
        if any(t.name in structured_output_tools and t.status != "error" for t in tool_messages):
            return end_destination

        return model_destination

    return tools_to_model

```

##### loop\_exit\_node \-\> tools / exit\_node / loop\_entry\_node

条件判断顺序如下：

- 存在 jump\_to：按 jump\_to 路由。

- 不存在 AIMessage：结束，转到 exit\_node。

- 没有工具调用（通过 len\(last\_ai\_message\.tool\_calls\) == 0 判断）：结束。

- 存在待执行工具调用：通过 Send 机制并发分发给 tools 节点。

- 结构化输出已生成：结束。

- 特殊情况需要重回模型入口：去 `loop_entry_node`。

```Python
model_to_tools_destinations = ["tools", exit_node]
if response_format or loop_exit_node != "model" or middleware_w_wrap_model_call:
    model_to_tools_destinations.append(loop_entry_node)

graph.add_conditional_edges(
    loop_exit_node,
    RunnableCallable(
        **_make_model_to_tools_edge**(
            model_destination=loop_entry_node,
            structured_output_tools=structured_output_tools,
            end_destination=exit_node,
        ),
        trace=False,
    ),
    model_to_tools_destinations,
)


def _make_model_to_tools_edge(
    *,
    model_destination: str,
    structured_output_tools: dict[str, OutputToolBinding[Any]],
    end_destination: str,
) -> Callable[[dict[str, Any]], str | list[Send] | None]:
    def model_to_tools(
        state: dict[str, Any],
    ) -> str | list[Send] | None:
        # 1. 存在 jump_to：按 jump_to 路由。
        if jump_to := state.get("jump_to"):
            return _resolve_jump(
                jump_to,
                model_destination=model_destination,
                end_destination=end_destination,
            )

        last_ai_message, tool_messages = _fetch_last_ai_and_tool_messages(state["messages"])

        # 2. 不存在AIMessage
        if last_ai_message is None:
            return end_destination

        tool_message_ids = [m.tool_call_id for m in tool_messages]

        # 3. 没有工具调用：结束，去 exit_node
        if len(last_ai_message.tool_calls) == 0:
            return end_destination

        pending_tool_calls = [
            c
            for c in last_ai_message.tool_calls
            if c["id"] not in tool_message_ids and c["name"] not in structured_output_tools
        ]

        # 4. 有普通工具调用：去 tools
        if pending_tool_calls:
            return [Send("tools", [tool_call]) for tool_call in pending_tool_calls]

        # 5. 结构化输出已生成：结束
        if state.get("structured_response") is not None:
            return end_destination

        # 6. 特殊情况需要重回模型入口：去 loop_entry_node
        return model_destination

    return model_to_tools


```

#### 没有 tool\_node，但有structured\_output\_tools：loop\_exit\_node\-\>loop\_entry\_node/exit\_node

1. 有 jump\_to \-\> 按 jump\_to 走

2. 有 structured\_response \-\> 结束

3. 否则 \-\> 回模型入口重试（结构化输出没有产出时，模型需要再试一轮）

```python
elif len(structured_output_tools) > 0:
    graph.add_conditional_edges(
        loop_exit_node,
        RunnableCallable(
            _make_model_to_model_edge(
                model_destination=loop_entry_node,
                end_destination=exit_node,
            ),
            trace=False,
        ),
        [loop_entry_node, exit_node],
    )

def _make_model_to_model_edge(
    *,
    model_destination: str,
    end_destination: str,
) -> Callable[[dict[str, Any]], str | list[Send] | None]:
    def model_to_model(
        state: dict[str, Any],
    ) -> str | list[Send] | None:
        # 1. 有 jump_to -> 按 jump_to 走
        if jump_to := state.get("jump_to"):
            return _resolve_jump(
                jump_to,
                model_destination=model_destination,
                end_destination=end_destination,
            )

        # 2. 有 structured_response -> 结束
        if state.get("structured_response") is not None:
            return end_destination

        # 3. 默认回模型入口重试
        return model_destination

    return model_to_model

```

#### 没有 tool\_node，也没有 after\_model：loop\_exit\_node\-\>exit\_node

如果 `loop_exit_node == "model"`，说明没有 `after_model` middleware。也没有工具、没有结构化输出，所以模型调用完就直接退出：

```python
elif loop_exit_node == "model":
    graph.add_edge(loop_exit_node, exit_node)
```

#### 没有 tool\_node，但有 after\_model：after\_model\[0\]\-\>exit\_node

没有工具时，`after_model` 默认执行完就去 `exit_node`。但如果这个 `after_model` hook 声明了 `can_jump_to=["model"]` 或 `can_jump_to=["end"]`，它就可以通过返回 `{"jump_to": "model"}` 把流程重新送回 `loop_entry_node`，再来一轮模型调用

```python
else:
    _add_middleware_edge(
        graph,
        name=f"{middleware_w_after_model[0].name}.after_model",
        default_destination=exit_node,
        model_destination=loop_entry_node,
        end_destination=exit_node,
        can_jump_to=_get_can_jump_to(middleware_w_after_model[0], "after_model"),
    )
```

### before\_agent正向和before\_agent\[\-1\]\-\>loop\_entry\_node\(before\_model\[0\]\)

```Python
if middleware_w_before_agent:
    # 多before_agent顺序执行
    for m1, m2 in itertools.pairwise(middleware_w_before_agent):
        _add_middleware_edge(
            graph,
            name=f"{m1.name}.before_agent",
            default_destination=f"{m2.name}.before_agent",
            model_destination=loop_entry_node,
            end_destination=exit_node,
            can_jump_to=_get_can_jump_to(m1, "before_agent"),
        )
    # 最后before_agent 到 loop_entry_node = before_model[0] or "model"
    _add_middleware_edge(
        graph,
        name=f"{middleware_w_before_agent[-1].name}.before_agent",
        default_destination=loop_entry_node,
        model_destination=loop_entry_node,
        end_destination=exit_node,
        can_jump_to=_get_can_jump_to(middleware_w_before_agent[-1], "before_agent"),
    )
```

### before\_model正向和before\_model\[\-1\]\-\>model\(包含wrap\_model\_call\)

```Python
if middleware_w_before_model:
    # 多before_model顺序执行
    for m1, m2 in itertools.pairwise(middleware_w_before_model):
        _add_middleware_edge(
            graph,
            name=f"{m1.name}.before_model",
            default_destination=f"{m2.name}.before_model",
            model_destination=loop_entry_node,
            end_destination=exit_node,
            can_jump_to=_get_can_jump_to(m1, "before_model"),
        )
    # 最后before_model 到 model"
    _add_middleware_edge(
        graph,
        name=f"{middleware_w_before_model[-1].name}.before_model",
        default_destination="model",
        model_destination=loop_entry_node,
        end_destination=exit_node,
        can_jump_to=_get_can_jump_to(middleware_w_before_model[-1], "before_model"),
    )
```

### model\-\>after\_model\[\-1\]和after\_model逆向

```Python
if middleware_w_after_model:
    # model->最后一个after_model
    graph.add_edge("model", f"{middleware_w_after_model[-1].name}.after_model")
    # after_model逆向加
    for idx in range(len(middleware_w_after_model) - 1, 0, -1):
        m1 = middleware_w_after_model[idx]
        m2 = middleware_w_after_model[idx - 1]
        _add_middleware_edge(
            graph,
            name=f"{m1.name}.after_model",
            default_destination=f"{m2.name}.after_model",
            model_destination=loop_entry_node,
            end_destination=exit_node,
            can_jump_to=_get_can_jump_to(m1, "after_model"),
        )
```

### after\_agent逆向和after\_agent\[0\] \-\> END

```Python
if middleware_w_after_agent:
    # after_agent逆向
    for idx in range(len(middleware_w_after_agent) - 1, 0, -1):
        m1 = middleware_w_after_agent[idx]
        m2 = middleware_w_after_agent[idx - 1]
        _add_middleware_edge(
            graph,
            name=f"{m1.name}.after_agent",
            default_destination=f"{m2.name}.after_agent",
            model_destination=loop_entry_node,
            end_destination=exit_node,
            can_jump_to=_get_can_jump_to(m1, "after_agent"),
        )

    # 最后的after_agent -> END
    _add_middleware_edge(
        graph,
        name=f"{middleware_w_after_agent[0].name}.after_agent",
        default_destination=END,
        model_destination=loop_entry_node,
        end_destination=exit_node,
        can_jump_to=_get_can_jump_to(middleware_w_after_agent[0], "after_agent"),
    )
```

# 总结

顺序链是在 `create_agent()` 编译 `StateGraph` 时拼出来的，`create_agent` 会先按类型筛出真正被重写的方法，再用两种机制完成编排：

1. `before_agent`、`before_model`、`model`、`after_model`、`tools`、`after_agent` 是 graph 节点。

2. `wrap_model_call` 和 `wrap_tool_call` 不是独立 graph node，而是分别包在 `model` node 和 `tools` node 的内部。

换句话说，顺序钩子（before\_\*和after\_\*）通过“图的边”串联，包装钩子（`wrap_`\*）通过“函数嵌套”串联。`wrap_*` 呈现“后进先出”的根本原因是 Python 调用栈，而不是框架额外维护了一套栈结构。

其他注意点：

1. `before_agent` 和 `before_model` 按注册顺序连接，`after_model` 和 `after_agent` 则按相反顺序连接。

2. `before_agent` 只从 `START` 进入一次；

3. `before_model` 是循环入口，所以每次工具执行完后都会回到它；

4. `after_agent` 只在最终结束前执行一次。

![结尾关注\.png](图片和附件/结尾关注.png)




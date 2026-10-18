from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, before_model
from langchain.messages import AIMessage
from langgraph.runtime import Runtime
from typing_extensions import NotRequired


class CountingState(AgentState):
    model_call_count: NotRequired[int]


@dataclass
class RequestContext:
    enable_limit: bool
    max_model_calls: int


@before_model(state_schema=CountingState, can_jump_to=["end"])
def count_and_limit(
    state: CountingState,
    runtime: Runtime[RequestContext],
) -> dict[str, Any] | None:
    # runtime.context 是本次调用传入的只读配置
    if not runtime.context.enable_limit:
        return None  # 不更新任何状态，继续正常调用模型

    # state 保存当前运行状态，也能读取自定义字段
    count = state.get("model_call_count", 0)
    if count >= runtime.context.max_model_calls:
        return {
            "messages": [AIMessage(content="模型调用次数已达上限。")],
            "jump_to": "end",  # 跳过后续模型调用，直接结束
        }

    # 返回 dict，其中字段会合入 agent state
    return {"model_call_count": count + 1}


agent = create_agent(
    model="openai:gpt-5.2",
    tools=[],
    middleware=[count_and_limit],
    context_schema=RequestContext,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "帮我分析这段代码"}]},
    context=RequestContext(enable_limit=True, max_model_calls=3),
)
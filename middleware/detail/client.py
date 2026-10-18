from typing import Callable
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    before_model,
    wrap_model_call,
)
from langgraph.runtime import Runtime
import os
from langchain.chat_models import init_chat_model


model = init_chat_model(
    model="gpt-5.5-2026-04-24",                 # model 参数会作为 azure_deployment 使用
    model_provider="azure_openai",
    api_version="2024-03-01-preview",
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    max_tokens=4096,
    max_retries=3,
)


# 写法一：继承 AgentMiddleware，把多个 hook 和配置收拢到一个对象中
class RetryLoggingMiddleware(AgentMiddleware):
    def __init__(self, max_retries: int = 3) -> None:
        self.max_retries = max_retries

    def before_model(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict | None:
        print(f"即将调用模型，消息数: {len(state['messages'])}")
        return None

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        for attempt in range(self.max_retries):
            try:
                return handler(request)
            except Exception:
                if attempt == self.max_retries - 1:
                    raise
                print(f"模型调用失败，准备第 {attempt + 2} 次调用")


class_agent = create_agent(
    model=model,
    tools=[],
    middleware=[RetryLoggingMiddleware(max_retries=3)],
)


# 写法二：用 Decorator 把无共享状态的 hook 拆成独立函数
MAX_RETRIES = 3


@before_model
def log_before_model(
    state: AgentState,
    runtime: Runtime,
) -> dict | None:
    print(f"即将调用模型，消息数: {len(state['messages'])}")
    return None


@wrap_model_call
def retry_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    for attempt in range(MAX_RETRIES):
        try:
            return handler(request)
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"模型调用失败，准备第 {attempt + 2} 次调用")



decorator_agent = create_agent(
    model=model,
    tools=[],
    middleware=[log_before_model, retry_model],
)
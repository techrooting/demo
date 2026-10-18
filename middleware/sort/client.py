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
    ChatOpenAI(model="gpt-5-nano", temperature=0),
    tools=[add],
    middleware=[TraceMiddleware()],
)

result = agent.invoke({"messages": [{"role": "user", "content": "必须调用 add 计算 2+3，只返回结果。"}]})
rich_print("final:", result)
# before_agent
# before_model
# wrap_model_call:before
# wrap_model_call:after 1.621s
# after_model
# wrap_tool_call:before add
# wrap_tool_call:after add
# before_model
# wrap_model_call:before
# wrap_model_call:after 0.983s
# after_model
# after_agent
# final:
# {
#     'messages': [
#         HumanMessage(
#             content='必须调用 add 计算 2+3，只返回结果。',
#             additional_kwargs={},
#             response_metadata={},
#             id='05e17444-3e7b-45f1-8b02-9df737b6097a'
#         ),
#         AIMessage(
#             content='',
#             additional_kwargs={'refusal': None},
#             response_metadata={
#                 'token_usage': {
#                     'completion_tokens': 20,
#                     'prompt_tokens': 141,
#                     'total_tokens': 161,
#                     'completion_tokens_details': {
#                         'accepted_prediction_tokens': None,
#                         'audio_tokens': 0,
#                         'reasoning_tokens': 0,
#                         'rejected_prediction_tokens': None,
#                         'text_tokens': None
#                     },
#                     'prompt_tokens_details': {
#                         'audio_tokens': 0,
#                         'cache_write_tokens': None,
#                         'cached_tokens': 0,
#                         'image_tokens': None,
#                         'text_tokens': None
#                     }
#                 },
#                 'model_provider': 'openai',
#                 'model_name': 'gpt-5.5-2026-04-24',
#                 'system_fingerprint': None,
#                 'id': 'chatcmpl-EZcUc1mjjMy4kPn1sXP0wqo53KOXO',
#                 'service_tier': 'default',
#                 'prompt_filter_results': None,
#                 'finish_reason': 'tool_calls',
#                 'logprobs': None,
#                 'content_filter_results': {}
#             },
#             id='lc_run--01a144f3-3aad-78e2-95bc-84ea51342f6c-0',
#             tool_calls=[
#                 {
#                     'name': 'add',
#                     'args': {'a': 2, 'b': 3},
#                     'id': 'call_jhcOONXc54YdWJfjeAMSCHcw',
#                     'type': 'tool_call'
#                 }
#             ],
#             invalid_tool_calls=[],
#             usage_metadata={
#                 'input_tokens': 141,
#                 'output_tokens': 20,
#                 'total_tokens': 161,
#                 'input_token_details': {'audio': 0, 'cache_read': 0},
#                 'output_token_details': {'audio': 0, 'reasoning': 0}
#             }
#         ),
#         ToolMessage(
#             content='5',
#             name='add',
#             id='53a22363-5186-45e6-b90c-8fa0b0c77f82',
#             tool_call_id='call_jhcOONXc54YdWJfjeAMSCHcw'
#         ),
#         AIMessage(
#             content='5',
#             additional_kwargs={'refusal': None},
#             response_metadata={
#                 'token_usage': {
#                     'completion_tokens': 4,
#                     'prompt_tokens': 172,
#                     'total_tokens': 176,
#                     'completion_tokens_details': {
#                         'accepted_prediction_tokens': None,
#                         'audio_tokens': 0,
#                         'reasoning_tokens': 0,
#                         'rejected_prediction_tokens': None,
#                         'text_tokens': None
#                     },
#                     'prompt_tokens_details': {
#                         'audio_tokens': 0,
#                         'cache_write_tokens': None,
#                         'cached_tokens': 0,
#                         'image_tokens': None,
#                         'text_tokens': None
#                     }
#                 },
#                 'model_provider': 'openai',
#                 'model_name': 'gpt-5.5-2026-04-24',
#                 'system_fingerprint': None,
#                 'id': 'chatcmpl-EZcUd5hT6uh298zFApD6WA4LZrnzn',
#                 'service_tier': 'default',
#                 'prompt_filter_results': None,
#                 'finish_reason': 'stop',
#                 'logprobs': None,
#                 'content_filter_results': {}
#             },
#             id='lc_run--01a144f3-4105-7e61-ae5c-6e6dd90159ed-0',
#             tool_calls=[],
#             invalid_tool_calls=[],
#             usage_metadata={
#                 'input_tokens': 172,
#                 'output_tokens': 4,
#                 'total_tokens': 176,
#                 'input_token_details': {'audio': 0, 'cache_read': 0},
#                 'output_token_details': {'audio': 0, 'reasoning': 0}
#             }
#         )
#     ]
# }
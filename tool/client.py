import os

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from rich import print as rich_print


@tool
def add(a: int, b: int) -> int:
    """计算两个整数之和。"""
    return a + b



model = init_chat_model(
    model="gpt-5.5-2026-04-24",                 # model 参数会作为 azure_deployment 使用
    model_provider="azure_openai",
    api_version="2024-03-01-preview",
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    max_tokens=4096,
    max_retries=3,
)

agent = create_agent(
    model=model,
    tools=[add],
)
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "1+2=?",
    }]
})

rich_print("返回结果：",result)

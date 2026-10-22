import sys
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

load_dotenv(find_dotenv(), override=True)

from langchain.agents import create_agent
from langchain_core.tools import tool
from model_factory import create_azure_openai_model
from rich import print as rich_print


@tool
def add(a: int, b: int) -> int:
    """计算两个整数之和。"""
    return a + b


model = create_azure_openai_model()

agent = create_agent(
    model=model,
    tools=[add],
)
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "请使用 add 工具计算 1+2=?",
    }]
})

rich_print("返回结果：",result)

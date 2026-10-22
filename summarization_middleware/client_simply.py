from pathlib import Path
import subprocess
import sys
from dotenv import load_dotenv
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
load_dotenv(project_root / ".env", override=True)
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.tools import tool
from model_factory import create_azure_openai_model
from rich import print as rich_print

@tool
def write_file(path: str, content: str) -> str:
    """写入 UTF-8 文本文件。"""
    Path(path).write_text(content, encoding="utf-8")
    return f"wrote {path}"

@tool
def run_tests(path: str) -> str:
    """运行指定测试文件。"""
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", path],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout + completed.stderr


model = create_azure_openai_model()

agent = create_agent(
    model=model,
    tools=[write_file, run_tests],
    middleware=[SummarizationMiddleware()],
)

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "请先使用 todo list 规划，在 demo.py 中写 hello_world 函数并运行测试",
    }]
})

rich_print("返回结果：",result)

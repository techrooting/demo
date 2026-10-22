import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from dotenv import find_dotenv, load_dotenv
from model_factory import create_azure_openai_model
from rich import print as rich_print

load_dotenv(find_dotenv(), override=True)


class TracingFilesystemBackend(FilesystemBackend):
    """FilesystemBackend wrapper that records when skill files are actually touched."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.events: list[dict[str, Any]] = []

    def _record(self, operation: str, path: str, result: Any) -> None:
        self.events.append(
            {
                "time": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                "operation": operation,
                "path": path,
                "result": self._summarize_result(result),
            }
        )

    def _summarize_result(self, result: Any) -> dict[str, Any]:
        if hasattr(result, "entries"):
            entries = result.entries or []
            return {
                "error": result.error,
                "entry_count": len(entries),
                "entries": [
                    {"path": item.get("path"), "is_dir": item.get("is_dir")}
                    for item in entries
                ],
            }

        if hasattr(result, "file_data"):
            content = result.file_data.get("content", "") if result.file_data else ""
            return {
                "error": result.error,
                "encoding": result.file_data.get("encoding") if result.file_data else None,
                "line_range": [result.start_line, result.end_line],
                "content_chars": len(content),
                "preview": content[:160],
            }

        if isinstance(result, list):
            return {
                "file_count": len(result),
                "files": [
                    {
                        "path": item.path,
                        "error": item.error,
                        "content_bytes": len(item.content or b""),
                    }
                    for item in result
                ],
            }

        return {"repr": repr(result)}

    def ls(self, path: str) -> Any:
        result = super().ls(path)
        self._record("ls", path, result)
        return result

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> Any:
        result = super().read(file_path, offset=offset, limit=limit)
        self._record("read", file_path, result)
        return result

    def download_files(self, paths: list[str]) -> Any:
        result = super().download_files(paths)
        self._record("download_files", ", ".join(paths), result)
        return result


def build_model() -> Any:
    return create_azure_openai_model()


def build_loading_report(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    report = []
    for event in events:
        path = event["path"]
        operation = event["operation"]
        if operation == "ls" and path.rstrip("/") == "/skills":
            phase = "metadata-source-scan"
            loaded = "skill source directory"
        elif operation == "ls" and path.rstrip("/") == "/skills/load-trace-demo":
            phase = "instruction-directory-inspection"
            loaded = "skill directory contents"
        elif operation == "download_files" and "SKILL.md" in path:
            phase = "metadata-frontmatter-parse"
            loaded = "skill metadata from SKILL.md frontmatter"
        elif operation == "read" and path.endswith("/SKILL.md"):
            phase = "instruction-progressive-disclosure"
            loaded = "full skill instructions"
        elif operation == "read" and "/references/" in path:
            phase = "support-reference"
            loaded = "reference document"
        elif operation == "read" and "/scripts/" in path:
            phase = "support-script"
            loaded = "helper script"
        elif operation == "read" and "/assets/" in path:
            phase = "support-asset"
            loaded = "static asset"
        else:
            phase = "other"
            loaded = "other backend resource"

        report.append(
            {
                "time": event["time"],
                "phase": phase,
                "operation": operation,
                "path": path,
                "loaded": loaded,
                "result": event["result"],
            }
        )
    return report


def main() -> None:
    backend = TracingFilesystemBackend(root_dir=Path(__file__).parent)
    deep_agent = create_deep_agent(
        model=build_model(),
        backend=backend,
        skills=["/skills"],
    )

    result = deep_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "请使用 load-trace-demo skill。"
                        "先读取它的 SKILL.md，再按 SKILL.md 要求读取 references、scripts、assets 下的文件。"
                        "最后用中文输出加载报告：怎么加载的、什么时间加载哪个目录/文件、每次加载结果是什么。"
                    ),
                }
            ]
        }
    )

    rich_print("\n[bold]=== Backend load trace (actual filesystem operations) ===[/bold]")
    rich_print(json.dumps(backend.events, ensure_ascii=False, indent=2))

    rich_print("\n[bold]=== Loading report from backend trace ===[/bold]")
    rich_print(json.dumps(build_loading_report(backend.events), ensure_ascii=False, indent=2))

    rich_print("\n[bold]=== Agent result ===[/bold]")
    rich_print(result)


if __name__ == "__main__":
    main()

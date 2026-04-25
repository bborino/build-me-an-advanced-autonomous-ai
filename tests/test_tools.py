from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autonomous_assistant.config import AssistantSettings
from autonomous_assistant.tools.base import ToolRuntimeContext
from autonomous_assistant.tools.filesystem import FilesystemTool
from autonomous_assistant.tools.shell import ShellTool


class ToolTests(unittest.TestCase):
    def create_context(self, root: Path) -> ToolRuntimeContext:
        data_dir = root / ".assistant_data"
        settings = AssistantSettings(
            workspace_root=root,
            data_dir=data_dir,
            memory_db_path=data_dir / "memory.sqlite3",
            shell_enabled=True,
        )
        settings.ensure_directories()
        return ToolRuntimeContext(workspace_root=root, settings=settings, session_id="test")

    def test_filesystem_write_and_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            tool = FilesystemTool()
            context = self.create_context(root)
            write_result = tool.execute(
                {"operation": "write", "path": "notes/test.txt", "content": "hello"},
                context,
            )
            self.assertTrue(write_result.ok)

            read_result = tool.execute(
                {"operation": "read", "path": "notes/test.txt", "start_line": 1, "end_line": 5},
                context,
            )
            self.assertTrue(read_result.ok)
            self.assertIn("hello", read_result.content["text"])

    def test_shell_blocks_destructive_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            tool = ShellTool()
            context = self.create_context(root)
            result = tool.execute({"command": "rm -rf ."}, context)
            self.assertFalse(result.ok)
            self.assertIn("blocked", result.summary.lower())


if __name__ == "__main__":
    unittest.main()

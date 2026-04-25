from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autonomous_assistant.assistant import AutonomousAssistant
from autonomous_assistant.config import AssistantSettings


class AssistantTests(unittest.TestCase):
    def create_settings(self, root: Path) -> AssistantSettings:
        data_dir = root / ".assistant_data"
        return AssistantSettings(
            workspace_root=root,
            data_dir=data_dir,
            memory_db_path=data_dir / "memory.sqlite3",
            shell_enabled=False,
        )

    def test_capabilities_include_core_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = self.create_settings(Path(tmp_dir))
            assistant = AutonomousAssistant(settings=settings)
            names = {tool["name"] for tool in assistant.capabilities()["tools"]}
            self.assertIn("filesystem", names)
            self.assertIn("web_search", names)
            self.assertIn("python", names)

    def test_assistant_run_completes_without_external_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = self.create_settings(Path(tmp_dir))
            assistant = AutonomousAssistant(settings=settings)
            report = assistant.run("Create a launch checklist for a demo day presentation.")
            self.assertTrue(report.final_output)
            self.assertIn(report.status.value, {"completed", "partial"})
            self.assertGreaterEqual(len(report.plan), 3)


if __name__ == "__main__":
    unittest.main()


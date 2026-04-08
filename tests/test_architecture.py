from __future__ import annotations

import ast
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "shadowgen_ml_service"


def module_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
    return imports


class ArchitectureTests(unittest.TestCase):
    def test_core_does_not_depend_on_http_or_frameworks(self) -> None:
        for path in (SRC_ROOT / "core").rglob("*.py"):
            imports = module_imports(path)
            self.assertFalse(any(item.startswith("fastapi") for item in imports), path)
            self.assertFalse(any(item.startswith("pydantic") for item in imports), path)
            self.assertFalse(any(item.startswith("shadowgen_ml_service.interfaces") for item in imports), path)

    def test_application_does_not_import_interface_modules(self) -> None:
        for path in (SRC_ROOT / "application").rglob("*.py"):
            imports = module_imports(path)
            self.assertFalse(any(item.startswith("shadowgen_ml_service.interfaces.http") for item in imports), path)

    def test_stage_implementations_do_not_import_other_stage_packages(self) -> None:
        stages_root = SRC_ROOT / "infrastructure" / "stages"
        for path in stages_root.rglob("*.py"):
            stage_name = path.parent.name
            imports = module_imports(path)
            forbidden = [
                item
                for item in imports
                if item.startswith("shadowgen_ml_service.infrastructure.stages.")
                and f".{stage_name}." not in item
                and not item.startswith("shadowgen_ml_service.infrastructure.stages.shared")
            ]
            self.assertEqual(forbidden, [], path)


if __name__ == "__main__":
    unittest.main()

"""Integration checks against the package as a consumer sees it.

These deliberately avoid reaching into private modules. An editable checkout can
hide a missing file or an unexported name; importing only the public surface and
running the shipped examples as subprocesses catches that.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import frechet_edit

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "examples"


class TestPublicSurface:
    def test_version_is_exposed(self):
        assert isinstance(frechet_edit.__version__, str)
        assert frechet_edit.__version__

    @pytest.mark.parametrize(
        "name",
        [
            "discrete_edit_distance",
            "ordinary_discrete_frechet",
            "verify_witness",
            "replay",
            "EditResult",
            "Deletion",
            "Insertion",
            "VerificationReport",
            "NumericallyAmbiguous",
            "UnsupportedDimensionError",
        ],
    )
    def test_documented_name_is_importable(self, name):
        assert hasattr(frechet_edit, name), f"{name} missing from the public API"
        assert name in frechet_edit.__all__, f"{name} missing from __all__"

    def test_all_entries_actually_exist(self):
        for name in frechet_edit.__all__:
            assert hasattr(frechet_edit, name), f"__all__ advertises missing {name}"

    def test_py_typed_marker_is_shipped(self):
        package_dir = Path(frechet_edit.__file__).parent
        assert (package_dir / "py.typed").exists(), "py.typed marker is not shipped"

    def test_readme_first_example_works_verbatim(self):
        """The exact snippet from README.md must produce the documented output."""
        reference = np.array([[0.0], [1.0], [2.0]])
        observation = np.array([[0.0], [1.0], [100.0], [2.0]])
        result = frechet_edit.discrete_edit_distance(
            reference, observation, delta=0.1, operations="delete", return_witness=True
        )
        assert result.cost == 1
        assert [e.as_dict() for e in result.edits] == [{"op": "delete", "index": 2}]
        assert frechet_edit.verify_witness(reference, observation, result).ok is True
        assert frechet_edit.ordinary_discrete_frechet(reference, observation) == 98.0

    def test_readme_insertion_example_works_verbatim(self):
        R, Q = np.array([[0.0], [2.0], [4.0]]), np.array([[0.0]])
        r = frechet_edit.discrete_edit_distance(
            R, Q, delta=1.0, operations="insert", return_witness=True
        )
        assert r.cost == 1
        assert r.edits[0].point == (3.0,)


class TestShippedExamples:
    @pytest.mark.parametrize("script", ["basic_edits.py", "route_audit.py"])
    def test_example_runs_offline(self, script, tmp_path):
        """Run from a DIFFERENT working directory, so nothing resolves by accident."""
        path = EXAMPLES / script
        assert path.exists(), f"documented example {script} is missing"
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, f"{script} failed:\n{proc.stdout}\n{proc.stderr}"
        assert proc.stdout.strip(), f"{script} produced no output"

    def test_examples_import_only_the_public_package(self):
        """An example that reaches into a private module would not survive install."""
        for script in EXAMPLES.glob("*.py"):
            text = script.read_text(encoding="utf-8")
            assert "frechet_edit._" not in text, f"{script.name} imports a private module"
            assert "from experiments" not in text, (
                f"{script.name} imports the repo-only experiments package, so it "
                "would not run from an installed wheel"
            )


class TestDocumentedFilesExist:
    @pytest.mark.parametrize(
        "relative",
        [
            "README.md",
            "LICENSE",
            "CITATION.cff",
            "CHANGELOG.md",
            "pyproject.toml",
            "docs/definition.md",
            "docs/recurrences.md",
            "docs/witness-invariants.md",
            "docs/numerics.md",
            "docs/oracle-protocol.md",
            "docs/paper-map.md",
            "docs/performance.md",
            "docs/limitations.md",
            "docs/data-and-labels.md",
            "docs/experiment-protocol.md",
        ],
    )
    def test_file_present(self, relative):
        assert (REPO_ROOT / relative).exists(), f"{relative} is referenced but missing"

    def test_readme_links_resolve(self):
        """Every relative markdown link in the README must point at a real file."""
        import re

        text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        missing = []
        for target in re.findall(r"\]\((?!https?:)([^)#]+)", text):
            if not (REPO_ROOT / target).exists():
                missing.append(target)
        assert not missing, f"README links to missing files: {missing}"

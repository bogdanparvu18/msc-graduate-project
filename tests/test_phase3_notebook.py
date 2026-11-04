"""Validate the runnable notebook structure without mounting private storage."""

import ast
from pathlib import Path

import nbformat


def test_bootstrap_notebook_has_valid_clean_executable_cells():
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / "notebooks/phase3/phase3_01_benchmark_bootstrap.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert cell.execution_count is None
            assert cell.outputs == []

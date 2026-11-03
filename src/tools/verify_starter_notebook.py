"""Run the Notebook's Python cells without requiring a Jupyter kernel."""

from pathlib import Path
import sys

import matplotlib
import nbformat


matplotlib.use("Agg")
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
notebook = nbformat.read(root / "docs" / "AI6124_Group1_Project_Starter.ipynb", as_version=4)
nbformat.validate(notebook)
namespace = {"__name__": "__main__"}
for number, cell in enumerate(notebook.cells, start=1):
    if cell.cell_type == "code":
        print(f"Running code cell {number}")
        exec(compile(cell.source, f"notebook-cell-{number}", "exec"), namespace)
print("Notebook code cells: pass")

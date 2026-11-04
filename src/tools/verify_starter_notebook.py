"""Verify the final Notebook was executed and saved by a Jupyter kernel."""

from pathlib import Path
import nbformat


root = Path(__file__).resolve().parents[2]
notebook = nbformat.read(root / "docs" / "AI6124_Group1_Final_Project.ipynb", as_version=4)
nbformat.validate(notebook)
codes = [cell for cell in notebook.cells if cell.cell_type == "code"]
if not codes or any(cell.execution_count is None for cell in codes):
    raise ValueError("Final Notebook has unexecuted code cells")
errors = [output for cell in codes for output in cell.outputs if output.output_type == "error"]
if errors:
    raise ValueError(f"Final Notebook contains {len(errors)} cell errors")
if sum(len(cell.outputs) for cell in codes) < len(codes):
    raise ValueError("Saved Notebook is missing expected results or figures")
print(f"Final Notebook: pass; {len(codes)} executed code cells, "
      f"{sum(len(cell.outputs) for cell in codes)} saved outputs")

import os
import sys
import json
import io
import contextlib
from pathlib import Path
import pandas as pd

project_dir = Path(r"f:\Praveen 3rd year-AI&DS\mint-leaf-ai")
notebooks_dir = project_dir / "notebooks"
nb_files = sorted(list(notebooks_dir.glob("*.ipynb")))

print(f"=======================================================")
print(f"🚀 EXECUTING ALL {len(nb_files)} NOTEBOOKS & CAPTURING CELL OUTPUTS")
print(f"=======================================================\n")

execution_results = {}

for nb_path in nb_files:
    print(f"📘 Notebook: {nb_path.name}")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
        
    global_env = {'__name__': '__main__'}
    cell_outputs = []
    
    code_cell_idx = 0
    for cell in nb_data.get("cells", []):
        if cell.get("cell_type") == "code":
            code_cell_idx += 1
            src_lines = cell.get("source", [])
            code = "".join(src_lines)
            
            # Clean colab magics for local execution
            clean_code = "\n".join([
                f"# {l}" if l.strip().startswith("!") or l.strip().startswith("%") else l
                for l in code.splitlines()
            ])
            
            # Capture stdout
            f_out = io.StringIO()
            error_msg = None
            
            try:
                with contextlib.redirect_stdout(f_out):
                    exec(clean_code, global_env)
                captured_text = f_out.getvalue().strip()
                status = "✅ SUCCESS"
            except Exception as e:
                captured_text = f_out.getvalue().strip()
                error_msg = f"{type(e).__name__}: {e}"
                status = f"❌ ERROR ({error_msg})"
                
            # Populate notebook cell output standard JSON structure
            output_text = captured_text if not error_msg else f"{captured_text}\n[ERROR]: {error_msg}".strip()
            cell["outputs"] = [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [line + "\n" for line in output_text.splitlines()]
                }
            ]
            cell["execution_count"] = code_cell_idx
            
            cell_outputs.append({
                "cell_number": code_cell_idx,
                "status": status,
                "output": output_text
            })
            print(f"   Cell {code_cell_idx}: {status}")
            
    # Save notebook with populated outputs
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=1)
        
    execution_results[nb_path.name] = cell_outputs
    print(f"💾 Updated {nb_path.name} with saved cell outputs!\n")

# Save complete execution log
log_path = project_dir / "outputs" / "reports" / "notebook_execution_results.json"
log_path.parent.mkdir(parents=True, exist_ok=True)

with open(log_path, "w", encoding="utf-8") as f:
    json.dump(execution_results, f, indent=4)

print(f"📄 Saved full execution report to: {log_path}")

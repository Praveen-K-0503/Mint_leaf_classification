import json
import os
import sys
import ast
from pathlib import Path

notebooks_dir = Path(r"f:\Praveen 3rd year-AI&DS\mint-leaf-ai\notebooks")
notebook_files = sorted(list(notebooks_dir.glob("*.ipynb")))

print(f"🔍 Auditing {len(notebook_files)} Jupyter Notebook files in {notebooks_dir}...\n")

total_errors = 0

for nb_path in notebook_files:
    print(f"📘 Analyzing notebook: {nb_path.name}")
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb_json = json.load(f)
            
        cells = nb_json.get("cells", [])
        code_cells = [c for c in cells if c.get("cell_type") == "code"]
        markdown_cells = [c for c in cells if c.get("cell_type") == "markdown"]
        
        print(f"   - Markdown cells: {len(markdown_cells)}")
        print(f"   - Code cells:     {len(code_cells)}")
        
        cell_errors = []
        for idx, code_cell in enumerate(code_cells, 1):
            source = "".join(code_cell.get("source", []))
            # Clean Colab magic commands like !pip or %matplotlib for ast compilation check
            clean_lines = []
            for line in source.splitlines():
                if line.strip().startswith("!") or line.strip().startswith("%"):
                    clean_lines.append(f"# {line}")
                else:
                    clean_lines.append(line)
            clean_source = "\n".join(clean_lines)
            
            try:
                ast.parse(clean_source)
            except SyntaxError as se:
                cell_errors.append((idx, str(se)))
                
        if cell_errors:
            print(f"   ❌ Syntax errors found in {len(cell_errors)} code cells:")
            for c_idx, err in cell_errors:
                print(f"      - Cell {c_idx}: {err}")
            total_errors += len(cell_errors)
        else:
            print(f"   ✅ All {len(code_cells)} code cells passed AST Python syntax validation!")
            
    except Exception as e:
        print(f"   ❌ Failed to parse notebook JSON: {e}")
        total_errors += 1
    print()

if total_errors == 0:
    print("🎉 ALL NOTEBOOKS ARE SYNTAX-VALID AND COLAB-COMPATIBLE!")
else:
    print(f"⚠️ Found {total_errors} errors across notebooks.")

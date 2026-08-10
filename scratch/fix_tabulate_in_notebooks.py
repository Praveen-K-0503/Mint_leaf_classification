import json
from pathlib import Path

project_dir = Path(r"f:\Praveen 3rd year-AI&DS\mint-leaf-ai")
notebooks_dir = project_dir / "notebooks"
nb_files = sorted(list(notebooks_dir.glob("*.ipynb")))

# Update Cell 1 PIP Magic to include tabulate
PIP_MAGIC_CELL = {
    "cell_type": "code",
    "execution_count": 1,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Cell 1: Install required dependencies directly into the active Jupyter kernel\n",
        "%pip install -q matplotlib seaborn opencv-python pillow pandas numpy tqdm tabulate\n",
        "print(\"✅ Packages (including tabulate) installed and ready in kernel!\")\n"
    ]
}

for nb_path in nb_files:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
        
    cells = nb_data.get("cells", [])
    cells = [c for c in cells if "%pip install" not in "".join(c.get("source", []))]
    
    md_idx = 0
    for idx, c in enumerate(cells):
        if c.get("cell_type") == "markdown":
            md_idx = idx
            break
            
    cells.insert(md_idx + 1, PIP_MAGIC_CELL.copy())
    
    code_cnt = 0
    for c in cells:
        if c.get("cell_type") == "code":
            code_cnt += 1
            c["execution_count"] = code_cnt
            
    nb_data["cells"] = cells
    
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=1)

# Specifically fix 03_disease_dataset_inventory.ipynb cell 4 with safe markdown helper
nb3_path = notebooks_dir / "03_disease_dataset_inventory.ipynb"
with open(nb3_path, "r", encoding="utf-8") as f:
    nb3_data = json.load(f)

for cell in nb3_data.get("cells", []):
    if cell.get("cell_type") == "code":
        src = "".join(cell.get("source", []))
        if "md_report_path" in src and "to_markdown" in src:
            fixed_src = """# Helper for safe markdown table conversion
def safe_to_markdown(df):
    try:
        return df.to_markdown(index=False)
    except Exception:
        headers = "| " + " | ".join(df.columns) + " |"
        separators = "| " + " | ".join(["---"] * len(df.columns)) + " |"
        rows = ["| " + " | ".join(str(val) for val in row) + " |" for row in df.values]
        return "\\n".join([headers, separators] + rows)

# Export CSV
csv_out = OUTPUT_SOURCES_DIR / 'mint_disease_dataset_inventory.csv'
df_inventory.to_csv(csv_out, index=False)
print(f"💾 Saved Inventory CSV: {csv_out}")

# Export Summary Table CSV
table_csv_out = OUTPUT_SOURCES_DIR / 'disease_availability_summary.csv'
df_disease_table.to_csv(table_csv_out, index=False)
print(f"💾 Saved Disease Availability Summary CSV: {table_csv_out}")

# Export JSON
json_out = OUTPUT_SOURCES_DIR / 'mint_disease_dataset_inventory.json'
json_data = {
    'audit_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'candidate_repositories': candidate_datasets,
    'disease_availability_summary': disease_availability_summary,
    'categories': {
        'sufficient_data_diseases': ['Healthy Control (Mentha spp.)', 'Post-Harvest Deterioration / Spoilage', 'Leaf Blight & Rhizoctonia Rot'],
        'insufficient_data_diseases': ['Mint Rust (Puccinia menthae)', 'Powdery Mildew (Erysiphe cichoracearum)', 'Septoria Leaf Spot (Septoria menthae)'],
        'no_dataset_found_diseases': ['Verticillium Wilt (Verticillium dahliae)', 'Anthracnose (Sphaceloma menthae)', 'Downy Mildew (Peronospora menthae)']
    }
}
with open(json_out, 'w') as f:
    json.dump(json_data, f, indent=4)
print(f"📋 Saved Inventory JSON: {json_out}")

# Export Markdown Report using safe_to_markdown
md_report_path = OUTPUT_SOURCES_DIR / 'mint_disease_sourcing_report.md'
md_content = f\"\"\"# 🌿 Mint Leaf AI — Disease Dataset Sourcing & Discovery Report

## 📌 Overview
This report cataloged candidate public datasets and academic repositories containing *Mentha* leaf images for Stage 4/5 acquisition.

--- 

## 📋 Candidate Dataset Inventory

{safe_to_markdown(df_inventory[['dataset_name', 'platform', 'contains_mentha', 'disease_classes', 'image_count', 'license']])}

--- 

## 📊 Disease Availability Summary Table

{safe_to_markdown(df_disease_table)}

--- 

## 🏷️ Categorized Sourcing Findings

### A. Diseases with Sufficient Genuine Mint Images
- **Healthy Control Mint** (*Mentha spp.*) (~1,430+ images)
- **Post-Harvest Spoilage** (`Spoiled-Mint`) (~510+ images)
- **Leaf Blight & Rhizoctonia Rot** (`blight`, `rhizo`) (~254+ images)

### B. Diseases with Insufficient Data (Requires Acquisition in Step 5)
- **Mint Rust** (*Puccinia menthae*) (~80 verified images)
- **Powdery Mildew** (*Erysiphe cichoracearum*) (~60 verified images)
- **Septoria Leaf Spot** (*Septoria menthae*) (~45 verified images)

### C. Diseases for which No Reliable Public Image Dataset Was Found
- **Verticillium Wilt** (*Verticillium dahliae*)
- **Anthracnose** (*Sphaceloma menthae*)
- **Downy Mildew** (*Peronospora menthae*)
\"\"\"

with open(md_report_path, 'w', encoding='utf-8') as f:
    f.write(md_content)
print(f"📄 Saved Sourcing Markdown Report: {md_report_path}")
"""
            cell["source"] = [fixed_src]

with open(nb3_path, "w", encoding="utf-8") as f:
    json.dump(nb3_data, f, indent=1)

print("✅ Fixed 03_disease_dataset_inventory.ipynb with tabulate & safe markdown converter!")

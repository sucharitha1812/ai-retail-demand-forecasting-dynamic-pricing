import json

notebook_path = "retail-demand-forecasting-dynamic-pricing.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Remove top-level widget metadata
if "metadata" in nb and "widgets" in nb["metadata"]:
    del nb["metadata"]["widgets"]

# 2. Clean cell-level metadata
for cell in nb.get("cells", []):
    if "metadata" in cell:
        # Remove colab + widget-related noise
        cell["metadata"].pop("id", None)
        cell["metadata"].pop("outputId", None)
        cell["metadata"].pop("executionInfo", None)
        cell["metadata"].pop("colab", None)

    # 3. Clean outputs 
    for output in cell.get("outputs", []):
        if isinstance(output, dict):
            # Remove widget view data
            if "data" in output:
                output["data"].pop("application/vnd.jupyter.widget-view+json", None)

            if "metadata" in output:
                output["metadata"].pop("application/vnd.jupyter.widget-view+json", None)

# Save cleaned notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("✅ Notebook cleaned for GitHub rendering")
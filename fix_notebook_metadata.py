import json

notebook_path = "retail-demand-forecasting-dynamic-pricing.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Remove top-level notebook metadata that can break GitHub preview
if "metadata" in nb:
    nb["metadata"].pop("widgets", None)
    nb["metadata"].pop("colab", None)

# Clean all cells
for cell in nb.get("cells", []):
    if "metadata" in cell:
        cell["metadata"].pop("id", None)
        cell["metadata"].pop("outputId", None)
        cell["metadata"].pop("executionInfo", None)
        cell["metadata"].pop("colab", None)
        cell["metadata"].pop("widgets", None)

    # Clean outputs
    cleaned_outputs = []
    for output in cell.get("outputs", []):
        if not isinstance(output, dict):
            cleaned_outputs.append(output)
            continue

        # Remove widget-view data from outputs
        if "data" in output:
            output["data"].pop("application/vnd.jupyter.widget-view+json", None)
            output["data"].pop("application/vnd.jupyter.widget-state+json", None)

        if "metadata" in output:
            output["metadata"].pop("application/vnd.jupyter.widget-view+json", None)
            output["metadata"].pop("application/vnd.jupyter.widget-state+json", None)

        cleaned_outputs.append(output)

    cell["outputs"] = cleaned_outputs

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook cleaned for GitHub rendering.")
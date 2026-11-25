import json
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = ROOT_DIR / "data" / "golden_dataset.json"

def load_dataset_by_metric(
        metric_name: str,
        dataset_path: Path = DEFAULT_DATASET_PATH
) -> List[Dict[str, Any]]:
    """Loads and filters test cases from the golden dataset by metric."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    target = metric_name.strip().lower()
    filtered = []

    for item in dataset:
        target_metrics = item.get("target_metrics", [])

        if isinstance(target_metrics, list):
            metrics_list = [str(m).strip().lower() for m in target_metrics if m]
        elif isinstance(target_metrics, str):
            metrics_list = [target_metrics.strip().lower()]
        else:
            metrics_list = []

        topic = str(item.get("topic", "")).lower()
        item_id = str(item.get("id", "")).lower()

        if target in metrics_list or target in topic or target in item_id:
            filtered.append(item)

    return filtered

"""Loading datasets from the Hugging Face Hub and loading the saved splits.

Two kinds of function live here:
  - load_banking77 / load_clinc_oos: download the raw data at a pinned revision.
    Only scripts/make_splits.py calls these.
  - load_split / load_eval_set: read the saved, preprocessed splits from
    data/processed/. Every system and the notebook use these, so they all see
    exactly the same data (rigour rule 6).
"""

import json
from pathlib import Path

import pandas as pd
import yaml
from datasets import load_dataset

# src/router/data.py -> parents[2] is the project root. Using an absolute root
# means paths work no matter which folder a script or notebook is run from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

OOS_LABEL = "oos"
OOS_LABEL_ID = -1  # OOS rows have no intent id. Never use label_id to index an array for them.

# Saved split name -> file name in data/processed/
SPLIT_FILES = {
    "train": "banking77_train.csv",
    "val": "banking77_val.csv",
    "test": "banking77_test.csv",
    "oos_val": "oos_val.csv",
    "oos_test": "oos_test.csv",
}
LABEL_MAP_FILE = "label_map.json"
REMOVED_ROWS_FILE = "removed_rows.csv"


def load_config(name: str = "data") -> dict:
    """Read configs/<name>.yaml into a dict."""
    with open(CONFIG_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Raw data from the Hub (used only by scripts/make_splits.py)
# ---------------------------------------------------------------------------

def load_banking77(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Download Banking77 at the pinned revision.

    Returns (train_df, test_df, label_names). label_names[i] is the intent name
    for Hub label i, so label_id in our files equals the Hub's own label number.
    """
    spec = cfg["banking77"]
    ds = load_dataset(spec["hub_id"], revision=spec["revision"], cache_dir=str(RAW_DIR))
    label_names: list[str] = ds["train"].features["label"].names

    frames = {}
    for split in ["train", "test"]:
        raw = ds[split].to_pandas()
        frames[split] = pd.DataFrame(
            {
                "text": raw["text"],
                "label": [label_names[i] for i in raw["label"]],
                "label_id": raw["label"].astype(int),
                "orig_index": range(len(raw)),
                "source": f"banking77_{split}",
            }
        )
    return frames["train"], frames["test"], label_names


def load_clinc_oos(cfg: dict) -> dict[str, pd.DataFrame]:
    """Download CLINC150 at the pinned revision and keep only the out-of-scope rows.

    Returns {"train": df, "validation": df, "test": df} of OOS rows.
    OOS rows are found by intent name (cfg oos_label), not by a hard-coded number.
    """
    spec = cfg["clinc150"]
    ds = load_dataset(
        spec["hub_id"], spec["config"], revision=spec["revision"], cache_dir=str(RAW_DIR)
    )
    intent_names: list[str] = ds["train"].features["intent"].names
    oos_id = intent_names.index(spec["oos_label"])

    pools = {}
    for split in ["train", "validation", "test"]:
        raw = ds[split].to_pandas()
        raw["orig_index"] = range(len(raw))  # row number in the full CLINC split
        raw = raw[raw["intent"] == oos_id]
        pools[split] = pd.DataFrame(
            {
                "text": raw["text"].values,
                "label": OOS_LABEL,
                "label_id": OOS_LABEL_ID,
                "orig_index": raw["orig_index"].values,
                "source": f"clinc_oos_{split}",
            }
        )
    return pools


# ---------------------------------------------------------------------------
# Label map
# ---------------------------------------------------------------------------

def save_label_map(label_names: list[str], path: Path | None = None) -> None:
    """Write {intent_name: id} to data/processed/label_map.json."""
    path = path or PROCESSED_DIR / LABEL_MAP_FILE
    label_map = {name: i for i, name in enumerate(label_names)}
    with open(path, "w") as f:
        json.dump(label_map, f, indent=2)


def load_label_map() -> dict[str, int]:
    """Read {intent_name: id} from data/processed/label_map.json."""
    with open(PROCESSED_DIR / LABEL_MAP_FILE) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Saved splits (used by every system, script and the notebook)
# ---------------------------------------------------------------------------

def load_split(name: str) -> pd.DataFrame:
    """Load one saved split: "train", "val", "test", "oos_val" or "oos_test"."""
    if name not in SPLIT_FILES:
        raise ValueError(f"Unknown split {name!r}. Choose from {list(SPLIT_FILES)}")
    path = PROCESSED_DIR / SPLIT_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: uv run python scripts/make_splits.py")
    # keep_default_na=False stops pandas turning a query like "NA" into a missing value.
    return pd.read_csv(path, keep_default_na=False)


def load_eval_set(split: str, with_oos: bool = True) -> pd.DataFrame:
    """Banking77 val or test, optionally with the matching OOS pool appended.

    split is "val" or "test". The result has an is_oos column (True for OOS rows).
    This is the mixed set the router policy is tuned on (val) and scored on (test).
    """
    if split not in ("val", "test"):
        raise ValueError("split must be 'val' or 'test'")
    in_scope = load_split(split)
    if not with_oos:
        return in_scope.assign(is_oos=False)
    oos = load_split(f"oos_{split}")
    mixed = pd.concat([in_scope.assign(is_oos=False), oos.assign(is_oos=True)])
    return mixed.reset_index(drop=True)

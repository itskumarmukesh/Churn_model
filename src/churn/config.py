from pathlib import Path
import yaml


def load_config(path):
    """Read a YAML config file and fill in default values for anything it leaves out."""
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as f:
        user_settings = yaml.safe_load(f)

    cfg = {
        "id_column": None,
        "drop_columns": [],
        "numeric_from_text": [],
        "zero_means_missing": [],
        "test_size": 0.2,
        "random_state": 42,
        "cv_folds": 5,
        "n_bootstrap": 1000,
    }
    business = {"value_saved": 800, "cost_contact": 100, "success_rate": 0.25}

    cfg.update(user_settings)
    business.update(user_settings.get("business", {}))
    cfg["business"] = business

    # paths in the config are relative to the repository root (the parent of the configs folder)
    cfg["_root"] = str(path.parent.parent)
    return cfg
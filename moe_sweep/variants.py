import json
import shutil
from pathlib import Path


def create_variants(base_path: str, routing_ks: list[int]) -> dict[int, Path]:
    """Create model variant directories with different num_experts_per_tok.

    Each variant symlinks all files from the base model except config.json,
    which is copied and edited with the target num_experts_per_tok.

    Returns mapping of k -> variant directory path.
    """
    base = Path(base_path)
    if not base.exists():
        raise FileNotFoundError(f"Base model path not found: {base}")

    config_path = base / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {base}")

    with open(config_path) as f:
        base_config = json.load(f)

    variants_dir = base / "variants"
    variants_dir.mkdir(exist_ok=True)
    result = {}

    for k in routing_ks:
        variant_dir = variants_dir / f"ept{k}"
        variant_dir.mkdir(exist_ok=True)

        # Symlink all files except config.json
        for item in base.iterdir():
            if item.name in ("variants", "config.json"):
                continue
            target = variant_dir / item.name
            if target.exists() or target.is_symlink():
                target.unlink() if target.is_file() or target.is_symlink() else shutil.rmtree(target)
            target.symlink_to(item)

        # Write modified config.json
        variant_config = base_config.copy()
        variant_config["num_experts_per_tok"] = k
        with open(variant_dir / "config.json", "w") as f:
            json.dump(variant_config, f, indent=2)

        result[k] = variant_dir
        print(f"Created variant ept{k} at {variant_dir}")

    return result


def verify_variants(base_path: str, routing_ks: list[int]) -> bool:
    """Verify all variants have correct num_experts_per_tok."""
    base = Path(base_path)
    all_ok = True
    for k in routing_ks:
        config_path = base / "variants" / f"ept{k}" / "config.json"
        if not config_path.exists():
            print(f"MISSING: {config_path}")
            all_ok = False
            continue
        with open(config_path) as f:
            config = json.load(f)
        actual = config.get("num_experts_per_tok")
        if actual != k:
            print(f"MISMATCH: ept{k} has num_experts_per_tok={actual}")
            all_ok = False
        else:
            print(f"OK: ept{k} num_experts_per_tok={k}")
    return all_ok

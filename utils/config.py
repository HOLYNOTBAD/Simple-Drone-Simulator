from __future__ import annotations

from pathlib import Path


def resolve_script_config(script_file: str | Path, config_arg: str | None = None) -> str:
    """Resolve a default config path for a simulation script.

    Resolution order:
    1. Explicit `config_arg` if provided
    2. `configs/<script_stem>.yaml`
    3. `configs/<script_stem_without__sim>.yaml`
    """
    if config_arg not in (None, ""):
        return str(config_arg)

    script_path = Path(script_file).resolve()
    root_dir = script_path.parents[1]
    cfg_dir = root_dir / "configs"
    stem = script_path.stem

    candidates = [cfg_dir / f"{stem}.yaml"]
    if stem.endswith("_sim"):
        candidates.append(cfg_dir / f"{stem[:-4]}.yaml")

    for path in candidates:
        if path.exists():
            return path.as_posix()

    candidate_text = ", ".join(p.as_posix() for p in candidates)
    raise FileNotFoundError(f"No matching config found for {script_path.name}. Tried: {candidate_text}")

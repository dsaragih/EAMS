import glob
import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


EXPERIMENT_OVERRIDE_ROOTS = {
    "checkpoint_path",
    "data",
    "debug",
    "matmul_precision",
    "paths",
    "seed",
    "trainer",
    "transforms",
}

ROBUSTNESS_CONFIG_DIR = (
    Path(__file__).resolve().parent.parent / "configs" / "robustness_eval"
)


def clone_cfg(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))


def _resolve_default_config_path(entry, base_dir: Path) -> Path:
    if isinstance(entry, str):
        default_name = entry
    elif isinstance(entry, dict) and len(entry) == 1:
        default_name = next(iter(entry.values()))
    else:
        raise ValueError(f"Unsupported defaults entry: {entry}")

    default_path = Path(str(default_name))
    if not default_path.is_absolute():
        default_path = base_dir / default_path
    if default_path.suffix != ".yaml":
        default_path = default_path.with_suffix(".yaml")
    return default_path


def _load_yaml_with_defaults(
    config_path: Path, visited: set[Path] | None = None
) -> DictConfig:
    resolved_path = config_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Config file not found: {resolved_path}")

    active_stack = set() if visited is None else set(visited)
    if resolved_path in active_stack:
        raise ValueError(f"Circular config defaults detected for {resolved_path}")
    active_stack.add(resolved_path)

    loaded_cfg = OmegaConf.load(resolved_path)
    cfg = clone_cfg(loaded_cfg)
    defaults_entries = list(cfg.get("defaults", [])) if "defaults" in cfg else []

    merged_cfg = OmegaConf.create()
    for entry in defaults_entries:
        default_path = _resolve_default_config_path(entry, resolved_path.parent)
        merged_cfg = OmegaConf.merge(
            merged_cfg,
            _load_yaml_with_defaults(default_path, active_stack),
        )

    if "defaults" in cfg:
        del cfg["defaults"]

    return OmegaConf.merge(merged_cfg, cfg)


def _copy_cfg_node(value):
    return OmegaConf.create(OmegaConf.to_container(value, resolve=False))


def _normalize_override_key(raw_override: str) -> str:
    key = raw_override.strip()
    while key.startswith(("+", "~")):
        key = key[1:]
    return key.split("=", 1)[0]


def load_robustness_eval_config(config_name_or_path: str) -> DictConfig:
    raw_path = Path(config_name_or_path).expanduser()
    if raw_path.is_file():
        config_path = raw_path
    else:
        config_path = ROBUSTNESS_CONFIG_DIR / config_name_or_path
        if config_path.suffix != ".yaml":
            config_path = config_path.with_suffix(".yaml")

    return _load_yaml_with_defaults(config_path)


def build_robustness_variants(
    robustness_eval_cfg: DictConfig,
    transforms_cfg: DictConfig,
) -> list[dict]:
    variants = []
    if robustness_eval_cfg.get("include_baseline", True):
        variants.append(
            {
                "condition_name": transforms_cfg.get("condition_name", "baseline"),
                "test_transforms": OmegaConf.to_container(
                    transforms_cfg.get("test_transforms", []), resolve=False
                ),
            }
        )

    configured_variants = robustness_eval_cfg.get("variants", [])
    for variant in configured_variants:
        variants.append(OmegaConf.to_container(variant, resolve=False))

    return variants


def locate_experiment_artifacts(experiment_dir: str) -> dict[str, Path]:
    root = Path(experiment_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {root}")

    outputs_dir = root / "outputs"
    checkpoints_dir = root / "checkpoints"
    logs_dir = root / "logs"
    results_dir = root / "results"

    hydra_dir_candidates = [outputs_dir / ".hydra", outputs_dir]
    config_path = None
    hydra_path = None
    overrides_path = None
    for hydra_dir in hydra_dir_candidates:
        candidate_config = hydra_dir / "config.yaml"
        candidate_hydra = hydra_dir / "hydra.yaml"
        candidate_overrides = hydra_dir / "overrides.yaml"
        if candidate_config.is_file():
            config_path = candidate_config
            hydra_path = candidate_hydra if candidate_hydra.is_file() else None
            overrides_path = (
                candidate_overrides if candidate_overrides.is_file() else None
            )
            break

    if config_path is None:
        raise FileNotFoundError(
            f"Could not find Hydra config.yaml under {outputs_dir} or {outputs_dir / '.hydra'}"
        )

    return {
        "root": root,
        "checkpoints_dir": checkpoints_dir,
        "config_path": config_path,
        "hydra_path": hydra_path,
        "logs_dir": logs_dir,
        "outputs_dir": outputs_dir,
        "overrides_path": overrides_path,
        "results_dir": results_dir,
    }


def load_experiment_overrides(overrides_path: Path | None) -> list[str]:
    if overrides_path is None or not overrides_path.is_file():
        return []
    overrides_cfg = OmegaConf.load(overrides_path)
    overrides = OmegaConf.to_container(overrides_cfg, resolve=True)
    return [str(item) for item in overrides] if overrides else []


def apply_runtime_experiment_overrides(
    saved_cfg: DictConfig,
    runtime_cfg: DictConfig,
    runtime_task_overrides: list[str],
) -> DictConfig:
    if not runtime_task_overrides:
        return saved_cfg

    for raw_override in runtime_task_overrides:
        key = _normalize_override_key(raw_override)
        if not key or key == "experiment_dir" or key.startswith("hydra."):
            continue

        root = key.split(".", 1)[0].split("/", 1)[0]
        if root not in EXPERIMENT_OVERRIDE_ROOTS:
            continue

        runtime_value = OmegaConf.select(runtime_cfg, key)
        if runtime_value is None and "." not in key and "/" not in key:
            runtime_value = runtime_cfg.get(key, None)
        if runtime_value is None:
            continue

        value_copy = _copy_cfg_node(runtime_value)
        OmegaConf.update(saved_cfg, key, value_copy, merge=True)

    return saved_cfg


def infer_fold_from_path(path: str | None) -> int | None:
    if not path:
        return None
    path_obj = Path(path)
    for part in path_obj.parts:
        if part.startswith("fold_"):
            suffix = part.split("fold_", 1)[1]
            if suffix.isdigit():
                return int(suffix)
    return None


def load_test_cfg_from_experiment_dir(
    experiment_dir: str,
    runtime_cfg: DictConfig | None = None,
    runtime_task_overrides: list[str] | None = None,
) -> tuple[DictConfig, dict[str, Path], list[str]]:
    artifacts = locate_experiment_artifacts(experiment_dir)
    saved_cfg = OmegaConf.load(artifacts["config_path"])
    saved_overrides = load_experiment_overrides(artifacts["overrides_path"])

    saved_cfg.paths.write_dir_prefix = str(artifacts["root"])
    saved_cfg.paths.checkpoints_dir = str(artifacts["checkpoints_dir"])
    saved_cfg.paths.results_dir = str(artifacts["results_dir"])
    saved_cfg.paths.logging_dir = str(artifacts["logs_dir"])

    if runtime_cfg is not None:
        saved_cfg = apply_runtime_experiment_overrides(
            saved_cfg,
            runtime_cfg=runtime_cfg,
            runtime_task_overrides=runtime_task_overrides or [],
        )

    if runtime_cfg is not None and runtime_cfg.get("checkpoint_path", None):
        saved_cfg.checkpoint_path = runtime_cfg.checkpoint_path
    if runtime_cfg is not None and runtime_cfg.get("debug", False):
        saved_cfg.debug = runtime_cfg.debug
    if runtime_cfg is not None and runtime_cfg.get("experiment_dir", None):
        saved_cfg.experiment_dir = runtime_cfg.experiment_dir

    checkpoint_path = saved_cfg.get("checkpoint_path", None)
    inferred_fold = infer_fold_from_path(checkpoint_path)
    if (
        inferred_fold is not None
        and hasattr(saved_cfg, "data")
        and "current_fold" in saved_cfg.data
    ):
        saved_cfg.data.current_fold = inferred_fold

    return saved_cfg, artifacts, saved_overrides


def find_best_checkpoint(checkpoint_dir: str, metric: str = "avg") -> str:
    pattern = os.path.join(checkpoint_dir, f"best-{metric}-*.ckpt")
    checkpoints = glob.glob(pattern)

    if not checkpoints:
        pattern = os.path.join(checkpoint_dir, "last.ckpt")
        checkpoints = glob.glob(pattern)

    if not checkpoints:
        pattern = os.path.join(checkpoint_dir, "**", "last.ckpt")
        checkpoints = glob.glob(pattern, recursive=True)

    if not checkpoints:
        pattern = os.path.join(checkpoint_dir, "**", "*.ckpt")
        checkpoints = glob.glob(pattern, recursive=True)

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]


def discover_checkpoint_paths(
    checkpoints_dir: str | Path,
    fold: int | None = None,
) -> list[tuple[int, Path]]:
    checkpoints_root = Path(checkpoints_dir).expanduser().resolve()
    if not checkpoints_root.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {checkpoints_root}"
        )

    selected_fold_dirs: list[tuple[int, Path]] = []
    for fold_dir in checkpoints_root.iterdir():
        if not fold_dir.is_dir() or not fold_dir.name.startswith("fold_"):
            continue
        suffix = fold_dir.name.split("fold_", 1)[1]
        if not suffix.isdigit():
            continue
        fold_idx = int(suffix)
        if fold is not None and fold_idx != fold:
            continue
        selected_fold_dirs.append((fold_idx, fold_dir))

    selected_fold_dirs.sort(key=lambda item: item[0])

    if fold is not None and not selected_fold_dirs:
        raise FileNotFoundError(
            f"No checkpoint directory found for fold_{fold} under {checkpoints_root}"
        )

    checkpoint_paths: list[tuple[int, Path]] = []
    for fold_idx, fold_dir in selected_fold_dirs:
        fold_checkpoints = sorted(fold_dir.glob("*.ckpt"), key=lambda path: path.name)
        checkpoint_paths.extend(
            (fold_idx, checkpoint_path) for checkpoint_path in fold_checkpoints
        )

    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoint files found under {checkpoints_root}")

    return checkpoint_paths

import importlib
from pathlib import Path


MODELS_PACKAGE = "spd.models"
MODELS_DIR = Path(__file__).resolve().parent


def _normalize_model_name(model_name):
    model_name = str(model_name)
    if model_name.endswith(".py"):
        model_name = model_name[:-3]
    return model_name.strip().strip("/\\").replace("\\", "/")


def _path_to_module(path):
    rel_path = path.relative_to(MODELS_DIR).with_suffix("")
    return ".".join((MODELS_PACKAGE, *rel_path.parts))


def _try_import(module_name):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        # Continue searching only when the requested module path is missing.
        # If the module exists but an import inside it fails, surface that error.
        if exc.name == module_name or module_name.startswith(f"{exc.name}."):
            return None
        raise


def _find_model_module(model_name):
    normalized_name = _normalize_model_name(model_name)
    dotted_name = normalized_name.replace("/", ".")

    direct_module = _try_import(f"{MODELS_PACKAGE}.{dotted_name}")
    if direct_module is not None:
        return direct_module

    target_stem = Path(normalized_name).name
    candidates = []
    for path in MODELS_DIR.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        rel_path = path.relative_to(MODELS_DIR).with_suffix("")
        rel_dotted = ".".join(rel_path.parts)
        if path.stem == target_stem or rel_dotted == dotted_name:
            candidates.append(path)

    if not candidates:
        raise ImportError(
            f"Cannot find model file for '{model_name}' under {MODELS_DIR}."
        )

    candidates.sort(key=lambda p: (len(p.relative_to(MODELS_DIR).parts), str(p)))
    module_name = _path_to_module(candidates[0])
    module = _try_import(module_name)
    if module is None:
        raise ImportError(
            f"Found model file {candidates[0]}, but could not import module {module_name}."
        )
    return module


def build_model(model_name, dataset_name, args):
    dataset_to_class = {
        "BNCI2014001": "BNCI2014Net",
        "BNCI2015001": "BNCI2015Net",
    }

    if dataset_name not in dataset_to_class:
        raise ValueError(f"Unregistered dataset type: {dataset_name}")

    class_name = dataset_to_class[dataset_name]
    module = _find_model_module(model_name)

    try:
        model_class = getattr(module, class_name)
    except AttributeError:
        raise AttributeError(
            f"Class {class_name} was not found in model module {module.__name__}."
        )

    slice = args.get("slice", None)
    if slice is None:
        return model_class()
    return model_class(slice)

import importlib

def dataset(name):
    """
    Local minimal dataset loader for DeepMind mathematics modules.
    Works with patched modules and modern SymPy.
    """
    module_name, fn = name.split("__", 1)

    # Map DeepMind module names to local modules
    module_map = {
        "algebra": "math_dataset.modules.algebra",
        "calculus": "math_dataset.modules.calculus",
        "geometry": "math_dataset.modules.geometry",
        "probability": "math_dataset.modules.probability",
    }

    if module_name not in module_map:
        raise ValueError(f"Unknown module: {module_name}")

    mod = importlib.import_module(module_map[module_name])
    return getattr(mod, fn)()

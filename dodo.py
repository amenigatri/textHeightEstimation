from pathlib import Path


DOIT_CONFIG = {
    "default_tasks": ["format", "lint"],
    "backend": "json",
}

HERE = Path(__file__).parent


def task_format():
    """Reformat all files using black."""
    print(HERE)
    return {"actions": [["black", HERE]], "verbosity": 1}


def task_format_check():
    """Check, but not change, formatting using black."""
    return {"actions": [["black", HERE, "--check"]], "verbosity": 1}


def task_lint():
    """Lint all files with Prospector."""
    return {"actions": [["prospector"]], "verbosity": 1}


def task_test():
    """Run Pytest with coverage."""
    return {"actions": [["pytest"]], "verbosity": 2}

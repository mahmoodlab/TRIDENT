import os


def env_flag(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


RUN_INTEGRATION_TESTS = env_flag("TRIDENT_RUN_INTEGRATION_TESTS", "0")
RUN_GPU_TESTS = env_flag("TRIDENT_RUN_GPU_TESTS", "0")

"""Baseline turn policy. Hard tool limits are enforced by the environment."""
MAX_STEPS = 12


def should_continue(steps: int, finished: bool) -> bool:
    return not finished and steps < MAX_STEPS

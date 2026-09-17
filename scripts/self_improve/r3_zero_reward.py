"""Agent-R1 custom-reward shim; task reward is supplied by AgentFlowOutput."""


def compute_score(**_kwargs) -> float:
    return 0.0


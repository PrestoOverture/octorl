"""Agent-side advisory pre-submit tool choice; verifier remains external."""


def check_request(changed_files: list[str]) -> dict:
    return {"name": "run_tests", "arguments": {}} if changed_files else {"name": "list_files", "arguments": {"path": "."}}

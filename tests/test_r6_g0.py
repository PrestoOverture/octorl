from pathlib import Path

from scripts.self_improve.r6_select import g0_regression


def test_all_36_r4r_stages_reproduce_exactly():
    result = g0_regression(Path.cwd())
    assert result["all_pass"]
    assert result["failure_driven_stages"] == result["fixed_stages"] == 18
    assert result["windows"] == 36

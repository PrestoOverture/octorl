"""R6 prompt, parser, aggregation, and selector synthetic controls."""

import hashlib
import json
import subprocess
import sys

import pytest

from src.curriculum.failure_driven import CELLS, PI0
from src.curriculum.model_chooser import (
    aggregate_samples, distribution_from_q, parse_completion, query_seed, render_prompt, tv_distance,
)


STATS = {
    CELLS[0]: {"n": 10, "fail": 2, "infra": 1},
    CELLS[1]: {"n": 8, "fail": 4, "infra": 0},
    CELLS[2]: {"n": 6, "fail": 3, "infra": 2},
}


def test_prompt_exact_table_hash_and_cross_process_determinism():
    rendered = render_prompt(STATS)
    expected = """| fault type | attempts | failures | failure rate |
|---|---|---|---|
| constraint_violation | 10 | 2 | 0.250 |
| missing_dependency | 8 | 4 | 0.500 |
| stale_version | 6 | 3 | 0.500 |"""
    assert expected in rendered["prompt"]
    assert "last 20 training updates" in rendered["prompt"]
    assert "has 40 practice tasks" in rendered["prompt"]
    assert rendered["prompt_sha256"] == hashlib.sha256(rendered["prompt"].encode()).hexdigest()
    code = """import json
from src.curriculum.model_chooser import render_prompt
s={'constraint_violation':{'n':10,'fail':2,'infra':1},'missing_dependency':{'n':8,'fail':4,'infra':0},'stale_version':{'n':6,'fail':3,'infra':2}}
print(json.dumps(render_prompt(s),sort_keys=True,separators=(',',':')))"""
    first = subprocess.check_output([sys.executable, "-c", code])
    second = subprocess.check_output([sys.executable, "-c", code])
    assert first == second


def test_query_seed_exact_formulas():
    for noise, prefix in ((False, "chooser"), (True, "noise")):
        expected = int(hashlib.sha256(f"r6|{prefix}|42|3|7".encode()).hexdigest()[:8], 16)
        assert query_seed(42, 3, 7, noise=noise) == expected


@pytest.mark.parametrize("completion,valid,unnormalised", [
    ('{"constraint_violation":40,"missing_dependency":30,"stale_version":30}', True, False),
    ('answer: {"constraint_violation":40,"missing_dependency":30,"stale_version":30} trailing', True, False),
    ('{"constraint_violation":1,"missing_dependency":1,"stale_version":1} '
     '{"constraint_violation":40,"missing_dependency":30,"stale_version":30}', True, False),
    ('{"constraint_violation":40,"missing_dependency":30}', False, False),
    ('{"constraint_violation":40,"missing_dependency":30,"stale_version":30,"extra":0}', False, False),
    ('{"constraint_violation":40,"missing_dependency":30,"stale_versoin":30}', False, False),
    ('{"constraint_violation":-1,"missing_dependency":30,"stale_version":30}', False, False),
    ('{"constraint_violation":true,"missing_dependency":30,"stale_version":30}', False, False),
    ('{"constraint_violation":NaN,"missing_dependency":30,"stale_version":30}', False, False),
    ('{"constraint_violation":0,"missing_dependency":0,"stale_version":0}', False, False),
    ('{"constraint_violation":39,"missing_dependency":30,"stale_version":30}', True, True),
    ('{"outer":{"x":1}} {"constraint_violation":40,"missing_dependency":30,"stale_version":30}', True, False),
    ('', False, False),
])
def test_parser_cases(completion, valid, unnormalised):
    parsed = parse_completion(completion)
    assert parsed["valid"] is valid
    assert parsed["unnormalised"] is unnormalised


def test_parser_last_parseable_dict_wins_even_after_invalid_braces():
    text = ('{"constraint_violation":40,"missing_dependency":30,"stale_version":30} '
            '{not json}')
    assert parse_completion(text)["valid"]


def test_parser_recovers_last_balanced_object_after_unmatched_prefix_brace():
    text = ('unfinished { text {"constraint_violation":40,"missing_dependency":30,'
            '"stale_version":30}')
    assert parse_completion(text)["valid"]


def test_five_of_eight_falls_back_six_does_not():
    valid = json.dumps(dict(zip(CELLS, (40, 30, 30))))
    five = aggregate_samples([valid] * 5 + [""] * 3)
    six = aggregate_samples([valid] * 6 + [""] * 2)
    assert five["fallback"] and five["q"] == PI0
    assert not six["fallback"] and six["q"] != PI0


@pytest.mark.parametrize("hot", CELLS)
def test_one_hot_q_respects_cap_floor_normalisation_and_count(hot):
    q = {cell: float(cell == hot) for cell in CELLS}
    p, counts = distribution_from_q(q)
    assert max(p.values()) <= 0.6
    assert all(p[cell] >= 0.4 * PI0[cell] for cell in CELLS)
    assert abs(sum(p.values()) - 1) <= 1e-12
    assert sum(counts.values()) == 40


def test_identical_chooser_tv_is_exactly_zero():
    q = dict(zip(CELLS, (0.2, 0.3, 0.5)))
    assert tv_distance(q, q) == 0

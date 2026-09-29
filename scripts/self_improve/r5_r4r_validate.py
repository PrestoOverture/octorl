#!/usr/bin/env python3
"""Record final offline validation, without running any experiment or bootstrap."""
import hashlib
import json
import re
import subprocess
import sys
from r5_diagnostics import ROOT, OUT, DIAG, write_json
from r5_verify_report import verify


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True)


def allowed(path):
    return (path in {'README.md', 'scripts/self_improve/r5_report.py', 'scripts/self_improve/r5_verify_report.py',
                     'scripts/self_improve/r5_diagnostics.py', 'tests/test_r5_deliverable.py',
                     'artifacts/self_improve/r5/R5_report.md', 'artifacts/self_improve/r5/checkpoint_manifest.json',
                     'artifacts/self_improve/r5/verification.json', 'artifacts/self_improve/r5/HANDOFF.md',
                     'artifacts/self_improve/r5/raw/pytest_final.log'}
            or path.startswith('artifacts/self_improve/r5/diagnostics/')
            or bool(re.fullmatch(r'scripts/self_improve/r5_r4r_[^/]+\.py', path))
            or bool(re.fullmatch(r'tests/test_r5_[^/]+\.py', path)))


def main():
    command = [sys.executable, str(ROOT / 'scripts/self_improve/r5_report.py')]
    subprocess.run(command, cwd=ROOT, check=True)
    first = (OUT / 'R5_report.md').read_bytes()
    subprocess.run(command, cwd=ROOT, check=True)
    assert first == (OUT / 'R5_report.md').read_bytes(), 'regeneration not idempotent'
    report = verify()
    before = json.loads((DIAG / 'r4r_protected_before.json').read_text())
    for path, digest in before.items():
        with (ROOT / path).open('rb') as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest() == digest, path
    staged = git('diff', '--cached', '--name-only').splitlines()
    assert not staged
    changed = git('diff', '--name-only').splitlines()
    untracked = git('ls-files', '--others', '--exclude-standard').splitlines()
    assert all(allowed(path) for path in changed + untracked if path != 'uv.lock')
    assert 'uv.lock' in untracked and 'uv.lock' not in changed
    # The README prefix outside the self-improve section must be byte-identical.
    heading = '## Self-improve training study: dev-tool fault recovery'
    original = git('show', 'HEAD:README.md')
    assert original.split(heading)[0] == (ROOT / 'README.md').read_text().split(heading)[0]
    log = (OUT / 'raw/pytest_final.log').read_text()
    match = re.search(r'(\d+) passed(?:, (\d+) skipped)? in ([\d.]+)s', log)
    assert match and 'failed' not in log and 'ERROR' not in log
    lr = json.loads((DIAG / 'r4r_lr_check.json').read_text())
    result = {'success': True, 'pytest_command': '.venv/bin/python -m pytest tests/test_r5_*.py tests/test_r4_*.py tests/test_r4r_*.py -q',
              'pytest_counts': {'passed': int(match[1]), 'skipped': int(match[2] or 0), 'seconds': float(match[3])},
              'pytest_log': 'artifacts/self_improve/r5/raw/pytest_final.log', 'pytest_output': log,
              'report': report, 'regeneration_byte_identical': True,
              'report_sha256': hashlib.sha256(first).hexdigest(),
              'protected_files_checked': len(before), 'protected_files_unchanged': True,
              'lr_steps_checked': lr['steps_checked'], 'lr_max_absolute_difference': lr['max_absolute_difference'],
              'staged_paths': staged, 'git_status': git('status', '--porcelain'), 'git_diff_stat': git('diff', '--stat'),
              'changed_files': changed, 'new_files': [p for p in untracked if p != 'uv.lock'],
              'preexisting_uv_lock_untouched': True, 'gpu_training_evaluation_or_full_load_run': False,
              'remote_access': False,
              'contract_reconciliations': ['raw/pytest_final.log is outside the general output allowlist but explicitly required by the Tests success condition.',
                 'The required R4 10% comparison sentence remains in section 3; all R4 source paths and historical verdicts are confined to section 4 or R4-labelled numeric rows, as specified by the verifier rules.']}
    write_json(OUT / 'verification.json', result)
    print(json.dumps({k: result[k] for k in ('success', 'pytest_counts', 'report', 'regeneration_byte_identical', 'protected_files_checked', 'protected_files_unchanged', 'lr_steps_checked', 'lr_max_absolute_difference')}, indent=2))


if __name__ == '__main__':
    main()

"""Build a transparent, authored development history for parcel billing helpers.

These are controlled micro-repository development examples, not mined upstream
bugs. The initial implementation is committed, its failures are recorded, then
corrections are committed. Catalog references are actual immutable git objects.
"""
from pathlib import Path
import json
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2] / 'repos/train/parcel_ledger'
PATH = 'parcel_ledger/billing_rules.py'
# kind, initial implementation, corrected implementation, feature implementation,
# feature description, visible assertion
ROWS = [
('condition-inversion',
 'def discount(amount, threshold):\n    if amount < threshold:\n        return amount // 10\n    return 0\n',
 'def discount(amount, threshold):\n    if amount >= threshold:\n        return amount // 10\n    return 0\n',
 'def discount(amount, threshold, *, member=False):\n    if member:\n        return amount // 5\n    if amount < threshold:\n        return amount // 10\n    return 0\n',
 'Add member discount at twenty percent; standard threshold accidentally inverted.', 'discount(200, 100) == 20'),
('off-by-one',
 'def invoice_ids(start, count):\n    return list(range(start, start + count + 1))\n',
 'def invoice_ids(start, count):\n    return list(range(start, start + count))\n',
 'def invoice_ids(start, count, *, prefix=None):\n    ids = list(range(start, start + count + 1))\n    return [prefix + str(i) for i in ids] if prefix is not None else ids\n',
 'Add optional string prefixes; inclusive-end regression in original integer IDs.', 'invoice_ids(7, 2) == [7, 8]'),
('variable-misuse',
 'def payer_amount(charged, refunded):\n    return refunded\n',
 'def payer_amount(charged, refunded):\n    return charged\n',
 'def payer_amount(charged, refunded, *, net=False):\n    if net:\n        return charged - refunded\n    return refunded\n',
 'Add net-payment reporting; gross branch uses refunded field.', 'payer_amount(200, 50) == 200'),
('wrong-return-value',
 'def fee_total(base, surcharge):\n    return base - surcharge\n',
 'def fee_total(base, surcharge):\n    return base + surcharge\n',
 'def fee_total(base, surcharge, *, waived=False):\n    if waived:\n        return base\n    return base - surcharge\n',
 'Add surcharge waiver; standard total incorrectly subtracts surcharge.', 'fee_total(200, 50) == 250'),
('missing-exception-handling',
 'def parse_quantity(text, fallback):\n    return int(text)\n',
 'def parse_quantity(text, fallback):\n    try:\n        return int(text)\n    except ValueError:\n        return fallback\n',
 'def parse_quantity(text, fallback, *, unlimited=False):\n    if unlimited and text == "unlimited":\n        return -1\n    return int(text)\n',
 'Add unlimited sentinel; refactoring loses invalid-text fallback.', 'parse_quantity("bad", 3) == 3'),
('api-parameter-error',
 'def prioritize_amounts(amounts):\n    return sorted(amounts, reverse=False)\n',
 'def prioritize_amounts(amounts):\n    return sorted(amounts, reverse=True)\n',
 'def prioritize_amounts(amounts, *, smallest_first=False):\n    if smallest_first:\n        return sorted(amounts)\n    return sorted(amounts, reverse=False)\n',
 'Add smallest-first option; existing descending order uses wrong reverse argument.', 'prioritize_amounts([1, 3, 2]) == [3, 2, 1]'),
('boundary-condition-omission',
 'def first_reference(references, default):\n    return references[0]\n',
 'def first_reference(references, default):\n    if not references:\n        return default\n    return references[0]\n',
 'def first_reference(references, default, *, last=False):\n    if last:\n        return references[-1] if references else default\n    return references[0]\n',
 'Add last-reference selection; first-reference empty-list guard disappears.', 'first_reference([], "none") == "none"'),
]

def git(*args):
    return subprocess.check_output(['git','-C',str(REPO),*args],text=True).strip()

def main():
    if (REPO/'injector_sources.json').exists():
        raise SystemExit('Catalog already exists; refusing to rewrite immutable history')
    git('init','-q')
    git('config','user.name','OctoRL authored fixtures')
    git('config','user.email','fixtures@example.invalid')
    path = REPO/PATH
    path.write_text('"""Parcel billing helpers: controlled development history, not upstream bugs."""\n\n'+'\n'.join(row[1] for row in ROWS))
    tests = REPO/'tests/test_billing_rules.py'
    tests.parent.mkdir(exist_ok=True)
    tests.write_text('from parcel_ledger.billing_rules import *\n\n'+'\n'.join(f'def test_rule_{i}():\n    assert {row[5]}\n' for i,row in enumerate(ROWS)))
    git('add',PATH,'tests/test_billing_rules.py')
    git('commit','-qm','Add initial authored billing helper implementations and specifications')
    initial = git('rev-parse','HEAD')
    first = subprocess.run([sys.executable,'-m','pytest','-q','tests/test_billing_rules.py'],cwd=REPO,capture_output=True,text=True)
    if first.returncode != 1:
        raise RuntimeError(first.stdout + first.stderr)
    path.write_text('"""Parcel billing helpers: controlled development history, not upstream bugs."""\n\n'+'\n'.join(row[2] for row in ROWS))
    # Ensure same-second code edits cannot use timestamp-based stale bytecode.
    import shutil
    shutil.rmtree(REPO/'parcel_ledger/__pycache__',ignore_errors=True)
    final = subprocess.run([sys.executable,'-m','pytest','-q','tests/test_billing_rules.py'],cwd=REPO,capture_output=True,text=True)
    if final.returncode:
        raise RuntimeError(final.stdout+final.stderr)
    git('add',PATH)
    git('commit','-qm','Correct threshold, ID bounds, payer field, fee, parse fallback, ordering and empty references')
    commit = git('rev-parse','HEAD')
    catalog = {'provenance':'Controlled authored development history, not naturally occurring or upstream-mined defects.',
        'initial_commit':initial,'fix_commit':commit,
        'history':[dict(type=r[0],path=PATH,fix_commit=commit,before=r[1],after=r[2]) for r in ROWS],
        'features':[dict(type=r[0],path=PATH,before=r[2],after=r[3],description=r[4]) for r in ROWS]}
    (REPO/'injector_sources.json').write_text(json.dumps(catalog,indent=2)+'\n')
    output = Path(__file__).parent/'injector_history_evidence.json'
    output.write_text(json.dumps({'initial_commit':initial,'fix_commit':commit,'initial_tests':first.stdout,'corrected_tests':final.stdout},indent=2)+'\n')
    git('bundle','create','injector_history.bundle','--all')

if __name__ == '__main__':
    main()

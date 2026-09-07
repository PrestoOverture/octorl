"""Does the hidden suite actually catch each injected bug?

PRD 4.2: hidden tests are a *strengthened superset* of visible tests. If visible
fails while hidden passes, the superset property is violated for that cell."""
import os, shutil, subprocess, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.environment.harness import package
from src.injector.catalog import inject_from_catalog
from src.injector.core import BUG_TYPES, Difficulty

def run(args, cwd):
    return subprocess.run([sys.executable,"-m","pytest","-p","hypothesis.extra.pytestplugin","-q",
                           "--tb=no","--hypothesis-seed=11",*args], cwd=cwd, capture_output=True,
                          env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","PYTEST_DISABLE_PLUGIN_AUTOLOAD":"1"}).returncode

ROOT = Path(__file__).resolve().parents[1]
repos = sorted(p for p in (ROOT/"repos").glob("*/*") if p.is_dir())
bad = []
rows_out = []
total = 0
for repo in repos:
    for t in BUG_TYPES:
        d = Difficulty("mutation", t, 1, "L0", "single-function")
        try: inject_from_catalog(repo, d, 11)
        except Exception: continue
        tmp = Path(tempfile.mkdtemp())
        inst = package(repo, d, 11, tmp)
        shutil.copytree(inst.hidden_tests_dir, inst.root/"hidden_tests")
        v, h = run(["tests/"], inst.root), run(["hidden_tests/"], inst.root)
        total += 1
        status = "ok" if h != 0 else ("HIDDEN-MISSES-BUG" if v != 0 else "NEITHER-CATCHES")
        rows_out.append({"repo": repo.name, "bug_type": t, "visible_rc": v, "hidden_rc": h, "status": status})
        if h == 0:
            bad.append((repo.name, t, v, h, status))
        print(f"  {repo.name:20} {t:28} visible_rc={v} hidden_rc={h}  {status}", flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
import json
(ROOT/"artifacts/p1/reward_signal_audit.json").write_text(json.dumps(
    {"cells": rows_out, "undetected": len(bad), "total": total}, indent=2)+"\n")
print(f"\n{len(bad)}/{total} cells where hidden tests do NOT detect the injected bug")
for b in bad: print("   ", b)

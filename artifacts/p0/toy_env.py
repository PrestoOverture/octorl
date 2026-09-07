"""Framework-neutral two-tool code-repair spike with seeded hidden properties."""
from __future__ import annotations
import json
import random
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
CLEAN = 'def clamp(value: int, lower: int, upper: int) -> int:\n    """Clamp value to inclusive ordered bounds."""\n    if lower > upper:\n        raise ValueError("inverted bounds")\n    return max(lower, min(value, upper))\n'
BUG = CLEAN.replace('min(value, upper)', 'max(value, upper)')
VISIBLE = '''import pytest
from clamp import clamp
@pytest.mark.parametrize("value,lo,hi,expected", [(2,0,5,2),(-2,0,5,0),(9,0,5,5),(0,0,0,0),(-4,-8,-2,-4)])
def test_clamp(value,lo,hi,expected):
    assert clamp(value,lo,hi)==expected

def test_invalid():
    with pytest.raises(ValueError): clamp(0,3,1)
'''
HIDDEN = VISIBLE + '''\nfrom hypothesis import given, settings, seed, strategies as st
@seed(20260906)
@settings(max_examples=50, deadline=None, database=None)
@given(st.integers(-100000,100000),st.integers(-100000,100000),st.integers(0,100000))
def test_property(value,lo,width):
    hi=lo+width
    result=clamp(value,lo,hi)
    assert result == (lo if value<lo else hi if value>hi else value)
    assert lo<=result<=hi
    assert clamp(result,lo,hi)==result
'''

class ToyEnvironment:
    def __init__(self, seed: int = 20260906):
        self.seed=seed
        self.path=Path(tempfile.mkdtemp(prefix='octorl-p0-'))
        (self.path/'clamp.py').write_text(BUG)
        (self.path/'test_visible.py').write_text(VISIBLE)
        self.steps=0
        self.events: list[dict[str,Any]]=[]

    def call(self, name: str, arguments: dict[str,Any]) -> dict[str,Any]:
        start=time.time(); stdout=''; stderr=''; code=0
        self.steps+=1
        try:
            if self.steps>12: raise ValueError('12-step limit reached')
            if name=='read_file':
                path=arguments['path']
                if path not in ('clamp.py','test_visible.py'): raise ValueError('path not allowed')
                lines=(self.path/path).read_text().splitlines(keepends=True)
                selected=arguments.get('range')
                if selected is not None:
                    if not isinstance(selected,list) or len(selected)!=2 or not all(isinstance(x,int) for x in selected) or selected[0]<1 or selected[1]<selected[0]: raise ValueError('range must be [first,last], 1-based inclusive')
                    lines=lines[selected[0]-1:selected[1]]
                stdout=''.join(lines)
            elif name=='apply_patch':
                diff=arguments['diff']
                old=re.findall(r'^--- (\S+)',diff,re.M); new=re.findall(r'^\+\+\+ (\S+)',diff,re.M)
                if old not in (['a/clamp.py'],['clamp.py']) or new not in (['b/clamp.py'],['clamp.py']): raise ValueError('only clamp.py can be patched')
                result=subprocess.run(['patch','--batch','--forward','-p1' if old==['a/clamp.py'] else '-p0'],input=diff,text=True,capture_output=True,cwd=self.path,timeout=30)
                stdout,stderr,code=result.stdout,result.stderr,result.returncode
            else: raise ValueError(f'unknown tool: {name}')
        except (ValueError,KeyError,TypeError,subprocess.TimeoutExpired) as exc:
            stderr=str(exc); code=1
        event=dict(name=name,arguments=arguments,stdout=stdout,stderr=stderr,exit_code=code,start=start,end=time.time())
        self.events.append(event)
        return event

    def verify(self, hidden: bool=True) -> dict[str,Any]:
        target='test_hidden.py' if hidden else 'test_visible.py'
        if hidden: (self.path/target).write_text(HIDDEN)
        start=time.time()
        result=subprocess.run(['python','-m','pytest','-q',target],capture_output=True,text=True,cwd=self.path,timeout=30,env={**__import__('os').environ,'PYTHONDONTWRITEBYTECODE':'1'})
        return dict(stdout=result.stdout,stderr=result.stderr,exit_code=result.returncode,passed=result.returncode==0,start=start,end=time.time())

    def close(self) -> None:
        shutil.rmtree(self.path)

if __name__=='__main__':
    ROOT.joinpath('toy').mkdir(exist_ok=True)
    for name,content in [('clamp.py',BUG),('test_visible.py',VISIBLE),('test_hidden.py',HIDDEN),('clamp.clean.py',CLEAN)]: ROOT.joinpath('toy',name).write_text(content)
    env=ToyEnvironment(); results={'seed':env.seed,'bug_visible':env.verify(False),'bug_hidden':env.verify()}
    (env.path/'clamp.py').write_text(CLEAN)
    results.update(clean_visible=env.verify(False),clean_hidden=env.verify())
    ROOT.joinpath('toy_validation.json').write_text(json.dumps(results,indent=2)); env.close()
    assert not results['bug_visible']['passed'] and not results['bug_hidden']['passed'] and results['clean_visible']['passed'] and results['clean_hidden']['passed']

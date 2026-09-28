"""Apply the incremental Agent-R1 patch and deploy only R4r scripts/input copies."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile

WS=Path(__file__).resolve().parents[2]
ART=WS/'artifacts/self_improve/r4r'


def remote(command, data=None):
    return subprocess.run(['ssh','-o','BatchMode=yes','autodl-r4',command],input=data,
                          text=True,capture_output=True,check=True).stdout


def main():
    evidence=json.loads((ART/'patch_hashes.json').read_text())
    paths=[f['path'] for f in evidence['files']]
    cmd='sha256sum '+' '.join(shlex.quote(p) for p in paths)
    before={line.split(maxsplit=1)[1].strip():line.split()[0] for line in remote(cmd).splitlines()}
    expected={f['path']:f['before_sha256'] for f in evidence['files']}
    after_expected={f['path']:f['after_sha256'] for f in evidence['files']}
    if before==expected:
        patch=(ART/'agent_r1_r4r.patch').read_text()
        remote('cd /root/Agent-R1 && git apply --check -',patch)
        remote('cd /root/Agent-R1 && git apply -',patch)
    elif before!=after_expected:
        raise RuntimeError('remote Agent-R1 files changed; refusing to overwrite')
    after={line.split(maxsplit=1)[1].strip():line.split()[0] for line in remote(cmd).splitlines()}
    assert after==after_expected
    files=sorted(p for p in (WS/'scripts/self_improve').glob('r4r_*') if p.is_file())
    files+=sorted(p for p in (WS/'scripts/self_improve/r4r_inputs').rglob('*') if p.is_file())
    with tempfile.NamedTemporaryFile(mode='w') as listing:
        listing.write(''.join(str(p.relative_to(WS))+'\n' for p in files));listing.flush()
        subprocess.run(['rsync','-rlt','--partial','--files-from='+listing.name,'-e','ssh -o BatchMode=yes',str(WS)+'/',
                        'autodl-r4:/root/octorl_r3/'],check=True)
    hashes={str(p.relative_to(WS)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    code='import json,hashlib,pathlib\nfiles='+repr(list(hashes))+'\nprint(json.dumps({p:hashlib.sha256((pathlib.Path("/root/octorl_r3")/p).read_bytes()).hexdigest() for p in files}))'
    deployed=json.loads(remote('export PATH=/root/miniconda3/bin:$PATH; python3 -',code))
    assert deployed==hashes
    report={'agent_r1':evidence,'verified_remote_after':after,
            'deployed_files':[{'path':p,'local_sha256':h,'remote_sha256':deployed[p]} for p,h in hashes.items()]}
    (ART/'deployment_hashes.json').write_text(json.dumps(report,indent=2)+'\n')
    output=remote('export PATH=/root/miniconda3/bin:$PATH; bash /root/octorl_r3/scripts/self_improve/r4r_preflight.sh --dry-run')
    (ART/'remote_preflight_dry_run.txt').write_text(output)
    print(output)
    print(f'{len(files)} deployed files verified; Agent-R1 hashes verified')


if __name__=='__main__': main()

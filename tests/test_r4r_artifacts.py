from collections import Counter
import json
from pathlib import Path
import random
import numpy as np
from scripts.self_improve.r4r_gates import sha256
from scripts.self_improve.r4r_analysis import regression, synthetic_power, calibrations, clustered_ci
from src.tasks.tool_recovery.generator import content_fingerprint

WS=Path.cwd();ART=WS/'artifacts/self_improve/r4r'


def test_test2_counts_families_fingerprints_and_seal():
    data=ART/'data';manifest=json.loads((data/'test2_manifest.json').read_text());seal=json.loads((data/'test2_seal.json').read_text())
    assert sha256(data/'test2_manifest.json')==seal['manifest_sha256']
    assert sha256(data/'test2.parquet')==seal['parquet_sha256']
    instances=manifest['instances']
    assert Counter(i['fault_type'] for i in instances)=={'constraint_violation':60,'missing_dependency':52,'stale_version':48,'normal':40}
    assert {i['family'] for i in instances}=={'deployment_service','notification_service'}
    def fingerprints(items):
        result=set()
        for i in items:
            actual=content_fingerprint(i['family'],i['presented_config'],i['external_state'],i['fault_type'],i['faulted_field'])
            assert i['fingerprint']==actual
            result.add(actual)
        return result
    fresh=fingerprints(instances);assert len(fresh)==200
    for split in ('train','dev','test'):
        old=json.loads((WS/f'artifacts/self_improve/r3/data/{split}_manifest.json').read_text())
        assert not fresh & fingerprints(old['instances'])
        assert manifest['generator_version']==old['generator_version']=='r2.0'
    assert manifest['start_seed']==300000


def test_analysis_exact_regression():
    assert regression(WS)['deep_equal_all_fields']


def test_power_frozen_report_and_aa():
    assert clustered_ci(np.zeros(160),seed=20260928)==(0.,0.)
    report=json.loads((ART/'prelaunch_power.json').read_text())
    assert report['reps']==1000 and report['seed']==20260928 and report['n']==160
    assert set(report['conditions'])=={'independent','coupled_q_hi','coupled_q_lo'}
    for name,condition in report['conditions'].items():
        assert set(condition['shifts'])=={'0.00','0.03','0.05','0.10'}
        coverage=condition['shifts']['0.00']['coverage_of_zero']
        type1=condition['shifts']['0.00']['detection_rate']
        # Decision D1: inherited null control is the independent Bernoulli null; coupled nulls over-cover
        # by construction and are accepted on Type I <= 0.05. Numbers are frozen (no seed tuning).
        assert report['null_coverage_pass'][name]==((.93 <= coverage <= .97) if name=='independent' else type1 <= .05)
        assert report['null_coverage_pass'][name]
    assert report['conditions']['coupled_q_lo']['shifts']['0.00']['detection_rate']==.014
    assert report['conditions']['independent']['shifts']['0.00']['coverage_of_zero']==.951
    assert report['conditions']['coupled_q_hi']['shifts']['0.00']['coverage_of_zero']==.943
    assert report['conditions']['coupled_q_lo']['shifts']['0.00']['coverage_of_zero']==.972


def test_archive_manifest_and_seeded_sample():
    manifest=json.loads((WS/'artifacts/self_improve/r4/checkpoint_archive_manifest.json').read_text())
    assert manifest['all_matched']
    assert len(manifest['files'])==manifest['total_files']
    assert sum(r['size'] for r in manifest['files'])==manifest['total_bytes']
    for row in manifest['files']:assert row['local_sha256']==row['remote_sha256']
    for scope,count in manifest['remote_find_counts'].items():
        assert sum(r['path'].startswith(scope+'/') for r in manifest['files'])==count
    for row in random.Random(20260928).sample(manifest['files'],50):
        assert sha256(WS/'artifacts/self_improve/r4/checkpoint_archive'/row['path'])==row['remote_sha256']
    assert (ART/'deletion_list.txt').read_text().splitlines()==['/root/autodl-tmp/octorl_r4/'+s for s in ('seed_42','seed_137','seed_2718','validation')]


def test_deployment_and_patch_hash_records():
    deployment=json.loads((ART/'deployment_hashes.json').read_text())
    for row in deployment['deployed_files']:
        assert sha256(WS/row['path'])==row['local_sha256']==row['remote_sha256']
    patch=deployment['agent_r1']
    assert sha256(ART/'agent_r1_r4r.patch')==patch['patch_sha256']
    for row in patch['files']:
        name=Path(row['path']).name
        assert sha256(ART/'source_before'/name)==row['before_sha256']
        assert sha256(ART/'source_after'/name)==row['after_sha256']
        assert deployment['verified_remote_after'][row['path']]==row['after_sha256']


def test_primary_analysis_160_instances_default_seed_and_aa():
    from scripts.self_improve.r4r_analysis import analyze
    records={}
    for fault,n in [('stale_version',160),('normal',40)]:
        for i in range(n):
            for r in range(4):
                task=f'{fault}_{i}'
                records[task,r]={'task_id':task,'rollout_index':r,'fault_type':fault,'binary_reward':int(i%2==0)}
    result=analyze({'base':records,'fixed':records})
    assert result['bootstrap_seed']==20260928
    assert len(result['Q1_by_seed']['single']['per_instance'])==160
    assert result['Q1_by_seed']['single']['ci95']==[0.,0.]

"""Record a reviewable submission source snapshot after evidence checks pass."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prismserve import __version__


def main():
    def load(path):
        return json.loads((ROOT/path).read_text())
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    v=load('results/capacity_frontier/validation.json')
    s=load('results/capacity_frontier/submission_check.json')
    d=load('results/final_demo/demo_validation.json')
    tests=load('results/validation.json')
    assert all(r['passed'] for r in (v,s,d)) and tests['successful']
    assert d['live_capacity_verified'] and d['concurrent_http_verified']
    manifest=load('results/capacity_frontier/manifest.json')
    assert manifest['status']=='complete' and manifest['version']==__version__=='1.0.0'
    for name,value in manifest['source_sha256'].items():
        assert digest(ROOT/'prismserve'/name)==value
    files=[]
    for directory in ('prismserve','configs','tests','scripts','docs','data'):
        files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    files.extend(ROOT/name for name in ('README.md','pyproject.toml','references.bib'))
    sources={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(files)}
    evidence_files=['results/capacity_frontier/manifest.json','results/capacity_frontier/validation.json',
                    'results/capacity_frontier/submission_check.json','results/validation.json',
                    'results/final_demo/demo_validation.json','results/validation_study_final/manifest.json']
    result={'version':__version__,'status':'submission_snapshot','created_utc':datetime.now(timezone.utc).isoformat(),
            'source_sha256':sources,'evidence_sha256':{name:digest(ROOT/name) for name in evidence_files},
            'tests_passed':tests['tests_run'],'frontier_serving_runs':v['serving_runs'],
            'notes':['Immutable by convention: subsequent edits should use a new version and result folder.',
                     'Final sources, exact experiment snapshots, raw results, report, slides and wheel accompany the ZIP.',
                     'Large pretrained weights are prepared separately using their pinned source/hash manifest.']}
    (ROOT/'deliverables/release.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(f"Frozen submission version {__version__}: {len(sources)} source/data files; {tests['tests_run']} tests passed")


if __name__=='__main__':
    main()

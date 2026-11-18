"""Read-only publication integrity checks; optional deliberate manifest refresh."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'repository_manifest.json'


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def files_for_publication():
    raw=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT)
    names={name.decode('utf-8') for name in raw.split(b'\0') if name}
    return [ROOT/name for name in sorted(names) if name!='repository_manifest.json' and (ROOT/name).is_file()]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--write-manifest',action='store_true')
    args=parser.parse_args()
    release=json.loads((ROOT/'deliverables/release.json').read_text())
    substitutions={'README.md':'docs/archive/submission_v1/README.md',
                   '.gitignore':'docs/archive/submission_v1/.gitignore'}
    for name,value in release['source_sha256'].items():
        assert digest(ROOT/substitutions.get(name,name))==value, f'frozen source changed: {name}'
    for name,value in release['evidence_sha256'].items():
        assert digest(ROOT/name)==value, f'frozen evidence changed: {name}'
    runtime=json.loads((ROOT/'results/capacity_frontier/manifest.json').read_text())
    for name,value in runtime['source_sha256'].items():
        assert digest(ROOT/'prismserve'/name)==value, f'benchmark runtime changed: {name}'
    if args.write_manifest:
        files=files_for_publication()
        payload={'repository':'https://github.com/Vamaxi/PrismServe','project_version':'1.0.0',
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'files':{p.relative_to(ROOT).as_posix():{'bytes':p.stat().st_size,'sha256':digest(p)} for p in files},
            'selection_notes':['Complete source, tests, docs, report assets and accepted/labeled diagnostic evidence.',
                'Pretrained weights, supplied PDFs/full extracts, caches, debug renders, logs and duplicate deliverable ZIPs excluded.',
                'Frozen README/.gitignore originals are archived separately; public versions are updated for GitHub.']}
        MANIFEST.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    manifest=json.loads(MANIFEST.read_text())
    for name,item in manifest['files'].items():
        assert digest(ROOT/name)==item['sha256'], f'publication file changed: {name}'
    files=files_for_publication()
    assert {p.relative_to(ROOT).as_posix() for p in files}==set(manifest['files']), 'unrecorded publication files'
    assert all(p.stat().st_size<100*2**20 for p in files),'file exceeds GitHub single-file size limit'
    assert not any(p.suffix=='.safetensors' for p in files),'pretrained weights selected unexpectedly'
    assert not any('debug' in p.name and 'reports' in p.parts for p in files),'render debug selected unexpectedly'
    print(json.dumps({'passed':True,'publication_files':len(files),
                      'uncompressed_mib':round(sum(p.stat().st_size for p in files)/2**20,2),
                      'frozen_runtime_source_verified':True,'original_submission_source_verified':True,
                      'model_weights_excluded':True,'publication_file_hashes_verified':True},indent=2))


if __name__=='__main__':main()

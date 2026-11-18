"""Check current report/deck/wheel and optionally every packaged file hash."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prismserve import __version__


def main():
    import pymupdf
    from pptx import Presentation
    parser=argparse.ArgumentParser()
    parser.add_argument('--package',action='store_true')
    args=parser.parse_args()
    out=ROOT/'results/capacity_frontier'
    validation=json.loads((out/'validation.json').read_text())
    assert validation['passed']
    manifest=json.loads((out/'manifest.json').read_text())
    with zipfile.ZipFile(ROOT/f'results/dist/prismserve-{__version__}-py3-none-any.whl') as wheel:
        for name,value in manifest['source_sha256'].items():
            assert hashlib.sha256(wheel.read('prismserve/'+name)).hexdigest()==value
    report=pymupdf.open(ROOT/'deliverables/PrismServe_Final_Report.pdf')
    slides=pymupdf.open(ROOT/'deliverables/PrismServe_Presentation.pdf')
    ppt=Presentation(ROOT/'deliverables/PrismServe_Presentation.pptx')
    assert len(slides)==len(ppt.slides)==22
    text=''.join(page.get_text() for page in report)
    assert 'v1.0' in text and '675' in text and '2048' in text
    assert all(len(page.get_text().strip())>50 for page in report)
    images=sum(len(page.get_images()) for page in report)
    assert images>=13
    record={'passed':True,'version':__version__,'report_pages':len(report),'report_embedded_images':images,
            'presentation_slides':len(slides),'wheel_runtime_matches_benchmark_snapshot':True}
    if args.package:
        release=json.loads((ROOT/'deliverables/release.json').read_text())
        assert release['version']==__version__
        for path,value in release['source_sha256'].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==value,path
        with zipfile.ZipFile(ROOT/'deliverables/PrismServe_Project.zip') as archive:
            package=json.loads(archive.read('PrismServe/deliverables/package_manifest.json'))
            for path,value in package['files'].items():
                assert hashlib.sha256(archive.read('PrismServe/'+path)).hexdigest()==value,path
                assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==value,path
            assert archive.testzip() is None
            assert not any(name.endswith('model.safetensors') for name in archive.namelist())
            record['packaged_files_verified']=len(package['files'])
    else:
        (out/'submission_check.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()

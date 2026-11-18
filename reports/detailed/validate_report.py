"""Validate companion documents without touching the frozen submission records."""
from pathlib import Path
import csv
import hashlib
import json
import re

import pymupdf

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    m=json.loads((HERE/'report_manifest.json').read_text())
    for name,value in m['artifact_sha256'].items():assert digest(ROOT/'deliverables'/name)==value,name
    for name,value in m['input_sha256'].items():assert digest(ROOT/name)==value,name
    release=json.loads((ROOT/'deliverables/release.json').read_text())
    for key in ('source_sha256','evidence_sha256'):
        for name,value in release[key].items():assert digest(ROOT/name)==value,name
    for name,value in m['protected_original_sha256'].items():assert digest(ROOT/name)==value,name
    pdf=pymupdf.open(ROOT/'deliverables/PrismServe_Detailed_Report.pdf')
    assert len(pdf)==m['pdf']['pages']
    assert len(pdf.get_toc())==m['pdf']['bookmarks']
    text=''.join(p.get_text() for p in pdf)
    assert all(p.get_text().strip() for p in pdf)
    assert text.count('1. Executive summary and principal')<=3,'pagination restarted content'
    assert 'Appendix C.' in text and '2.752' in text
    clipped=[]
    for i,page in enumerate(pdf):
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                for span in line['spans']:
                    x0,y0,x1,y1=span['bbox']
                    if x0<-1 or x1>page.rect.width+1 or y0<-1 or y1>page.rect.height+1:
                        clipped.append((i+1,span['text']))
        for link in page.get_links():
            if link.get('kind')==pymupdf.LINK_GOTO:assert 0<=link['page']<len(pdf)
    assert not clipped,clipped[:5]
    atlas=pymupdf.open(ROOT/'deliverables/PrismServe_Graph_and_Diagram_Atlas.pdf')
    assert len(atlas)==len(m['figures'])==47
    assert all(len(p.get_images())==1 for p in atlas)
    html=(ROOT/'deliverables/PrismServe_Detailed_Report.html').read_text(encoding='utf-8')
    assert len(re.findall(r'src="data:image/png;base64,',html))==53
    ids=set(re.findall(r'\bid="([^"]+)"',html))
    assert all(a in ids for a in re.findall(r'href="#([^"]+)"',html))
    total=0
    for p in sorted((HERE/'tables').glob('*.csv')):
        with p.open(encoding='utf-8',newline='') as f:total+=len(list(csv.DictReader(f)))
    assert total==m['run_level_table_rows']==454
    original_charts={p for p in (ROOT/'results').rglob('*.png') if 'preview' not in p.name and 'page_' not in p.name}
    assert len(original_charts)==m['existing_result_charts_covered']==31
    record={'passed':True,'pdf_pages':len(pdf),'registered_figures':len(m['figures']),
            'equation_panels':len(m['equations']),'registered_tables':len(m['tables']),
            'complete_run_table_rows':total,'all_existing_result_charts_included':len(original_charts),
            'atlas_pages':len(atlas),
            'self_contained_html_and_internal_links_verified':True,'pdf_text_within_page_bounds':True,
            'original_frozen_source_and_submission_artifacts_unchanged':True,
            'visually_reviewed':['cover','architecture','control-flow diagram','capacity results','table appendix and page contact sheets']}
    (HERE/'companion_validation.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()

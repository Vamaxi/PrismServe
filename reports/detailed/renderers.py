"""Local PDF/HTML exporters for the report companion; no new packages."""
from pathlib import Path
import base64
import html
import json
import re

import markdown
import pymupdf

CSS='''
body {font-family:sans-serif;font-size:10pt;line-height:1.35;color:#20374a;}
h1 {font-size:29pt;color:#123b50;} h2 {font-size:17pt;color:#176b87;page-break-before:always;margin-bottom:13pt;}
h3 {font-size:12pt;color:#176b87;margin-top:14pt;} h4 {font-size:10.5pt;color:#123b50;}
p {margin:7pt 0;} li {margin:4pt 0;} a {color:#176b87;text-decoration:none;}
table {border-collapse:collapse;width:100%;font-size:7.9pt;margin:8pt 0 13pt;}
th,td {border:.4pt solid #bfced6;padding:4pt;} th {font-weight:bold;color:#123b50;}
pre {font-family:monospace;font-size:8pt;border-left:2pt solid #91b7c8;padding:8pt;white-space:pre-wrap;}
code {font-family:monospace;font-size:8pt;} img {max-width:100%;}
.figure {margin:12pt 0;page-break-inside:avoid;} .caption {font-size:8.5pt;color:#456071;margin:5pt 0 10pt;}
.cover {margin-top:48pt;} .eyebrow {font-size:10pt;color:#176b87;letter-spacing:1pt;}
.subtitle {font-size:19pt;color:#35566b;line-height:1.3;} .cover-note {font-size:11pt;line-height:1.6;}
.toc {font-size:10pt;} .small {font-size:8.5pt;color:#526b7c;}
.figure-frame {border:0;margin:12pt 0;width:100%;} .figure-frame td {border:0;padding:0;}
.diagram-frame {page-break-before:always;}
'''


def prepare_html(md,cover):
    body=markdown.markdown(md,extensions=['tables','fenced_code'],output_format='xhtml')
    headings=[]
    def heading(match):
        level=int(match.group(1));inner=match.group(2)
        text=re.sub('<[^>]+>','',inner)
        identifier=f'section-{len(headings)+1}'
        headings.append((level,identifier,html.unescape(text)))
        return f'<h{level} id="{identifier}">{inner}</h{level}>'
    body=re.sub(r'<h([234])>(.*?)</h\1>',heading,body,flags=re.S)
    toc='<h2 id="contents">Contents</h2><div class="toc"><ol>'
    toc+=''.join(f'<li><a href="#{key}">{html.escape(text)}</a></li>' for level,key,text in headings if level==2)
    toc+='</ol></div>'
    return cover+toc+body,headings


def export_pdf(body,destination,root):
    positions=[]
    def position(p):
        if p.open_close&1 and p.heading and p.id:
            positions.append((p.heading,p.text,p.page_num,p.id))
    from PIL import Image
    # Bound intrinsic raster sizes explicitly. A high-resolution figure must
    # not create a block too tall to place in the current page rectangle.
    def size_image(match):
        src=match.group(1)
        with Image.open(root/src) as img:width,height=img.size
        scale=min(495/width,390/height)
        return f'<img src="{src}" style="width:{width*scale:.1f}pt;height:{height*scale:.1f}pt;"'
    pdfbody=re.sub(r'<img src="([^"]+)"',size_image,body)
    def figure_frame(match):
        identifier=match.group(1) or ''
        content=match.group(2)
        classes='figure-frame'+(' diagram-frame' if 'Original ' in content else '')
        return f'<table class="{classes}"{identifier}><tr><td>{content}</td></tr></table>'
    pdfbody=re.sub(r'<div class="figure"( id="[^"]+")?>(.*?)</div>',figure_frame,pdfbody,flags=re.S)
    story=pymupdf.Story(html=pdfbody,user_css=CSS,archive=pymupdf.Archive(str(root)))
    page=pymupdf.Rect(0,0,595,842);box=pymupdf.Rect(45,45,550,791)
    def rectangle(n,filled):
        if n>160:raise RuntimeError('Unexpected report pagination exceeds 160 pages; last headings: '+repr(positions[-4:])+'; filled='+str(filled))
        return page,box,None
    doc=story.write_with_links(rectangle,positionfn=position)
    toc=[]
    for level,text,pageno,identifier in positions:
        if level>=2 and identifier!='contents':
            target=min(level-1,(toc[-1][0]+1) if toc else 1)
            toc.append([target,text,pageno])
    if toc:doc.set_toc(toc)
    for i,p in enumerate(doc):
        if i:
            p.insert_text((45,28),'PRISMSERVE | DETAILED TECHNICAL REPORT',fontsize=7.2,color=(.3,.43,.5))
        p.draw_line((45,806),(550,806),color=(.72,.8,.84),width=.5)
        p.insert_text((45,821),'DA4022 | Companion to submission v1.0.0 | 18 November 2026',fontsize=7.2,color=(.3,.43,.5))
        p.insert_text((527,821),str(i+1),fontsize=8,color=(.3,.43,.5))
    doc.set_metadata({'title':'PrismServe: Comprehensive Technical Report','author':'DA4022 project team',
                      'subject':'Architecture, algorithms, all experiment graphs, numerical results and reproducibility'})
    doc.save(destination,deflate=True)
    summary={'pages':len(doc),'embedded_image_placements':sum(len(p.get_images()) for p in doc),
             'bookmarks':len(toc),'internal_links':sum(len(p.get_links()) for p in doc)}
    doc.close()
    return summary


def export_html(body,destination,root):
    def embed(match):
        data=(root/match.group(1)).read_bytes()
        return 'src="data:image/png;base64,'+base64.b64encode(data).decode()+'"'
    body=re.sub(r'src="([^"]+)"',embed,body)
    screen='body{max-width:1060px;margin:2.5rem auto;padding:0 1.5rem}h2{border-top:1px solid #cfdee5;padding-top:1.4rem}table{font-size:.88rem}img{height:auto}.figure{margin:1.4rem 0}pre{overflow:auto}html{scroll-behavior:smooth}'
    destination.write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>PrismServe Comprehensive Technical Report</title><style>'+CSS+screen+'</style></head><body>'+body+'</body></html>',encoding='utf-8')


def export_atlas(figures,destination,root):
    from PIL import Image
    doc=pymupdf.open();toc=[]
    for f in figures:
        path=root/f['asset'];w,h=Image.open(path).size
        landscape=w/h>1.45
        page=doc.new_page(width=842 if landscape else 595,height=595 if landscape else 842)
        width,height=page.rect.width,page.rect.height
        page.insert_text((35,28),f"Figure {f['number']} | {f['kind']}",fontsize=12,color=(.07,.23,.31))
        rect=pymupdf.Rect(35,45,width-35,height-125)
        page.insert_image(rect,filename=str(path),keep_proportion=True)
        page.insert_textbox(pymupdf.Rect(35,height-115,width-35,height-20),f['caption']+'\nSource: '+f['source'],fontsize=9,color=(.15,.25,.31))
        toc.append([1,f"{f['number']}. {f['caption'][:95]}",len(doc)])
    doc.set_toc(toc);doc.set_metadata({'title':'PrismServe Graph and Diagram Atlas','author':'DA4022 project team'})
    doc.save(destination,deflate=True);count=len(doc);doc.close();return count

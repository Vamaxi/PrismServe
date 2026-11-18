"""Local PDF/HTML/DOCX exporters for the report companion; no new packages."""
from pathlib import Path
import base64
import html
import json
import re
import zipfile
from xml.etree import ElementTree as ET

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


W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
WP='http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
A='http://schemas.openxmlformats.org/drawingml/2006/main'
PIC='http://schemas.openxmlformats.org/drawingml/2006/picture'
for prefix,uri in [('w',W),('r',R),('wp',WP),('a',A),('pic',PIC)]:ET.register_namespace(prefix,uri)


def node(parent,tag,attrs=None,text=None):
    prefix,name=tag.split(':');uri={'w':W,'r':R,'wp':WP,'a':A,'pic':PIC}[prefix]
    converted={}
    for k,v in (attrs or {}).items():
        if ':' in k:
            p,n=k.split(':');k='{'+{'w':W,'r':R,'wp':WP,'a':A,'pic':PIC}[p]+'}'+n
        converted[k]=str(v)
    result=ET.SubElement(parent,'{'+uri+'}'+name,converted)
    result.text=text
    return result


def export_docx(body,destination,root):
    """Generate standard editable OOXML paragraphs/tables with embedded figures."""
    from PIL import Image
    document=ET.Element('{'+W+'}document');content=node(document,'w:body')
    images=[];links=[];bookmark_counter=0
    def paragraph(parent,style='Normal',identifier=None):
        nonlocal bookmark_counter
        p=node(parent,'w:p');pr=node(p,'w:pPr');node(pr,'w:pStyle',{'w:val':style})
        if identifier:
            bookmark_counter+=1
            node(p,'w:bookmarkStart',{'w:id':bookmark_counter,'w:name':identifier.replace('-','_')})
            node(p,'w:bookmarkEnd',{'w:id':bookmark_counter})
        return p
    def run(parent,text,bold=False,italic=False,code=False):
        if not text:return
        r=node(parent,'w:r');pr=node(r,'w:rPr')
        if bold:node(pr,'w:b')
        if italic:node(pr,'w:i')
        if code:node(pr,'w:rFonts',{'w:ascii':'Consolas','w:hAnsi':'Consolas'})
        for index,line in enumerate(text.split('\n')):
            if index:node(r,'w:br')
            t=node(r,'w:t',text=line);t.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
    def inline(element,p,bold=False,italic=False,code=False):
        run(p,element.text,bold,italic,code)
        for child in element:
            if child.tag=='br':node(node(p,'w:r'),'w:br')
            elif child.tag=='a':
                href=child.get('href','')
                if href.startswith('#'):link=node(p,'w:hyperlink',{'w:anchor':href[1:].replace('-','_')})
                else:
                    rid=f'rId{1000+len(links)}';links.append((rid,href));link=node(p,'w:hyperlink',{'r:id':rid})
                inline(child,link,bold,italic,code)
            else:inline(child,p,bold or child.tag in ('strong','b'),italic or child.tag in ('em','i'),code or child.tag=='code')
            run(p,child.tail,bold,italic,code)
    def image(element,parent,identifier=None):
        path=root/element.get('src')
        width,height=Image.open(path).size
        scale=min(6.65*914400/width,7.1*914400/height)
        cx,cy=int(width*scale),int(height*scale)
        index=len(images)+1;rid=f'rId{10+index}';images.append((rid,path))
        p=paragraph(parent,'Figure',identifier);pr=p.find('{'+W+'}pPr');node(pr,'w:keepNext')
        drawing=node(node(p,'w:r'),'w:drawing');frame=node(drawing,'wp:inline',{'distT':0,'distB':0,'distL':0,'distR':0})
        node(frame,'wp:extent',{'cx':cx,'cy':cy});node(frame,'wp:docPr',{'id':index,'name':path.stem,'descr':element.get('alt','')})
        g=node(frame,'a:graphic');gd=node(g,'a:graphicData',{'uri':PIC});pic=node(gd,'pic:pic')
        nv=node(pic,'pic:nvPicPr');node(nv,'pic:cNvPr',{'id':index,'name':path.name});node(nv,'pic:cNvPicPr')
        fill=node(pic,'pic:blipFill');node(fill,'a:blip',{'r:embed':rid});node(node(fill,'a:stretch'),'a:fillRect')
        sp=node(pic,'pic:spPr');tr=node(sp,'a:xfrm');node(tr,'a:off',{'x':0,'y':0});node(tr,'a:ext',{'cx':cx,'cy':cy})
        node(node(sp,'a:prstGeom',{'prst':'rect'}),'a:avLst')
    def blocks(element,parent):
        for e in element:
            if e.tag in ('h1','h2','h3','h4'):
                style={'h1':'Title','h2':'Heading1','h3':'Heading2','h4':'Heading3'}[e.tag]
                p=paragraph(parent,style,e.get('id'));inline(e,p)
            elif e.tag=='p':
                img=e.find('img')
                if img is not None:image(img,parent,e.get('id'));continue
                cls=e.get('class','');style='Caption' if 'caption' in cls else 'Subtitle' if 'subtitle' in cls else 'Normal'
                p=paragraph(parent,style,e.get('id'));inline(e,p)
            elif e.tag=='img':image(e,parent,e.get('id'))
            elif e.tag=='table':
                table=node(parent,'w:tbl');pr=node(table,'w:tblPr');node(pr,'w:tblW',{'w:w':0,'w:type':'auto'})
                borders=node(pr,'w:tblBorders')
                for edge in ('top','left','bottom','right','insideH','insideV'):node(borders,'w:'+edge,{'w:val':'single','w:sz':4,'w:color':'BFCDD5'})
                rows=e.findall('.//tr')
                for row in rows:
                    tr=node(table,'w:tr');header=row.find('th') is not None
                    if header:node(node(tr,'w:trPr'),'w:tblHeader')
                    for cell in row:
                        tc=node(tr,'w:tc');props=node(tc,'w:tcPr')
                        node(props,'w:tcW',{'w:w':int(9500/max(1,len(row))),'w:type':'dxa'})
                        p=paragraph(tc,'TableText');inline(cell,p,bold=header)
                paragraph(parent,'Normal')
            elif e.tag in ('ul','ol'):
                for i,item in enumerate(e.findall('li'),1):
                    p=paragraph(parent,'Normal');run(p,f'{i}. ' if e.tag=='ol' else '\u2022 ');inline(item,p)
            elif e.tag=='pre':
                p=paragraph(parent,'Code');run(p,''.join(e.itertext()),code=True)
            elif e.tag=='div' and 'figure' in e.get('class',''):
                img=e.find('img')
                if img is not None:image(img,parent,e.get('id'))
                for caption in e.findall('p'):
                    p=paragraph(parent,'Caption');inline(caption,p)
            else:blocks(e,parent)
    tree=ET.fromstring('<root>'+body+'</root>');blocks(tree,content)
    sect=node(content,'w:sectPr');node(sect,'w:footerReference',{'w:type':'default','r:id':'rId3'})
    node(sect,'w:pgSz',{'w:w':11906,'w:h':16838});node(sect,'w:pgMar',{'w:top':900,'w:bottom':900,'w:left':950,'w:right':950,'w:header':360,'w:footer':360})
    styles=ET.Element('{'+W+'}styles')
    for name,size,color in [('Normal',21,'20374A'),('Title',52,'123B50'),('Subtitle',32,'35566B'),('Heading1',32,'176B87'),('Heading2',25,'176B87'),('Heading3',22,'123B50'),('Caption',18,'456071'),('TableText',17,'20374A'),('Code',17,'20374A'),('Figure',18,'20374A')]:
        st=node(styles,'w:style',{'w:type':'paragraph','w:styleId':name});node(st,'w:name',{'w:val':name})
        pp=node(st,'w:pPr');node(pp,'w:spacing',{'w:after':100,'w:line':276,'w:lineRule':'auto'});node(pp,'w:widowControl')
        if name.startswith('Heading'):
            node(pp,'w:keepNext');node(pp,'w:outlineLvl',{'w:val':int(name[-1])-1})
        if name=='Heading1':node(pp,'w:pageBreakBefore')
        if name=='Figure':node(pp,'w:jc',{'w:val':'center'})
        rp=node(st,'w:rPr');node(rp,'w:rFonts',{'w:ascii':'Calibri','w:hAnsi':'Calibri'})
        node(rp,'w:sz',{'w:val':size});node(rp,'w:color',{'w:val':color})
        if name.startswith('Heading') or name=='Title':node(rp,'w:b')
    footer=ET.Element('{'+W+'}ftr');p=paragraph(footer,'Caption');run(p,'PrismServe | Detailed technical report | Page ')
    rr=node(p,'w:r');node(rr,'w:fldChar',{'w:fldCharType':'begin'});node(node(p,'w:r'),'w:instrText',text=' PAGE ')
    node(node(p,'w:r'),'w:fldChar',{'w:fldCharType':'end'})
    rels=[('rId1','styles','styles.xml',None),('rId2','settings','settings.xml',None),('rId3','footer','footer1.xml',None)]
    rels += [(rid,'image','media/'+path.name,None) for rid,path in images]
    rels += [(rid,'hyperlink',href,'External') for rid,href in links]
    def relationships(items):
        rootrel=ET.Element('Relationships',{'xmlns':'http://schemas.openxmlformats.org/package/2006/relationships'})
        for rid,kind,target,mode in items:
            at={'Id':rid,'Type':'http://schemas.openxmlformats.org/officeDocument/2006/relationships/'+kind,'Target':target}
            if mode:at['TargetMode']=mode
            ET.SubElement(rootrel,'Relationship',at)
        return ET.tostring(rootrel,encoding='utf-8',xml_declaration=True)
    types='''<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/><Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/><Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/></Types>'''
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml',types)
        z.writestr('_rels/.rels',relationships([('rId1','officeDocument','word/document.xml',None)]))
        z.writestr('word/document.xml',ET.tostring(document,encoding='utf-8',xml_declaration=True))
        z.writestr('word/styles.xml',ET.tostring(styles,encoding='utf-8',xml_declaration=True))
        z.writestr('word/settings.xml',f'<w:settings xmlns:w="{W}"><w:updateFields w:val="true"/></w:settings>')
        z.writestr('word/footer1.xml',ET.tostring(footer,encoding='utf-8',xml_declaration=True))
        z.writestr('word/_rels/document.xml.rels',relationships(rels))
        for path in dict.fromkeys(path for _,path in images):z.write(path,'word/media/'+path.name)
    return {'image_placements':len(images),'tables':len(document.findall('.//{'+W+'}tbl')),'bookmarks':bookmark_counter}


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

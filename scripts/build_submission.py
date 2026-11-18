"""Export authored report/slides and measured figures; no external service needed."""
from __future__ import annotations

from pathlib import Path
import importlib.metadata
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def _html(markdown_text):
    import markdown
    return markdown.markdown(markdown_text, extensions=["tables", "fenced_code"])


def report_pdf(source: Path, destination: Path):
    import pymupdf
    content = source.read_text(encoding="utf-8")
    css = """
    body { font-family: sans-serif; font-size: 10pt; line-height: 1.38; color: #162e40; }
    h1 { font-size: 25pt; color: #12394a; margin-bottom: 14pt; }
    h2 { font-size: 14pt; color: #176b87; margin-top: 16pt; }
    h3 { font-size: 11pt; color: #176b87; }
    p { margin: 6pt 0; } li { margin: 3pt 0; }
    table { border-collapse: collapse; width: 100%; margin: 9pt 0; font-size: 8pt; }
    th, td { border: .4pt solid #b7c9d0; padding: 4pt; }
    th { font-weight: bold; color: #176b87; } code { font-family: monospace; font-size: 8pt; }
    pre { border-left: 2pt solid #b7c9d0; padding: 8pt; white-space: pre-wrap; }
    a { color: #176b87; } img { max-width: 100%; }
    """
    html = _html(content)
    # Documents are published in deliverables; fix relative links for the HTML
    # copy, while Story resolves image files against the source's own directory.
    def image_source(match):
        value = match.group(1)
        if value.startswith(("http://", "https://", "data:")):
            return match.group(0)
        relative = (source.parent / value).resolve().relative_to(ROOT).as_posix()
        return f'src="{relative}"'
    pdf_html = re.sub(r'src="([^"]+)"', image_source, html)
    story = pymupdf.Story(html=pdf_html, user_css=css, archive=pymupdf.Archive(str(ROOT)))
    page = pymupdf.Rect(0, 0, 595, 842)
    rectangle = pymupdf.Rect(45, 43, 550, 795)
    doc = story.write_with_links(lambda number, filled: (page, rectangle, None))
    for index, pdf_page in enumerate(doc):
        pdf_page.draw_line((45, 809), (550, 809), color=(.7, .8, .83), width=.5)
        pdf_page.insert_text((45, 823), f"PrismServe | DA4022 | 18 November 2026 | {index + 1}", fontsize=8, color=(.25, .4, .47))
    doc.set_metadata({"title": "PrismServe: final technical report", "author": "DA4022 project team"})
    doc.save(destination, deflate=True)
    doc.close()
    standalone = "<!doctype html><meta charset='utf-8'><title>PrismServe report</title><style>" + css + "body{max-width:1000px;margin:3em auto;padding:0 2em}</style>" + html
    standalone = re.sub(r'(href|src)="\.\./', r'\1="../', standalone)
    (destination.with_suffix(".html")).write_text(standalone, encoding="utf-8")


def _slides(source: Path):
    raw = source.read_text(encoding="utf-8")
    raw = re.sub(r"\A---\n.*?\n---\n", "", raw, flags=re.S)
    slides = []
    for section in re.split(r"\n---\s*\n", raw):
        notes_match = re.search(r"<!--\s*Speaker notes:\s*(.*?)\s*-->", section, flags=re.S)
        notes = notes_match.group(1) if notes_match else ""
        text = re.sub(r"<!--.*?-->", "", section, flags=re.S).strip()
        title_match = re.search(r"^# (.+)$", text, flags=re.M)
        if not title_match:
            continue
        title = title_match.group(1)
        body = text[title_match.end():].strip()
        slides.append((title, body, notes))
    return slides


def _plain(text):
    text = re.sub(r"```\w*\n?", "", text)
    text = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("**", "").replace("`", "")
    text = re.sub(r"^#{1,6} ", "", text, flags=re.M)
    return text


def presentation(source: Path, destination: Path):
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.text import MSO_AUTO_SIZE
    from pptx.util import Inches, Pt
    import pymupdf

    slides = _slides(source)
    deck = Presentation()
    deck.slide_width, deck.slide_height = Inches(13.333), Inches(7.5)
    pdf = pymupdf.open()
    for number, (title, body, notes) in enumerate(slides, 1):
        slide = deck.slides.add_slide(deck.slide_layouts[6])
        background = slide.background.fill
        background.solid()
        background.fore_color.rgb = RGBColor.from_string("F6F9FB")
        banner = slide.shapes.add_shape(1, 0, 0, deck.slide_width, Inches(.15))
        banner.fill.solid()
        banner.fill.fore_color.rgb = RGBColor.from_string("176B87")
        banner.line.fill.background()
        box = slide.shapes.add_textbox(Inches(.6), Inches(.45), Inches(12), Inches(.8))
        paragraph = box.text_frame.paragraphs[0]
        paragraph.text = title
        paragraph.font.name = "Aptos Display"
        paragraph.font.size = Pt(32)
        paragraph.font.bold = True
        paragraph.font.color.rgb = RGBColor.from_string("12394A")
        table_lines = [line.strip() for line in body.splitlines()
                       if line.strip().startswith("|") and line.count("|") >= 2]
        body_without_table = "\n".join(line for line in body.splitlines() if line.strip() not in table_lines)
        body_top = 1.4
        if table_lines:
            matrix = [[cell.strip().replace("**", "").replace("`", "") for cell in line.strip("|").split("|")]
                      for line in table_lines if not re.fullmatch(r"[|:\- ]+", line)]
            columns = max(len(row) for row in matrix)
            table_height = min(3.3, .52 * len(matrix))
            shape = slide.shapes.add_table(len(matrix), columns, Inches(.7), Inches(body_top), Inches(11.9), Inches(table_height))
            for row_index, row in enumerate(matrix):
                for column_index in range(columns):
                    cell = shape.table.cell(row_index, column_index)
                    cell.text = row[column_index] if column_index < len(row) else ""
                    cell.margin_left, cell.margin_right = Inches(.1), Inches(.1)
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = RGBColor.from_string("176B87" if row_index == 0 else "E6F0F4")
                    for paragraph in cell.text_frame.paragraphs:
                        paragraph.font.name = "Aptos"
                        paragraph.font.size = Pt(16 if columns >= 4 else 18)
                        paragraph.font.bold = row_index == 0
                        paragraph.font.color.rgb = RGBColor.from_string("FFFFFF" if row_index == 0 else "203C4D")
            body_top += table_height + .25
        body_box = slide.shapes.add_textbox(Inches(.7), Inches(body_top), Inches(11.9), Inches(6.7 - body_top))
        text_frame = body_box.text_frame
        text_frame.word_wrap = True
        text_frame.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
        plain = _plain(body_without_table)
        lines = [line for line in plain.splitlines() if line.strip() and not re.fullmatch(r"[|:\- ]+", line)]
        for i, line in enumerate(lines):
            paragraph = text_frame.paragraphs[0] if i == 0 else text_frame.add_paragraph()
            paragraph.text = line
            paragraph.font.name = "Consolas" if line.startswith("|") else "Aptos"
            paragraph.font.size = Pt(17 if line.startswith("|") else 22)
            paragraph.font.color.rgb = RGBColor.from_string("203C4D")
            paragraph.space_after = Pt(11)
        footer = slide.shapes.add_textbox(Inches(.7), Inches(7), Inches(12), Inches(.25))
        footer.text = f"PRISMSERVE  |  DA4022  |  REAL GPT-2 + TINY EXECUTION + LABELED SIMULATION                         {number:02d}"
        footer.text_frame.paragraphs[0].font.size = Pt(9)
        footer.text_frame.paragraphs[0].font.color.rgb = RGBColor.from_string("547382")
        if notes:
            slide.notes_slide.notes_text_frame.text = notes
        pdf_page = pdf.new_page(width=960, height=540)
        pdf_page.draw_rect((0, 0, 960, 11), color=None, fill=(.09, .42, .53))
        pdf_page.insert_textbox((42, 30, 920, 92), title, fontsize=27, fontname="hebo", color=(.07, .22, .29))
        css = "body{font-family:sans-serif;font-size:17pt;line-height:1.2;color:#203c4d}h2{font-size:22pt}li{margin:8pt 0}p{margin:8pt 0}table{width:100%;font-size:14pt;border-collapse:collapse}td,th{padding:6pt;border-bottom:1pt solid #b7c9d0}pre{font-family:monospace;font-size:13pt}"
        # HTML box scales down only if needed, giving a readable complete PDF.
        pdf_page.insert_htmlbox((48, 104, 914, 483), _html(body), css=css)
        pdf_page.insert_text((48, 519), f"PRISMSERVE | DA4022 | real GPT-2 + tiny execution + labeled simulation | {number:02d}", fontsize=8, color=(.3, .45, .5))
    deck.save(destination)
    pdf.set_metadata({"title": "PrismServe project presentation", "author": "DA4022 project team"})
    pdf.save(destination.with_suffix(".pdf"), deflate=True)
    pdf.close()


def measured_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = ROOT / "results/demo/real_backend.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in data["measurements"] if r["device"] == "cuda"]
    if not rows:
        rows = [r for r in data["measurements"] if r["device"] == "cpu"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    labels = [r["variant"] for r in rows]
    axes[0].bar(labels, [r["resident_model_bytes"] / 1024 for r in rows], color="#176b87")
    axes[0].set_ylabel("Resident model tensors (KiB)")
    axes[1].bar(labels, [r["tbt_ms_mean"] for r in rows], color="#cf864b")
    axes[1].set_ylabel("Measured mean TBT (ms)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("REAL tiny-model execution — storage savings do not guarantee speedups")
    fig.tight_layout()
    figure_dir = ROOT / "results/demo/figures"
    figure_dir.mkdir(exist_ok=True)
    fig.savefig(figure_dir / "measured_storage_latency.png", dpi=170)
    plt.close(fig)


def environment():
    import platform
    packages = {}
    for name in ["torch", "numpy", "psutil", "fastapi", "uvicorn", "pydantic", "matplotlib", "python-pptx", "pymupdf", "markdown", "transformers", "safetensors", "tokenizers"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    data = {"python": sys.version, "platform": platform.platform(), "packages": packages}
    (ROOT / "results/environment.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    output = ROOT / "deliverables"
    output.mkdir(exist_ok=True)
    environment()
    measured_figure()
    report_pdf(ROOT / "docs/final_report.md", output / "PrismServe_Final_Report.pdf")
    presentation(ROOT / "docs/presentation.md", output / "PrismServe_Presentation.pptx")
    print(f"Exported report and presentation to {output}")


if __name__ == "__main__":
    main()

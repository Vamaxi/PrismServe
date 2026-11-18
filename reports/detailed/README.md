# Comprehensive report companion

This companion expands the frozen PrismServe submission v1.0.0 without changing its code, original report/slides, release manifest or ZIP. It regenerates presentation artifacts from archived evidence; it does not rerun inference or create additional benchmark observations.

The report includes the implementation architecture, request/control flowcharts, selective batching, CPU-attention data movement, precision restoration, memory accounting, equations, protocols, every archived result chart, unfavorable results, diagnostic sessions, and complete numerical tables for all 454 main-suite runs. It separates actual execution from analytical and diagnostic evidence.

## Outputs

- `deliverables/PrismServe_Detailed_Report.pdf`: full report, clickable contents/figure/table indices and PDF bookmarks.
- `deliverables/PrismServe_Detailed_Report.html`: self-contained HTML with embedded images and internal links; usable offline.
- `deliverables/PrismServe_Graph_and_Diagram_Atlas.pdf`: one high-resolution registered figure per page, with caption and provenance.
- `deliverables/PrismServe_Detailed_Report_Package.zip`: all three outputs, editable report Markdown, generation code, PNG/SVG assets, CSV tables and manifest.

`report_manifest.json` records figure/table/equation registries, original input hashes, generated artifact hashes and the unchanged original-release check. `companion_validation.json` records PDF/HTML validation and original source/release integrity. Figures use originals copied from the result folders or new plots derived from raw archived records. Schematics are original code-native diagrams; they are not physical measurements. Diagnostic plots are labeled and placed separately from accepted benchmark comparisons.

## Regenerate in the original project tree

```powershell
python reports/detailed/build_report.py
python reports/detailed/validate_report.py
```

Dependencies already used by the project: PyMuPDF, Markdown, Matplotlib and Pillow. The Markdown report is the editable source. The extracted report bundle contains the generator and CSV exports but not every underlying raw project record; full regeneration needs the original PrismServe project and its result folders. Restore these companion sources into `reports/detailed/` relative to that project root.

Temporary pagination checks and page previews are excluded from the report package. The original `PrismServe_Project.zip` remains unchanged; use the separate detailed report package to share this expanded documentation.

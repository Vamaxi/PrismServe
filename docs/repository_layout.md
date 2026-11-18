# Public repository contents and provenance

The GitHub repository is the publication view of PrismServe. It preserves the working runtime, tests, configurations, accepted experiment records, labeled diagnostics, figures and readable/editable reports. The original local submission and its ZIP remain unchanged in the author's workspace.

## Included files

| Path | Reason for inclusion |
|---|---|
| `prismserve/`, `pyproject.toml` | Complete installable implementation; runtime modules match the final experiment's source hashes |
| `tests/`, `configs/`, `scripts/` | Numerical/resource correctness, experiment protocols, live demos, verification and document generation |
| `docs/`, `references.bib` | Design, mechanisms, methods, results, presentation notes and viva guide |
| `data/wikitext2/` | Small reproducible validation/test prefixes, response payloads, attribution and hashes |
| `models/gpt2/prismserve_source.json` | Pinned checkpoint revision, asset sizes/hashes and model attribution |
| `results/` | Actual/analytical observations, all outcomes, separate diagnostics and exact source snapshots |
| `results/demo/tiny_checkpoint.pt` | Small independently trained decoder artifact supporting earlier functional evidence |
| `reports/detailed/` | Detailed report generator, Markdown, PNG/SVG schematics and full CSV result tables |
| `deliverables/` | Detailed/concise reports, HTML version, graph atlas, editable slides and release metadata |
| `artifacts/source_text/manifest.json` | Original input-document page/hash provenance without redistributing those PDFs or full extracts |

## Excluded files

- Pretrained GPT-2 weights and tokenizer assets: prepare the pinned public model explicitly with `pretrained-prepare --download`.
- The four original supplied PDF inputs and their full text extracts: paper links and citation metadata are retained; these files are not needed for inference or saved-result inspection.
- Python/virtual-environment caches, build directories and editable-install metadata.
- Pagination-debug PDFs, page-preview images, transient server logs and repeated personal demo sessions.
- Deliverable ZIP archives and the generated package manifest, which duplicate the repository's sources/documents. Experiment `source_snapshot.zip` files and historical wheels remain because they identify the implementation that produced earlier measurements.

No experiment is relabeled to hide a failure. `results/validation_study/` is a mixed-power diagnostic; `results/validation_study_ac/` is an incomplete diagnostic; `results/validation_study_final/` is the final accepted AC-conditioned comparison. Their STATUS files explain the distinction. The first two capacity pilots are also retained. Preview images and logs do not supply missing successful measurements.

## Verify the published project

From the repository root:

```powershell
python -m unittest discover -s tests -v
python scripts/check_capacity_frontier.py --check-current
python scripts/check_validation_study.py
python scripts/check_controller_results.py
python scripts/check_live_results.py
python scripts/check_repository.py
```

The unit-test command leaves the saved test record unchanged. The experiment checkers verify archived/current runtime source, matching policy inputs, memory quotas and completion/rejection/quality accounting as applicable. The repository checker verifies the selected publication files and frozen runtime/data hashes without requiring excluded model assets, supplied PDFs or ZIP archives.

`repository_manifest.json` describes the publication selection and hashes every included project file except itself and Git metadata. After editing project files, regenerate it deliberately with `python scripts/check_repository.py --write-manifest`; the new state is distinct from the historical measurements. Git itself records commits and history.

## Original freeze versus public README

`deliverables/release.json` describes the exact local submission v1.0.0. Its README and `.gitignore` hashes refer to the originals copied into `docs/archive/submission_v1/`. GitHub has an expanded publication README and housekeeping rules. Runtime source, recorded data and experiment snapshots are not changed to accommodate that documentation difference. Public checker `check_repository.py` resolves those two originals through the archive before checking the frozen source list.

`check_submission.py --package`, `reports/detailed/validate_report.py`, and a full detailed-report rebuild have stricter **original-workspace** requirements: excluded submission ZIPs, original PDF inputs and/or the exact original README are part of their full hash scopes. They are retained as source tools, but their original-workspace assertions should not be interpreted as a fresh-clone prerequisite. Inspect the included reports directly or use the repository/experiment checks above. To recreate that original workflow, supply the omitted original inputs and use the archived submission README/ignore files in a separate checkout.

Reproduction runs must write to new result folders, keeping the published observations unchanged. Any actual CPU/CUDA performance comparison on another machine needs fresh device-specific calibration. A new machine's result is new evidence, not an update to the historical laptop measurements.

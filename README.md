# Search for dark matter produced in association with a single top quark or a top quark pair in proton-proton collisions at √s = 13 TeV

Documentation site (MkDocs Material) for the reproducible CMS Open Data re-implementation of
CMS-EXO-18-010. Published at <https://cms-epn.github.io/search-for-dark-matter-docs>; the analysis
code lives in [CMS-EPN/search-for-dark-matter](https://github.com/CMS-EPN/search-for-dark-matter).

```bash
uv run mkdocs serve            # preview on http://127.0.0.1:8000
uv run mkdocs build --strict   # what CI must pass
```

Pushing to `main` deploys to GitHub Pages (`.github/workflows/ci.yml`).

## Layout

| Path | Content |
|---|---|
| `docs/getting-started/` | How to run, troubleshooting, repository and pipeline |
| `docs/results/` | Corrected plots per channel and the paper comparison. Images in `results/img/` are extracted from `results/*_run.ipynb` of the code repository |
| `docs/changes/` | Web version of the code repository's `docs/RUNBOOK.md` (COMP / PHYS / PLOT, convergence, open questions) |
| `docs/Single_Lepton/`, `docs/All_Hadronic/` | The original upstream analysis, unchanged |

> `docs/Single_Lepton/Analisis_muon_final_version.md` is the **source** from which the code
> repository's `src/build_single_lepton.py` generates `notebooks/Single_Lepton.ipynb`. Do not
> rename or restructure it. `src/make_report.py` also reads the upstream PNGs in both folders.

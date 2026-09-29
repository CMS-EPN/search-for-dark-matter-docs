# Repository and pipeline

## Layout

```text
search-for-dark-matter/
├── notebooks/                 ← run these
│   ├── Single_Lepton.ipynb        single-lepton analysis (muon + electron) — generated
│   ├── All_Hadronic.ipynb         all-hadronic (0-lepton) analysis — generated
│   └── upstream/                  original DPOA notebooks, reference only
├── results/                   ← already-run, CERN-approved output
│   ├── comparacion.pdf            paper vs. before vs. corrected, every channel and region
│   ├── Single_Lepton_run.ipynb    notebook exactly as executed, plots included
│   └── All_Hadronic_run.ipynb
├── src/                       the library the notebooks import
│   ├── dpoa_io.py                 downloads, XRootD retries, caching — no physics
│   ├── dpoa_datasets.py           physics inputs: records, cross sections, luminosity, provenance
│   ├── dpoa_workshop.py           upstream dataset registry, left untouched
│   ├── build_single_lepton.py     regenerates notebooks/Single_Lepton.ipynb
│   ├── build_all_hadronic.py      regenerates notebooks/All_Hadronic.ipynb
│   ├── region_plot_cell.py        the signal-region plot, shared by both builders
│   └── make_report.py             builds results/comparacion.pdf
├── docs/
│   ├── RUNBOOK.md                 every change vs. upstream (mirrored on this site)
│   └── paper.pdf                  CMS-EXO-18-010
├── run_analysis.sh            headless run that resumes after network drops
├── pixi.toml / pixi.lock      the reproducible environment
└── output_raw/                per-dataset parquet cache (created on first run, git-ignored)
```

## How a run flows

```text
dpoa_workshop registry ──► dpoa_datasets.build_registry()   fixes Zvv, adds 7 missing samples
                                     │
                                     ▼
                        dpoa_io.build_ntuples()            file index per dataset → ntuples.json
                                     │
                                     ▼
       process_file_* (notebook) ─► dpoa_io.process_dataset()   XRootD stream, 4 threads, retries,
                                     │                          per-file failure report
                                     ▼
                        output_raw/<dataset><suffix>.parquet  + .manifest.json (N_FILES per result)
                                     │
                                     ▼
       dpoa_io.sum_gen_weights()  +  dpoa_datasets.luminosity_pb()
                                     │   w = σ · L / Σ genWeight   (same file slice as the events)
                                     ▼
                        region_plot_cell.plot_region_stack()  paper order, binning, overflow
```

Output suffixes keep the channels apart: `_raw` (SL muon), `_electron_raw` (SL electron),
`_0lep_raw` (AH). Before this was fixed both channels wrote the same filename and silently
reused each other's events ([COMP-05](../changes/computational.md#comp-05-per-channel-output-files)).

## Where the physics lives

Every selection cut, b-tagging working point and kinematic definition is **in the notebook**,
in plain sight. `src/dpoa_io.py` deliberately contains none of it — only networking and
bookkeeping. Physics *inputs* that need provenance (Open Data record IDs, cross sections with
their XSDB source, the luminosity derivation) live in `src/dpoa_datasets.py`, each value
commented with where it came from.

## Notebooks are generated

Neither runnable notebook is edited by hand:

| Notebook | Source | Builder |
|---|---|---|
| `Single_Lepton.ipynb` | `docs/Single_Lepton/Analisis_muon_final_version.md` in **this documentation repository** — the single-lepton analysis never existed upstream as a notebook | `src/build_single_lepton.py` |
| `All_Hadronic.ipynb` | `notebooks/upstream/All_Hadronic_20files.ipynb` | `src/build_all_hadronic.py` |

Each builder applies a list of named patches, each with a comment saying what was wrong, and
exits non-zero if a patch does not find its target cell. The physics cells that are not patched
pass through byte-identical, which is what makes the upstream physics auditable.

!!! warning "The single-lepton source lives here"
    `build_single_lepton.py` reads the markdown export under `docs/Single_Lepton/` of this
    documentation repository, so it must be cloned **next to** the code repository, and that
    file must not be renamed or restructured.

```bash
pixi run build        # both notebooks
pixi run build-sl     # single lepton only
pixi run build-ah     # all hadronic only
```

## Rebuilding the comparison report

`src/make_report.py` extracts every plot from the executed notebooks, places it next to the
paper's figure and the original ("before") plot, and writes `results/comparacion.pdf`. It
expects the paper crops in `_report_scratch/paperfigs/` (git-ignored).

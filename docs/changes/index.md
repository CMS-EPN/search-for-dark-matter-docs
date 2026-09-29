# Changes to the upstream analysis

This section is the web version of `docs/RUNBOOK.md` in the code repository: **every deviation
from the DPOA collaboration's original notebooks**, recorded so a physicist can review the
physics without reading diffs, and so anything can be reverted. Each entry states what changed,
why, the source that justifies it, how to verify it, and — where it was measured — its effect.

## Three classes of change

| Class | Meaning | Who should review |
|---|---|---|
| [**COMP**](computational.md) | Computational. Changes *how* the code runs; selected events and numbers are identical. | Computing |
| [**PHYS**](physics.md) | Physics. Changes *which* events are selected or *how* they are weighted; the numbers change. | **Physics — needs sign-off** |
| [**PLOT**](plots.md) | Presentation. Makes our histograms comparable to the paper's; no event moves. | Anyone comparing plots |

"Applied" means the code does it, not that a physicist has signed it off. The upstream behaviour
stays reproducible: the original notebooks are kept untouched in `notebooks/upstream/`.

## Status

| Phase | What | State |
|---|---|---|
| A | Reproduce the upstream plots faithfully, verify the pipeline end to end | ✅ 2026-09-03 |
| B | Apply the PHYS changes and converge on the paper | ✅ PHYS-01…14 applied |
| C | Make our histograms comparable to the paper's | ✅ PLOT-01…05 applied |
| D | Single-lepton channel with the same corrections | ✅ 2026-09-07 |
| — | Repository reorganised (`notebooks/`, `src/`, `docs/`, `results/`) | ✅ 2026-09-23 |

**Phase A evidence.** The upstream output was reproduced exactly before any physics change:
the stored upstream output reads `SingleMuon_raw.parquet with 25932 events` and our run reported
`SingleMuon: 20/20 files ok, 25,932 events`; the $m_T^W$, $N_b$ (muon) and SR-2b (AH) plots
matched the reference PNGs bin for bin. 648/648 ROOT files read, 0 network failures, 1 h 50 min.

## Change index

| ID | Class | Title | Applied |
|---|---|---|---|
| [COMP-01](computational.md#comp-01-pinned-reproducible-environment) | COMP | Pinned reproducible environment | ✅ |
| [COMP-02](computational.md#comp-02-failures-are-reported-instead-of-swallowed) | COMP | Failures reported, not swallowed | ✅ |
| [COMP-03](computational.md#comp-03-one-n_files-constant) | COMP | One `N_FILES` constant | ✅ |
| [COMP-04](computational.md#comp-04-one-luminosity-definition) | COMP | One luminosity definition | ✅ |
| [COMP-05](computational.md#comp-05-per-channel-output-files) | COMP | Per-channel output files | ✅ |
| [COMP-06](computational.md#comp-06-cache-invalidated-when-n_files-changes) | COMP | Cache invalidated on `N_FILES` change | ✅ |
| [COMP-07](computational.md#comp-07-robust-file-index-fetching) | COMP | Robust file-index fetching | ✅ |
| [COMP-08](computational.md#comp-08-xrootd-retries) | COMP | XRootD retries | ✅ |
| [COMP-09](computational.md#comp-09-parallel-file-processing) | COMP | Parallel file processing | ✅ |
| [COMP-10](computational.md#comp-10-wget-replaced-with-python) | COMP | `!wget` replaced with Python | ✅ |
| [COMP-11](computational.md#comp-11-missing-reference-figures-do-not-abort-a-run) | COMP | Missing figures do not abort | ✅ |
| [COMP-12](computational.md#comp-12-notebooks-are-generated-not-hand-edited) | COMP | Notebooks generated from build scripts | ✅ |
| [COMP-13](computational.md#comp-13-dead-cells-removed) | COMP | Dead cells removed | ✅ |
| [COMP-14](computational.md#comp-14-plot-functions-take-their-data-explicitly) | COMP | Plot functions take data explicitly | ✅ |
| [COMP-15](computational.md#comp-15-runs-resume-after-a-network-drop) | COMP | Runs resume after a network drop | ✅ |
| [COMP-16](computational.md#comp-16-paths-resolved-from-the-repository-root) | COMP | Paths resolved from the repository root | ✅ |
| [PHYS-01](physics.md#phys-01-zvv-points-at-an-unrelated-signal-sample) | PHYS | `Zvv` → real Z→νν sample | ✅ |
| [PHYS-02](physics.md#phys-02-luminosity-derived-per-channel) | PHYS | Per-channel luminosity | ✅ |
| [PHYS-03](physics.md#phys-03-baseline-selection-missing-cuts) | PHYS | Missing baseline cuts (Table 10) | ✅ |
| [PHYS-04](physics.md#phys-04-mindphi-over-two-jets-not-four) | PHYS | min∆φ over 2 jets, not 4 | ✅ |
| [PHYS-05](physics.md#phys-05-signal-region-cuts-computed-but-never-applied) | PHYS | Signal-region cuts (Table 13) | ✅ except $M_{T2}^W$ |
| [PHYS-06](physics.md#phys-06-missing-background-samples) | PHYS | Missing background samples | ✅ |
| [PHYS-07](physics.md#phys-07-met-filters-and-golden-json) | PHYS | MET filters, golden JSON | ✅ |
| [PHYS-08](physics.md#phys-08-b-tagging-working-point) | PHYS | b-tagging working point | ✅ superseded by PHYS-10 |
| [PHYS-09](physics.md#phys-09-the-all-hadronic-control-region-is-not-in-the-paper) | PHYS | AH control region not in the paper | ✅ identified |
| [PHYS-10](physics.md#phys-10-b-tagging-switched-to-csvv2-medium) | PHYS | CSVv2 medium b-tagging | ✅ |
| [PHYS-11](physics.md#phys-11-wz-added-to-the-diboson-stack) | PHYS | WZ added | ✅ |
| [PHYS-12](physics.md#phys-12-zvv-low-pt-bin-added) | PHYS | Z(νν) low-$p_T$ bin added | ✅ |
| [PHYS-13](physics.md#phys-13-zvv-cross-section-use-the-unfiltered-value) | PHYS | Z(νν) cross section 4.201 pb | ✅ |
| [PHYS-14](physics.md#phys-14-top-pt-reweighting) | PHYS | Top $p_T$ reweighting | ✅ |
| [PHYS-15](physics.md#phys-15-single-top-tw-s-channel-and-ttv-samples-missing) | PHYS | tW / s-channel / tt̄V samples | ⬜ needs record IDs |
| [PLOT-01](plots.md#plot-01-stacking-order) | PLOT | Paper stacking order | ✅ |
| [PLOT-02](plots.md#plot-02-binning) | PLOT | Paper binning | ✅ |
| [PLOT-03](plots.md#plot-03-the-last-bin-is-an-overflow-bin) | PLOT | Last bin holds the overflow | ✅ |
| [PLOT-04](plots.md#plot-04-axis-ranges) | PLOT | Fixed relative axis ranges | ✅ |
| [PLOT-05](plots.md#plot-05-legend-order) | PLOT | Legend order matches the stack | ✅ |

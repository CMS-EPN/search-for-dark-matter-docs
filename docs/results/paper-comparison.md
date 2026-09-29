# Paper vs. before vs. corrected

This is the comparison CERN/DPOA reviewed and approved. Each row puts three plots side by side:

| Column | What it is |
|---|---|
| **PAPER** | The corresponding panel of the paper's Figure 4 (single lepton) or Figure 5 (all hadronic), 35.9 fb⁻¹, post-fit |
| **ANTES** (before) | The same plot from the original DPOA notebooks, as published in the earlier version of this site |
| **CORREGIDA** (corrected) | The same plot after every change in [Changes](../changes/index.md), 2.4–4.9 fb⁻¹, pre-fit |

A dash (—) means that plot does not exist in that version: the paper shows only the signal
regions, and the original notebooks never produced some of the regions.

[:material-file-pdf-box: Download the full comparison (PDF, 1.8 MB)](files/comparison.pdf){ .md-button }

## What was corrected

![Summary of corrections: before vs corrected](img/comparison/1-summary.png)

## All hadronic

### Signal regions

![All-hadronic signal regions: paper, before, corrected](img/comparison/2-ah_signal_regions.png)

### Baseline

![All-hadronic baseline distributions: before and corrected](img/comparison/3-ah_baseline.png)

## Single lepton

### Muon — signal regions

![Single-lepton muon signal regions: paper, before, corrected](img/comparison/4-sl_muon_signal_regions.png)

### Electron — signal regions

![Single-lepton electron signal regions: paper and corrected](img/comparison/5-sl_electron_signal_regions.png)

### Muon — baseline

![Single-lepton muon baseline distributions: before and corrected](img/comparison/6-sl_muon_baseline.png)

### Electron — baseline

![Single-lepton electron baseline distributions: before and corrected](img/comparison/7-sl_electron_baseline.png)

## What changed visually, and why

| Visible difference in "before" | Cause | Fix |
|---|---|---|
| Spiky, near-empty blue Z/γ* band in AH | `Zvv` pointed at a long-lived-particle **signal** sample | [PHYS-01](../changes/physics.md#phys-01-zvv-points-at-an-unrelated-signal-sample), [PHYS-13](../changes/physics.md#phys-13-zvv-cross-section-use-the-unfiltered-value) |
| Data ~10× above MC in AH | missing baseline cuts, wrong luminosity, missing Z(νν) | [PHYS-02](../changes/physics.md#phys-02-luminosity-derived-per-channel), [PHYS-03](../changes/physics.md#phys-03-baseline-selection-missing-cuts) |
| tt̄ at the bottom of the SL stack | processes sorted by yield | [PLOT-01](../changes/plots.md#plot-01-stacking-order) |
| No bin edge shared with the paper | different binning | [PLOT-02](../changes/plots.md#plot-02-binning) |
| Rightmost bin falls away instead of rising | overflow was dropped | [PLOT-03](../changes/plots.md#plot-03-the-last-bin-is-an-overflow-bin) |
| No ratio panel | never drawn | added; it shows the ratio is **flat** |

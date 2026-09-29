# Search for dark matter with top quarks — CMS Open Data

A reproducible re-implementation, from **CMS Open Data (2016, Run2016H)**, of the CMS analysis
*"Search for dark matter produced in association with a single top quark or a top quark pair
in proton–proton collisions at $\sqrt{s} = 13 \TeV$"* (CMS-EXO-18-010).

Part of the **CERN DPOA** effort, carried out by the EPN CMS group, to make the analysis runnable
by anyone with a Linux machine and a network connection.

!!! success "Current state — 2026-09"
    Both channels run end to end on 20 files per dataset with **0 errors**
    (all-hadronic 28/28 cells, single-lepton 35/35 cells). The corrected plots were reviewed and
    approved by CERN/DPOA. Every change to the original notebooks is documented, justified and
    reversible.

## At a glance

| | Upstream notebooks | Now | Paper / analysis note |
|---|---:|---:|---:|
| **All-hadronic** data/MC | ~10 | **1.34** | 1.06 (note Table 12) |
| All-hadronic Z(νν) share of background | ~0 % (wrong sample) | **24.1 %** | 29.7 % |
| All-hadronic tt̄ share of background | 73 % | **59.1 %** | 45.5 % |
| **Single-lepton** data/MC, muon | — | **1.075** | 1.118 (note Table 11) |
| Single-lepton data/MC, electron | — | **0.942** | 1.118 |

The residual disagreement is not a mystery: it decomposes into a small number of **named
inputs** that are not published in the analysis note or not yet located in Open Data (V+jets NLO
k-factors, the VH and tW samples, and the $M_{T2}^W$ variable). See
[Open questions](changes/open-questions.md).

## Where to go

<div class="grid cards" markdown>

- **I want to see the results**

    The corrected plots for both channels, and the side-by-side comparison with the paper.

    [:octicons-arrow-right-24: Results](results/index.md)

- **I want to run it myself**

    One tool (pixi), one command (`pixi run lab`).

    [:octicons-arrow-right-24: Getting started](getting-started/index.md)

- **I want to review what was changed**

    Every computational, physics and plotting change, with source and measured effect.

    [:octicons-arrow-right-24: Changes (runbook)](changes/index.md)

- **I want the original notebooks**

    The DPOA collaboration's analysis as originally written, kept for reference.

    [:octicons-arrow-right-24: Original analysis](original/index.md)

</div>

## The physics in one paragraph

In simplified dark-matter models a new mediator ($\phi$ or $a$) couples to Standard Model fermions
proportionally to their mass, so it is produced most readily together with top quarks
($gg \to t\bar{t}\phi$, $gb \to tW\phi$, $qq' \to tj\phi$) and decays invisibly to a pair of
dark-matter particles $\chi\bar{\chi}$. The detector signature is therefore top-quark decay
products plus large missing transverse momentum $p_T^{\text{miss}}$. The analysis splits events
by the number of isolated leptons into the **single-lepton (SL)** channel — one muon or electron —
and the **all-hadronic (AH)** channel — no leptons — and in each one compares the observed
$p_T^{\text{miss}}$ spectrum in signal regions to the Standard Model background prediction.

## Sources

| Source | What it is |
|---|---|
| *paper* | CMS-EXO-18-010, the published CMS paper (in the code repository as `docs/paper.pdf`) |
| *note* | The CMS analysis note; tables cited by number. Collaboration-internal — ask a DPOA mentor for a copy |
| *record N* | CMS Open Data record, `https://opendata.cern.ch/record/N` |
| Code | [github.com/CMS-EPN/search-for-dark-matter](https://github.com/CMS-EPN/search-for-dark-matter) |

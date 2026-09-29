# Physics changes (PHYS)

These change **which events are selected or how they are weighted**. All are applied in the
current notebooks; each still needs physics sign-off. Measured effects are in the
[convergence log](convergence.md).

!!! abstract "The benchmark"
    Note **Table 12** is the all-hadronic cut-flow with absolute yields at 35.9 fb⁻¹. After the
    AH baseline it gives, as a share of total background: tt̄(1ℓ) 41.7 %, **Z(νν) 29.5 %**,
    W+jets 15.1 %, single top 5.1 %, tt̄(2ℓ) 3.8 %, VV 2.7 %, QCD 1.1 %, tt̄V 0.9 %,
    Z(ℓℓ) 0.2 % — and **data/MC = 1.06**. Note **Table 11** plays the same role for the single
    lepton channel. When this list was started, the upstream data/MC was ≈ 7–10.

## PHYS-01 · Zvv points at an unrelated signal sample

**Impact: largest.** Z(νν) is ~30 % of the all-hadronic background.

**Was:** `dpoa_workshop.py` mapped `Zvv` to `ggH_HToSSTo4l_lowctau_MH-500_MS-150_ctauS-100` — a
long-lived-particle **signal** sample (932 events in its first file) — with the Z→νν cross
section 77.3 pb (recognisable as note Table 6's `ZJetsToNuNu HT-200To400` = 77.67 pb, so the
intent was clearly Z→νν). Symptom: the spiky, near-empty blue Z/γ* band in the old AH plots.

**Now:** `dpoa_datasets.build_registry()` replaces it with **record 74910**,
`ZJetsToNuNu_Zpt-200toInf_BPSFilter_TuneCP5` UL16 NanoAODv9 (7 files, 453 967 events). The
original `dpoa_workshop.py` is left untouched. Cross section: [PHYS-13](#phys-13-zvv-cross-section-use-the-unfiltered-value).

## PHYS-02 · Luminosity derived per channel

**Was:** `LUM = 3400 pb⁻¹` for every channel — chosen to make the plots agree, not derived. The
original comment reads `(20/82)*8900`, but 82 is the *SingleMuon* file count, applied to the
*MET* channel, which has 32 files.

**Now:** Open Data exposes only Run2016H = 8.9 fb⁻¹ (note Table 2).
`dpoa_datasets.luminosity_pb()` measures the processed fraction **by bytes** of the record —
exact metadata, and a better proxy than file count since file sizes vary by more than 2×:

| Dataset | Record | Files | Fraction | $L$ [pb⁻¹] | vs 3400 |
|---|---:|---:|---:|---:|---|
| `met` (AH) | 30559 | 20/32 | 0.5514 | **4907** | MC was 31 % too low |
| `SingleMuon` | 30563 | 20/82 | 0.2663 | **2370** | MC was 43 % too high |
| `SingleElectron` | 30562 | 20/80 | 0.3254 | **2896** | MC was 17 % too high |

**Assumption to review:** bytes ∝ recorded luminosity. A physicist may prefer summing the
golden-JSON lumi sections actually present.

## PHYS-03 · Baseline selection: missing cuts

| Baseline cut | Note Table 10 | Upstream |
|---|---|---|
| $n_b$ | **≥ 1** | not applied |
| $p_T^{\text{miss}}$ (AH) | **≥ 250 GeV** | 240 |
| $p_T^{\text{miss}}$ (SL) | **≥ 160 GeV** | 150 |
| $\min\Delta\phi(j_{1,2}, p_T^{\text{miss}})$ (AH) | **> 0.4** | not applied |

$n_b \ge 1$ and $\min\Delta\phi > 0.4$ are the cuts that remove QCD multijet and fake
$p_T^{\text{miss}}$. Both inflate **data** far more than MC, so omitting them was a large part of
the upstream data/MC ≈ 7–10.

## PHYS-04 · min∆φ over two jets, not four { #phys-04-mindphi-over-two-jets-not-four }

**Was:** the AH processor minimised ∆φ over the **top four** central jets
(`jets_top4 = f_central[:, :4]`). **Paper:** $\min\Delta\phi(j_{1,2}, p_T^{\text{miss}})$ — the two
leading jets (paper Table 1, note Table 13). A minimum over more jets is always ≤ the minimum
over fewer, so the same numerical cut was systematically tighter. Required reprocessing.

## PHYS-05 · Signal-region cuts computed but never applied

| Cut (note Table 13) | AH | SL | Upstream |
|---|---|---|---|
| $\min\Delta\phi(j_{1,2})$ | ≥ 1.0 | ≥ 1.2 | AH ok; SL used **0.5** |
| $m_T^b$ | ≥ 180 GeV | ≥ 180 GeV | **computed, never applied** |
| $p_T(j_1)/H_T$ | ≤ 0.5, $n_b \ge 2$ only | — | **never computed** |
| $m_T$ | — | ≥ 160 GeV | not applied |
| $M_{T2}^W$ | — | ≥ 200 GeV | **not implemented** |

All applied now, **except $M_{T2}^W$**: it needs a kinematic minimisation over jet–lepton
assignments and is the largest single cut in the SL selection —
[OPEN-3](open-questions.md#open-3-mt2w-the-largest-cut-in-the-sl-selection).

## PHYS-06 · Missing background samples

| Sample added | Record | Why |
|---|---|---|
| W+jets HT-600to800, 800to1200, 1200to2500, 2500toInf | 69731, 69735, 69723, 69727 | the high-HT tail is what survives $p_T^{\text{miss}} > 250$ GeV |
| `TTTo2L2Nu` | 67801 | 3.8 % of the AH background, dominant in SL |

**Deliberately not added — `TTToHadronic` (record 67841).** It is absent from note Table 6, and
Table 12 has no fully-hadronic tt̄ row. Without genuine $p_T^{\text{miss}}$ it behaves like QCD
multijet, which $\min\Delta\phi > 0.4$ removes. *(An earlier draft wrongly called it a major missing
background; recorded so the error is not repeated.)*

**QCD multijet not added:** 1.1 % of the background, and Open Data ships `QCD_Pt_*` rather than
the `QCD_HT_*` binning the note used.

## PHYS-07 · MET filters and golden JSON

Note §2.1 requires, on data and simulation, `goodVertices`, `HBHENoiseFilter`,
`HBHENoiseIsoFilter`, `EcalDeadCellTriggerPrimitiveFilter`, `globalTightHalo2016Filter`,
`BadPFMuonFilter`, `BadChargedCandidateFilter`; on data only `eeBadScFilter`; plus the
golden-JSON good-lumi list. Upstream applied none (`build_lumi_mask` was imported and never
called). **All are applied now.**

Measured effect, small: the filters remove 0.05–1.8 % of events, the MET triggers 0.64 % at
$p_T^{\text{miss}} > 250$ GeV. Neither explains the data excess — see
[hypotheses rejected](convergence.md#hypotheses-tested-and-rejected).

## PHYS-08 · b-tagging working point

**Was:** `Jet_btagDeepFlavB > 0.2770`, the DeepJet medium working point for **2017**. For UL2016
postVFP (Run2016H) it is **0.2489**. Superseded by PHYS-10, since the paper does not use DeepJet
at all.

## PHYS-09 · The all-hadronic control region is not in the paper

Upstream `filter_cr_wlnu` selects 0 leptons, $n_b = 0$, $n_{\text{jet}} \ge 3$, labelled `AH0lWR`.
Note Table 14 defines the AH control regions as `AH1eTR`, `AH1mTR`, `AH1eWR`, `AH1mWR`,
`AH2eZR`, `AH2mZR` — **all requiring one or two leptons**. They cannot be built from the MET
dataset at all; they need SingleElectron/SingleMuon, i.e. a cross-channel restructuring.
No AH control-region plot is produced.

## PHYS-10 · b-tagging switched to CSVv2 medium

`Jet_btagCSVV2 > 0.8484` replaces DeepJet: note Table 10 specifies **CSVM**, which is what the
paper uses. Both counts are stored in the parquet (`nBTag_csv`, `nBTag_deepjet`), so the choice
can be revisited without reprocessing.

**A prediction that was wrong, recorded on purpose.** The expectation was that CSVv2's higher
mistag rate would raise Z(νν) and W+jets (which enter $n_b \ge 1$ only through mistags). Measured
on the same events:

| Sample | DeepJet | CSVv2 | Ratio |
|---|---:|---:|---:|
| tt̄ semileptonic | 55 938 | 48 172 | 0.86 |
| tt̄ dileptonic | 20 830 | 17 796 | 0.85 |
| Z(νν) | 35 379 | 28 827 | **0.81** |

Every process goes *down*. The change is kept because it is what the paper does, but it does
**not** explain the Z(νν) deficit.

## PHYS-11 · WZ added to the diboson stack

`WZ_TuneCP5` inclusive, **record 72754**, σ = 47.13 pb. Note Table 6 lists WW, WZ and ZZ;
upstream had WW and ZZ only. Diboson share 0.9 % → 1.1 % (Table 12: 2.7 %).

## PHYS-12 · Zvv low-pT bin added

`ZJetsToNuNu_Zpt-100to200_BPSFilter`, **record 74908**. Reconstructed $p_T^{\text{miss}}$ is not
generator $p_T(Z)$ — jet mismeasurement and recoil migrate events upward — so the 100–200 GeV bin
is not empty above 250 GeV.

## PHYS-13 · Zvv cross section: use the unfiltered value

**The single largest correction of the whole exercise.** XSDB returns two official values per
$p_T(Z)$ bin, differing by 5.7×:

| XSDB `process_name` | σ [pb] | Campaign |
|---|---:|---|
| `ZJetsToNuNu_Zpt-200toInf_BPSFilter_TuneCP5` | 0.7333 | UL17 |
| `ZJetsToNuNu_Zpt-200toInf_TuneCUETP8M1` | **4.201** | Summer16 |
| `ZJetsToNuNu_Zpt-100to200_BPSFilter_TuneCP5` | 5.002 | UL17 |
| `ZJetsToNuNu_Zpt-100to200_TuneCUETP8M1` | **35.99** | Summer16 |

The Open Data sample *is* the BPSFilter one, so 0.7333 pb looks obvious — and gives Z(νν) at
4.6 % against the required 29.7 %. **A direct measurement settles it**, reading the `Runs` tree:

| Sample | Events | $\sum w_{\text{gen}}$ | Mean genWeight |
|---|---:|---:|---:|
| `ZJetsToNuNu_Zpt-200toInf_BPSFilter` | 453 967 | 23 817 | **0.0525** |
| `WJetsToLNu_HT-400To600` (unfiltered) | 652 621 | 652 621 | **1.0000** |

An unfiltered LO madgraphMLM sample has mean genWeight exactly 1; this one has 0.0525
(1/0.0525 = 19). **The filter acceptance is already folded into the per-event weights**, so with
$w = \sigma \cdot L \cdot w_{\text{gen}} / \sum w_{\text{gen}}$ the σ to pair is the *unfiltered* one.

Cross-check against Table 12 (8716 Z(νν) events at 35.9 fb⁻¹ → 1191 at 4.907 fb⁻¹):

| σ used | Z(νν) events | vs expected |
|---|---:|---:|
| 0.7333 pb | 262 | 0.22× |
| **4.201 pb** | **1501** | **1.26×** |

**Known residual:** 4.201 overshoots by 26 %. Three uncorrected effects all push that way — tune
(CUETP8M1 vs CP5), $p_T(Z)$- vs HT-binned phase space, and the missing NLO/LO k-factors. Treat the
Z→νν normalisation as good to tens of percent.

**Also from XSDB — Drell–Yan.** `DYJetsToLL_M-50_Zpt-200toInf_BPSFilter_TuneCP5` is listed at
**0.393 pb for UL16** (exact campaign) and 0.3919 pb for UL17. Upstream used **1.27 pb**, 3.2× too
high, in both channels. The 0.3 % UL16/UL17 difference also confirms cross sections here do not
depend on the campaign.

## PHYS-14 · Top pT reweighting

The one MC correction the note specifies completely (section *Corrections for MC samples*): the
generated top $p_T$ spectrum is harder than observed, and each tt̄ event is reweighted by

$$
\rho(p_T) = e^{\,0.0615 - 0.0005\, p_T}, \qquad
w_{\text{top}} = \sqrt{\rho(p_T^{t_1})\,\rho(p_T^{t_2})}
$$

with $p_T$ at matrix-element level. **Implementation:** the two generated tops are the last
copies in `GenPart` (`|pdgId| == 6`, statusFlags bit 13); events without exactly two get weight 1.
`GenPart_*` branches are read only for `ttbar*` datasets. The weight is stored as a
`topPtWeight` column; older parquet files without it are read as "no correction".

Measured on `TTToSemiLeptonic` (1 233 000 events):

| Quantity | Value |
|---|---:|
| ⟨w⟩, all events | **1.0005** |
| ⟨w⟩, $p_T^{\text{miss}} > 250$ GeV | **0.890** |
| Range of w | 0.440 – 1.063 |
| Events with exactly two generated tops | 100 % |
| ⟨w⟩ on WW (control, no tops) | 1.0000 |

Normalisation-preserving but **not** shape-preserving — which is the point: it removes ~11 % of
tt̄ specifically in the boosted signal region. Using hard-process copies (bit 7) instead gives the
same weight to four decimals.

!!! bug "One implementation bug, caught loudly"
    The first version filtered the weight as a plain numpy array and applied only the second of
    the processor's two masks: `IndexError: boolean index did not match indexed array along axis
    0; size of axis is 1233000 but size of corresponding boolean axis is 5150`. `top_w` is now an
    `ak.Array` masked with exactly the same syntax, on the same lines, as `genWeight`.
    COMP-02 made it a hard error instead of silently wrong weights.

Measured effect in the full AH run: tt̄ × **0.854**, share 62.8 % → **59.1 %** — within 3 % of the
0.83 the gap analysis required. See the [convergence log](convergence.md#phys-14-measured).

## PHYS-15 · Single-top tW, s-channel and ttV samples missing

**Status: identified, not applied — needs record IDs.** Note Table 6 lists:

| Sample | σ [pb] | Present |
|---|---:|---|
| ST t-channel top | 136.02 | ✅ |
| ST t-channel antitop | 80.95 | ✅ |
| **ST tW top** | **35.85** | ❌ |
| **ST tW antitop** | **35.85** | ❌ |
| ST s-channel | 3.36 | ❌ |
| TTWJetsToLNu | 0.2043 | ✅ (as `ttW`) |
| TTWJetsToQQ | 0.4062 | ❌ |
| TTZToQQ | 0.5297 | ❌ |
| TTZToLLNuNu | 0.2529 | ❌ |

**tW is the one that matters.** The SL signal region requires $m_T > 160$ GeV; t-channel single
top has one W and is removed at the Jacobian edge, exactly like tt̄(1ℓ). tW has a second W, two
neutrinos and real $p_T^{\text{miss}}$, and survives. That accounts for the SL t+X factor of 28
(0.5 % vs 14.1 %), and is consistent with AH, where t+X is low by only 2.7× because t-channel
still contributes without a lepton requirement.

A portal free-text search returned nothing for these names — but it also returns nothing for
`TTToSemiLeptonic`, which this analysis reads every run, so the index cannot be trusted for MC
names. [OPEN-6](open-questions.md#open-6-record-ids-for-tw-s-channel-and-ttv).

# Convergence log

How the all-hadronic channel moved from the upstream output to the current result, one change at
a time, against note Table 12 rescaled to our 4.907 fb⁻¹ — and what the single-lepton channel is
measured against.

## All hadronic, step by step

| # | Change | data/MC | Z(νν) share | tt̄ share |
|---|---|---:|---:|---:|
| — | upstream | ~10 | ~0 % (wrong sample) | 73 % |
| 1 | PHYS-01…08 applied | 1.615 | 4.6 % | |
| 2 | + CSVv2 b-tagging, + WZ, + Z(νν) low-$p_T$ bin | 1.485 | 4.6 % | |
| 3 | + corrected Z(νν) cross section (PHYS-13) | 1.217 | 21.9 % | 62.8 % |
| 4 | + top $p_T$ reweighting (PHYS-14) | **1.340** | **24.1 %** | **59.1 %** |
| — | **note Table 12** | **1.06** | **29.7 %** | **45.5 %** |

!!! question "Why the Z(νν) cross section had to be tested, not assumed"
    With 0.7333 pb the equivalent luminosity of the sample is 619 fb⁻¹ — 17× the full 2016
    dataset, unusually generous for a background. The prediction written down beforehand was:
    *if Z(νν) lands near 5 %, the pairing is wrong.* A 1-file test gave 4.5 %; the full 20-file
    run gave **4.6 %** — systematic, not statistical. The resolution is in
    [PHYS-13](physics.md#phys-13-zvv-cross-section-use-the-unfiltered-value).

## PHYS-14, measured

Full 20-file run, 28/28 cells, 0 errors.

| Group | Before | After | Factor | % before | % after | Table 12 |
|---|---:|---:|---:|---:|---:|---:|
| tt̄ | 4333.5 | 3698.8 | **0.854** | 62.8 % | **59.1 %** | 45.5 % |
| Z(νν) | 1508.4 | 1508.4 | 1.000 | 21.9 % | **24.1 %** | 29.7 % |
| W+jets | 870.0 | 870.0 | 1.000 | 12.6 % | **13.9 %** | 15.1 % |
| t+X | 116.8 | 116.8 | 1.000 | 1.7 % | 1.9 % | — |
| VV, VH | 64.1 | 64.1 | 1.000 | 0.9 % | 1.0 % | 2.7 % |
| **Total MC** | **6895.9** | **6261.2** | | | | |
| **data/MC** | **1.217** | **1.340** | | | | **1.06** |

**The prediction held.** The gap analysis required tt̄ × 0.83; the note's formula, applied with no
tuning against the target, delivered **0.854**. Every composition row moved toward Table 12.

**And data/MC got worse, as it should.** The correction only *removes* tt̄; the corrections that
would *add* events — V+jets k-factors ([OPEN-4](open-questions.md#open-4-vjets-nlolo-k-factors))
and VH ([OPEN-5](open-questions.md#open-5-vh-is-missing-from-the-diboson-group)) — are the ones
not available. **Composition is the diagnostic that responds to what was fixed; normalisation is
the one still waiting on what is missing.** Reporting only data/MC would have made a correct
change look like a mistake.

## Where the remaining gap is, per process

Target total = data / 1.06 = 8392 / 1.06 = **7917** events against 6896 (before PHYS-14):
everything has to grow by 1.15× while tt̄'s share falls.

| Process | Now | Needed | Factor | What would supply it |
|---|---:|---:|---:|---|
| tt̄ | 4333.5 | 3602.2 | **0.83** | top $p_T$ reweighting — measured **0.854**, applied |
| Z(νν) | 1508.4 | 2351.3 | **1.56** | V+jets NLO/LO k-factor |
| W+jets | 870.0 | 1195.5 | **1.37** | V+jets NLO/LO k-factor |
| VV, VH | 64.1 | 213.8 | **3.33** | **VH is absent from our stack entirely** |

1. **tt̄** — covered by PHYS-14.
2. **Z(νν) and W+jets** need 1.56 and 1.37. QCD NLO/LO k-factors for V+jets at high boson $p_T$
   are typically 1.2–1.5 — the right magnitude, on exactly the processes the note applies them to.
3. **VV, VH** needs 3.33, too much for a k-factor: the paper's group is "VV, **VH**" and ours has
   no VH. It is the smallest absolute term, though — 150 events of 7917, against +843 for Z(νν).

The remaining disagreement decomposes into one correction applied, one that cannot be derived
from the note, and one missing sample. No third unknown effect is needed.

### What closure would look like

Applying the implied factors for the two missing inputs (Z(νν) ×1.56, W+jets ×1.37, VV/VH ×3.33)
on top of the measured result gives data/MC **1.108**, tt̄ **48.8 %**, Z(νν) **31.1 %**, W+jets
**15.7 %** — against 1.06 / 45.5 / 29.7 / 15.1 in the paper.

!!! danger "Arithmetic, not a result — do not quote it"
    The factors were derived from Table 12, so landing near Table 12 is circular. It shows only
    that the residual is *consistent* with two named inputs. The honest current state remains
    **data/MC = 1.34, tt̄ at 59.1 %**.

## What the ratio panel rules out

The Data/Bkg ratio (added with the PLOT pass) is **flat** across $p_T^{\text{miss}}$. That kills the
hypothesis carried through most of this work — that the disagreement came from Open Data being
the UL2016 reconstruction while the paper used the 2016 legacy reprocessing with a different MET
calibration. A calibration difference distorts the shape; a flat ratio means the shape is right
and only the normalisation is off.

## Hypotheses tested and rejected

| Hypothesis | Measured effect | Verdict |
|---|---:|---|
| Missing MET triggers | 0.64 % of events at $p_T^{\text{miss}} > 250$ | rejected |
| Missing MET noise filters | 0.05–1.8 % | rejected |
| b-tagger mistag rate | −19 %, wrong direction | rejected |
| UL vs legacy MET calibration | ratio is flat | rejected |
| Z(νν) cross-section pairing | **4.55×** | **confirmed** |

## MC corrections: applied and not

| Correction | Status | Why |
|---|---|---|
| Top $p_T$ reweighting | **applied** | formula given in full in the note |
| V+jets NLO/LO k-factors | not applied | computed with MG5_aMC@NLO vs boson $p_T$, not tabulated in the note |
| b-tagging scale factors | not applied | per-jet, per-flavour BTV POG tables |
| Lepton ID / iso / tracking SFs | not applied | EGAMMA / MUON POG tag-and-probe tables |
| Trigger scale factors | not applied | measured per $p_T$, η |
| Pileup reweighting | not applied | needs the data pileup profile |
| Global fit across control regions | not implemented | the paper's figures are **post-fit** |

## Single lepton: the benchmark, read in advance

Note **Table 11** after the full SL selection:

| Process | Events | Share |
|---|---:|---:|
| **tt̄ (2ℓ)** | 772.81 | **61.4 %** |
| W+jets | 185.79 | 14.8 % |
| t+X | 177.17 | 14.1 % |
| tt̄+V | 52.73 | 4.2 % |
| VV, VH | 44.00 | 3.5 % |
| **tt̄ (1ℓ)** | 20.77 | **1.7 %** |
| Z(ℓℓ) | 2.88 | 0.2 % |
| Z(νν) | 1.52 | 0.1 % |
| **Total / data** | **1257.68 / 1406** | **data/MC = 1.118** |

**Dileptonic tt̄ dominates, 37 to 1 over semileptonic**, because $m_T > 160$ GeV removes genuine
one-lepton tt̄ at the Jacobian edge; what survives lost its second lepton. Z(νν), which dominated
the AH effort, is 0.1 % here.

**$M_{T2}^W$ is the largest cut in the selection:**

| Cut | Total background | Factor |
|---|---:|---:|
| after $m_T$ | 17 489.14 | |
| after **$M_{T2}^W$** | 2 713.20 | **×0.16** |
| after $\min\Delta\phi$ | 1 659.28 | ×0.61 |
| after $m_T^b$ | 1 257.68 | ×0.76 |

It removes 84 % of the background — more than every other SR cut combined — and it is selective:
×9.3 on tt̄(2ℓ), ×1.7 on W+jets. Without it our SL signal regions are several times too full,
overwhelmingly with tt̄(2ℓ). Our measured SL result is on the
[single-lepton results page](../results/single-lepton.md#background-composition-in-the-signal-region).

### What the single-lepton channel needs, in order

1. **[OPEN-6]** the tW record IDs — one factor-28 row, mechanically understood.
2. **[OPEN-3]** $M_{T2}^W$ — why tt̄(2ℓ) sits at 93 % instead of 61 %.
3. **[OPEN-4]** the V+jets k-factors — W+jets at 3.9 % against 14.8 %, same direction as AH.

# Results

All plots on these pages come from the **full run** — 20 ROOT files per dataset, 22 datasets —
executed with the corrected notebooks and reviewed by CERN/DPOA. They are the outputs stored in
`results/All_Hadronic_run.ipynb` and `results/Single_Lepton_run.ipynb` of the code repository.

| Channel | Cells | Plots | Errors | Luminosity of the processed data |
|---|---:|---:|---:|---:|
| [All hadronic](all-hadronic.md) (MET dataset) | 28/28 | 6 | 0 | 4.9 fb⁻¹ |
| [Single lepton](single-lepton.md), muon (SingleMuon) | 35/35 | 7 | 0 | 2.4 fb⁻¹ |
| [Single lepton](single-lepton.md), electron (SingleElectron) | — | 7 | 0 | 2.9 fb⁻¹ |

The [paper comparison](paper-comparison.md) puts every plot next to the paper's figure and the
original, uncorrected version.

## Headline numbers

=== "All hadronic — against note Table 12"

    | Background group | Events | Share | Note Table 12 |
    |---|---:|---:|---:|
    | tt̄ | 3698.8 | 59.1 % | 45.5 % |
    | Z(νν) + jets | 1508.4 | 24.1 % | 29.7 % |
    | W(ℓν) + jets | 870.0 | 13.9 % | 15.1 % |
    | t + X | 116.8 | 1.9 % | — |
    | VV, VH | 64.1 | 1.0 % | 2.7 % |
    | Z(ℓℓ) + jets | 3.1 | 0.0 % | — |
    | **Total MC** | **6261.0** | | |
    | **Data** | **8392** | | |
    | **data/MC** | **1.340** | | **1.06** |

    After the baseline selection, 4.907 fb⁻¹.

=== "Single lepton — against note Table 11"

    | Process | Muon | Electron | Note Table 11 |
    |---|---:|---:|---:|
    | tt̄ (2ℓ) | 93.3 % | 94.2 % | 61.4 % |
    | W(ℓν) + jets | 3.9 % | 3.3 % | 14.8 % |
    | tt̄ (1ℓ) | 1.6 % | 1.3 % | 1.7 % |
    | VV, VH | 0.6 % | 0.7 % | 3.5 % |
    | t + X | 0.5 % | 0.5 % | 14.1 % |
    | Z(νν) + jets | 0.1 % | 0.1 % | 0.1 % |
    | Z(ℓℓ) + jets | 0.0 % | 0.0 % | 0.2 % |
    | **Total MC** | **205.5** | **179.3** | |
    | **Data** | **221** | **169** | |
    | **data/MC** | **1.075** | **0.942** | **1.118** |

    After the full signal selection **except $M_{T2}^W$**, which is not implemented.

## How to read these plots

!!! info "What matches the paper, and what cannot"
    - **Same presentation.** Binning, stacking order, legend order, overflow in the last bin and
      the relative axis range all follow the paper's Figures 4 and 5
      ([PLOT-01…05](../changes/plots.md)). Bins can be compared one to one.
    - **Lower absolute scale.** Open Data exposes only Run2016H; we process 2.4–4.9 fb⁻¹ against
      the paper's 35.9 fb⁻¹, so every stack sits roughly an order of magnitude lower.
    - **Pre-fit, not post-fit.** The paper's backgrounds are normalised by a simultaneous fit
      across control regions. Ours are pure simulation at fixed cross section, so an exact match
      was never reachable without implementing the fit.
    - **Shape is the robust comparison.** The Data/Bkg ratio panels are flat across
      $p_T^{\text{miss}}$: the shapes are right and what remains is normalisation, traced to the
      specific missing inputs in [Open questions](../changes/open-questions.md).

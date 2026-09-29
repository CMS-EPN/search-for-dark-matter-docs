# Plotting changes (PLOT)

Even where the numbers were right, our plots and the paper's could not be laid side by side,
because they were not the same histogram. These fixes move **no event**: yields, composition
and data/MC are unchanged. They matter because until they were in place, every judgement of
"these look different" was partly measuring our own plotting.

The mismatches were found by cropping the six panels of Figure 4 (single lepton, p. 16) and the
three of Figure 5 (all hadronic, p. 17) out of the paper at 300 dpi and reading axes and stack
off the pixels. The plot function now lives in one file, `src/region_plot_cell.py`, embedded by
both builders — it used to be duplicated as a string literal in each, so a fix to one silently
left the other behind.

## PLOT-01 · Stacking order

**Was:** processes sorted by yield, smallest on top.
**Paper:** a fixed order, bottom first:

```text
VV/VH → t+X → tt̄ → W(ℓν) → Z(νν) → Z(ℓℓ)
```

In the all-hadronic channel both orders happen to agree. In the single-lepton channel they do
not: tt̄ is ~75 % of the SR, so yield-sorting dropped it to the *bottom*, while the paper keeps it
third. The same numbers produced a completely different silhouette — a large part of why the SL
comparison "looked nothing alike". **Now:** `PAPER_STACK_ORDER`.

## PLOT-02 · Binning

| | Before | Paper (now) |
|---|---|---|
| All hadronic | 250–600 GeV, 10 bins of 30 GeV | **250–550 GeV, 15 bins of 20 GeV** |
| Single lepton | 160–600 GeV, 10 bins of 44 GeV | **160–520 GeV, 9 bins of 40 GeV** |

Not one bin edge was shared with the paper. Edges were measured from the marker positions in the
cropped panels (AH markers at 260, 280, …, 540; SL at 180, 220, …, 500).

## PLOT-03 · The last bin is an overflow bin

Both captions say *"The last bin contains overflow events."* We were dropping everything above
the axis maximum, so our rightmost bin **fell away** where the paper's **rises** — the most
conspicuous shape difference in the comparison, and not physics at all. Values are now clipped
just below the top edge before filling, for data and MC.

## PLOT-04 · Axis ranges

**Was:** a y ceiling of `max(counts) × 300`, so every panel chose its own decades.

Copying the paper's absolute range would be wrong: with 2.4–4.9 fb⁻¹ against 35.9 fb⁻¹, our
stacks sit about an order of magnitude lower and would be pushed into an empty frame. What is
copied is the paper's **proportions**: limits at fixed multiples of the stack peak — 3.5 decades
below, 2 above for the legend. Every panel in a channel now has the paper's dynamic range without
pretending to have its luminosity.

## PLOT-05 · Legend order

The legend came from matplotlib's return order, which turned out to be bottom-up, while the
paper's reads top-down: Data, Z(ℓℓ), Z(νν), W(ℓν), tt̄, t+X, VV/VH. It is now built from the same
`order` list that draws the stack, reversed once, so it cannot drift again.

## Also in this pass

- **Ratio panel added** (Data/Bkg, as in paper Figure 5). It showed the ratio is **flat** — see
  [what the ratio panel rules out](convergence.md#what-the-ratio-panel-rules-out).
- **Region labels** read as the paper's (`0l, SR, 0 FJ`, `1mu, 2 b tag, SR`, …) instead of
  internal shorthand (`SR 1b 0f`).
- **The CMS luminosity label** is passed explicitly per channel instead of being derived from
  whichever luminosity was largest.

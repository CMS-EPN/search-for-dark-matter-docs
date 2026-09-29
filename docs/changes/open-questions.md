# Open questions for the collaboration

Everything still open is a **specific missing input**, not an unexplained discrepancy. This list
is what to ask the collaboration for.

| ID | What is needed | Channel | Status |
|---|---|---|---|
| OPEN-1 | Z→νν and DY cross sections | both | ✅ closed |
| OPEN-2 | Follow the paper, or stay faithful to upstream? | both | ✅ both kept |
| **OPEN-3** | An implementation of $M_{T2}^W$, or agreement to report SL as incomplete | SL | ⬜ open |
| **OPEN-4** | V+jets NLO/LO k-factor tables | both | ⬜ open — biggest single input |
| **OPEN-5** | UL16 VH record IDs | AH | ⬜ open |
| **OPEN-6** | UL16 record IDs for tW, s-channel, tt̄V | SL | ⬜ open |

## OPEN-1 · Cross sections for Z→νν and DY

**Closed 2026-09-03** — queried from XSDB with a CMS account. Initially the BPSFilter row
(0.7333 pb, UL17) looked like an exact match for record 74910; the full-statistics run proved
otherwise, and **4.201 pb** is used — full argument in
[PHYS-13](physics.md#phys-13-zvv-cross-section-use-the-unfiltered-value). DY: **0.393 pb**
(UL16 row), replacing upstream's 1.27 pb.

## OPEN-2 · Paper or upstream?

Applying the PHYS changes moves the plots toward the paper and away from a faithful reproduction
of the DPOA notebooks. **Resolution:** keep both. The upstream notebooks are preserved untouched
in `notebooks/upstream/` and the Phase A reproduction proved the pipeline reproduces them exactly;
the main notebooks carry the corrections.

## OPEN-3 · MT2W: the largest cut in the SL selection

Not implemented — it needs a kinematic minimisation over jet–lepton assignments (note ref. [41]).
From note Table 11 it removes 84 % of the SL background on its own (17 489 → 2 713, ×0.16), and
×9.3 on tt̄(2ℓ), which is 61 % of this channel's background, against ×1.7 on W+jets.

**Question:** is an approximation acceptable, or should the SL signal regions be reported as
explicitly incomplete?

## OPEN-4 · V+jets NLO/LO k-factors

The note says W+jets and Z+jets are corrected with electroweak and QCD NLO/LO k-factors computed
with MG5_aMC@NLO v2.2.2 as a function of generated boson $p_T$. The table itself is not in the
note.

**Why this one matters most:** the two processes still short against Table 12 are exactly the
two these k-factors act on — Z(νν) needs ×1.56 and W+jets ×1.37, inside the 1.2–1.5 range
published k-factors take. The same shortfall shows in SL (W+jets 3.9 % vs 14.8 %).

**Needed:** the numerical tables (or the ROOT file) vs generated boson $p_T$, QCD and EW parts,
for W+jets and Z+jets. Guessing values would move the plots toward the paper for the wrong reason,
so nothing is applied.

## OPEN-5 · VH is missing from the diboson group

The paper's group is "VV, VH"; ours has WW, WZ, ZZ. Table 12 puts the group at 2.7 %; we have
1.0 %. A portal name search for VH samples in RunIISummer20UL16 NanoAODv9 returned nothing — but
the same search returns nothing for `TTToSemiLeptonic` either, so it is not evidence of absence.
The one UL16 Higgs record it surfaced, `HTo2LongLivedTo2mu2jets_MH-125_...` (records
41334/41335), is precisely the kind of long-lived-particle signal sample that upstream had
wired in as Z→νν, and is not relevant.

**Needed:** the UL16 VH record IDs, or confirmation they are not published — in which case the
component stays documented as unreproducible from Open Data.

## OPEN-6 · Record IDs for tW, s-channel and ttV

`ST tW top`, `ST tW antitop`, `ST s-channel`, `TTWJetsToQQ`, `TTZToQQ`, `TTZToLLNuNu`. tW alone
explains the SL t+X factor of 28 — see
[PHYS-15](physics.md#phys-15-single-top-tw-s-channel-and-ttv-samples-missing). Every sample whose
record ID was found (Zvv, WZ, the low-$p_T$ Z bin, the W+jets tail) was fixed the moment it was
known.

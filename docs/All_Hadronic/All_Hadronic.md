# All hadronic chanel
## Physics Motivation and Channel Strategy: All-Hadronic Channel

The Large Hadron Collider (LHC) collides protons at center-of-mass energies high enough to probe physics beyond the Standard Model. Although the protons are composite objects, the relevant hard scatterings occur between their constituents — quarks and gluons. In the context of simplified dark matter models, these partonic interactions can produce top quarks together with a new mediator particle (commonly denoted φ for scalar or a for pseudoscalar). The mediator then decays invisibly into a pair of dark matter candidates (χχ̄). At the detector level, this results in events with multiple top quarks plus significant missing transverse momentum (p_T^miss), the latter coming from the invisible χ particles.

The production mechanisms of interest include:

• **Gluon fusion:**
  $$ gg \to t \bar{t}\,\phi \to t \bar{t} + \chi \bar{\chi} $$

• **Single top associated production:**
  $$ gb \to t \phi \to t + \chi \bar{\chi} $$

• **t–channel production:**
  $$ qq' \to tb \phi \to tb + \chi \bar{\chi} $$

In all cases, the top quarks decay via $t \to W b$. Each W boson subsequently decays either leptonically ($W \to \ell \nu$) or hadronically ($W \to q \bar{q}'$). Thus, the final states contain a mixture of b-tagged jets, light-flavor jets, charged leptons (electrons or muons), and genuine $p_T^{\text{miss}}$.

## Channel Strategy

Because of the different W decay modes, analyses are divided into channels, each defined by the number of isolated charged leptons:

• **Single-lepton (SL):** one isolated electron or muon, several jets (including ≥1 b-tag), and nonzero $p_T^{\text{miss}}$. This channel is statistically powerful and relatively clean, striking a balance between signal sensitivity and manageable backgrounds.

• **All-hadronic (AH):** no isolated leptons, many jets including b-tagged jets, and $p_T^{\text{miss}}$. While it has the largest raw yield, it suffers from overwhelming QCD multijet background, which can fake $p_T^{\text{miss}}$.

• **Dilepton:** two isolated leptons, large $p_T^{\text{miss}}$, and multiple jets. It provides a very clean signal region but is limited by low branching fraction, hence low statistics.

**In this notebook, we concentrate on the all-hadronic channel with no isolated leptons.**

There are both theoretical and practical reasons for this choice:

- **From the physics side:** The AH channel has the highest branching ratio (~46% for $t\bar{t}$) since both W bosons decay hadronically. This provides maximum statistical power despite the challenging QCD background.

- **From the experimental side:** Dedicated MET-based triggers and stringent angular cuts can effectively suppress QCD contamination. The presence of multiple b-jets further enhances signal-to-background discrimination.

This focus allows us to demonstrate the full workflow — from event selection to histograms — in a setting where the interplay between signal characteristics and background processes can be clearly explained. Splitting into channels is therefore not a stylistic decision but a physics necessity: each final state probes the same underlying processes under different background conditions and detector signatures.

---

# Software Setup and Package Imports

Before we start analyzing data, we need to set up the software environment. This section imports the Python packages that allow us to read CMS NanoAOD files, manipulate event data, and produce plots in a reproducible and physics-oriented way.

• **Core utilities:**
  - `os`, `time`, `json`, `logging`, `asyncio` → for file handling, timing, and bookkeeping.
  - These are standard Python libraries that help organize the workflow and log progress.

• **Numerical analysis:**
  - `numpy` (`np`) → fundamental for vectorized calculations on arrays.
  - `pandas` (`pd`) → convenient for storing metadata (cross sections, cutflows, etc.) in table form.

• **Visualization:**
  - `matplotlib` (`mpl`, `plt`) → general-purpose plotting.
  - `hist` → modern histogramming library designed for HEP, integrates smoothly with `mplhep`.

• **HEP-specific data access:**
  - `uproot` → reads CMS `.root` files into Python without needing C++/ROOT. Essential for open data workflows.
  - `awkward` (`ak`) → handles "jagged arrays" (variable-length collections per event), e.g. different numbers of jets per event.
  - `vector` → enables 4-vector operations (pT, eta, phi, invariant masses) in a NumPy/Awkward-friendly way. We register it with Awkward to use directly on event data.

• **Coffea ecosystem:**
  - `coffea.processor` → framework to run HEP analyses at scale.
  - `NanoAODSchema` and `NanoEventsFactory` → provide a high-level interface to CMS NanoAOD, automatically building event objects like muons, jets, MET, etc.
  - `transforms` and `methods.base/vector` → ensure physics-style behavior (Lorentz vectors, masks, object methods) are available on Awkward arrays.

Finally, we configure Coffea to ignore unusual cross-references in CMS open data (they don't affect our analysis) and print the versions of the key packages. This ensures reproducibility: anyone re-running the notebook can confirm they are using the same software environment.

```python
import asyncio, logging, os, time, json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import hist
import uproot
import awkward as ak
import vector as v
v.register_awkward()

# Coffea
from coffea import processor
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.nanoevents import transforms
from coffea.nanoevents.methods import base, vector as cvector

# Ensure Coffea behaviors are available on Awkward objects
ak.behavior.update(base.behavior)
ak.behavior.update(cvector.behavior)

# Be gentle with open-data odd cross-refs
NanoAODSchema.warn_missing_crossrefs = False

import coffea
print("versions:",
      "coffea", coffea.__version__,
      "| awkward", ak.__version__,
      "| uproot", uproot.__version__)
```

---

# Dataset Definitions and Luminosity Masks

The CMS Open Data portal provides access to collision and simulation datasets in the NanoAOD format. Each dataset (e.g. MET, TTToHadronic, W+jets) is split across many `.root` files, sometimes hundreds per process. To manage this efficiently, we build a Python dictionary `nanoaod_filenames` that maps process names to their corresponding file index lists. Each entry points to a `.txt` file containing the remote paths (via XRootD) of the NanoAOD files.

**Examples:**

• **Data (collision events):** `MET`. These correspond to recorded events with MET-based trigger paths.

• **Signal MC:** e.g. hypothetical $t\bar{t} + DM$ samples.

• **Background MC:** top pair production (hadronic, semileptonic, dileptonic), single top (t-channel, tW), and electroweak processes like W+jets, Z→νν, or dibosons (WW, ZZ).

This separation is not just organizational:

- **From a physics perspective,** each dataset represents a different process contributing to the observed events. Signal vs. background categories are crucial for defining the search strategy.

- **From a programming perspective,** keeping datasets in a dictionary allows us to iterate over them in loops, automate file loading, and apply the same selections consistently.

## Luminosity Masks

Real CMS data are recorded in luminosity sections (blocks of events). Not all sections are usable: some are flagged as problematic by the detector monitoring. To ensure reproducibility, CMS provides certified luminosity JSON files, which specify the "good" sections.

The function `build_lumi_mask()` implements this filter:

• Reads the certified JSON file.
• Compares the `run` and `luminosityBlock` of each event to the approved ranges.
• Returns a boolean mask selecting only events in certified sections.

This step is critical in real data analysis:

- **Physics motivation:** prevents contamination from detector malfunctions.

## Helper Functions

• `get_files_for_dataset(dataset, random=False, n=0)` → loads a subset of filenames for a given dataset. Useful when testing code with fewer files to save time.

• `pretty_print(fields, ...)` → formats lists of branches or variables, making the NanoAOD structure easier to inspect.

Together, these utilities allow us to handle dozens of datasets and millions of events in a manageable, modular way.

```python
import dpoa_workshop
from dpoa_workshop import nanoaod_filenames
from dpoa_workshop import get_files_for_dataset
from dpoa_workshop import pretty_print
from dpoa_workshop import build_lumi_mask
```

---

# Building the Ntuple File Index

CMS Open Data provides file index text files (`file_index.txt`) for each dataset. These contain the actual XRootD paths to the NanoAOD `.root` files, along with metadata such as the number of events per file.

To streamline the workflow:

• We define a function `download_and_parse_fileindex(url)` that fetches each `file_index.txt` via HTTP and extracts only the ROOT file paths.

• We loop over all entries in `nanoaod_filenames` (the dictionary we built earlier) and collect the full list of ROOT files per dataset.

• The result is stored in a new dictionary `ntuples_simple`, which maps dataset → list of ROOT file paths.

• Finally, we save this as a JSON file (`ntuples_simple.json`) for later reuse.

```python
import os, json, requests
from dpoa_workshop import nanoaod_filenames

def download_and_parse_fileindex(url):
    """Download a file_index.txt and return list of ROOT paths."""
    r = requests.get(url)
    lines = [ln.strip() for ln in r.text.splitlines() if ln.strip()]
    # Each line is: root://... nevts=N
    paths = [ln.split()[0] for ln in lines]
    return paths

ntuples_simple = {}

# Loop through nanoaod_filenames and save only paths
for dataset, urls in nanoaod_filenames.items():
    all_paths = []
    for url in urls:
        try:
            all_paths.extend(download_and_parse_fileindex(url))
        except Exception as e:
            print(f"[warn] {dataset} {url} -> {e}")
    
    ntuples_simple[dataset] = all_paths

# Save the JSON
with open("ntuples_simple.json", "w") as f:
    json.dump(ntuples_simple, f, indent=2)

print("ntuples_simple.json created with datasets:", list(ntuples_simple.keys()))
```

---

# Analysis Configuration

We first define which datasets to analyze. This includes data (`MET`) and several MC backgrounds (ttbar, single top, W+jets, dibosons, Z→νν, etc.).

We also set controls for testing:

• `N_FILES_MAX_PER_SAMPLE`: how many ROOT files per dataset.
• `MAX_EVENTS_PER_FILE`: how many events to read per file.

This allows us to run quickly on subsets of data before scaling to the full analysis.

```python
# ================================
# Initial configuration
# ================================
DATASETS_TO_RUN = [
    "MET",
    "ttbar-hadronic",
    "ttbar-semileptonic",
    "t-channel-top",
    "ttW",
    "WJets-HT400to600",
    "WJets-2J-FxFx",
    "DYJets-Zpt200",
    "Zvv",
    "ZZ",
    "WW",
]

# Limit ROOTs and events
N_FILES_MAX_PER_SAMPLE = 1  # use only 1 ROOT file per dataset
MAX_EVENTS_PER_FILE = 50000  # None = use all events

# Luminosity (fb^-1)
LUMI_FB = 1.0

# JSON with paths to the ROOTs
NTUPLES_JSON = "ntuples_simple.json"
```

---

# Constructing the Fileset

We construct a fileset, which is a dictionary that maps each dataset name to:

• A list of ROOT file paths.
• Metadata such as process name, cross section, and variation.

This abstraction allows us to loop over datasets uniformly later, regardless of whether they are data or MC.

```python
import json

def construct_fileset_simple(
    n_files_max_per_sample=1,
    ntuples_json="ntuples_simple.json"
):
    with open(ntuples_json) as f:
        info = json.load(f)
    
    fileset = {}
    for key, files in info.items():
        if key not in DATASETS_TO_RUN:
            continue
        
        # limit number of files
        if n_files_max_per_sample == -1:
            use_files = files
        else:
            use_files = files[:n_files_max_per_sample]
        
        # normalize file list
        file_list = [f["path"] if isinstance(f, dict) else f for f in use_files]
        
        fileset[key] = {
            "files": file_list,
            "metadata": {
                "process": key,
                "dataset": key,
                "variation": "nominal",
            }
        }
    
    return fileset

# --- construct fileset ---
fileset = construct_fileset_simple(
    n_files_max_per_sample=N_FILES_MAX_PER_SAMPLE,
    ntuples_json=NTUPLES_JSON
)

print(f"[OK] Fileset constructed with {len(fileset)} datasets")
for k, pack in fileset.items():
    print(f"  {k}: {len(pack['files'])} file(s)")
```

---

# Baseline Event Selection: All-Hadronic Channel

We now define the baseline cuts for the all-hadronic channel with no isolated leptons. These are motivated by CMS dark matter searches and are designed to suppress dominant Standard Model backgrounds (mainly QCD multijet, $t\bar{t}$, and W/Z+jets) while enhancing sensitivity to dark matter signals with real $p_T^{\text{miss}}$.

## 1. Trigger Requirements

Events are selected if they fire any of the following HLT paths:

• `HLT_PFMETNoMu120`
• `HLT_PFMETNoMu90_PFMHTNoMu90_IDTight`
• `HLT_PFMETNoMu110_PFMHTNoMu110_IDTight`
• `HLT_PFMETNoMu120_PFMHTNoMu120_IDTight`
• `HLT_PFMETNoMu90_JetIdCleaned_PFMHTNoMu90_IDTight`
• `HLT_PFMETNoMu120_JetIdCleaned_PFMHTNoMu120_IDTight`
• `HLT_PFMET120_PFMHT120`
• `HLT_PFMET110_PFMHT110_IDTight`
• `HLT_PFMET120_PFMHT120_IDTight`
• `HLT_PFMET170`
• `HLT_PFMET170_NoiseCleaned`
• `HLT_PFMET170_HBHECleaned`
• `HLT_PFMET170_HBHE_BeamHaloCleaned`

**Motivation:** These MET-based triggers are efficient for hadronic final states with genuine missing energy.

## 2. Event Cleaning Flags

The following filters must be applied to both data and simulation:

• `HBHENoiseFilter`
• `HBHENoiseIsoFilter`
• `ECALDeadCellFilter`
• `GlobalTightHalo2016Filter`
• `BadPFMuonFilter`
• `BadChargedHadronFilter`

The following filters are applied to data only:

• `EEBadScFilter`
• `BadMuons`
• `DuplicateMuons`

**Motivation:** Remove events with detector noise and instrumental fake MET.

## 3. Lepton Veto

• **No "Veto" Leptons:** Events must contain no "Veto" leptons.
  - **Electrons:** No electrons with $p_T > 10$ GeV, $|\eta| < 2.5$, and loose ID.
  - **Muons:** No muons with $p_T > 10$ GeV, $|\eta| < 2.4$, and loose ID.

**Motivation:** Suppress semileptonic $t\bar{t}$, W+jets, and dilepton backgrounds.

## 4. Jet Selection

• **Jet $p_T$:** > 30 GeV
• **Jet Eta:**
  - Central jets: $|\eta| < 2.4$
  - Forward jets: $2.4 < |\eta| < 5$ (optional, depending on analysis)
• **Jet ID:** Loose jet ID requirements
• **Overlap Removal:** Jet objects are not considered if they are within $\Delta R < 0.4$ of a "Tight" electron or muon.

**Motivation:** Ensure well-reconstructed jets while avoiding double-counting with leptons.

## 5. Minimum Number of Jets

• **Baseline:** ≥ 3 jets

**Motivation:** Top decays produce at least 6 quarks in the $t\bar{t}$ fully hadronic channel (2 b-quarks + 4 light quarks from W decays). Requiring ≥3 jets ensures we capture the event topology.

## 6. Minimum Number of b-tagged Jets

• **Baseline:** ≥ 1 b-tagged jet (CSVM working point) with $p_T > 30$ GeV
• **Categorization:**
  - $n_b = 1$ (for single top + DM events)
  - $n_b ≥ 2$ (for $t\bar{t}$ + DM events)

**Motivation:** Top-quark decays always produce b-jets, so this suppresses W+light-flavor jets and QCD.

## 7. MET Requirements

• **Baseline:** $p_T^{\text{miss}} ≥ 250$ GeV

**Motivation:** Dark matter escapes undetected → large genuine MET. This high threshold strongly suppresses QCD multijet background.

## 8. Angular Separation: Δφ Cuts

• **Baseline:** $\min\Delta\phi(j_{1,2}, p_T^{\text{miss}}) > 0.4$
• **Optimized Selection:** $\min\Delta\phi(j_{1,2}, p_T^{\text{miss}}) > 1.0$

**Motivation:** Reduces QCD multijet events with mismeasured MET aligned with jets. True MET from dark matter is typically more isotropic.

## 9. Additional Kinematic Variables

• **b–MET Transverse Mass ($M_{bT}$):**
  - $M_{bT} > 180$ GeV
  - **Motivation:** Further discriminates signal from $t\bar{t}$ backgrounds by exploiting the kinematics of b-quarks and MET.

• **Jet 1 $p_T$ / $H_T$:**
  - ≤ 0.5 (specifically for $n_b ≥ 2$ category)
  - **Motivation:** Reduces events where a single jet dominates the hadronic activity, which is characteristic of QCD.

---

## Summary

The baseline selection region is defined by:

• MET-based triggers (no leptons required)
• Event cleaning filters
• **No veto leptons** (electrons or muons with $p_T > 10$ GeV)
• ≥ 3 jets with ≥ 1 b-tag
• Large MET ($p_T^{\text{miss}} ≥ 250$ GeV)
• Angular separation to suppress QCD ($\min\Delta\phi(j_{1,2}, p_T^{\text{miss}}) > 0.4$ or $> 1.0$)
• Additional kinematic cuts ($M_{bT} > 180$ GeV, jet $p_T$ / $H_T$ ≤ 0.5)

Together, these cuts target signal-like topologies while removing the bulk of Standard Model backgrounds, especially the overwhelming QCD multijet contamination.

---

# Analysis Implementation

This class (`DMAnalysisAllHadronic`) encodes the physics selection of the analysis for the all-hadronic channel:

• **Object-level cuts:**
  - Veto loose electrons: $p_T > 10$ GeV, $|\eta| < 2.5$, loose ID.
  - Veto loose muons: $p_T > 10$ GeV, $|\eta| < 2.4$, loose ID.
  - Jets: $p_T > 30$ GeV, $|\eta| < 2.4$, good jetID.
  - b-jets: tagged with DeepCSV/DeepFlav WP.

• **Event-level cuts:**
  - Pass MET-based triggers (HLT).
  - No veto leptons (0 electrons, 0 muons).
  - ≥3 jets, ≥1 b-jet.
  - $p_T^{\text{miss}} ≥ 250$ GeV.
  - $\min\Delta\phi(j_{1,2}, p_T^{\text{miss}}) > 0.4$.

• **Outputs:**
  - Histograms for physics variables.
  - Cutflow (number of events passing each step).

```python
import time, pickle
from pathlib import Path
import numpy as np
import awkward as ak
import hist
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory

# === Function to read events from a ROOT file with optional limit ===
def events_from_file(path, metadata=None, schemaclass=NanoAODSchema, max_events=None):
    """
    Load events from a ROOT file.
    If max_events is not None, only read up to that number of events (useful for quick tests).
    """
    factory = NanoEventsFactory.from_root(
        {path: "Events"},
        schemaclass=schemaclass,
        metadata=(metadata or {}),
        entry_stop=max_events,
    )
    return factory.events()

# === Main class for Dark Matter analysis (All-Hadronic channel) ===
class DMAnalysisAllHadronic:
    def __init__(self, DATASET, lumi_fb):
        self.DATASET = DATASET
        self.lumi_fb = float(lumi_fb)
        
        # Histogram axes
        self.process_cat = hist.axis.StrCategory([], name="process", label="Process", growth=True)
        self.variation_cat = hist.axis.StrCategory([], name="variation", label="Variation", growth=True)
        
        # Physics variables
        num_axis = hist.axis.Regular(15, 0, 15, name="var")      # njets, nbjets
        met_axis = hist.axis.Regular(30, 0, 600, name="var")     # MET
        dphi_axis = hist.axis.Regular(32, 0, 3.2, name="var")    # min deltaphi
        ht_axis = hist.axis.Regular(40, 0, 2000, name="var")     # HT
        mbt_axis = hist.axis.Regular(30, 0, 600, name="var")     # M_bT
        
        # Histograms
        self.h = {
            'njets':    hist.Hist(num_axis, self.process_cat, self.variation_cat, storage=hist.storage.Weight()),
            'nbjets':   hist.Hist(num_axis, self.process_cat, self.variation_cat, storage=hist.storage.Weight()),
            'met':      hist.Hist(met_axis, self.process_cat, self.variation_cat, storage=hist.storage.Weight()),
            'min_dphi': hist.Hist(dphi_axis, self.process_cat, self.variation_cat, storage=hist.storage.Weight()),
            'ht':       hist.Hist(ht_axis, self.process_cat, self.variation_cat, storage=hist.storage.Weight()),
            'mbt':      hist.Hist(mbt_axis, self.process_cat, self.variation_cat, storage=hist.storage.Weight()),
        }
        
        # Cutflow
        self.cut_flow = {
            "All": 0,
            "HLT_MET": 0,
            "Filters": 0,
            "LeptonVeto": 0,
            ">=3jets": 0,
            ">=1btag": 0,
            "MET>=250": 0,
            "minDPhi>0.4": 0,
        }
    
    # ---------- helpers ----------
    def _pass_met_hlt(self, events):
        """Apply MET-based HLT paths."""
        hlt = getattr(events, "HLT", None)
        if hlt is None:
            return ak.ones_like(events.event, dtype=bool)
        
        hlt_paths = [
            "PFMETNoMu120",
            "PFMETNoMu90_PFMHTNoMu90_IDTight",
            "PFMETNoMu110_PFMHTNoMu110_IDTight",
            "PFMETNoMu120_PFMHTNoMu120_IDTight",
            "PFMET120_PFMHT120",
            "PFMET110_PFMHT110_IDTight",
            "PFMET120_PFMHT120_IDTight",
            "PFMET170",
        ]
        
        masks = []
        for name in hlt_paths:
            if hasattr(hlt, name):
                masks.append(ak.values_astype(getattr(hlt, name), bool))
        
        if not masks:
            return ak.ones_like(events.event, dtype=bool)
        
        # Logical OR of all triggers
        combined = masks[0]
        for m in masks[1:]:
            combined = combined | m
        return combined
    
    def _pass_event_filters(self, events):
        """Apply event cleaning flags."""
        flag = getattr(events, "Flag", None)
        if flag is None:
            return ak.ones_like(events.event, dtype=bool)
        
        required_flags = [
            "HBHENoiseFilter",
            "HBHENoiseIsoFilter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "globalSuperTightHalo2016Filter",
            "BadPFMuonFilter",
        ]
        
        mask = ak.ones_like(events.event, dtype=bool)
        for fname in required_flags:
            if hasattr(flag, fname):
                mask = mask & ak.values_astype(getattr(flag, fname), bool)
        
        return mask
    
    def _select_veto_electrons(self, events):
        """Veto loose electrons: pt > 10, |eta| < 2.5."""
        el = events.Electron
        if hasattr(el, "cutBased"):
            return el[(el.pt > 10) & (abs(el.eta) < 2.5) & (el.cutBased >= 1)]
        return el[(el.pt > 10) & (abs(el.eta) < 2.5)]
    
    def _select_veto_muons(self, events):
        """Veto loose muons: pt > 10, |eta| < 2.4."""
        mu = events.Muon
        if hasattr(mu, "looseId"):
            return mu[(mu.pt > 10) & (abs(mu.eta) < 2.4) & (mu.looseId == True)]
        return mu[(mu.pt > 10) & (abs(mu.eta) < 2.4)]
    
    def _select_good_jets(self, events):
        """Jets: pt > 30, |eta| < 2.4, jetId >= 2."""
        jets = events.Jet
        jetid = getattr(jets, "jetId", ak.zeros_like(jets.pt, dtype=np.int32) + 2)
        return jets[(jets.pt > 30) & (abs(jets.eta) < 2.4) & (jetid >= 2)]
    
    def _count_bjets(self, jets):
        """Count b-tagged jets (DeepFlav or DeepCSV medium WP)."""
        if hasattr(jets, "btagDeepFlavB"):
            mask = jets.btagDeepFlavB > 0.3093  # medium WP
        elif hasattr(jets, "btagDeepB"):
            mask = jets.btagDeepB > 0.6321      # medium WP
        else:
            mask = ak.zeros_like(jets.pt, dtype=bool)
        return ak.sum(mask, axis=1), mask
    
    def _compute_min_dphi(self, jets, met):
        """Compute min delta phi between leading 2 jets and MET."""
        if ak.any(ak.num(jets) >= 2):
            jet1 = jets[:, 0]
            jet2 = jets[:, 1]
            dphi1 = abs(jet1.phi - met.phi)
            dphi2 = abs(jet2.phi - met.phi)
            dphi1 = ak.where(dphi1 > np.pi, 2*np.pi - dphi1, dphi1)
            dphi2 = ak.where(dphi2 > np.pi, 2*np.pi - dphi2, dphi2)
            min_dphi = ak.where(dphi1 < dphi2, dphi1, dphi2)
        else:
            min_dphi = ak.zeros_like(met.pt) + 999.0
        return min_dphi
    
    def _compute_ht(self, jets):
        """Compute HT = scalar sum of jet pT."""
        return ak.sum(jets.pt, axis=1)
    
    def _compute_mbt(self, bjets, met):
        """Compute transverse mass between leading b-jet and MET."""
        if ak.any(ak.num(bjets) >= 1):
            bjet_lead = bjets[:, 0]
            mbt = np.sqrt(2.0 * bjet_lead.pt * met.pt * 
                         (1.0 - np.cos(bjet_lead.phi - met.phi)))
        else:
            mbt = ak.zeros_like(met.pt)
        return mbt
    
    def _weights(self, events):
        """Weights disabled: return 1 for each event."""
        return np.ones(len(events), dtype="float64")
    
    def _labels_array(self, label, n):
        return np.array([label] * int(n), dtype=object)
    
    def _fill_event_hist(self, H, var_np, process, variation, w_np):
        H.fill(
            var=var_np,
            process=self._labels_array(process, len(var_np)),
            variation=self._labels_array(variation, len(var_np)),
            weight=w_np
        )
    
    # ---------- main pipeline ----------
    def process(self, events):
        process = events.metadata.get("process", "unknown")
        variation = events.metadata.get("variation", "nominal")
        
        w_evt = self._weights(events)
        self.cut_flow["All"] += len(events)
        
        # HLT
        hltmask = self._pass_met_hlt(events)
        events = events[hltmask]
        w_evt = w_evt[ak.to_numpy(hltmask)]
```python
        self.cut_flow["HLT_MET"] += len(events)
        
        # Event filters
        filtermask = self._pass_event_filters(events)
        events = events[filtermask]
        w_evt = w_evt[ak.to_numpy(filtermask)]
        self.cut_flow["Filters"] += len(events)
        
        # Lepton veto (no veto electrons, no veto muons)
        el_veto = self._select_veto_electrons(events)
        mu_veto = self._select_veto_muons(events)
        mask_lep = (ak.num(el_veto) == 0) & (ak.num(mu_veto) == 0)
        events = events[mask_lep]
        w_evt = w_evt[ak.to_numpy(mask_lep)]
        self.cut_flow["LeptonVeto"] += len(events)
        
        # Jets
        jets = self._select_good_jets(events)
        nj = ak.num(jets)
        
        # >= 3 jets
        jets_ok = (nj >= 3)
        events = events[jets_ok]
        jets = jets[jets_ok]
        nj = nj[jets_ok]
        w_evt = w_evt[ak.to_numpy(jets_ok)]
        self.cut_flow[">=3jets"] += len(events)
        
        if len(events) == 0:
            return {"nevents": {process: 0}, "hists": self.h}
        
        # b-jets
        nb, btag_mask = self._count_bjets(jets)
        bjets = jets[btag_mask]
        
        # >= 1 b-tag
        btag_ok = (nb >= 1)
        events = events[btag_ok]
        jets = jets[btag_ok]
        bjets = bjets[btag_ok]
        nj = nj[btag_ok]
        nb = nb[btag_ok]
        w_evt = w_evt[ak.to_numpy(btag_ok)]
        self.cut_flow[">=1btag"] += len(events)
        
        if len(events) == 0:
            return {"nevents": {process: 0}, "hists": self.h}
        
        # MET
        met = getattr(events, "MET", None)
        met_pt = met.pt if (met is not None and hasattr(met, "pt")) else ak.zeros_like(events.event, dtype=float)
        
        # MET >= 250 GeV
        met_ok = (met_pt >= 250)
        events = events[met_ok]
        jets = jets[met_ok]
        bjets = bjets[met_ok]
        nj = nj[met_ok]
        nb = nb[met_ok]
        met_pt = met_pt[met_ok]
        met = met[met_ok]
        w_evt = w_evt[ak.to_numpy(met_ok)]
        self.cut_flow["MET>=250"] += len(events)
        
        if len(events) == 0:
            return {"nevents": {process: 0}, "hists": self.h}
        
        # min delta phi
        min_dphi = self._compute_min_dphi(jets, met)
        
        # min delta phi > 0.4
        dphi_ok = (min_dphi > 0.4)
        events = events[dphi_ok]
        jets = jets[dphi_ok]
        bjets = bjets[dphi_ok]
        nj = nj[dphi_ok]
        nb = nb[dphi_ok]
        met_pt = met_pt[dphi_ok]
        met = met[dphi_ok]
        min_dphi = min_dphi[dphi_ok]
        w_evt = w_evt[ak.to_numpy(dphi_ok)]
        self.cut_flow["minDPhi>0.4"] += len(events)
        
        if len(events) == 0:
            return {"nevents": {process: 0}, "hists": self.h}
        
        # Additional kinematic variables
        ht = self._compute_ht(jets)
        mbt = self._compute_mbt(bjets, met)
        
        # Fill histograms
        w_evt_np = ak.to_numpy(w_evt)
        self._fill_event_hist(self.h['njets'], ak.to_numpy(nj), process, variation, w_evt_np)
        self._fill_event_hist(self.h['nbjets'], ak.to_numpy(nb), process, variation, w_evt_np)
        self._fill_event_hist(self.h['met'], ak.to_numpy(met_pt), process, variation, w_evt_np)
        self._fill_event_hist(self.h['min_dphi'], ak.to_numpy(min_dphi), process, variation, w_evt_np)
        self._fill_event_hist(self.h['ht'], ak.to_numpy(ht), process, variation, w_evt_np)
        self._fill_event_hist(self.h['mbt'], ak.to_numpy(mbt), process, variation, w_evt_np)
        
        # Return both histos and filtered events
        return {
            "nevents": {process: int(len(w_evt_np))},
            "hists": self.h,
            "selected_events": events,
            "jets": jets,
            "bjets": bjets,
            "njets": nj,
            "nbjets": nb,
            "met": met_pt,
            "min_dphi": min_dphi,
            "ht": ht,
            "mbt": mbt,
        }
    
    def postprocess(self, accumulator):
        return accumulator
```

---

# Running the Analysis

Now we run the main loop:

1. Iterate over each dataset in the fileset.
2. Read events from ROOT files with `events_from_file`.
3. Apply the physics cuts with `analysis.process`.
4. Save results:
   • Per-dataset CSVs with selected event variables.
   • Histograms (pickled) for plotting later.
   • Cutflow table to verify event selection efficiency.
   • Event counts for normalization.

This is the driver stage where the analysis is executed.

```python
from pathlib import Path
import pandas as pd
import awkward as ak
import pickle, time
import numpy as np

OUT_DIR = Path("./outputs_ah")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_HISTOS = OUT_DIR / "dm_histograms_ah.pkl"
OUT_CUTFLOW = OUT_DIR / "dm_cutflow_ah.csv"
OUT_NEVENTS = OUT_DIR / "dm_nevents_ah.csv"

analysis = DMAnalysisAllHadronic(DATASET="MET", lumi_fb=LUMI_FB)

entries_total = 0
nevents_rows = []

t0 = time.monotonic()

for key, pack in fileset.items():
    flist = pack["files"]
    meta = pack["metadata"]
    
    if not flist:
        continue
    
    events_all = []
    
    for path in flist:
        md = dict(meta)
        
        try:
            ev = events_from_file(path, metadata=md, max_events=MAX_EVENTS_PER_FILE)
        except Exception as e:
            print(f"[warn] skip {path} -> {e}")
            continue
        
        entries_total += len(ev)
        
        # --- run analysis ---
        out = analysis.process(ev)
        nsel = list(out["nevents"].values())[0]
        
        if nsel == 0:
            continue
        
        # --- use filtered events for CSV ---
        ev_sel = out["selected_events"]
        jets = out["jets"]
        bjets = out["bjets"]
        nj = out["njets"]
        nb = out["nbjets"]
        met_pt = out["met"]
        min_dphi = out["min_dphi"]
        ht = out["ht"]
        mbt = out["mbt"]
        
        df_ev = pd.DataFrame({
            "process": [md["process"]]*len(ev_sel),
            "dataset": [md["dataset"]]*len(ev_sel),
            "njets": ak.to_numpy(nj),
            "nbjets": ak.to_numpy(nb),
            "met": ak.to_numpy(met_pt),
            "min_dphi": ak.to_numpy(min_dphi),
            "ht": ak.to_numpy(ht),
            "mbt": ak.to_numpy(mbt),
        })
        
        events_all.append(df_ev)
        
        nevents_rows.append({
            "key": key,
            "process": md["process"],
            "dataset": md["dataset"],
            "file": path,
            "selected_events": int(len(df_ev)),
            "entries_in_file": int(len(ev)),
            "xsec": md.get("xsec", None),
        })
    
    # Export CSV for this dataset
    if events_all:
        df_all = pd.concat(events_all, ignore_index=True)
        outfile = OUT_DIR / f"{key}_processed.csv"
        df_all.to_csv(outfile, index=False)
        print(f"[OK] Wrote {len(df_all)} events -> {outfile}")

# Save global results
elapsed = time.monotonic() - t0

with open(OUT_HISTOS, "wb") as f:
    pickle.dump(analysis.h, f, protocol=pickle.HIGHEST_PROTOCOL)

pd.DataFrame(list(analysis.cut_flow.items()), columns=["cut","events"]).to_csv(OUT_CUTFLOW, index=False)
pd.DataFrame(nevents_rows).to_csv(OUT_NEVENTS, index=False)

print(f"\n[OK] Total entries processed: {entries_total}")
print(f"[OK] Global cutflow:\n{analysis.cut_flow}")
print(f"[timing] Elapsed: {elapsed:.2f} s")
```

---

# Visualization: Raw Event Counts

```python
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import mplhep as hep

# Apply CMS style
plt.style.use(hep.style.CMS)

# Load histograms and nevents
df_nevents = pd.read_csv(OUT_NEVENTS)

with open(OUT_HISTOS, "rb") as f:
    hdict = pickle.load(f)

def plot_all_variables_raw(hdict, lumi_fb):
    """
    Plot all variables without any normalization:
    - MET as data (points with error bars).
    - All other datasets as MC (stacked histograms).
    """
    variables = list(hdict.keys())
    print("Available variables:", variables)
    
    for var in variables:
        h = hdict[var]
        processes = list(h.axes["process"])
        edges = h.axes["var"].edges
        centers = 0.5*(edges[:-1] + edges[1:])
        width = np.diff(edges)
        
        # Explicitly define data vs MC
        data_procs = ["MET"]
        mc_procs = [p for p in processes if p not in data_procs]
        
        plt.figure(figsize=(7,5))
        bottom = np.zeros(len(edges)-1)
        
        # --- MC stacked ---
        for proc in mc_procs:
            vals = h[{"process": proc}].values().ravel()
            if np.any(vals):
                plt.bar(edges[:-1], vals, width=width, bottom=bottom,
                       align="edge", alpha=0.7, label=proc)
                bottom += vals
        
        # --- Data as points ---
        for proc in data_procs:
            if proc in processes:
                vals = h[{"process": proc}].values().ravel()
                if np.any(vals):
                    yerr = np.sqrt(vals)
                    plt.errorbar(centers, vals, yerr=yerr,
                               fmt="o", color="black", label=proc)
        
        plt.xlabel(var)
        plt.ylabel("Raw events")
        plt.legend(fontsize=8, frameon=False)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()

# === Run ===
plot_all_variables_raw(hdict, LUMI_FB)
```

---

# Cross-Sections and Normalization

## Why Normalization is Necessary

When comparing data and Monte Carlo (MC) simulations, the raw event counts are not directly comparable:

• **Data:**
  - Events are collected with a detector during a given time period.
  - The "size" of the dataset is controlled by the integrated luminosity (L).
  - Example: UL2016 MET dataset corresponds to ~35.9 fb⁻¹.
  - There is no cross section attached — it is just what was recorded.

• **MC (simulations):**
  - Each simulated dataset corresponds to a particular physics process (e.g. $t\bar{t}$, W+jets).
  - Generators simulate a finite number of events (N_gen) with a known theoretical cross section (σ).
  - By construction, MC samples may represent more or fewer events than would be seen in real data.
  - Therefore, they must be scaled.

## The Normalization Formula

To make MC comparable to data, we apply a per-event weight:

$$ w = \frac{\sigma \cdot L}{N_\text{gen}} $$

Where:

• σ (pb) = process cross section (from theory/measurements).
• L (fb⁻¹) = integrated luminosity of the data sample.
  - Convert to pb⁻¹ by multiplying by 1000.
• N_gen = total number of generated MC events (before cuts).

This weight ensures that when we sum the MC events after applying cuts, the histograms reflect the expected yield in the same luminosity as the data.

## Step 1 — Define Cross-Sections

For each simulated process (MC), we need the theoretical cross section (σ) in pb. This will later be combined with the luminosity (L) and the number of generated events (N_gen) to normalize MC histograms.

```python
# ================================
# Cross-sections (pb) — official values
# ================================
XSEC_PB = {
    # Data-like (no cross section)
    ("MET", None): None,
    ("SingleMuon", None): None,
    ("SingleElectron", None): None,
    
    # ttbar
    ("ttbar-semileptonic", None): 831.76,
    ("ttbar-hadronic", None): 831.76,
    
    # single top
    ("t-channel-top", None): 136.0,
    ("t-channel-antitop", None): 81.0,
    ("tW-top", None): 71.7,
    
    # ttV
    ("ttW", None): 0.204,
    ("ttZ", None): 0.252,
    
    # W+jets
    ("WJets-HT400to600", None): 48.9,
    ("WJets-2J-FxFx", None): 615.7,
    
    # DY+jets
    ("DYJets-inclusive", None): 6025.0,
    ("DYJets-Zpt200", None): 1.27,
    
    # Diboson + Z→νν
    ("Zvv", None): 77.3,
    ("WW", None): 118.7,
    ("ZZ", None): 16.6,
}

def get_xsec(proc: str, subgroup: str | None = None):
    """
    Return cross-section (pb) for (proc, subgroup).
    Data-like samples (MET, SingleMuon, SingleElectron) return None.
    """
    key = (proc, subgroup)
    if key in XSEC_PB:
        return XSEC_PB[key]
    return XSEC_PB.get((proc, None), None)

print("=== Cross-sections (pb) ===")
for (proc, subgroup), xsec in XSEC_PB.items():
    name = proc if subgroup is None else f"{proc} ({subgroup})"
    print(f"{name:20s} : {xsec}")
```

## Step 2 — Count Generated Events

We need to know how many events were generated (N_gen) for each dataset. This is critical to compute normalization factors.

```python
import uproot

def count_events_one_file(root_path):
    try:
        with uproot.open(root_path) as f:
            return f["Events"].num_entries
    except Exception as e:
        print(f"[warn] could not open {root_path} -> {e}")
        return 0

# Dictionary to store counts
events_count = {}

for key, pack in fileset.items():
    fpaths = pack["files"][:N_FILES_MAX_PER_SAMPLE]
    if not fpaths:
        continue
    
    total_events = 0
    file_events = []
    
    for path in fpaths:
        nevts = count_events_one_file(path)
        total_events += nevts
        file_events.append((path, nevts))
        print(f"{key}: {path}, events={nevts}")
    
    events_count[key] = {
        "files": fpaths,
        "file_events": file_events,
        "total_events": total_events,
    }
    
    print(f"--> {key}: total_events={total_events}\n")

# --- Final summary ---
print("\nSummary (per dataset):")
grand_total = 0
for k, v in events_count.items():
    print(f"  {k:25s} → total_events={v['total_events']} (from {len(v['files'])} file(s))")
    grand_total += v["total_events"]

print(f"\n[OK] Grand total across datasets = {grand_total}")
```

---

# Normalized Plots

```python
# ================================
# Normalization + CMS-style plots
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Set your analysis luminosity (fb^-1)
LUMI_FB = 0.01

# Load histograms
with open(OUT_HISTOS, "rb") as f:
    hdict = pickle.load(f)

# Explicit N_gen dictionary (truth-level generated events)
N_EVENTS_GEN = {
    "MET": 1654969,
    "SingleMuon": 14113,
    "SingleElectron": 2338304,
    "ttbar-semileptonic": 1233000,
    "ttbar-hadronic": 1344000,
    "t-channel-top": 168000,
    "ttW": 9451,
    "WJets-HT400to600": 41364,
    "WJets-2J-FxFx": 1766801,
    "DYJets-Zpt200": 9748,
    "WW": 2016000,
    "ZZ": 4000,
    "Zvv": 932,
}

def get_neq_gen(proc: str) -> int|None:
    """N_gen = from fixed dictionary."""
    return N_EVENTS_GEN.get(proc, None)

def norm_factor(proc: str, lumi_fb: float) -> float:
    if proc in ("SingleMuon", "SingleElectron", "MET"):
        return 1.0
    
    xsec = get_xsec(proc)
    ngen = get_neq_gen(proc)
    
    if (xsec is None) or (ngen is None) or (ngen <= 0):
        return 1.0
    
    return (xsec * lumi_fb * 1e3) / float(ngen)

def build_norm_report(hdict, lumi_fb):
    some_var = next(iter(hdict.keys()))
    processes = list(hdict[some_var].axes["process"])
    
    rows = []
    for proc in processes:
        xsec = get_xsec(proc)
        ngen = get_neq_gen(proc)
        factor = norm_factor(proc, lumi_fb)
        
        rows.append({
            "process": proc,
            "xsec_pb": xsec,
            "lumi_fb": lumi_fb if xsec is not None else None,
            "N_gen_used": ngen,
            "scale_factor": factor,
            "is_data": proc in ("SingleMuon", "SingleElectron", "MET"),
        })
    
    return pd.DataFrame(rows)

def plot_all_variables_normalized(hdict, lumi_fb):
    rep = build_norm_report(hdict, lumi_fb)
    print("\n=== Normalization report (All-Hadronic Channel) ===")
    print(rep.to_string(index=False))
    
    variables = list(hdict.keys())
    
    for var in variables:
        h = hdict[var]
        processes = list(h.axes["process"])
        edges = h.axes["var"].edges
        centers = 0.5*(edges[:-1] + edges[1:])
        width = np.diff(edges)
        
        data_procs = [p for p in processes if p in ("MET", "SingleMuon", "SingleElectron")]
        mc_procs = [p for p in processes if p not in data_procs]
        
        plt.figure(figsize=(7,5))
        bottom = np.zeros(len(edges)-1)
        
        for proc in mc_procs:
            vals = h[{"process": proc}].values().ravel()
            vals *= norm_factor(proc, lumi_fb)
            if np.any(vals):
                plt.bar(edges[:-1], vals, width=width, bottom=bottom,
                       align="edge", alpha=0.7, label=proc)
                bottom += vals
        
        for proc in data_procs:
            vals = h[{"process": proc}].values().ravel()
            if np.any(vals):
                yerr = np.sqrt(vals)
                plt.errorbar(centers, vals, yerr=yerr, fmt="o", color="black", label=proc)
        
        plt.xlabel(var)
        plt.ylabel("Events (MC scaled, Data raw)")
        plt.legend(fontsize=8, frameon=False)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()

# === Run ===
plot_all_variables_normalized(hdict, LUMI_FB)
```

---

This completes the all-hadronic channel analysis workflow with proper event selection, MET-based triggers, lepton veto, angular cuts to suppress QCD, and normalized MC-to-data comparisons.
![](https://cernbox-codimd.web.cern.ch/uploads/upload_e6d2ccd0232ead0e9dace280397c1025.png)
![](https://cernbox-codimd.web.cern.ch/uploads/upload_2bbea8e7d9227f60d5ec884e8b46ce5d.png)
![](https://cernbox-codimd.web.cern.ch/uploads/upload_eee2846d99d4ce904bb97e28bd589184.png)


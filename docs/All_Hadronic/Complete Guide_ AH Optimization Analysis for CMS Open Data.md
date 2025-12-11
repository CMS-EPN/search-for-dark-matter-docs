# Complete Guide: AH Optimization Analysis for CMS Open Data

## Overview
This guide provides step-by-step instructions to generate the All-Hadronic (AH) optimization plots using CMS Open Data. The analysis processes TTToSemiLeptonic Monte Carlo events to produce distributions of key kinematic variables for signal region optimization.
# Physics Motivation and Channel Strategy: All-Hadronic Channel

The Large Hadron Collider (LHC) collides protons at center-of-mass energies high enough to probe physics beyond the Standard Model. In the context of simplified dark matter models, partonic interactions can produce top quarks together with a new mediator particle that decays invisibly into dark matter candidates (χχ̄). At the detector level, this results in events with multiple top quarks plus significant missing transverse momentum (p_T^miss).

The production mechanisms of interest include:
- **Gluon fusion:** $$gg → tt̄φ → tt̄ + χχ̄ $$
- **Single top associated production:** $$gb → tφ → t + χχ̄  $$
- **t-channel production:** $$qq' → tbφ → tb + χχ̄ $$

In all cases, the top quarks decay via $t \to W b$. In the **all-hadronic (AH) channel**, both W bosons decay hadronically $(W → qq̄')$, resulting in a fully hadronic final state with no isolated leptons.

## Channel Characteristics

The all-hadronic channel is defined by:
- **No isolated leptons** (lepton veto applied)
- **Multiple jets** (≥3 jets with ≥1 b-tagged jet)
- **Large missing transverse energy** from dark matter particles and detector resolution

**Advantages:**
- Highest branching fraction (~46% for tt̄ → all hadronic)
- Largest raw event yield
- Sensitive to highly boosted topologies

**Challenges:**
- Overwhelming QCD multijet background
- Instrumental MET from jet mismeasurements
- Requires sophisticated background estimation techniques

The all-hadronic channel complements lepton channels by providing additional sensitivity in high-MET regions where QCD background can be controlled through kinematic selections.
## Prerequisites

### 1. System Requirements
- Python 3.8 or higher
- 8+ GB RAM recommended
- Stable internet connection (for XRootD access)
- 5+ GB free disk space

### 2. Python Package Installation
```bash
# Create and activate virtual environment (optional but recommended)
python -m venv cms_analysis
source cms_analysis/bin/activate  # On Windows: cms_analysis\Scripts\activate

# Install required packages
pip install uproot awkward numpy matplotlib
```
## Selection Criteria

### 1. Trigger Requirements (HLT)
Events must fire at least one of the following MET-based triggers:
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

### 2. Event Cleaning Flags
**Applied to both data and MC:**
- HBHENoiseFilter
- HBHENoiseIsoFilter  
- ECALDeadCellFilter
- GlobalTightHalo2016Filter
- BadPFMuonFilter
- BadChargedHadronFilter

**Applied to data only:**
- EEBadScFilter

### 3. Lepton Veto
**No veto leptons allowed:**
- **Electrons:** No electrons with pT > 10 GeV, |η| < 2.5
- **Muons:** No muons with pT > 10 GeV, |η| < 2.4

### 4. Jet Selection
- **pT:** > 30 GeV
- **|η|:** < 2.4 (central jets)
- **Jet ID:** Loose working point
- **Overlap removal:** Jets within ΔR < 0.4 of tight leptons are removed

### 5. Event-Level Requirements
- **Number of jets:** ≥ 3
- **b-tagged jets:** ≥ 1 (DeepCSV medium WP: 0.6321)
- **Missing ET:** pT^miss ≥ 250 GeV
- **Δφ(jet1,2, MET):** > 0.4 (baseline), > 1.0 (optimized)

### 6. Additional Kinematic Variables (Optimized Selection)
- **Transverse bottom mass:** M_bT > 180 GeV
- **Jet fraction:** pT(j1)/HT ≤ 0.5 (for n_b ≥ 2 category)

## Step-by-Step Analysis

### Step 1: Create the Analysis Script

Create a file named `ah_optimization_analysis.py` with the following complete code:

```python
import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from matplotlib import rcParams
import time
import math

# Set plot style for publication quality
rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (15, 10),
    'figure.autolayout': True
})
plt.style.use('seaborn-v0_8-whitegrid')

class AHOptimizationAnalyzer:
    """Analyzer for All-Hadronic optimization plots"""
    
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.events = None
        self.jet_info = None
        
        # Selection parameters for AH baseline
        self.params = {
            'jet_pt_min': 30,
            'jet_eta_max': 2.4,
            'bjet_wp': 0.2783,  # DeepJet medium working point
            'met_min': 30,
            'min_dphi': 1.0,
            'mt_min': 180,
            'pt_ht_ratio_max': 0.5,
        }
    
    def load_events(self, max_events_per_file=None):
        """Load events from multiple remote files"""
        print("="*70)
        print("LOADING EVENTS FROM XROOTD SERVERS")
        print("="*70)
        
        all_events = []
        total_files = len(self.file_paths)
        
        for i, file_path in enumerate(self.file_paths):
            print(f"\n[{i+1}/{total_files}] Processing: {file_path.split('/')[-1]}")
            
            try:
                start_time = time.time()
                
                # Open remote file
                file = uproot.open(file_path)
                tree = file["Events"]
                
                # Get total entries
                n_entries = tree.num_entries
                print(f"   Total entries: {n_entries:,}")
                
                # Determine entries to read
                if max_events_per_file and max_events_per_file < n_entries:
                    entries_to_read = max_events_per_file
                else:
                    entries_to_read = n_entries
                
                # Read essential branches
                print(f"   Reading {entries_to_read:,} events...")
                events = tree.arrays([
                    "Jet_pt", "Jet_eta", "Jet_phi", "Jet_btagDeepFlavB",
                    "MET_pt", "MET_phi",
                    "Muon_pt", "Muon_eta", "Muon_tightId", "Muon_pfRelIso04_all",
                    "Electron_pt", "Electron_eta", "Electron_cutBased", "Electron_pfRelIso03_all",
                    "nJet", "nMuon", "nElectron"
                ], entry_stop=entries_to_read, library="ak")
                
                all_events.append(events)
                
                elapsed = time.time() - start_time
                print(f"   ✓ Loaded {len(events):,} events in {elapsed:.1f}s ({len(events)/elapsed:.0f} events/s)")
                
            except Exception as e:
                print(f"   ✗ Error: {e}")
                continue
        
        if all_events:
            self.events = ak.concatenate(all_events)
            print(f"\n✓ SUCCESS: Loaded {len(self.events):,} total events from {len(all_events)} files")
        else:
            raise ValueError("Failed to load any events")
    
    def apply_ah_baseline_selection(self):
        """Apply AH baseline selection (no isolated leptons, kinematic cuts)"""
        print("\n" + "="*70)
        print("APPLYING AH BASELINE SELECTION")
        print("="*70)
        
        # 1. Lepton veto
        print("\n1. Applying lepton veto...")
        lepton_veto = self.apply_lepton_veto()
        print(f"   Events with 0 isolated leptons: {ak.sum(lepton_veto):,} ({ak.sum(lepton_veto)/len(self.events)*100:.1f}%)")
        
        # 2. Jet selection
        print("\n2. Selecting jets...")
        self.jet_info = self.select_jets()
        
        # 3. Calculate kinematic variables
        print("\n3. Calculating kinematic variables...")
        self.calculate_kinematic_variables()
        
        # 4. Apply baseline kinematic cuts
        print("\n4. Applying kinematic cuts...")
        baseline_mask = self.apply_baseline_cuts(lepton_veto)
        
        print(f"\n✓ AH Baseline selection complete")
        print(f"   Events passing baseline: {ak.sum(baseline_mask):,} ({ak.sum(baseline_mask)/len(self.events)*100:.1f}%)")
        
        return baseline_mask
    
    def apply_lepton_veto(self):
        """Veto events with isolated leptons"""
        
        # Initialize counters
        n_iso_muons = ak.zeros_like(self.events.MET_pt, dtype=int)
        n_iso_electrons = ak.zeros_like(self.events.MET_pt, dtype=int)
        
        # Muon selection: pT > 10 GeV, |eta| < 2.4, tight ID, isolation < 0.15
        if 'Muon_pt' in self.events.fields:
            muon_pt = ak.fill_none(self.events.Muon_pt, 0)
            muon_eta = ak.fill_none(self.events.Muon_eta, 100)
            muon_tightId = ak.fill_none(self.events.Muon_tightId, 0)
            muon_iso = ak.fill_none(self.events.Muon_pfRelIso04_all, 100)
            
            muon_mask = (
                (muon_pt > 10) & 
                (np.abs(muon_eta) < 2.4) &
                (muon_tightId == 1) &
                (muon_iso < 0.15)
            )
            n_iso_muons = ak.sum(muon_mask, axis=1)
        
        # Electron selection: pT > 10 GeV, |eta| < 2.5, tight ID, isolation < 0.10
        if 'Electron_pt' in self.events.fields:
            electron_pt = ak.fill_none(self.events.Electron_pt, 0)
            electron_eta = ak.fill_none(self.events.Electron_eta, 100)
            electron_cutBased = ak.fill_none(self.events.Electron_cutBased, 0)
            electron_iso = ak.fill_none(self.events.Electron_pfRelIso03_all, 100)
            
            electron_mask = (
                (electron_pt > 10) & 
                (np.abs(electron_eta) < 2.5) &
                (electron_cutBased >= 3) &  # 3 = Tight, 4 = SuperTight
                (electron_iso < 0.10)
            )
            n_iso_electrons = ak.sum(electron_mask, axis=1)
        
        return (n_iso_muons == 0) & (n_iso_electrons == 0)
    
    def select_jets(self):
        """Select and count jets"""
        
        # Basic jet selection
        jet_pt = ak.fill_none(self.events.Jet_pt, 0)
        jet_eta = ak.fill_none(self.events.Jet_eta, 100)
        jet_phi = ak.fill_none(self.events.Jet_phi, 0)
        
        # Good jets: pT > 30 GeV, |eta| < 2.4
        good_jet_mask = (
            (jet_pt > self.params['jet_pt_min']) &
            (np.abs(jet_eta) < self.params['jet_eta_max'])
        )
        
        # Count jets
        n_jets = ak.sum(good_jet_mask, axis=1)
        
        # Count b-jets (DeepJet medium WP)
        if 'Jet_btagDeepFlavB' in self.events.fields:
            btag = ak.fill_none(self.events.Jet_btagDeepFlavB, -1)
            bjet_mask = good_jet_mask & (btag > self.params['bjet_wp'])
            n_bjets = ak.sum(bjet_mask, axis=1)
        else:
            n_bjets = ak.zeros_like(n_jets)
        
        print(f"   Events with ≥4 jets: {ak.sum(n_jets >= 4):,} ({ak.sum(n_jets >= 4)/len(self.events)*100:.1f}%)")
        print(f"   Events with ≥1 b-jets: {ak.sum(n_bjets >= 1):,} ({ak.sum(n_bjets >= 1)/len(self.events)*100:.1f}%)")
        print(f"   Events with ≥2 b-jets: {ak.sum(n_bjets >= 2):,} ({ak.sum(n_bjets >= 2)/len(self.events)*100:.1f}%)")
        
        return {
            'n_jets': n_jets,
            'n_bjets': n_bjets,
            'good_jet_mask': good_jet_mask,
            'jet_pt': jet_pt,
            'jet_eta': jet_eta,
            'jet_phi': jet_phi
        }
    
    def calculate_kinematic_variables(self):
        """Calculate minΔφ, MT, HT, and pT/HT ratio"""
        
        print("   Calculating minΔφ(j1,2, MET)...")
        print("   Calculating MT...")
        print("   Calculating HT...")
        print("   Calculating pT/HT ratio...")
        
        # Get jet information
        good_jet_mask = self.jet_info['good_jet_mask']
        jet_pt = self.jet_info['jet_pt'][good_jet_mask]
        jet_phi = self.jet_info['jet_phi'][good_jet_mask]
        
        # Calculate HT (scalar sum of jet pT)
        self.HT = ak.sum(jet_pt, axis=1)
        
        # Initialize arrays
        n_events = len(self.events)
        self.min_dphi = np.zeros(n_events)
        self.MT = np.zeros(n_events)
        self.pt_ht_ratio = np.zeros(n_events)
        
        # Convert to numpy for faster processing
        met_pt = ak.to_numpy(self.events.MET_pt)
        met_phi = ak.to_numpy(self.events.MET_phi)
        n_jets = ak.to_numpy(self.jet_info['n_jets'])
        
        # Process each event
        for i in range(n_events):
            if n_jets[i] >= 2:
                # Get jets for this event
                event_jets_pt = ak.to_numpy(jet_pt[i])
                event_jets_phi = ak.to_numpy(jet_phi[i])
                
                if len(event_jets_pt) >= 2:
                    # Sort by pT (descending)
                    sorted_idx = np.argsort(event_jets_pt)[::-1]
                    
                    # Get two leading jets
                    jet1_phi = event_jets_phi[sorted_idx[0]]
                    jet2_phi = event_jets_phi[sorted_idx[1]]
                    jet1_pt = event_jets_pt[sorted_idx[0]]
                    
                    # Calculate Δφ
                    delta_phi1 = abs(jet1_phi - met_phi[i])
                    delta_phi2 = abs(jet2_phi - met_phi[i])
                    
                    # Normalize to [0, π]
                    if delta_phi1 > math.pi:
                        delta_phi1 = 2*math.pi - delta_phi1
                    if delta_phi2 > math.pi:
                        delta_phi2 = 2*math.pi - delta_phi2
                    
                    # minΔφ
                    self.min_dphi[i] = min(delta_phi1, delta_phi2)
                    
                    # Calculate MT (transverse mass)
                    self.MT[i] = math.sqrt(2 * jet1_pt * met_pt[i] * (1 - math.cos(delta_phi1)))
                    
                    # Calculate pT/HT ratio
                    if self.HT[i] > 0:
                        self.pt_ht_ratio[i] = jet1_pt / self.HT[i]
        
        print(f"   ✓ Variables calculated for {n_events:,} events")
    
    def apply_baseline_cuts(self, lepton_veto):
        """Apply AH baseline cuts"""
        
        n_jets = self.jet_info['n_jets']
        
        # Basic AH baseline selection
        baseline_mask = (
            lepton_veto &
            (n_jets >= 4) &
            (self.events.MET_pt > self.params['met_min'])
        )
        
        return baseline_mask
    
    def get_category_masks(self, baseline_mask):
        """Get masks for nb = 1 and nb ≥ 2 categories"""
        
        n_bjets = self.jet_info['n_bjets']
        
        # Events in baseline selection
        in_baseline = baseline_mask
        
        # Category masks
        nb_eq_1_mask = in_baseline & (n_bjets == 1)
        nb_ge_2_mask = in_baseline & (n_bjets >= 2)
        
        print(f"\nCategory statistics after baseline:")
        print(f"   nb = 1 events: {ak.sum(nb_eq_1_mask):,}")
        print(f"   nb ≥ 2 events: {ak.sum(nb_ge_2_mask):,}")
        
        return nb_eq_1_mask, nb_ge_2_mask
    
    def plot_optimization_distributions(self, nb_eq_1_mask, nb_ge_2_mask):
        """
        Create the 5 optimization plots as shown in the paper
        """
        print("\n" + "="*70)
        print("GENERATING AH OPTIMIZATION PLOTS")
        print("="*70)
        
        # Create figure with 2x3 layout (5 plots + 1 empty)
        fig = plt.figure(figsize=(16, 12))
        
        # Define subplot positions
        ax1 = plt.subplot(2, 3, 1)  # (a) minΔφ for nb = 1
        ax2 = plt.subplot(2, 3, 2)  # (b) MT for nb = 1
        ax3 = plt.subplot(2, 3, 4)  # (c) minΔφ for nb ≥ 2
        ax4 = plt.subplot(2, 3, 5)  # (d) MT for nb ≥ 2
        ax5 = plt.subplot(2, 3, 6)  # (e) pT/HT for nb ≥ 2
        
        # Top-right subplot is empty for title
        ax_empty = plt.subplot(2, 3, 3)
        ax_empty.axis('off')
        
        # Get data for each category
        min_dphi_nb1 = self.min_dphi[ak.to_numpy(nb_eq_1_mask)]
        mt_nb1 = self.MT[ak.to_numpy(nb_eq_1_mask)]
        
        min_dphi_nb2 = self.min_dphi[ak.to_numpy(nb_ge_2_mask)]
        mt_nb2 = self.MT[ak.to_numpy(nb_ge_2_mask)]
        pt_ht_nb2 = self.pt_ht_ratio[ak.to_numpy(nb_ge_2_mask)]
        
        # Filter out zeros
        min_dphi_nb1 = min_dphi_nb1[min_dphi_nb1 > 0]
        mt_nb1 = mt_nb1[mt_nb1 > 0]
        min_dphi_nb2 = min_dphi_nb2[min_dphi_nb2 > 0]
        mt_nb2 = mt_nb2[mt_nb2 > 0]
        pt_ht_nb2 = pt_ht_nb2[pt_ht_nb2 > 0]
        
        # Plot (a): minΔφ for nb = 1
        print("Plotting (a) minΔφ for nb = 1...")
        if len(min_dphi_nb1) > 0:
            ax1.hist(min_dphi_nb1, bins=50, range=(0, 3.2),
                    histtype='step', linewidth=2, color='blue',
                    density=True)
            ax1.axvline(self.params['min_dphi'], color='red', linestyle='--',
                       linewidth=1.5, alpha=0.8, label=f'Cut: >{self.params["min_dphi"]} rad')
            ax1.legend(loc='upper right')
        
        ax1.set_xlabel(r'min$\Delta\phi(j_{1,2}, p_T^{miss})$ [rad]')
        ax1.set_ylabel('Normalized Events')
        ax1.set_title(r'(a) min$\Delta\phi(j_{1,2}, p_T^{miss})$ for $n_b = 1$')
        ax1.grid(True, alpha=0.3)
        ax1.text(0.05, 0.95, f'Entries: {len(min_dphi_nb1):,}',
                transform=ax1.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot (b): MT for nb = 1
        print("Plotting (b) MT for nb = 1...")
        if len(mt_nb1) > 0:
            ax2.hist(mt_nb1, bins=50, range=(0, 500),
                    histtype='step', linewidth=2, color='green',
                    density=True)
            ax2.axvline(self.params['mt_min'], color='red', linestyle='--',
                       linewidth=1.5, alpha=0.8, label=f'Cut: >{self.params["mt_min"]} GeV')
            ax2.legend(loc='upper right')
        
        ax2.set_xlabel(r'$M_T$ [GeV]')
        ax2.set_ylabel('Normalized Events')
        ax2.set_title(r'(b) $M_T$ for $n_b = 1$')
        ax2.grid(True, alpha=0.3)
        ax2.text(0.05, 0.95, f'Entries: {len(mt_nb1):,}',
                transform=ax2.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot (c): minΔφ for nb ≥ 2
        print("Plotting (c) minΔφ for nb ≥ 2...")
        if len(min_dphi_nb2) > 0:
            ax3.hist(min_dphi_nb2, bins=50, range=(0, 3.2),
                    histtype='step', linewidth=2, color='red',
                    density=True)
            ax3.axvline(self.params['min_dphi'], color='blue', linestyle='--',
                       linewidth=1.5, alpha=0.8, label=f'Cut: >{self.params["min_dphi"]} rad')
            ax3.legend(loc='upper right')
        
        ax3.set_xlabel(r'min$\Delta\phi(j_{1,2}, p_T^{miss})$ [rad]')
        ax3.set_ylabel('Normalized Events')
        ax3.set_title(r'(c) min$\Delta\phi(j_{1,2}, p_T^{miss})$ for $n_b \geq 2$')
        ax3.grid(True, alpha=0.3)
        ax3.text(0.05, 0.95, f'Entries: {len(min_dphi_nb2):,}',
                transform=ax3.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot (d): MT for nb ≥ 2
        print("Plotting (d) MT for nb ≥ 2...")
        if len(mt_nb2) > 0:
            ax4.hist(mt_nb2, bins=50, range=(0, 500),
                    histtype='step', linewidth=2, color='purple',
                    density=True)
            ax4.axvline(self.params['mt_min'], color='blue', linestyle='--',
                       linewidth=1.5, alpha=0.8, label=f'Cut: >{self.params["mt_min"]} GeV')
            ax4.legend(loc='upper right')
        
        ax4.set_xlabel(r'$M_T$ [GeV]')
        ax4.set_ylabel('Normalized Events')
        ax4.set_title(r'(d) $M_T$ for $n_b \geq 2$')
        ax4.grid(True, alpha=0.3)
        ax4.text(0.05, 0.95, f'Entries: {len(mt_nb2):,}',
                transform=ax4.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot (e): pT/HT for nb ≥ 2
        print("Plotting (e) pT/HT for nb ≥ 2...")
        if len(pt_ht_nb2) > 0:
            ax5.hist(pt_ht_nb2, bins=50, range=(0, 1),
                    histtype='step', linewidth=2, color='orange',
                    density=True)
            ax5.axvline(self.params['pt_ht_ratio_max'], color='red', linestyle='--',
                       linewidth=1.5, alpha=0.8, label=f'Cut: <{self.params["pt_ht_ratio_max"]}')
            ax5.legend(loc='upper right')
        
        ax5.set_xlabel(r'$p_T^{jet1} / H_T$')
        ax5.set_ylabel('Normalized Events')
        ax5.set_title(r'(e) $p_T^{jet1} / H_T$ for $n_b \geq 2$')
        ax5.grid(True, alpha=0.3)
        ax5.text(0.05, 0.95, f'Entries: {len(pt_ht_nb2):,}',
                transform=ax5.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Add main title
        ax_empty.text(0.5, 0.5, 'AH Optimization Distributions\n(after baseline selection)\n' +
                     'TTToSemiLeptonic MC\n' +
                     f'Total events: {len(self.events):,}',
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=ax_empty.transAxes,
                     fontsize=14, fontweight='bold')
        
        # Add statistics box
        stats_text = (
            f'AH Baseline Selection:\n'
            f'• 0 isolated leptons\n'
            f'• ≥4 jets (pT > 30 GeV)\n'
            f'• MET > 30 GeV\n'
            f'• nb = 1: {ak.sum(nb_eq_1_mask):,} events\n'
            f'• nb ≥ 2: {ak.sum(nb_ge_2_mask):,} events'
        )
        
        fig.text(0.02, 0.02, stats_text, fontsize=10,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f'ah_optimization_plots_{timestamp}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ Plots saved to: {output_file}")
        
        plt.show()
        
        # Also create summary statistics plot
        self.plot_summary_statistics(nb_eq_1_mask, nb_ge_2_mask)
    
    def plot_summary_statistics(self, nb_eq_1_mask, nb_ge_2_mask):
        """Create additional summary plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Event counts by category
        categories = ['nb = 1', 'nb ≥ 2', 'Total Baseline']
        counts = [
            ak.sum(nb_eq_1_mask),
            ak.sum(nb_ge_2_mask),
            ak.sum(nb_eq_1_mask) + ak.sum(nb_ge_2_mask)
        ]
        
        bars = axes[0, 0].bar(categories, counts, 
                             color=['skyblue', 'lightgreen', 'salmon'],
                             edgecolor='black')
        axes[0, 0].set_ylabel('Number of Events')
        axes[0, 0].set_title('Event Counts by Category')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        for i, (bar, count) in enumerate(zip(bars, counts)):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                          f'{count:,}', ha='center', va='bottom', fontsize=11)
        
        # 2. MET distribution comparison
        if ak.sum(nb_eq_1_mask) > 0:
            met_nb1 = ak.to_numpy(self.events.MET_pt[nb_eq_1_mask])
            axes[0, 1].hist(met_nb1, bins=40, range=(0, 400), histtype='step',
                          linewidth=2, color='blue', density=True, label='nb = 1')
        
        if ak.sum(nb_ge_2_mask) > 0:
            met_nb2 = ak.to_numpy(self.events.MET_pt[nb_ge_2_mask])
            axes[0, 1].hist(met_nb2, bins=40, range=(0, 400), histtype='step',
                          linewidth=2, color='red', density=True, label='nb ≥ 2')
        
        axes[0, 1].set_xlabel(r'$p_T^{miss}$ [GeV]')
        axes[0, 1].set_ylabel('Normalized Events')
        axes[0, 1].set_title(r'$p_T^{miss}$ Distribution by Category')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. HT distribution
        if ak.sum(nb_eq_1_mask) > 0:
            ht_nb1 = self.HT[ak.to_numpy(nb_eq_1_mask)]
            axes[1, 0].hist(ht_nb1, bins=50, range=(0, 1500), histtype='step',
                          linewidth=2, color='blue', density=True, label='nb = 1')
        
        if ak.sum(nb_ge_2_mask) > 0:
            ht_nb2 = self.HT[ak.to_numpy(nb_ge_2_mask)]
            axes[1, 0].hist(ht_nb2, bins=50, range=(0, 1500), histtype='step',
                          linewidth=2, color='red', density=True, label='nb ≥ 2')
        
        axes[1, 0].set_xlabel(r'$H_T$ [GeV]')
        axes[1, 0].set_ylabel('Normalized Events')
        axes[1, 0].set_title(r'$H_T$ Distribution by Category')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Jet multiplicity
        n_jets_nb1 = ak.to_numpy(self.jet_info['n_jets'][nb_eq_1_mask])
        n_jets_nb2 = ak.to_numpy(self.jet_info['n_jets'][nb_ge_2_mask])
        
        axes[1, 1].hist([n_jets_nb1, n_jets_nb2], bins=range(4, 16),
                       histtype='step', linewidth=2, density=True,
                       label=['nb = 1', 'nb ≥ 2'], color=['blue', 'red'])
        axes[1, 1].set_xlabel('Number of Jets')
        axes[1, 1].set_ylabel('Normalized Events')
        axes[1, 1].set_title('Jet Multiplicity by Category')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('AH Analysis Summary - TTToSemiLeptonic MC', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save summary plot
        summary_file = f'ah_summary_plots_{time.strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(summary_file, dpi=150, bbox_inches='tight')
        print(f"✓ Summary plots saved to: {summary_file}")
        
        plt.show()

def main():
    """Main execution function"""
    
    print("="*70)
    print("AH OPTIMIZATION ANALYSIS - CMS OPEN DATA")
    print("="*70)
    
    # All remote file URLs
    file_paths = [
        # Original files
        "root://eospublic.cern.ch//eos/opendata/cms/mc/RunIISummer20UL16NanoAODv9/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1/120000/08FCB2ED-176B-064B-85AB-37B898773B98.root",
        "root://eospublic.cern.ch//eos/opendata/cms/mc/RunIISummer20UL16NanoAODv9/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1/120000/0BD60695-8388-5141-B157-32AE1A3B4885.root",
        "root://eospublic.cern.ch//eos/opendata/cms/mc/RunIISummer20UL16NanoAODv9/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1/120000/4F3C361D-258C-1D41-AEEA-48CB87D3839A.root",
        
        # Additional files
        "root://eospublic.cern.ch//eos/opendata/cms/mc/RunIISummer20UL16NanoAODv9/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1/120000/F6C0248E-6AC1-CE45-BEFA-56A735AA214A.root",
        "root://eospublic.cern.ch//eos/opendata/cms/mc/RunIISummer20UL16NanoAODv9/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1/120000/FA5B9B55-06B4-A640-AF3C-7B44552E2393.root",
        "root://eospublic.cern.ch//eos/opendata/cms/mc/RunIISummer20UL16NanoAODv9/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1/120000/FB5A9307-B677-B947-8970-21DA6BD7C9C2.root",
        "root://eospublic.cern.ch//eos/opendata/cms/mc/RunIISummer20UL16NanoAODv9/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1/120000/1D63950A-E444-E049-BFF0-D33296A8A6CA.root",
        "root://eospublic.cern.ch//eos/opendata/cms/mc/RunIISummer20UL16NanoAODv9/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1/120000/FFA621C8-C16B-5740-AB60-84246D9B2FD1.root",
        "root://eospublic.cern.ch//eos/opendata/cms/mc/RunIISummer20UL16NanoAODv9/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1/120000/2D439FBF-CF8D-654F-93B1-2F7D0A74B0CB.root",
        "root://eospublic.cern.ch//eos/opendata/cms/mc/RunIISummer20UL16NanoAODv9/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1/120000/2E85B521-A37E-0044-8662-BFB0C1291422.root",
        "root://eospublic.cern.ch//eos/opendata/cms/mc/RunIISummer20UL16NanoAODv9/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1/120000/301EA765-5A14-1B43-ADED-D3BE6147134B.root"
    ]
    
    print(f"\n📊 Total files to process: {len(file_paths)}")
    
    try:
        # Initialize analyzer
        analyzer = AHOptimizationAnalyzer(file_paths)
        
        # Load events (limit to 10,000 per file for reasonable runtime)
        print("\n⏳ Loading events (this may take several minutes)...")
        analyzer.load_events(max_events_per_file=10000)
        
        # Apply AH baseline selection
        baseline_mask = analyzer.apply_ah_baseline_selection()
        
        # Get category masks
        nb_eq_1_mask, nb_ge_2_mask = analyzer.get_category_masks(baseline_mask)
        
        # Generate optimization plots
        analyzer.plot_optimization_distributions(nb_eq_1_mask, nb_ge_2_mask)
        
        # Print final summary
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"\n📈 FINAL STATISTICS:")
        print(f"   Total events processed: {len(analyzer.events):,}")
        print(f"   Events in baseline selection: {ak.sum(baseline_mask):,}")
        print(f"   nb = 1 events: {ak.sum(nb_eq_1_mask):,}")
        print(f"   nb ≥ 2 events: {ak.sum(nb_ge_2_mask):,}")
        print(f"\n✅ Plots have been generated and saved to disk")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   1. Check internet connection")
        print("   2. Try with fewer files first")
        print("   3. Reduce max_events_per_file parameter")
        print("   4. Ensure all dependencies are installed:")
        print("      pip install uproot awkward numpy matplotlib")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Start timer
    start_time = time.time()
    
    # Run analysis
    main()
    
    # Print execution time
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total execution time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
```

### Step 2: Run the Analysis

```bash
# Execute the analysis script
python ah_optimization_analysis.py
```

## Expected Output
You will get a result output similar to this:
![](https://cernbox-codimd.web.cern.ch/uploads/upload_90b2d138e817d960112aef95ca20fc9d.png)

### Console Output
The script will display progress messages for each step:

1. **Loading phase**: Shows progress for each file with loading speed
2. **Selection phase**: Displays statistics after each cut
3. **Plotting phase**: Confirms generation of each plot
4. **Final statistics**: Summary of events processed

### Generated Files
Two PNG files will be created with timestamps:

1. **Main optimization plots** (`ah_optimization_plots_YYYYMMDD_HHMMSS.png`):
   - 5 plots in 2×3 layout matching the paper
   - Normalized distributions
   - Cut lines shown
   - Event counts displayed

2. **Summary plots** (`ah_summary_plots_YYYYMMDD_HHMMSS.png`):
   - Event counts by category
   - MET distribution comparison
   - HT distribution comparison
   - Jet multiplicity

## Analysis Parameters

### Selection Criteria
| Parameter | Value | Description |
|-----------|-------|-------------|
| Jet pT | > 30 GeV | Minimum transverse momentum |
| Jet |η| | < 2.4 | Pseudorapidity acceptance |
| b-tagging WP | 0.2783 | DeepJet medium working point |
| MET | > 30 GeV | Missing transverse momentum |
| minΔφ cut | > 1.0 rad | Minimum angle between jets and MET |
| MT cut | > 180 GeV | Transverse mass cut |
| pT/HT cut | < 0.5 | Ratio for nb ≥ 2 category |

### Event Categories
- **nb = 1**: Events with exactly 1 b-tagged jet
- **nb ≥ 2**: Events with 2 or more b-tagged jets

## Troubleshooting

### Common Issues and Solutions

#### 1. Connection Errors
```python
# If XRootD connection fails, try:
# Option A: Reduce number of files
file_paths = file_paths[:3]  # Use only first 3 files

# Option B: Reduce events per file
analyzer.load_events(max_events_per_file=1000)
```

#### 2. Memory Issues
```python
# Reduce memory usage:
# 1. Process fewer events
analyzer.load_events(max_events_per_file=5000)

# 2. Use fewer files initially
file_paths = file_paths[:5]
```

#### 3. Slow Performance
- The analysis processes ~110,000 events from 11 files
- Expected runtime: 10-30 minutes depending on connection
- For faster testing, reduce `max_events_per_file` to 1000

## Understanding the Output Plots
![](https://cernbox-codimd.web.cern.ch/uploads/upload_2686853b00b671b4e6223f000eb9cd9b.png)

### Plot (a): minΔφ(j₁,₂, pₜᵐⁱˢˢ) for nb = 1
- **Purpose**: Show angular separation between leading jets and MET
- **Interpretation**: QCD multijet background tends to have small Δφ
- **Cut**: Events with minΔφ > 1.0 rad are selected

### Plot (b): Mₜ for nb = 1
- **Purpose**: Transverse mass distribution
- **Interpretation**: Signal events have larger MT values
- **Cut**: Events with MT > 180 GeV are selected

### Plot (c): minΔφ(j₁,₂, pₜᵐⁱˢˢ) for nb ≥ 2
- **Same as (a)** but for events with ≥2 b-jets

### Plot (d): Mₜ for nb ≥ 2
- **Same as (b)** but for events with ≥2 b-jets

### Plot (e): pₜʲᵉᵗ¹ / Hₜ for nb ≥ 2
- **Purpose**: Ratio of leading jet pT to total hadronic activity
- **Interpretation**: Signal events have more balanced jet energy distribution
- **Cut**: Events with ratio < 0.5 are selected

## Physics Context

### All-Hadronic Channel Characteristics
- **Branching ratio**: ~46% (highest for tt̄)
- **Background**: Dominated by QCD multijet production
- **Challenges**: MET can be faked by jet mismeasurement
- **Advantages**: Maximum statistical power

### Optimization Strategy
The cuts shown in the plots are optimized to:
1. Suppress QCD background using angular correlations (minΔφ)
2. Enhance signal using transverse mass (MT)
3. Further discriminate signal using jet energy balance (pT/HT)

## Customization Options

### Modify Selection Criteria
```python
# In the __init__ method, modify parameters:
self.params = {
    'jet_pt_min': 40,  # Increase jet pT threshold
    'met_min': 50,     # Increase MET threshold
    'min_dphi': 0.8,   # Relax angular cut
    'mt_min': 200,     # Increase MT cut
}
```

### Add Additional Plots
```python
def plot_additional_variables(self, nb_eq_1_mask, nb_ge_2_mask):
    """Add custom plots"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot b-jet multiplicity
    n_bjets_nb1 = ak.to_numpy(self.jet_info['n_bjets'][nb_eq_1_mask])
    n_bjets_nb2 = ak.to_numpy(self.jet_info['n_bjets'][nb_ge_2_mask])
    
    # Your custom plotting code here
```
![](https://cernbox-codimd.web.cern.ch/uploads/upload_85a2a2294c653087061f8e532bfead2e.png)
![](https://cernbox-codimd.web.cern.ch/uploads/upload_4e87fc17eb4373af623757534d186326.png)
![](https://cernbox-codimd.web.cern.ch/uploads/upload_3743167f816947f599f2068d7fde7699.png)

## Performance Notes

### Expected Resource Usage
- **Memory**: ~2-4 GB for 100,000 events
- **Time**: 10-30 minutes for full analysis
- **Network**: ~500 MB data transfer from CERN servers

### Optimization Tips
1. For development, use fewer files: `file_paths[:3]`
2. For quick tests, reduce events: `max_events_per_file=1000`
3. Save intermediate results to avoid re-downloading

## Citation and References

### Data Source
```bibtex
@misc{cms_opendata_2024,
  title = {Simulated dataset TTToSemiLeptonic in NANOAODSIM format},
  author = {CMS Collaboration},
  year = {2024},
  doi = {10.7483/OPENDATA.CMS.4J3Y.1CME},
  url = {https://opendata.cern.ch/record/67993}
}
```

### Related Papers
- CMS Collaboration, "Search for dark matter produced in association with a single top quark or a top quark pair in proton-proton collisions at √s = 13 TeV", JHEP 03 (2019) 141
- Original analysis methodology from arXiv:1901.01553

## Support

For issues or questions:
1. Check the console error messages
2. Verify internet connectivity to CERN servers
3. Ensure all Python packages are up to date
4. Consult the [CMS Open Data documentation](http://opendata.cern.ch/docs)

This complete guide provides everything needed to reproduce the AH optimization plots using CMS Open Data. The analysis follows the same methodology as the original CMS paper while being accessible through public data and open-source tools.
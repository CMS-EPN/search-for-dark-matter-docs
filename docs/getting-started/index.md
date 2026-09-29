# Run the analysis

!!! tip "Only want the results?"
    Nothing needs to be run to see them. The approved plots are on the
    [Results](../results/index.md) pages, and the code repository's `results/` folder holds the
    comparison PDF and both notebooks exactly as executed.

## Quick start

You need **one** tool: [pixi](https://pixi.sh). It installs the whole scientific stack —
including the XRootD client needed to stream CMS Open Data — into a folder inside the
repository, and touches nothing else on your system.

```bash
# 1. install pixi (once per machine)
curl -fsSL https://pixi.sh/install.sh | bash

# 2. clone and enter
git clone https://github.com/CMS-EPN/search-for-dark-matter.git
cd search-for-dark-matter

# 3. start JupyterLab — dependencies install automatically on first run
pixi run lab
```

Open <http://localhost:8888/lab> and run **`notebooks/Single_Lepton.ipynb`** or
**`notebooks/All_Hadronic.ipynb`** top to bottom.

!!! warning "Do not install the dependencies with pip"
    The XRootD client required to read `root://` URLs is not practically pip-installable. Its
    absence was the "download error" originally reported by the collaboration, and it is still
    the single most common reason the analysis fails to fetch data.

!!! note "Platform"
    Tested on Linux x86-64; the lockfile pins that platform. macOS and other platforms are
    untested and may need `pixi.toml` adjusted.

## Step by step

### 0 — Install pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Open a new terminal (or `source ~/.bashrc`) and check it with `pixi --version`.

### 1 — Start JupyterLab

```bash
pixi run lab
```

The **first** run downloads and installs the environment (~1.2 GB, a few minutes); later runs
start immediately. The environment, the package cache and Jupyter's own configuration are all
written inside the repository, never to your home partition.

### 2 — Pick a notebook

| Notebook | Channel | Paper section |
|---|---|---|
| `notebooks/Single_Lepton.ipynb` | 1 lepton (muon **and** electron) | 4.1, 4.3.1 |
| `notebooks/All_Hadronic.ipynb` | 0 leptons | 4.2, 4.3.2 |

They are independent and write to different files in `output_raw/`, so either can go first.
The notebooks under `notebooks/upstream/` are the originals, kept for reference — do not run
those.

### 3 — Smoke test first (recommended)

The full run streams 20 ROOT files per dataset and takes hours. Check the whole notebook
executes first, on a single file per dataset:

```bash
pixi run smoke-sl     # single-lepton, ~15 min
pixi run smoke-ah     # all-hadronic, ~15 min
```

A clean run ends with `[NbConvertApp] Writing ... bytes`. The resulting plots are
statistically meaningless — they only prove the machinery works. The one-file results left in
`output_raw/` are safe: the cache records how many files each result used, so the full run
detects them as stale and reprocesses them.

### 4 — Run the analysis

In JupyterLab, **Run ▸ Run All Cells**. What happens, in order:

| Stage | Time | What it does |
|---|---|---|
| Setup | seconds | Loads the stack and `dpoa_io`; builds the dataset registry (22 datasets); fetches the file index per dataset into `ntuples.json`; downloads the 2016 golden JSON. |
| Cross sections and a first read | seconds | Builds the `fileset` (dataset ↔ cross section ↔ data/MC) and opens one file over XRootD. If this works, downloads work. |
| **Event processing** | hours | Streams the first `N_FILES` files of each dataset, applies the selection, writes survivors to `output_raw/`. |
| Normalisation | 10–20 min | Reads `Runs/genEventSumw` from the *same* files to compute $w = \sigma \cdot L / \sum w_{\text{gen}}$; derives the luminosity of the processed data slice. |
| Plots | fast | Reads the cached parquet files and draws the CMS-style stacks. |

During event processing, read the `N/M files ok` column:

```text
=== MUON CHANNEL - MC (20 files per dataset) ===
  ttbar-semileptonic: 20/20 files ok, 214,560 events -> ttbar-semileptonic_raw.parquet
```

Failed files are listed with their reason. A dataset where **every** file fails stops the
notebook on purpose: a silently empty background is worse than an error.

| Notebook | Datasets | Approximate wall time |
|---|---|---|
| Single lepton | MC + SingleMuon + SingleElectron, two channels | 3–5 h |
| All hadronic | MC + MET, one channel | 1.5–3 h |

The step is almost entirely network I/O against CERN, so times vary with your connection.
**You can stop and resume at any point**: results are cached per dataset, and re-running skips
whatever already finished.

### Unattended runs

`run_analysis.sh` executes the notebooks headlessly and **resumes automatically** after network
drops. It checks that `eospublic.cern.ch:1094` is reachable before spending an attempt, tells a
firewall block apart from a CERN outage, and keeps the machine awake while running.

```bash
./run_analysis.sh Single_Lepton All_Hadronic
```

```bash
MAX_ATTEMPTS=12 COOLDOWN=300 ./run_analysis.sh Single_Lepton
```

## Configuration

| Knob | Where | Default | Notes |
|---|---|---|---|
| `N_FILES` | `src/dpoa_io.py`, or env `DPOA_N_FILES` | 20 | Files per dataset. Drives **both** event processing and the `sumGenWeights` sum, which must agree. |
| Luminosity | derived per channel by `dpoa_datasets.luminosity_pb()` | — | Byte fraction of Run2016H actually processed × 8.9 fb⁻¹. See [PHYS-02](../changes/physics.md#phys-02-luminosity-derived-per-channel). |
| `LUMI_PB` | `src/dpoa_io.py` | 3400 | The upstream fitted value. Kept only so the notebooks can print what upstream used; it no longer normalises anything. |

For a quicker, noisier run:

```bash
DPOA_N_FILES=5 pixi run lab
```

## Where the output goes

```text
output_raw/
├── .manifest.json                       # how many files each result used
├── ttbar-semileptonic_raw.parquet       # single-lepton, muon
├── ttbar-semileptonic_electron_raw.parquet
└── ttbar-semileptonic_0lep_raw.parquet  # all-hadronic
```

Plots render inline. To save one, replace `plt.show()` with `plt.savefig("name.pdf")` in the
plotting cell, or right-click the image in JupyterLab.

## Requirements

- Linux x86-64
- ~2 GB of disk for the environment, plus space for `output_raw/`
- Network access to `opendata.cern.ch` (HTTPS) and `eospublic.cern.ch` on the XRootD port **1094**

`pixi.lock` pins every package to an exact version, build and hash, so two people running
`pixi run lab` a year apart get byte-identical environments.

# Troubleshooting

## Setup

**`pixi: command not found`**
: Open a new terminal after installing, or `source ~/.bashrc`.

**`ModuleNotFoundError: No module named 'dpoa_io'`**
: You are not inside the pixi environment. `src/` is only on the Python path when the process is
  launched through pixi. Start Jupyter with `pixi run lab`, not with a `jupyter lab` installed
  elsewhere.

**Downloads fail with `root://` URLs and you are not using pixi**
: A plain `pip install` environment cannot read XRootD URLs. Use `pixi run lab`.

## Downloads

**`OSError: Failed to open file ... [3011] Unable to open file`**
: The cached file paths are stale. Delete `ntuples.json` at the repository root and re-run the
  index cell.

**A dataset reports `all N file(s) failed`, or `could not open root://... after 6 attempts`**
: Two different causes look identical from the notebook. Tell them apart:

    ```bash
    python -c "import XRootD; print('xrootd OK')"
    ```

    ```bash
    timeout 10 bash -c 'exec 3<>/dev/tcp/eospublic.cern.ch/1094' && echo "port 1094 OK" || echo "port 1094 BLOCKED"
    ```

    - **`ModuleNotFoundError: XRootD`** — the environment is wrong; use pixi.
    - **`port 1094 BLOCKED`** — your *network* blocks XRootD. Many corporate, hotel and some
      ISP networks drop everything outside the usual web ports. Use an academic network or a VPN
      (eduVPN). Control test: port 443 to the same host should connect in well under a second.
      If 443 answers instantly and 1094 times out, it is the network, not the code.
    - Also check **IPv6**: `eospublic.cern.ch` publishes both A and AAAA records, and a network
      with broken IPv6 fails even when IPv4 is healthy (`getent ahosts eospublic.cern.ch`).

    You do not need to start over. Every finished dataset is cached in `output_raw/`, and
    `./run_analysis.sh` waits for the port and resumes from the cache.

**It is taking forever**
: It is network-bound, not CPU-bound (measured: 18 % of one core, 44 Mbit/s sustained; adding
  more download streams made it *slower*). Lower `N_FILES`: `DPOA_N_FILES=5` gives usable shapes
  in about a quarter of the time.

## Results

**Plots are empty or a background is missing**
: Look for the `NOT loaded (...)` line printed by the loading cell. It names every dataset
  that had no parquet file or no usable scale factor.

**A dataset shows 0 files in the index table**
: Its index URL is stale. The run continues, but that background will be missing.

**Single-lepton control region prints `No events passed CR W(lν)`**
: Expected. The region requires $n_b = 0$, but the processors keep only $n_b \ge 1$ events when
  writing the cache. See the note at the end of the
  [single-lepton results](../results/single-lepton.md#reading-the-result).

## Editing

**I edited a notebook and lost my changes**
: Both notebooks are **generated**. `pixi run build-sl` / `pixi run build-ah` overwrite them.
  Put lasting edits in `src/build_single_lepton.py` or `src/build_all_hadronic.py`.

**I changed the signal-region plot and only one channel picked it up**
: The plot lives in `src/region_plot_cell.py` and both builders embed that same file. Edit it,
  then rebuild both notebooks (`pixi run build`).

# Computational changes (COMP)

These change **how** the code runs, not what it computes. Each was verified by running the
notebooks end to end; the selected events are identical.

## COMP-01 · Pinned, reproducible environment

**Was:** a `pip install uproot awkward numpy matplotlib vector hist mplhep dpoa_workshop` cell.
**Now:** `pixi.toml` + `pixi.lock` (conda-forge, linux-64, Python 3.11); one command, `pixi run lab`.

**Why:** a pip environment cannot read `root://` URLs — the XRootD client is not practically
pip-installable. This was the reported "download error". The lockfile pins every package to an
exact build and hash.

## COMP-02 · Failures are reported instead of swallowed

**Was:**

```python
def process_file_muon(...):
    try:
        ...
    except Exception as e:
        print(f"Error procesando {filename}: {e}")
        return None
```

followed by `pd.DataFrame(data_dict)` in the caller.

**Now:** the processing functions do not catch their own exceptions; `dpoa_io.process_dataset`
catches, counts and reports them per file, and raises if **every** file of a dataset failed.

**Why — the most consequential COMP change.** `pd.DataFrame(None)` is an empty DataFrame, not
an error. A file that failed to download was indistinguishable from a file with no selected
events: the dataset was written with fewer events than it should have, while `sumGenWeights`
was still computed over all 20 files — so that background was silently mis-scaled.

**Verify:** each processing cell prints `N/M files ok`; `M − N` failed and each is listed.

## COMP-03 · One `N_FILES` constant

**Was:** `N_FILES = 20` in the processing cell and `N_FILES_MC = 10` in the `sumGenWeights` cell.
**Now:** both read `dpoa_io.N_FILES`.

**Why:** the MC weight is $\sigma \cdot L / \sum w_{\text{gen}}$. Events from 20 files normalised
by the weight sum of 10 is off by a factor of 2. **This was a live bug** in the 10-file
all-hadronic notebook.

## COMP-04 · One luminosity definition

**Was:** `LUM` set in three cells — `(20/82)*8900 = 2170.7`, `35.9` and `3400.0`.
**Now:** one definition per channel. The baseline and signal-region plots of the same notebook
had been normalised to **different luminosities**. Whether the value is *correct* is a physics
question: [PHYS-02](physics.md#phys-02-luminosity-derived-per-channel).

## COMP-05 · Per-channel output files

**Was:** both channels wrote `output_raw/<dataset>_raw.parquet`.
**Now:** `_raw` (SL muon), `_electron_raw` (SL electron), `_0lep_raw` (AH).

**Why:** the two selections keep **different events from the same datasets** under the same
filename. Running one notebook after the other made the second silently reuse the first's
events — every plot of the second channel built from the wrong selection, with no error.

## COMP-06 · Cache invalidated when `N_FILES` changes

Results are cached per dataset, and `output_raw/.manifest.json` records how many files each was
built from; a mismatch triggers reprocessing. Without it, a `DPOA_N_FILES=1` smoke test would
poison the full run (1-file events normalised by a 20-file weight sum).

**Verify:** run with `DPOA_N_FILES=1`, then with 20 — the log says
`cache was built from 1 file(s), now asking for 20 - reprocessing`.

## COMP-07 · Robust file-index fetching

**Was:** `paths = [ln.split()[0] for ln in r.text.splitlines()]`, no status check.
**Now:** `dpoa_io._fetch_index` checks the HTTP status, rejects HTML responses and validates
every path as a ROOT URI. A stale index URL returns a 404 **HTML page**, which the original code
parsed into a path list starting with `<!DOCTYPE` and failed much later on a nonsense filename.

## COMP-08 · XRootD retries

`dpoa_io.open_events` retries with exponential backoff (6 attempts, ~310 s in total). The public
`eospublic.cern.ch` redirector intermittently drops connections, and one transient failure used
to cost a whole dataset through COMP-02's silent path.

## COMP-09 · Parallel file processing

A serial loop became a `ThreadPoolExecutor` with 4 workers. The workload is network I/O and
uproot releases the GIL while waiting; 4 is polite to the redirector. Results are
order-independent, so no output changes.

## COMP-10 · `!wget` replaced with Python

`wget` is not present everywhere, re-downloaded the golden JSON on every run, and its failures
are invisible inside Jupyter.

## COMP-11 · Missing reference figures do not abort a run

Five `display(Image(filename="Table_10.png"))` cells raised `FileNotFoundError` because the PNGs
were never committed. `dpoa_io.show_table(...)` prints a note instead. They are screenshots of
paper tables and carry no physics.

## COMP-12 · Notebooks are generated, not hand-edited

`pixi run build-sl` / `build-ah` regenerate the notebooks from `notebooks/upstream/` and from
this site's single-lepton markdown. Each patch is a named block with a comment saying what was
wrong, and the builders verify that the unpatched physics cells — 14 (SL) and 16 (AH) — pass
through byte-identical. Editing the `.ipynb` directly is lost on the next build.

## COMP-13 · Dead cells removed

Three trailing "troubleshooting" cells in the all-hadronic notebook broke a Run All (one
referenced an undefined `file_paths`; two reset `N_FILES = 3` and re-ran the processing loop).
They are now markdown notes. Stale `raw` output cells and dead single-lepton code were removed.

## COMP-14 · Plot functions take their data explicitly

`plot_grouped_stack` read a module-level `all_dfs` global that a later cell reassigned to the
electron dictionary, so re-running an earlier cell silently plotted the other channel. It now
takes `dfs=` explicitly.

## COMP-15 · Runs resume after a network drop

A run pulls tens of gigabytes over hours. An outage longer than the per-file retry budget fails
every file of a dataset and aborts the notebook — although every dataset already processed is
safely cached. Stretching the retries would make genuine failures slow to surface, so instead:

- **Re-running resumes.** The cache is keyed on `N_FILES` (COMP-06), so the right response to
  an abort is simply to run again. `run_analysis.sh` does that: up to `MAX_ATTEMPTS` (default 8)
  with a `COOLDOWN` (default 120 s).
- **It checks the port before spending an attempt.** An early version spent ~15 min per attempt
  rediscovering that every file open times out. It now probes `eospublic.cern.ch:1094` and waits
  (once a minute, up to `PORT_WAIT` = 4 h), restarting within a minute of the link coming back.
- **It says which failure it is.** If port 443 to the same host answers but 1094 does not, it
  is a firewall, not a CERN outage. Measured: 443 connected in **0.19 s** while 1094 **timed out
  after 12 s**; IPv6 was independently broken on the same network.
- `systemd-inhibit` keeps the machine awake; an earlier run died when the laptop suspended.

### Note on runtime — measured, not guessed

| Measurement | Value |
|---|---|
| Kernel CPU | **18 %** of one core |
| Sustained network | **44 Mbit/s** |
| Adding 6 download streams alongside the running job | total fell to **14 Mbit/s** |

The run is network-bound and the path to `eospublic` is saturated, so more parallelism makes it
slower. The one real saving is structural: the muon and electron processors each download the
same MC samples, so **half the bytes are duplicates**. A single pass writing both channels would
roughly halve the wall time — recorded for later, not done, to avoid risking the pipeline while
it was converging.

## COMP-16 · Paths resolved from the repository root

After the repository was reorganised into `notebooks/ src/ docs/ results/` (2026-09-23), a
notebook's kernel working directory became `notebooks/`, not the repository root.

- `dpoa_io` now resolves `output_raw/`, `ntuples.json` and the golden JSON from its **own file
  location**, not the current directory; `pixi.toml` exports `PYTHONPATH=$PIXI_PROJECT_ROOT/src`
  so `import dpoa_io` works from anywhere.
- The generated cells that wrote and read `ntuples.json` still used a bare relative path, so the
  cache silently went to `notebooks/ntuples.json`. Both builders now route through
  `dpoa_io.NTUPLES_JSON` (fixed 2026-09-24).

**Verified** from a clean state (no `ntuples.json` anywhere): both notebooks write and read it at
the repository root; full smoke tests passed — All_Hadronic 6 plots, Single_Lepton 14 plots,
0 errors.

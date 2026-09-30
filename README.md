# Code and data for "A Critical-Path Bound for Skipping Delay Evaluations in Multi-Robot Plan Search"

This folder contains the code, task data, and raw experiment records behind the manuscript. Every number, table, and figure in the paper can be regenerated from the stored records with the commands below. Repository: https://github.com/Shengda21/muti-robot.

Authors: Shengda Liu (Institute of Automation, Chinese Academy of Sciences), Xu Liang (Beijing Jiaotong University, corresponding author).

## Contents

```text
src/        scheduling, expected-completion scoring and the bound, task/plan compiler, AI2-THOR adapter
tools/      experiment runners, analysis, audits, asset generation (see below)
tests/      unit tests (41 tests)
data/       scene inventories and task suites
            transfer_*        30 held-out scenes, 81 tasks (main experiment)
            confirm_*         24 further scenes, 66 tasks (confirmation study)
results/    raw records and analyses (see "Results layout")
docs/       pre-specified protocols and the software environment record
requirements.txt
```

## Environment

Python 3.12. The search, scoring, audits and analysis use only the standard library; plotting needs Matplotlib. Stored records make a GPU, a simulator, or a language-model service unnecessary for the steps below. `requirements.txt` lists the versions used for the full pipeline (AI2-THOR 5.0.0 is only needed to regenerate scene inventories).

## Reproduce the reported results from the stored records

Run from this folder.

```text
python -m pytest tests -q

# Main experiment (Section V-A/B): analysis, independent audit of every skipped candidate, event-by-event replay
mkdir -p reproduced
python tools/analyze_robust_search.py --output reproduced/analysis --figure reproduced/search_effects
python tools/audit_robust_search.py             # rewrites results/major_revision/analysis/independent_audit.json
python tools/validate_major_revision_replay.py  # rewrites results/major_revision/replay_validation.json

# Confirmation study (Section V-C): analysis, then table, figure and LaTeX macros
python tools/analyze_closure.py --input results/closure/confirm_scaling --output reproduced/analysis_confirm_scaling
python tools/analyze_closure.py --input results/closure/confirm_cost    --output reproduced/analysis_confirm_cost
python tools/make_closure_assets.py --scaling reproduced/analysis_confirm_scaling \
       --cost reproduced/analysis_confirm_cost --explore results/closure/analysis_scaling2 \
       --figure reproduced/closure_effects.pdf --tex reproduced/results_closure_table.tex --macros reproduced/results_closure_numbers.tex
```

The analysis averages seeds within a task, then task sizes within a scene, and bootstraps scenes with 10,000 paired draws. Stored CPU times were measured on the original machines and cannot be reproduced exactly on other hardware. The regenerated `results_closure_table.tex` and `results_closure_numbers.tex` are identical to the files used in the manuscript. The audit reports 81 tasks, 1,701 search runs, 94,185 skipped candidates re-checked, and no false skip or trajectory mismatch.

## Rerun the searches

```text
# main experiment on the saved LLM candidates (defaults: data/transfer_tasks.json, results/major_revision/candidates)
python tools/run_robust_search.py --output reproduced/search --workers 4

# confirmation study (run on Linux for valid CPU timing); the settings used are in results/closure/confirm_*/config.json
python tools/run_scaling_search.py --mode scaling --tasks data/confirm_tasks.json --output reproduced/confirm_scaling --workers 4
python tools/run_scaling_search.py --mode cost    --tasks data/confirm_tasks.json --output reproduced/confirm_cost    --workers 1
```

Each search records every proposal, its score (or that it was skipped), the beam after every step, and, after timing stops, an exact re-evaluation of every skipped candidate. The runners assert that no skipped candidate could have entered the beam.

## Regenerate the upstream inputs (optional)

```text
# scene inventories, reachable grids, native-move calibration, task suites (needs AI2-THOR 5.0.0 and a display/Unity runtime)
python tools/collect_transfer_scenes.py --ranges 201:218,301:318 --output data/transfer_inventory.json --tasks data/transfer_tasks.json --calibration results/transfer_scene_calibration.json
python tools/collect_transfer_scenes.py --ranges 219:230,319:330 --output data/confirm_inventory.json --tasks data/confirm_tasks.json --calibration results/confirm_calibration.json

# LLM candidates: needs an OpenAI-compatible endpoint at http://127.0.0.1:1234/v1/chat/completions serving qwen3-vl-8b-instruct
python tools/run_transfer_candidates.py --output reproduced/candidates
```

The reported model was Qwen3-VL-8B-Instruct (Q4_K_M, temperature 0.7, fixed task seed). New generations can differ by runtime, so use the stored candidates in `results/major_revision/candidates` (one JSON per task, with raw responses) to reproduce the paper.

## Results layout

| Path | What it is | Used for |
| --- | --- | --- |
| `results/major_revision/candidates/` | stored LLM candidates and raw responses, one file per task | main experiment inputs |
| `results/major_revision/search/` | per-task search records (all methods and seeds) | main experiment, Table I/II, Fig. 2 |
| `results/major_revision/analysis/` | summaries, per-task and per-run metrics, independent audit | numbers in Section V-A/B |
| `results/major_revision/replay_validation.json` | event-by-event replay check (4,767 pause cases) | Section IV-D |
| `results/major_revision/development_search/` | runs on the six development scenes | method development only |
| `results/closure/confirm_scaling/`, `confirm_cost/` | confirmation study raw records | Section V-C, Fig. 3, Tables |
| `results/closure/analysis_confirm_*/` | analysis of the confirmation records | Section V-C |
| `results/closure/scaling2/`, `cost2/`, `analysis_scaling2/`, `analysis_cost2/` | exploration on the 30 earlier scenes (with tie-break baseline) | design and exploratory statements |
| `results/closure/sens_abs/`, `sens_rnd/`, `analysis_sens_*/` | exploratory sensitivity runs (fixed pause lengths; random start cells) | Section V-C |
| `results/closure/scaling/`, `cost/`, `analysis_scaling/`, `analysis_cost/` | first exploration runs, superseded before the confirmation data existed (no tie-break baseline, earlier start rule) | kept for transparency |

`docs/closure_protocol.md` is the pre-specified protocol of the confirmation study together with its change log; `docs/major_revision_protocol.md` is the protocol of the main experiment.

## Notes

- Tasks are named `t<scene>_n<objects>`. `data/transfer_tasks.json` lists 18 development tasks (FloorPlan201-203, 301-303) and 90 evaluation tasks; nine evaluation tasks are excluded by predefined rules (too few reachable objects or destinations), which leaves the 81 tasks of the paper. Search records exist only for these 81.
- Durations are static reachable-grid moves plus one manipulation step, not seconds; turns and collisions are ignored.
- No model weights, simulator assets, credentials, or server configuration are included.

## License

- Code (`src/`, `tools/`, `tests/`): MIT License, see `LICENSE`.
- Data and records (`data/`, `results/`, `docs/`): Creative Commons Attribution 4.0 International (CC BY 4.0), see `LICENSE-DATA`. Please cite the manuscript when reusing them.
- Scene and object descriptions in `data/` were extracted from AI2-THOR scenes; AI2-THOR itself is not redistributed here and remains under its own license.

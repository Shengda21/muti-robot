# Overall redesign and prospective experiment protocol

2026-09-29. This replaces the V2 research storyline, not its historical evidence. It is fixed before optimizing or inspecting outcomes in the new evaluation rooms. Existing kitchen results are discovery evidence only. The IEEE review and its five major concerns motivate this redesign.

## Research question and contribution

When is a candidate's extra nominal completion time too expensive to be recovered by resistance to a robot interruption? We study this as a decision and search problem. Language models supply legal task allocations/orders; a numerical planner has access to scene costs. The work does not claim that combining an LLM with scheduling is new.

For a fixed acyclic earliest-start plan, one uniformly selected robot is interrupted once. A nominal critical path contains an executing action at every time before completion, so the mean makespan increase over robots is at least the pause duration divided by team size. This creates a lower bound on the disturbed objective and a safe rejection test against a scored incumbent. The generalized bound includes the probability that an onset precedes completion when a common observation horizon exceeds a candidate's makespan. Exact single-perturbation evaluation is related to classical robust scheduling; the experiment must show whether this inexpensive bound is useful inside a fully specified search.

The three evidence questions are: (1) does pruning preserve the selected plan and search trajectory while reducing evaluation effort; (2) what changes when search optimizes nominal or interrupted completion under the same proposal budget; (3) what is attributable to LLM seeds rather than subsequent numerical optimization? Success does not require all comparisons to favor the proposed configuration.

## Data and independence

AI2-THOR 5.0.0 living rooms and bedrooms, with native inventories and reachable grids. Development rooms: FloorPlan201–203 and FloorPlan301–303. Fresh evaluation rooms: FloorPlan204–218 and FloorPlan304–318 (30 scene clusters). Tasks require 4, 6, or 8 distinct object transfers to two predetermined receptacles, with typed open/close actions where relevant. Initial containment already satisfying a requested transfer is excluded when building jobs. Tasks with insufficient reachable objects or destinations are excluded by the builder before any candidate scoring; exclusions and reasons are retained.

No room is excluded because of poor scores or failures. These are new scene/task outcomes, unlike the V2 post-hoc reanalysis. Development observations may repair implementation errors; any substantive algorithm change is recorded before evaluation starts. The task inventory and cost model are fixed before LLM generation.

Costs are shortest paths on the simulator's reachable grid plus one manipulation step. Units are discrete action steps, not seconds. A calibration run checks native navigation steps. This supports scene-grounded scheduling, not claims of collision-free simultaneous navigation or physical execution success. Old teleport-based success rates will not be presented as validation of the new method.

## Methods and comparable information

Each task has one 12-proposal Qwen3-VL-8B-Instruct Q4_K_M seed batch using grounded jobs, capabilities and the new task schema. Invalid proposals remain recorded; no silent random replacement. A separately labeled uniform generator samples 12 legal assignment/order/door combinations. Both seeds pass through the same compiler and receive the same numerical scene costs during search.

The common onset horizon T is the best nominal completion time in the union of these two initial pools. Every compared method therefore uses the same physical onset distribution. If an LLM request fails, service-only retries use the identical prompt. If no legal LLM plans remain, the task is retained for non-LLM comparisons and omitted only from explicitly paired LLM comparisons, with denominator reported.

Search uses a beam of four candidates and a budget of 256 unique symbolic proposals including its seeds. At each step it proposes a mutation of a uniformly selected beam member (change one job's robot, swap two job priorities, or change a door operator); one fifth of proposals are fresh uniform candidates. Duplicate assignment/order/door-operator tuples are skipped; different symbolic tuples can compile to equivalent graphs. Seeds 6101, 6102, 6103 provide three paired search replicates. The budget sweep uses prefixes at 32, 64, 128 and 256. Tie-breaking is objective, nominal cost, then first occurrence.

Compare:

1. LLM pool: choose its best nominal plan and its best interrupted-objective plan.
2. Uniform proposals: same total unique-proposal budget, choose minimum interrupted objective.
3. Nominal search: beam expansion ranked by nominal makespan; re-rank all its proposals by interrupted objective at the end.
4. Interruption-aware search: beam expansion and output ranked by interrupted objective.
5. Pruned interruption-aware search: identical proposals and decisions, but use the bound to skip candidates that cannot enter the current beam. This is a computational ablation, not a different optimization objective.
6. Repeat the two numerical searches with uniform initial seeds to separate seed source from search mechanism.

Before evaluation, include the necessary pruning control: the trivial lower bound J>=C. Compare no pruning, nominal-only pruning, and critical-path pruning using the current worst beam score (not the best incumbent) as threshold, so every possible beam entrant is retained and the subsequent proposal stream remains identical. The new bound's savings must be distinguished from ordinary nominal-cost rejection.

The closest published methods often change temporal edges, repair execution, or solve different task models. Do not label our controlled selectors as reimplementations of those systems. Include the strongest directly applicable nominal/search controls and compare scientific assumptions in related work.

## Score, evaluation, and statistics

Selection minimizes nominal makespan plus the exact expected increase for one pause, with uniform robot and continuous onset in [0,T]. Selection pause durations are {0.1T,0.2T,0.4T}, equally weighted. Fresh evaluation uses {0.15T,0.3T,0.6T}, also uniformly weighted and integrated over the same onset range. This tests duration transfer within the specified pause model; it is not an unseen physical fault class.

Primary correctness endpoint: pruned and unpruned search must return identical objectives and candidate identities for all task/seed pairs; every pruned proposal must be ineligible for the beam under its exact score when independently checked. Primary efficiency endpoints: end-to-end CPU time, number of exact risk evaluations, and fraction rejected. Compilation and bound overhead count in time. Report score-only and complete-search times separately.

Primary quality comparison: interrupted-completion search versus nominal-search-plus-reranking under the same seed pool, proposal budget, and search seed. Report nominal completion, mean disturbed completion, changes relative to nominal pool selection, and whether seed source changes the conclusion. Metrics in steps and normalized by common T; no task success claim inferred from makespan.

Average the three search seeds and task sizes within each scene before paired bootstrap (10,000 draws, seed 20260930). Pool 30 scene clusters for the primary estimate; report living-room and bedroom summaries as descriptive domain checks. Report confidence intervals and effect sizes, including unfavorable or near-zero outcomes. Counts of plans and pause evaluations are computational sample sizes, not independent experimental units.

## Validation and manuscript design

Use exact replay cross-checks for the new risk formula and pruning, and a native-navigation calibration for the new cost model. Development checks are not claimed as held-out benchmark improvements. Store original candidate responses, exclusion reasons, all candidate score traces needed for the pruning audit, per-task outcomes and the analysis script.

Rewrite the existing canonical four-page LaTeX manuscript in place after results are known; keep the pre-revision package as a recoverable historical snapshot. Plan approximately 0.8 page for problem/related work, 1.1 for model/bound/search, 1.3 for design/results, 0.2 for conclusion, and 0.6 for references. Use one TikZ method diagram and one evidence plot. Remove the old multi-selector catalogue, organization-theory framing and unrelated success narrative from the main argument. Retain old evidence in its supporting files.

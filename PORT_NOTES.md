# Port notes: CaP-X BEHAVIOR integration → BEHAVIOR-1K v3.9.3 (R1Pro)

Branch `b1k-v3.9.3-r1pro`. Upstream CaP-X targets BEHAVIOR v3.7.2 / Python 3.10 /
Isaac Sim 4.5, which has no RTX 50-series support, so its BEHAVIOR path cannot run on the
target machine (RTX 5090, driver 580.178.04) at all. Porting forward is forced, and upstream's
own BEHAVIOR numbers cannot be reproduced here as a check on this port.

Only two files carry OmniGibson coupling: `capx/envs/simulators/r1pro_b1k.py` and
`capx/integrations/r1pro/control.py` (plus one import in `capx/integrations/r1pro/utils.py`).
Everything above them — trial loop, multi-turn, visual differencing, LLM client, artifact
saving, metric aggregation — is simulator-agnostic and is left untouched.

## Policy

Fix what crashes or is unreachable. **Do not touch anything the model sees**: docstrings,
type annotations and task prompts are the agent's input, so changing them changes the
baseline's capability. Every deviation from upstream is listed below.

## Defects found and deliberately NOT fixed

Upstream's executor catches `BaseException`, so these surface as `sandbox_rc=1` with a
traceback in the trial artifacts. That is data about the baseline, not a lost run.

| where | defect |
|---|---|
| `capx/integrations/r1pro/control.py:1124-1128` | `solve_ik` catches `Exception`, prints, then falls through to an unbound local `cfg` → `UnboundLocalError` |
| `capx/envs/simulators/r1pro_b1k.py:674` | `_release_object` calls `execute_controller(ctrl_gen, env)` with a free variable `env` → `NameError` |
| `capx/envs/trial.py:925` | `TrialSummary.success = sandbox_rc == 0`, i.e. "the generated code ran", not task success. Task success is `info_step["task_completed"]`. Matters when comparing against upstream's published success rates |
| `capx/integrations/r1pro/control.py:98` | Class docstring is stale: documents a `goto_pose` that does not exist, omits 20 of the 25 exposed functions |
| `control.py` throughout | Return annotations are wrong on roughly a third of the API (`-> None` on functions returning `bool`; `get_sam3_mask` is annotated `-> np.ndarray` but returns an `int` pixel count). These are generated into the prompt by `ApiBase.combined_doc()`, so the model is told the wrong types |
| `capx/llm/client.py:247-257` | Retries 5xx forever with `sleep(240 ± 90)` against a 1000 s SIGALRM trial budget in `runner.py` |
| `capx/skills/` | A cross-episode skill library exists and is dead code: gated on `evolve_skill_library`, which no config sets and `_load_config` never reads into the merged config. `get_promoted_skills`, `inject_into_namespace` and the prompt formatters have no callers |
| `capx/integrations/r1pro/control.py:141` | `init_molmo()` is called unconditionally in `__init__`, making a Molmo server a hard dependency even when `point_prompt_molmo` is never used |

## Deviations from upstream (deliberate)

| change | why |
|---|---|
| `.openaikey` added to `.gitignore` | upstream ignores `.openrouterkey` but not this; the key was one `git add -A` from being published |
| `capx/third_party/b1k` repointed from `qingh097/b1k` to `felixzheng02/BEHAVIOR-1K`, pinned to `314bf7a8d`, branch `v3.9.3-uv-5090` | upstream's pin is BEHAVIOR v3.7.2 / Isaac Sim 4.5, which cannot run on RTX 50-series. Kept as a provenance pin only — not checked out, not installed from. The environment imports the editable install at `~/BEHAVIOR-1K` |

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
| `capx/third_party/b1k` repointed from `qingh097/b1k` to `felixzheng02/BEHAVIOR-1K`, pinned to `314bf7a8d`, branch `v3.9.3-uv-5090` | upstream's pin is BEHAVIOR v3.7.2 / Isaac Sim 4.5, which cannot run on RTX 50-series. Kept as a provenance pin only — not checked out, not installed from. The environment imports the editable install at `~/behavior_sim/BEHAVIOR-1K` |

## Port: what each v3.9.3 delta became

| upstream | v3.9.3 | resolution |
|---|---|---|
| `holonomic_base_command_in_world_frame(robot, q)` wrapping every `q_to_action` call (7 sites) | helper removed | Dropped the wrapper. v3.7's helper converted root-frame → world-frame because that release's `HolonomicBaseJointController` wanted world-frame; v3.9.3's `q_to_action` documents that it takes **root-frame** `q` and does the base-footprint conversion itself. Input convention is unchanged, so passing `q` straight through is equivalent |
| `StarterSemanticActionPrimitives.overwrite_arm(idx)` (5 sites) | `arm` is a read-only property | `ArmSwitchablePrimitives` in `capx/envs/simulators/_b1k_compat.py`. `overwrite_arm` was **not** stock BEHAVIOR: their fork commented out the `arm` property to make it assignable, so the shim restores their behaviour without patching BEHAVIOR |
| `env.robots[0]`, `env.task.object_scope`, `env.external_sensors[...]`, `env.step()`/`get_obs()` returns | all per-parallel-env now | Indexed with `[0]`; this wrapper drives a single env |
| `omnigibson.learning.utils.eval_utils` | `omnigibson.eval.utils` | re-pointed |
| hand-rolled aggregation of `2025-challenge-task-instances/metadata/episodes.jsonl` | `2026-challenge-task-instances/metadata/task.jsonl` | replaced with the stock `score_utils.load_human_stats(task_name)`, which returns the same four keys |
| hand-built 2025 instance paths | `get_task_instance_path(scene, instance, mode)` | mirrors how `omnigibson/eval/evaluator.py` builds the same path |
| trial index used directly as an instance id | public test ids are 301-320, hidden 321-340 | `instance_mode` kwarg (default `public_test`); trial `n` maps to `TEST_INSTANCE_IDS[n]`, so runs line up with the challenge's `--instance-indices` |
| `robot.model_name` | `robot.model` | renamed |
| `from omnigibson import ROBOT_ASSETS_ROOT` | not exported | `get_dataset_path("omnigibson-robot-assets")`, keeping the module-level name so `import *` consumers are unaffected. A stray `print(ROBOT_ASSETS_ROOT)` at import time was dropped with it |
| `models/r1pro/urdf/r1pro_ik.urdf` | not in the v3.9.3 dataset | reconstructed exactly — see below |
| `controller_cfg="r1pro_primitives.yaml"` | that file now ends in `task: type: DummyTask` | default changed to cap-x's own `capx/envs/configs/b1k/r1pro_behavior_capx.yaml`; the loader now accepts a path as well as a bare name |

### The reconstructed IK URDF

`R1ProControlApi` loads an 18-DoF `r1pro_ik.urdf` that ships with their BEHAVIOR fork but not with
v3.9.3, whose dataset has only the full 28-DoF `r1pro.urdf`. Substituting the 28-DoF file would be
**silently wrong**: `pyroki_joint_mapping` and `controller_joint_mapping` are hardcoded 18-element
permutations, so a 28-joint IK solution would land on the wrong controller joints.

The file is exactly reconstructable. Fixing the 6 wheel/steer joints and the 4 gripper finger
joints in the stock URDF leaves exactly 18 actuated joints in document order —
`torso_joint1..4, left_arm_joint1..7, right_arm_joint1..7` — which is precisely the ordering both
hardcoded mappings imply. `capx/integrations/r1pro/ik_urdf.py` generates it and asserts that
order, failing loudly rather than silently if the stock URDF ever changes.

### Ground-truth perception

`capx/integrations/r1pro/control_privileged.py` subclasses `R1ProControlApi` and overrides only
the detection methods, resolving objects through the task object scope. Function names,
signatures and docstrings are inherited unchanged, which matters because `ApiBase.combined_doc()`
generates the prompt from them. It stubs the perception-server constructors for the duration of
the parent's `__init__` rather than reimplementing it, so the pyroki context, joint mappings and
thresholds stay exactly as upstream set them. No SAM3 / ContactGraspNet / Molmo servers needed.

### LLM access

`.openaikey` (git-ignored, same convention as their `.openrouterkey`) is read by
`resolve_api_key`. `register_model` adds a command-line model id to `GPT_MODELS` and
`VLM_MODELS`, which otherwise silently route an unknown id down the wrong payload shape and trip
the visual-feedback assert. The `openai/` vendor prefix is stripped when posting to the OpenAI
API, which rejects it. All three apply to both query paths in `client.py`.

### Environment

cap-x is installed into the BEHAVIOR venv (Python 3.11, inside its own `>=3.10,<3.13`) with
`--no-deps`, because it declares `torch>=2.8` while Isaac Sim 5.1 here is built against
`torch==2.7.0+cu128`, and because its `sam3` path dependency points at an unchecked-out
submodule. Added by hand under a constraints file holding torch/numpy: `tyro`, `rich`, `mediapy`,
`msgpack_numpy`, `open3d`, `yourdfpy`, `robot_descriptions`, `viser`, `pyroki`. Verified
afterwards that `torch` is still `2.7.0+cu128`, `numpy` still `1.26.0`, and that the BEHAVIOR
smoke test still renders. Side effects to be aware of: `websockets` 17.4 → 16.1.1 and `psutil`
5.9.8 → 7.2.2.

## Further defects found during the port — document, do not fix

| where | defect |
|---|---|
| `control.py:813` (`sample_grasp_pose_simple`) | passes the natural-language `object_name` into `_sample_grasp_pose`, which does `scene.object_registry("name", obj_name)`. Any name the model would actually write resolves to `None` and crashes in `get_grasp_poses_for_object_sticky`. The privileged API sidesteps it by passing the resolved `obj.name` |
| all 48 `env_configs/r1pro/b1k_*.yaml` | none sets a scene, so every task not living in the base config's scene fails at env creation. Worked around by resolving `scene_model` from the challenge metadata, the way BEHAVIOR's evaluator does |
| `r1pro_b1k.py` `__init__` | builds `StarterSemanticActionPrimitives` — and therefore a cuRobo motion generator — unconditionally, although only `move_hand` and `_navigate_to_pose` use it. Grasping goes through pyroki. So cuRobo is a hard dependency of merely constructing the environment |

## Deltas found by running it (probe sequence, each verified)

| symptom | cause | fix |
|---|---|---|
| `KeyError: 'r1pro'` in `load_task_instance` | v3.9.3 instances store robot poses under a generic `"robot"` key; only 2025 carry-overs key them by model, which upstream assumed unconditionally | use the stock `get_presampled_robot_poses`, plus a `presampled_pose_key` kwarg for the carry-over case |
| `AttributeError: 'Robot' object has no attribute 'is_system'` | scope entities lost `is_system`/`exists`, and scope values can now be `None` | decide from the BDDL instance name with `is_system_bddl_inst(inst)`, exactly as `eval/evaluator.py` does |
| `AttributeError: 'AgentMetric' object has no attribute 'end_callback'` | the metric lifecycle was renamed: `start_callback`/`step_callback`/`end_callback`/`gather_results` → `reset`/`step`/`aggregate` | ported all four call sites; `step` now receives action/obs/reward/terminated/truncated/info as v3.9.3 expects |
| `KeyError: 'depth_linear'` in `get_env_observation` | their fork's config requested `depth_linear`; the stock configs do not | added to `obs_modalities` |
| `AssertionError: Controller [gripper_left] should be a JointController ... use_delta_commands=False` | the config was based on `r1pro_behavior.yaml`, whose controllers are the defaults. cap-x streams absolute joint positions through `q_to_action`, which requires the primitives-style controllers | rebuilt `r1pro_behavior_capx.yaml` from `r1pro_primitives.yaml` and grafted the BehaviorTask block in |

### Verified working end to end (ground-truth perception, cuRobo skipped)

`bringing_water`, public_test instance 301, in 166 s: env build, `reset`, `step`,
`get_object_pose` (a bottle at the right pose, 0.075 x 0.075 x 0.259 m), `get_env_observation`
(128x128 rgb + depth), `get_robot_position`, `get_current_joint_positions` (28),
`check_object_in_hand` (which exercises `_settle_robot` -> `q_to_action`), `compute_reward`
returning BEHAVIOR's q_score (0.0), and `task_completed` (False).

The task's real goal, worth knowing before reading any score:
`for every bottle, the bottle is on top of coffee_table1` **and** `electric_refrigerator1 is
not open` — two bottles out of a fridge, onto a table in another room, with the fridge closed
again. It is not a one-pick task.

### Open: cuRobo

`R1ProBehaviourLowLevel.__init__` builds `StarterSemanticActionPrimitives`, which builds a
`CuRoboMotionGenerator` unconditionally, so **cuRobo is required merely to construct the
environment** — even though only `move_hand` and `navigate_to_pose` use it and grasping goes
through pyroki. `nvidia-curobo` is not installed here and there is no `nvcc` / CUDA toolkit on
this machine, so installing it means a per-user CUDA 12.8 toolkit first. The probes above pass
`skip_curobo_initilization=True`, which is not a usable configuration for a real run on this
task because it needs cross-room navigation.

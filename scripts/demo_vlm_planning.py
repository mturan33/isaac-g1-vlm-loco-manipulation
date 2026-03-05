#!/usr/bin/env python3
"""
VLM Planning Demo - Pick-and-Place (Locomanipulation G1 Style)
================================================================
Demonstrates the full VLM planning pipeline:
    1. Create scene (robot + PackingTable + steering wheel)
    2. SemanticMap reads object positions (ground truth)
    3. Planner generates skill sequence from natural language task
    4. SkillExecutor runs the plan on the environment

Scene matches PickPlace-Locomanipulation-G1-Abs-v0:
    PackingTable with built-in basket, steering wheel on surface.

Planners:
    --planner simple : Rule-based (no VLM needed, for testing)
    --planner vlm    : Qwen2.5-VL via Ollama (requires running Ollama)

Usage (from C:\\IsaacLab):
    .\\isaaclab.bat -p source\\isaaclab_tasks\\isaaclab_tasks\\direct\\high_low_hierarchical_g1\\scripts\\demo_vlm_planning.py ^
        --num_envs 4 ^
        --checkpoint C:\\IsaacLab\\logs\\ulc\\g1_unified_stage1_2026-02-27_00-05-20\\model_best.pt ^
        --arm_checkpoint C:\\IsaacLab\\logs\\ulc\\ulc_g1_stage7_antigaming_2026-02-06_17-41-47\\model_best.pt ^
        --task "Pick up the steering wheel from the table" ^
        --planner simple
"""

# ============================================================================
# AppLauncher MUST be created before any Isaac Lab imports
# ============================================================================
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="VLM Planning Demo - Pick-and-Place")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
parser.add_argument(
    "--checkpoint", type=str, required=True,
    help="Path to V6.2 loco policy checkpoint (.pt file)",
)
parser.add_argument(
    "--arm_checkpoint", type=str, default=None,
    help="Path to Stage 7 arm policy checkpoint (.pt file)",
)
parser.add_argument(
    "--task", type=str,
    default="Pick up the steering wheel from the table",
    help="Natural language task description",
)
parser.add_argument(
    "--planner", type=str, default="simple", choices=["simple", "vlm"],
    help="Planner type: 'simple' (rule-based) or 'vlm' (Ollama Qwen2.5-VL)",
)
parser.add_argument(
    "--ollama_model", type=str, default="qwen2.5vl:7b",
    help="Ollama model name for VLM planner",
)
parser.add_argument(
    "--ollama_url", type=str, default="http://localhost:11434",
    help="Ollama API URL",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================================
# Isaac Lab imports (AFTER AppLauncher)
# ============================================================================
import os
import sys
import time
import json
import torch

# Force unbuffered stdout for real-time output in headless mode
sys.stdout.reconfigure(line_buffering=True)

import isaaclab.sim as sim_utils

# Add parent of high_low_hierarchical_g1 to path so package imports work
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG_PARENT = os.path.dirname(_PKG_DIR)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from high_low_hierarchical_g1.envs.hierarchical_env import (
    HierarchicalG1Env, HierarchicalSceneCfg, PHYSICS_DT,
)
from high_low_hierarchical_g1.planning.semantic_map import SemanticMap
from high_low_hierarchical_g1.planning.vlm_planner import VLMPlanner, SimplePlanner
from high_low_hierarchical_g1.planning.skill_executor import SkillExecutor


def main():
    """Main VLM planning demo."""
    num_envs = args_cli.num_envs
    device = "cuda:0"

    print("=" * 60)
    print("  VLM Planning Demo - Pick-and-Place")
    print("=" * 60)
    print(f"  Task       : {args_cli.task}")
    print(f"  Planner    : {args_cli.planner}")
    print(f"  Environments: {num_envs}")
    print(f"  Loco ckpt  : {args_cli.checkpoint}")
    print(f"  Arm ckpt   : {args_cli.arm_checkpoint or 'None (heuristic)'}")
    if args_cli.planner == "vlm":
        print(f"  VLM model  : {args_cli.ollama_model}")
        print(f"  Ollama URL : {args_cli.ollama_url}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Create simulation context
    # ------------------------------------------------------------------
    sim_cfg = sim_utils.SimulationCfg(
        dt=PHYSICS_DT,
        device=device,
        gravity=(0.0, 0.0, -9.81),
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            max_position_iteration_count=4,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.5,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[5.0, -5.0, 4.0], target=[1.0, 0.0, 0.5])

    # ------------------------------------------------------------------
    # 2. Create hierarchical environment
    # ------------------------------------------------------------------
    scene_cfg = HierarchicalSceneCfg()
    env = HierarchicalG1Env(
        sim=sim,
        scene_cfg=scene_cfg,
        checkpoint_path=args_cli.checkpoint,
        num_envs=num_envs,
        device=device,
        arm_checkpoint_path=args_cli.arm_checkpoint,
    )

    # ------------------------------------------------------------------
    # 3. Reset environment
    # ------------------------------------------------------------------
    print("\n[Demo] Resetting environment...")
    obs = env.reset()

    # Let the robot stabilize for a moment
    stand_cmd = torch.zeros(num_envs, 3, device=device)
    for _ in range(50):
        if not simulation_app.is_running():
            return
        obs = env.step(stand_cmd)

    # ------------------------------------------------------------------
    # 4. Create semantic map (ground truth mode)
    # ------------------------------------------------------------------
    print("\n[Demo] Creating semantic map (ground truth)...")
    semantic_map = SemanticMap(mode="ground_truth", env=env)
    semantic_map.update()

    # Print world state
    world_json = semantic_map.get_json()
    print(f"\n[Demo] World state:")
    print(json.dumps(world_json, indent=2, default=str))

    # ------------------------------------------------------------------
    # 5. Generate plan
    # ------------------------------------------------------------------
    print(f"\n[Demo] Planning: \"{args_cli.task}\"")
    plan = None

    if args_cli.planner == "vlm":
        print("[Demo] Using VLM planner (Ollama)...")
        vlm = VLMPlanner(
            model=args_cli.ollama_model,
            ollama_url=args_cli.ollama_url,
        )
        plan = vlm.plan(args_cli.task, world_json)

        if plan is None:
            print("[Demo] VLM planner failed, falling back to SimplePlanner")

    if plan is None:
        print("[Demo] Using SimplePlanner (rule-based)...")
        simple = SimplePlanner()
        plan = simple.plan(args_cli.task, world_json)

    if not plan:
        print("[Demo] ERROR: No plan generated!")
        simulation_app.close()
        return

    print(f"\n[Demo] Generated plan ({len(plan)} steps):")
    for i, step in enumerate(plan):
        print(f"  {i+1}. {step['skill']}({step.get('params', {})})")

    # ------------------------------------------------------------------
    # 6. Execute plan
    # ------------------------------------------------------------------
    executor = SkillExecutor(
        env=env,
        semantic_map=semantic_map,
        simulation_app=simulation_app,
    )

    start_time = time.time()
    result = executor.execute_plan(plan)
    total_time = time.time() - start_time

    # ------------------------------------------------------------------
    # 7. Print results
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  DEMO RESULTS")
    print(f"{'='*60}")
    print(f"  Task      : {args_cli.task}")
    print(f"  Planner   : {args_cli.planner}")
    print(f"  Completed : {result['completed']}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Steps:")
    for r in result["plan_results"]:
        status = r["result"]["status"]
        reason = r["result"].get("reason", "")
        symbol = "+" if status == "success" else "x"
        print(f"    [{symbol}] {r['skill']}: {status} - {reason}")

    # Final robot state
    final_height = obs["base_height"].mean().item()
    standing = (obs["base_height"] > 0.5).sum().item()
    print(f"\n  Final height: {final_height:.2f}m")
    print(f"  Standing: {standing}/{num_envs}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 8. Keep sim running briefly for visual inspection
    # ------------------------------------------------------------------
    print("\n[Demo] Holding for 3 seconds...")
    for i in range(150):  # ~3 seconds at 50Hz
        if not simulation_app.is_running():
            break
        obs = env.step(stand_cmd)

    print("[Demo] Done. Closing simulation...")
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        # Force exit -- Isaac Sim background threads can prevent clean shutdown
        import os
        os._exit(0)

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
    help="Path to Stage 2 arm policy checkpoint (.pt file)",
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
parser.add_argument(
    "--record", action="store_true",
    help="Record video frames (viewport capture → ffmpeg merge)",
)
parser.add_argument(
    "--record_dir", type=str, default="videos/demo_frames",
    help="Directory to save recorded frames",
)
parser.add_argument(
    "--record_fps", type=int, default=25,
    help="Target FPS for recording (captures every N-th sim step)",
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


# ============================================================================
# Video Recorder (Yöntem C: viewport frame capture → ffmpeg merge)
# ============================================================================
class VideoRecorder:
    """Captures viewport frames at target FPS, then merges with ffmpeg."""

    def __init__(self, output_dir: str, fps: int = 25, sim_dt: float = 0.02):
        self.output_dir = output_dir
        self.fps = fps
        self.frame_dir = os.path.join(output_dir, "frames")
        os.makedirs(self.frame_dir, exist_ok=True)
        self.frame_count = 0
        self._sim_step = 0
        # Capture every N-th sim step to achieve target FPS
        # sim runs at 1/sim_dt Hz (e.g., 50Hz), target is fps (e.g., 25)
        self._capture_interval = max(1, int(1.0 / (sim_dt * fps)))

        from omni.kit.viewport.utility import get_active_viewport
        self.viewport = get_active_viewport()
        print(f"[VIDEO] Recorder initialized: {fps} FPS, "
              f"capture every {self._capture_interval} sim steps, "
              f"frames -> {self.frame_dir}")

    def on_step(self):
        """Call after every sim step. Captures frame at correct interval."""
        self._sim_step += 1
        if self._sim_step % self._capture_interval == 0:
            from omni.kit.viewport.utility import capture_viewport_to_file
            frame_path = os.path.join(
                self.frame_dir, f"frame_{self.frame_count:06d}.png"
            )
            capture_viewport_to_file(self.viewport, frame_path)
            self.frame_count += 1

    def finalize(self, output_name: str = "demo_pickup.mp4"):
        """Merge frames into MP4 with ffmpeg."""
        import subprocess

        if self.frame_count == 0:
            print("[VIDEO] No frames captured!")
            return None

        output_path = os.path.join(self.output_dir, output_name)
        frame_pattern = os.path.join(self.frame_dir, "frame_%06d.png")

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(self.fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            output_path,
        ]

        print(f"\n[VIDEO] Converting {self.frame_count} frames to MP4...")
        print(f"[VIDEO] Command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"[VIDEO] Saved: {output_path}")

            import shutil
            shutil.rmtree(self.frame_dir)
            print(f"[VIDEO] Cleaned up frames")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"[VIDEO] ffmpeg error: {e.stderr.decode()[:500]}")
            print(f"[VIDEO] Frames saved in: {self.frame_dir}")
            print(f"[VIDEO] Manual merge: ffmpeg -framerate {self.fps} "
                  f"-i {frame_pattern} -c:v libx264 -pix_fmt yuv420p {output_path}")
            return None
        except FileNotFoundError:
            print(f"[VIDEO] ffmpeg not found! Install: pip install ffmpeg-python or system ffmpeg")
            print(f"[VIDEO] Frames saved in: {self.frame_dir}")
            print(f"[VIDEO] Manual merge: ffmpeg -framerate {self.fps} "
                  f"-i {frame_pattern} -c:v libx264 -pix_fmt yuv420p {output_path}")
            return None


def _wrap_env_for_recording(env, recorder: VideoRecorder):
    """Monkey-patch env step methods to call recorder.on_step() after each step.

    This avoids modifying skill_executor.py — all ~22 step calls automatically
    get frame capture without any changes.
    """
    original_step = env.step
    original_step_manipulation = env.step_manipulation
    original_step_arm_policy = env.step_arm_policy

    def _step_with_record(*args, **kwargs):
        result = original_step(*args, **kwargs)
        recorder.on_step()
        return result

    def _step_manipulation_with_record(*args, **kwargs):
        result = original_step_manipulation(*args, **kwargs)
        recorder.on_step()
        return result

    def _step_arm_policy_with_record(*args, **kwargs):
        result = original_step_arm_policy(*args, **kwargs)
        recorder.on_step()
        return result

    env.step = _step_with_record
    env.step_manipulation = _step_manipulation_with_record
    env.step_arm_policy = _step_arm_policy_with_record
    print("[VIDEO] Env step methods wrapped for recording")

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
    if args_cli.record:
        print(f"  Recording  : {args_cli.record_dir} @ {args_cli.record_fps} FPS")
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
    # 2b. Setup video recording (wraps env step methods)
    # ------------------------------------------------------------------
    recorder = None
    if args_cli.record:
        print("\n[Demo] Setting up video recording...")
        recorder = VideoRecorder(
            output_dir=args_cli.record_dir,
            fps=args_cli.record_fps,
            sim_dt=PHYSICS_DT,
        )
        _wrap_env_for_recording(env, recorder)

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
    hold_start = time.time()
    for i in range(150):  # ~3 seconds at 50Hz
        if not simulation_app.is_running():
            break
        if time.time() - hold_start > 5.0:  # Wall-clock timeout (safety)
            print("[Demo] Hold timeout reached")
            break
        obs = env.step(stand_cmd)

    # ------------------------------------------------------------------
    # 9. Finalize video recording
    # ------------------------------------------------------------------
    if recorder is not None:
        print(f"\n[Demo] Finalizing video ({recorder.frame_count} frames)...")
        recorder.finalize(output_name="demo_pickup.mp4")

    print("[Demo] Done.")


if __name__ == "__main__":
    import threading

    try:
        main()
    except (SystemExit, KeyboardInterrupt):
        pass

    # Force exit with watchdog -- simulation_app.close() hangs on Windows
    def _force_exit():
        print("[Demo] Watchdog: forcing exit after timeout")
        os._exit(0)

    timer = threading.Timer(5.0, _force_exit)
    timer.daemon = True
    timer.start()

    try:
        simulation_app.close()
    except Exception:
        pass

    os._exit(0)

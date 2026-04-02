"""
VLM Task Planner
=================
Local VLM-based task planner using Qwen3-VL via Ollama (streaming),
plus a rule-based SimplePlanner fallback.

OllamaVLMPlanner:
    - Connects to local Ollama instance via `ollama` Python package
    - Sends semantic map JSON + task as structured messages
    - Streams reasoning (<think> blocks) live to terminal
    - Receives structured skill plan as JSON

SimplePlanner:
    - Rule-based fallback (no VLM needed)
    - Keyword matching for pick-and-place tasks
    - Uses semantic map to find objects and surfaces
"""

from __future__ import annotations

import json
import re
import sys
import time
from typing import Optional


# Available skills for validation
AVAILABLE_SKILLS = {
    "walk_to", "pre_reach", "reach", "grasp", "lift",
    "lateral_walk", "lower", "place", "walk_to_position",
}

# ============================================================================
# System prompt — skill library + constraints for VLM
# ============================================================================

VLM_SYSTEM_PROMPT = """\
You are a Unitree G1 humanoid robot task planner operating in NVIDIA Isaac Sim.
You receive the current world state (robot position, objects, surfaces) and must
generate a sequence of skill primitives to accomplish the given task.

AVAILABLE SKILLS (use exactly these names):

1. pre_reach(target: str)
   Raise arm to high position before approaching the table.
   MUST be called BEFORE the first walk_to to avoid arm-table collision.
   target: object ID to prepare for (e.g. "object_01")

2. walk_to(target: str, stop_distance: float, hold_arm: bool)
   Walk to an object or surface.
   - target: object or surface ID (e.g. "object_01", "table_01")
   - stop_distance: how far to stop from target center.
     Use 0.40 for approaching objects on a table.
     Use 0.20 for basket/placement approach (need to get close).
   - hold_arm: true to keep arm frozen in current position during walk.
     MUST be true after pre_reach and when carrying an object.

3. reach(target: str)
   Extend right arm toward target object using RL arm policy.
   Magnetic grasp triggers automatically at 0.21m distance.
   target: object ID to reach for.

4. grasp()
   Close fingers around the object. Call after reach.
   No parameters needed.

5. lift()
   Raise arm straight up after grasping, lifting the object above table height.
   No parameters needed.

6. lower()
   Lower arm to table/basket level for placing the object.
   No parameters needed.

7. place()
   Open fingers to release held object, return arm to default position.
   No parameters needed.

CRITICAL CONSTRAINTS:
- ALWAYS call pre_reach BEFORE the first walk_to (arm must be raised before approaching table)
- Walk with hold_arm=true after pre_reach or when carrying an object
- Robot can only carry ONE object at a time
- After lift, use walk_to with target="table_01" and stop_distance=0.20 to reach the basket
  (basket is ON the table, so table is the walk target)
- The standard pick-and-place sequence is:
  pre_reach -> walk_to(object) -> reach -> grasp -> lift -> walk_to(table) -> lower -> place
- Use EXACT IDs from the world state (e.g. "object_01", "table_01"), not abbreviations

OUTPUT FORMAT:
Return ONLY a JSON object with a "plan" key containing an array of steps:
{"plan": [{"skill": "skill_name", "params": {"param1": "value1", ...}}, ...]}

Do NOT include any text outside the JSON object.\
"""


# ============================================================================
# OllamaVLMPlanner — streaming VLM with live reasoning display
# ============================================================================

class OllamaVLMPlanner:
    """Task planner using local VLM (Qwen3-VL) via Ollama with streaming reasoning.

    Args:
        model: Ollama model name (default: "qwen3-vl:4b")
        stream_reasoning: If True, print <think> blocks live to stderr
    """

    def __init__(
        self,
        model: str = "qwen3-vl:4b",
        stream_reasoning: bool = True,
    ):
        self.model = model
        self.stream_reasoning = stream_reasoning

        try:
            # Fix: Isaac Sim sets SSL_CERT_FILE to a non-existent path,
            # which causes httpx (used by ollama) to fail. Remove it so
            # Python's default SSL context is used instead.
            import os
            _ssl_cert = os.environ.pop("SSL_CERT_FILE", None)
            _ssl_cert_dir = os.environ.pop("SSL_CERT_DIR", None)
            try:
                import ollama as _ollama
                self._ollama = _ollama
            finally:
                # Restore if they existed
                if _ssl_cert is not None:
                    os.environ["SSL_CERT_FILE"] = _ssl_cert
                if _ssl_cert_dir is not None:
                    os.environ["SSL_CERT_DIR"] = _ssl_cert_dir
        except ImportError:
            print("[VLMPlanner] ERROR: 'ollama' package not installed. Run: pip install ollama")
            self._ollama = None

    def plan(
        self,
        task: str,
        semantic_map_json: dict,
        image_path: Optional[str] = None,
    ) -> Optional[list]:
        """Generate a skill plan from a natural language task.

        Args:
            task: Natural language task description
            semantic_map_json: World state from SemanticMap.get_json()
            image_path: Optional path to RGB image from viewport

        Returns:
            List of skill steps [{"skill": ..., "params": {...}}, ...] or None
        """
        if self._ollama is None:
            return None

        messages = self._build_messages(task, semantic_map_json, image_path)

        try:
            t0 = time.time()
            full_response = self._stream_chat(messages)
            elapsed = time.time() - t0
            print(f"\n[VLM] Response received in {elapsed:.1f}s", file=sys.stderr)

            plan = self._parse_response(full_response)
            if plan and self._validate_plan(plan, task):
                print(f"[VLM] Valid plan: {len(plan)} steps", file=sys.stderr)
                for i, step in enumerate(plan):
                    params_str = ", ".join(f"{k}={v}" for k, v in step.get("params", {}).items())
                    print(f"  {i+1}. {step['skill']}({params_str})", file=sys.stderr)
                self._unload_model()
                return plan
            else:
                print("[VLM] Plan validation failed", file=sys.stderr)
                self._unload_model()
                return None

        except Exception as e:
            print(f"[VLM] Error: {e}", file=sys.stderr)
            self._unload_model()
            return None

    def _unload_model(self):
        """Unload model from GPU to free VRAM for Isaac Sim."""
        try:
            self._ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": "x"}],
                keep_alive="0",
            )
            print(f"[VLM] Model unloaded from GPU", file=sys.stderr)
        except Exception:
            pass  # Best effort

    def _build_messages(self, task: str, semantic_map: dict, image_path: Optional[str]) -> list:
        """Build chat messages with system prompt + user context."""
        user_content = f"""CURRENT WORLD STATE:
{json.dumps(semantic_map, indent=2)}

TASK: {task}

Generate the skill plan as JSON."""

        user_msg = {"role": "user", "content": user_content}

        # Attach image if provided
        if image_path is not None:
            user_msg["images"] = [image_path]

        return [
            {"role": "system", "content": VLM_SYSTEM_PROMPT},
            user_msg,
        ]

    def _stream_chat(self, messages: list) -> str:
        """Stream response from Ollama, display reasoning live, return full text."""
        full_response = ""
        in_think = False
        think_printed_header = False

        for chunk in self._ollama.chat(model=self.model, messages=messages, stream=True):
            token = chunk['message']['content']
            full_response += token

            if self.stream_reasoning:
                # Detect <think> blocks (Qwen3 reasoning)
                if '<think>' in token:
                    in_think = True
                    if not think_printed_header:
                        print("\n\033[90m[VLM Reasoning]\033[0m", file=sys.stderr)
                        think_printed_header = True
                    # Print remainder after <think> tag
                    after = token.split('<think>', 1)[1]
                    if after:
                        print(f"\033[90m{after}\033[0m", end='', flush=True, file=sys.stderr)
                elif '</think>' in token:
                    # Print text before </think>
                    before = token.split('</think>', 1)[0]
                    if before:
                        print(f"\033[90m{before}\033[0m", end='', flush=True, file=sys.stderr)
                    in_think = False
                    print(f"\n\033[92m[VLM Plan Output]\033[0m", file=sys.stderr)
                elif in_think:
                    print(f"\033[90m{token}\033[0m", end='', flush=True, file=sys.stderr)

        return full_response

    def _parse_response(self, text: str) -> Optional[list]:
        """Parse JSON plan from VLM response (handles <think> blocks)."""
        # Remove <think>...</think> blocks
        clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        # Try to find JSON object
        # Method 1: Direct parse
        try:
            data = json.loads(clean)
            return self._extract_plan(data)
        except json.JSONDecodeError:
            pass

        # Method 2: Find {...} block
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return self._extract_plan(data)
            except json.JSONDecodeError:
                pass

        # Method 3: Find ```json ... ``` block
        match = re.search(r'```(?:json)?\s*(.*?)```', clean, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                return self._extract_plan(data)
            except json.JSONDecodeError:
                pass

        # Method 4: Find [...] array
        match = re.search(r'\[.*\]', clean, re.DOTALL)
        if match:
            try:
                arr = json.loads(match.group())
                if isinstance(arr, list):
                    return arr
            except json.JSONDecodeError:
                pass

        print(f"[VLM] Could not parse JSON from response (len={len(clean)}):\n{clean[:800]}", file=sys.stderr)
        return None

    def _extract_plan(self, data) -> Optional[list]:
        """Extract plan list from parsed JSON (handles various formats)."""
        if isinstance(data, dict) and "plan" in data:
            return data["plan"]
        elif isinstance(data, list):
            return data
        return None

    def _validate_plan(self, plan: list, task: str) -> bool:
        """Validate plan structure and required fields."""
        if not plan or len(plan) < 2:
            print(f"[VLM] Plan too short ({len(plan) if plan else 0} steps)", file=sys.stderr)
            return False

        validated = []
        for i, step in enumerate(plan):
            skill = step.get("skill", "")
            if skill not in AVAILABLE_SKILLS:
                print(f"[VLM] Step {i}: unknown skill '{skill}', removing", file=sys.stderr)
                continue

            # Check required params
            params = step.get("params", {})
            if skill in ("walk_to", "reach", "pre_reach") and "target" not in params:
                print(f"[VLM] Step {i}: {skill} missing 'target' param", file=sys.stderr)
                continue

            validated.append({"skill": skill, "params": params})

        if len(validated) < 2:
            return False

        # Overwrite plan with validated steps
        plan.clear()
        plan.extend(validated)
        return True


# ============================================================================
# SimplePlanner — Rule-based fallback (unchanged)
# ============================================================================

class SimplePlanner:
    """Rule-based fallback planner. No VLM required.

    Generates standard skill sequences from keyword matching
    on the task string and the semantic map contents.
    """

    def plan(self, task: str, semantic_map_json: dict) -> list:
        """Generate a skill plan from task keywords and semantic map.

        Args:
            task: Natural language task (e.g., "Pick up the steering wheel from the table")
            semantic_map_json: World state from SemanticMap.get_json()

        Returns:
            List of skill steps [{skill, params}, ...]
        """
        task_lower = task.lower()
        objects = semantic_map_json.get("objects", [])
        surfaces = semantic_map_json.get("surfaces", [])

        # Detect task type
        is_pick = any(w in task_lower for w in ["pick", "grab", "grasp", "get", "take"])
        is_place = any(w in task_lower for w in ["place", "put", "set", "drop"])

        if is_pick:
            return self._plan_pick(task_lower, objects)
        elif is_place:
            return self._plan_place(task_lower, surfaces)
        else:
            # Default: walk to first object
            if objects:
                return [
                    {"skill": "walk_to", "params": {"target": objects[0]["id"], "stop_distance": 0.30}},
                ]
            return []

    def _plan_pick(self, task: str, objects: list) -> list:
        """Pick from table and place in basket:
        pre_reach -> walk(hold_arm) -> reach -> grasp -> lift -> lateral_walk -> lower -> place.

        Pre_reach raises arm HIGH while still far from table (no collision).
        Walk approaches with arm held up.  Reach descends from above.
        """
        target_obj = self._find_target_object(task, objects)
        if target_obj is None:
            print("[SimplePlanner] No graspable object found in scene")
            return []
        return [
            # 1. Raise arm HIGH before approaching table (avoids collision)
            {"skill": "pre_reach", "params": {"target": target_obj["id"]}},
            # 2. Walk to object with arm held up (0.25m -- table blocks closer)
            {"skill": "walk_to", "params": {"target": target_obj["id"], "stop_distance": 0.40, "hold_arm": True}},
            # 3. Reach down to object and magnetically attach (10cm threshold)
            {"skill": "reach", "params": {"target": target_obj["id"]}},
            # 4. Close fingers around object
            {"skill": "grasp", "params": {}},
            # 5. Lift arm straight up above basket height
            {"skill": "lift", "params": {}},
            # 6. Walk laterally to basket (carry override keeps X, moves Y only)
            # stop_distance=0.20 -- get closer to basket center before placing
            {"skill": "walk_to", "params": {"target": "table_01", "stop_distance": 0.20, "hold_arm": True}},
            # 7. Lower arm into basket
            {"skill": "lower", "params": {}},
            # 8. Release into basket
            {"skill": "place", "params": {}},
        ]

    def _plan_place(self, task: str, surfaces: list) -> list:
        """Place only (assumes already holding): walk -> place."""
        target_surface = self._find_target_surface(task, surfaces)
        if target_surface is None:
            return []
        return [
            {"skill": "walk_to", "params": {"target": target_surface["id"], "stop_distance": 0.55, "hold_arm": True}},
            {"skill": "place", "params": {}},
        ]

    def _find_target_object(self, task: str, objects: list) -> Optional[dict]:
        """Find the most relevant graspable object from the task string."""
        # Try matching object class names to task keywords
        for obj in objects:
            if not obj.get("graspable", False):
                continue
            obj_class = obj["class"].lower()
            # Match any word in the class name
            for word in obj_class.split("_"):
                if word in task:
                    return obj

        # Fallback: first graspable object
        for obj in objects:
            if obj.get("graspable", False):
                return obj
        return None

    def _find_target_surface(self, task: str, surfaces: list) -> Optional[dict]:
        """Find the target surface from the task string."""
        # Check for surface class match
        for surf in surfaces:
            surf_class = surf["class"].lower()
            if surf_class in task:
                return surf

        # Fallback: first surface
        if surfaces:
            return surfaces[0]
        return None

"""Shared driver for the scene-perception + planning ablation study.

Goal: isolate `scene_description` (+ optionally `scene_description_full`,
i.e. the `scene_enrichment_simulation` accessibility/reachability
enrichment) and the planning stack (`vlm_planning` + `simultaneous_actions`)
from the rest of the production pipeline
(`run_validation_loop_replan_with_feedback_soft_guidance.py`), so the effect
of scene_enrichment on the resulting plan can be measured directly.

This is a SINGLE-PASS inspection tool, not a validation loop: for each image
in --frames-dir it runs perception (+ optional enrichment) and planning
exactly once and saves the outputs. There is no execution, no PRE/POST
validation, and no recovery -- those all depend on scene_enrichment /
planning already having run, so they are out of scope for this ablation.

The two entry-point scripts
(`run_scene_perception_planning_with_enrichment.py` and
`run_scene_perception_planning_no_enrichment.py`) both call `run()` below
with only `include_enrichment` flipped. Every other step function, prompt
version, model, and sampling parameter is identical between the two --
that is what makes the comparison an isolation of scene_enrichment's effect
on planning rather than two scripts that happen to differ in several ways.
Keeping the logic in one place (instead of copy-pasting into two scripts)
prevents the two variants from silently drifting apart over time.

Reuses the tested step functions from the production module unchanged
(scene_description, scene_description_full, vlm_planning,
simultaneous_actions), so per-step raw artifacts (prompts, raw model
responses, run_info) land in the same `outputs/<module_name>/<scenario>/...`
layout the full pipeline uses. This script additionally saves, per
scenario/run, under:

    humanoids_results/ablation_study/scene/<yes_enrichment|no_enrichment>/<scenario>/<loop_timestamp>/

    - terminal_log.txt  -- a verbatim capture of everything printed to the
      terminal during the run (stdout + stderr), the primary artifact for
      reading what happened.
    - run_summary.json  -- the same information structured as JSON (every
      step's parsed output inline, plus pointers into
      `outputs/<module_name>/...` for the raw files), for programmatic diffing
      between the two variants.

Usage (with enrichment, first frame only -- the scenario's initial scene
state):

    PYTHONPATH=.:src python -m src.ablation.run_scene_perception_planning_with_enrichment \\
        --scenario two_two_tennis --frames-dir image_data/two_two_tennis \\
        --scene-v v6 --plan-v v6 --sim-v v8 \\
        --scene-model gemini-robotics-er-1.6-preview \\
        --plan-model gemini-robotics-er-1.6-preview \\
        --sim-model gemini-robotics-er-1.6-preview \\
        --poses-by-image-path image_data/two_two_tennis/poses_by_image.json \\
        --first-frame-only

Usage (no enrichment -- same flags minus --poses-by-image-path/--grounding-*,
which do not apply):

    PYTHONPATH=.:src python -m src.ablation.run_scene_perception_planning_no_enrichment \\
        --scenario two_two_tennis --frames-dir image_data/two_two_tennis \\
        --scene-v v6 --plan-v v6 --sim-v v8 \\
        --scene-model gemini-robotics-er-1.6-preview \\
        --plan-model gemini-robotics-er-1.6-preview \\
        --sim-model gemini-robotics-er-1.6-preview \\
        --first-frame-only

Pass --first-frame-only to process only the scenario's first frame (natural
sort order in --frames-dir). Pass --frames to restrict to specific frame
filenames instead. Default (neither flag): every image in --frames-dir,
each treated as an independent scene snapshot -- there is no I_post/
execution step chaining frames together here.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from src.scenario_loader import load_scenario
from src.settings import load_settings
from src.utils import make_cycle_name, make_experiment_timestamp

from src.humanoids.run_validation_loop_replan_with_feedback_soft_guidance import (
    SUPPORTED_MODELS,
    execute_scene_description_full_step,
    execute_scene_description_step,
    execute_simultaneous_actions_step,
    execute_vlm_planning_step,
    extract_stages,
    list_frame_paths,
    load_poses_by_image_map,
    make_scenario_context,
    resolve_poses_by_image_path,
)


def print_output_block(title: str, payload: Any) -> None:
    """Print a step's parsed JSON output to the console (not just its saved path)."""
    print(f"[OUTPUT] {title}:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


class _TeeStream:
    """Mirrors every write to the real terminal stream and to a shared log
    file, so terminal_log.txt is a verbatim capture of what was printed."""

    def __init__(self, real_stream: Any, log_file: Any) -> None:
        self._real_stream = real_stream
        self._log_file = log_file

    def write(self, data: str) -> int:
        self._real_stream.write(data)
        self._log_file.write(data)
        return len(data)

    def flush(self) -> None:
        self._real_stream.flush()
        self._log_file.flush()


def build_parser(include_enrichment: bool) -> argparse.ArgumentParser:
    variant = "WITH scene_enrichment (scene_description_full)" if include_enrichment else "WITHOUT scene_enrichment (perception only)"
    sibling = "no-enrichment" if include_enrichment else "with-enrichment"

    parser = argparse.ArgumentParser(
        description=(
            f"Scene-perception + planning ablation -- {variant}. Single-pass: "
            "for each image in --frames-dir, runs scene_description"
            + (" + scene_description_full (scene_enrichment)" if include_enrichment else "")
            + " + vlm_planning + simultaneous_actions. No execution, "
            "validation, or recovery loop. Run together with the "
            f"{sibling} sibling script (same flags otherwise) to isolate "
            "scene_enrichment's effect on the resulting plan."
        )
    )

    parser.add_argument("--scenario", type=str, required=True)
    parser.add_argument("--frames-dir", type=str, required=True)
    parser.add_argument(
        "--frames",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Restrict to these frame filenames (basenames) inside "
            "--frames-dir. Default: every image in --frames-dir, each "
            "processed independently."
        ),
    )
    parser.add_argument(
        "--first-frame-only",
        action="store_true",
        help=(
            "Process only the first frame in --frames-dir (natural sort "
            "order), i.e. only the scenario's initial scene state. Takes "
            "precedence over --frames."
        ),
    )

    if include_enrichment:
        parser.add_argument("--poses-by-image-path", type=str, default=None)
        parser.add_argument("--grounding-safety-threshold", type=float, default=0.21)
        parser.add_argument("--grounding-debug-mapping", action="store_true")

    parser.add_argument("--scene-v", type=str, required=True)
    parser.add_argument("--plan-v", type=str, required=True)
    parser.add_argument("--sim-v", type=str, required=True)

    parser.add_argument("--scene-model", type=str, required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--plan-model", type=str, required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--sim-model", type=str, required=True, choices=SUPPORTED_MODELS)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)

    return parser


def build_pipeline_config(args: argparse.Namespace, include_enrichment: bool) -> dict[str, Any]:
    config: dict[str, Any] = {
        "ablation_variant": "yes_enrichment" if include_enrichment else "no_enrichment",
        "include_scene_enrichment": include_enrichment,
        "sampling": {"temperature": args.temperature, "top_p": args.top_p},
        "scene_description": {"prompt_version": args.scene_v, "model": args.scene_model},
        "vlm_planning": {"prompt_version": args.plan_v, "model": args.plan_model},
        "simultaneous_actions": {"prompt_version": args.sim_v, "model": args.sim_model},
    }
    if include_enrichment:
        config["scene_description_full"] = {
            "grounding_safety_threshold": args.grounding_safety_threshold,
            "grounding_debug_mapping": args.grounding_debug_mapping,
        }
    return config


def resolve_frame_paths(args: argparse.Namespace) -> list[str]:
    frame_paths = list_frame_paths(args.frames_dir)
    if args.first_frame_only:
        frame_paths = frame_paths[:1]
    elif args.frames:
        wanted = set(args.frames)
        selected = [p for p in frame_paths if Path(p).name in wanted]
        missing = wanted - {Path(p).name for p in selected}
        if missing:
            raise ValueError(f"--frames not found in --frames-dir: {sorted(missing)}")
        frame_paths = selected
    if not frame_paths:
        raise ValueError("No frames selected to process.")
    return frame_paths


def run(args: argparse.Namespace, include_enrichment: bool) -> None:
    settings = load_settings()
    scenario_data = load_scenario(settings, args.scenario)

    poses_by_image: dict[str, dict[str, list[float]]] | None = None
    if include_enrichment:
        poses_by_image_path = resolve_poses_by_image_path(
            settings=settings,
            scenario_name=args.scenario,
            explicit_path=args.poses_by_image_path,
        )
        poses_by_image = load_poses_by_image_map(poses_by_image_path)

    frame_paths = resolve_frame_paths(args)
    loop_timestamp = make_experiment_timestamp()
    variant_tag = "yes_enrichment" if include_enrichment else "no_enrichment"
    pipeline_config = build_pipeline_config(args, include_enrichment)

    output_dir = (
        settings.project_root
        / "humanoids_results"
        / "ablation_study"
        / "scene"
        / variant_tag
        / args.scenario
        / loop_timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "terminal_log.txt"

    with open(log_path, "w", encoding="utf-8", buffering=1) as log_file:
        tee_out = _TeeStream(sys.stdout, log_file)
        tee_err = _TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            _run_body(
                args=args,
                include_enrichment=include_enrichment,
                settings=settings,
                scenario_data=scenario_data,
                poses_by_image=poses_by_image,
                frame_paths=frame_paths,
                loop_timestamp=loop_timestamp,
                variant_tag=variant_tag,
                pipeline_config=pipeline_config,
                output_dir=output_dir,
            )

    print(f"[OK] Terminal log saved to: {log_path}")


def _run_body(
    args: argparse.Namespace,
    include_enrichment: bool,
    settings,
    scenario_data: dict[str, Any],
    poses_by_image: dict[str, dict[str, list[float]]] | None,
    frame_paths: list[str],
    loop_timestamp: str,
    variant_tag: str,
    pipeline_config: dict[str, Any],
    output_dir: Path,
) -> None:
    print("\n======================================================")
    print("SCENE-PERCEPTION + PLANNING ABLATION (single-pass, no execution/validation)")
    print(f"Scenario:            {args.scenario}")
    print(f"Variant:             {variant_tag}")
    print(f"Frames to process:   {len(frame_paths)}")
    print("======================================================")

    run_summary: dict[str, Any] = {
        "module": "ablation_scene_perception_planning_run_summary",
        "variant": variant_tag,
        "include_scene_enrichment": include_enrichment,
        "scenario_name": args.scenario,
        "loop_timestamp": loop_timestamp,
        "timestamp": datetime.now().isoformat(),
        "config": pipeline_config,
        "frames": [],
    }

    for cycle_idx, image_path in enumerate(frame_paths, start=1):
        cycle_name = make_cycle_name(cycle_idx)
        cycle_timestamp = make_experiment_timestamp()
        image_name = Path(image_path).name

        print("\n------------------------------------------------------")
        print(f"FRAME {cycle_idx}/{len(frame_paths)} | image={image_name}")
        print("------------------------------------------------------")

        scenario_context = make_scenario_context(scenario_data, image_path)
        frame_record: dict[str, Any] = {
            "cycle_index": cycle_idx,
            "cycle_name": cycle_name,
            "image": image_name,
            "error": None,
        }

        try:
            scene_description_artifact = execute_scene_description_step(
                settings=settings,
                scenario_name=args.scenario,
                scenario_context=scenario_context,
                version=args.scene_v,
                model_name=args.scene_model,
                loop_timestamp=loop_timestamp,
                cycle_name=cycle_name,
                cycle_idx=cycle_idx,
                cycle_timestamp=cycle_timestamp,
                pipeline_config=pipeline_config,
                image_path=image_path,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print_output_block("scene_description", scene_description_artifact["output"])
            frame_record["scene_description"] = {
                "output": scene_description_artifact["output"],
                "paths": scene_description_artifact["paths"],
            }

            if include_enrichment:
                scene_description_full_artifact = execute_scene_description_full_step(
                    settings=settings,
                    scenario_name=args.scenario,
                    scenario_context=scenario_context,
                    version=args.scene_v,
                    model_name=args.scene_model,
                    loop_timestamp=loop_timestamp,
                    cycle_name=cycle_name,
                    cycle_idx=cycle_idx,
                    cycle_timestamp=cycle_timestamp,
                    scene_description=scene_description_artifact["output"],
                    pipeline_config=pipeline_config,
                    image_path=image_path,
                    poses_by_image=poses_by_image,
                    safety_threshold=args.grounding_safety_threshold,
                    include_debug_mapping=args.grounding_debug_mapping,
                )
                scene_full_output = scene_description_full_artifact["output"]
                print_output_block("scene_description_full (scene_enrichment)", scene_full_output)
                frame_record["scene_description_full"] = {
                    "output": scene_full_output,
                    "paths": scene_description_full_artifact["paths"],
                }
            else:
                scene_full_output = scene_description_artifact["output"]
                print(
                    "[ABLATION] scene_description_full skipped -- planner "
                    "receives the plain scene_description object list only."
                )

            sequential_plan_artifact = execute_vlm_planning_step(
                settings=settings,
                scenario_name=args.scenario,
                scenario_context=scenario_context,
                version=args.plan_v,
                model_name=args.plan_model,
                loop_timestamp=loop_timestamp,
                cycle_name=cycle_name,
                cycle_idx=cycle_idx,
                cycle_timestamp=cycle_timestamp,
                scene_description_full=scene_full_output,
                scene_version=args.scene_v,
                scene_model=args.scene_model,
                pipeline_config=pipeline_config,
                temperature=args.temperature,
                top_p=args.top_p,
                recovery_context=None,
            )
            sequential_plan_output = sequential_plan_artifact["output"]
            print_output_block("vlm_planning (sequential_plan)", sequential_plan_output)
            frame_record["vlm_planning"] = {
                "output": sequential_plan_output,
                "paths": sequential_plan_artifact["paths"],
            }

            simultaneous_actions_artifact = execute_simultaneous_actions_step(
                settings=settings,
                scenario_name=args.scenario,
                scenario_context=scenario_context,
                version=args.sim_v,
                model_name=args.sim_model,
                loop_timestamp=loop_timestamp,
                cycle_name=cycle_name,
                cycle_idx=cycle_idx,
                cycle_timestamp=cycle_timestamp,
                scene_description_full=scene_full_output,
                sequential_plan=sequential_plan_output,
                scene_version=args.scene_v,
                scene_model=args.scene_model,
                plan_version=args.plan_v,
                plan_model=args.plan_model,
                pipeline_config=pipeline_config,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print_output_block(
                "simultaneous_actions (raw model output)",
                simultaneous_actions_artifact["output"],
            )
            frame_record["simultaneous_actions"] = {
                "output": simultaneous_actions_artifact["output"],
                "paths": simultaneous_actions_artifact["paths"],
            }

            try:
                stages = extract_stages(simultaneous_actions_artifact["output"])
                frame_record["stages"] = stages
                print_output_block("stages (normalized, with Preconditions/Postconditions)", stages)
            except ValueError as exc:
                frame_record["stages"] = None
                frame_record["stages_error"] = str(exc)
                print(f"[WARN] Could not normalize stages for {image_name}: {exc}")

        except Exception as exc:
            frame_record["error"] = str(exc)
            print(f"[ERROR] Frame {image_name} failed: {exc}")

        run_summary["frames"].append(frame_record)

    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    errored = sum(1 for f in run_summary["frames"] if f.get("error"))
    print("\n======================================================")
    print("ABLATION RUN SUMMARY")
    print(f"Variant:          {variant_tag}")
    print(f"Frames run:       {len(frame_paths)}")
    print(f"Frames errored:   {errored}")
    print(f"run_summary.json: {summary_path}")
    print(f"terminal_log.txt: {output_dir / 'terminal_log.txt'}")
    print("======================================================")

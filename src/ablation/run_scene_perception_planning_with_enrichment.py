"""Ablation entry point: scene_description + scene_description_full
(scene_enrichment) + vlm_planning + simultaneous_actions.

Single-pass inspection tool -- no execution, validation, or recovery. See
`scene_perception_planning_common.py` for the shared driver and full
docstring, and run alongside `run_scene_perception_planning_no_enrichment.py`
(identical flags, minus --poses-by-image-path/--grounding-*) to isolate the
effect of scene_enrichment on the resulting plan.
"""

from __future__ import annotations

from src.ablation.scene_perception_planning_common import build_parser, run


def main() -> None:
    args = build_parser(include_enrichment=True).parse_args()
    run(args, include_enrichment=True)


if __name__ == "__main__":
    main()

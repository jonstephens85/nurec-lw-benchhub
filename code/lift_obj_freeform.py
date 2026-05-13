# Copyright 2026 Jonathan Stephens
# Licensed under the Apache License, Version 2.0
#
# A LiftObj variant for custom USD scenes (e.g. NuRec/3DGS captures) that
# have no robocasa fixtures. Differences from LiftObj:
#
# - Skips robocasa fixture lookup (no counter required)
# - Cube spawn pose is set in absolute world coordinates via fix_object_pose_cfg
# - Robot spawn pose is supplied via the CSV (matched on layout = USD path)
# - Success threshold uses an absolute z value in the scene's world frame
# - env_spacing is widened to keep splat-backed envs from bleeding into each other
#
# To use with your own scene, edit the class-level constants below.

import torch

from lw_benchhub.utils.env import ExecuteMode
from lw_benchhub_tasks.lightwheel_robocasa_tasks.single_stage.lift_obj import LiftObj


class LiftObjFreeform(LiftObj):
    """LiftObj variant for custom USD scenes with no robocasa fixtures."""

    task_name: str = "LiftObjFreeform"

    # ============================================================
    # Edit these per scene
    # ============================================================
    # Cube spawn position in your scene's world coords (meters)
    CUBE_WORLD_POS = (-0.30, -0.06, 0.755)

    # Cube z must exceed this threshold to count as "lifted"
    # Set to roughly CUBE_WORLD_POS[2] + 0.10
    SUCCESS_HEIGHT_Z = 0.855

    # Distance between parallel envs in multi-env training
    # 30m default is too tight for splats with far-flung stray Gaussians
    ENV_SPACING = 50.0
    # ============================================================

    def __init__(self):
        super().__init__()
        # Override the kitchen-coords placement inherited from LiftObj
        # with absolute world coords for our scene
        self.fix_object_pose_cfg = {"object": {"pos": self.CUBE_WORLD_POS}}

    def _setup_kitchen_references(self, scene):
        """Skip the robocasa fixture lookup.

        The base LiftObj registers self.counter and sets init_robot_base_ref
        to it. Our scene has no counter fixture, so we leave both unset and
        rely on the CSV row to provide the robot's spawn pose directly.
        """
        # Intentionally do NOT call super() - don't try to find a counter
        # but DO initialize fixture_refs so _init_ref_fixtures has something
        # to iterate over (empty is fine)
        self.fixture_refs = {}
        self.init_robot_base_ref = None

    def _get_obj_cfgs(self):
        """Return an object cfg with no fixture reference.

        The placement sampler will still run on this and produce some default
        position, but _apply_object_placements overrides it with
        fix_object_pose_cfg afterward, so the sampler's output is discarded.

        The placement region needs to be non-degenerate (a 10cm x 10cm area)
        or the sampler will fail validation in an infinite retry loop.
        """
        return [
            dict(
                name="object",
                obj_groups="cube",
                graspable=True,
                placement=dict(
                    size=(0.10, 0.10),
                    pos=(self.CUBE_WORLD_POS[0], self.CUBE_WORLD_POS[1]),
                    offset=(0, 0),
                ),
            )
        ]

    def modify_env_cfg(self, env_cfg):
        """Widen env_spacing for multi-env splat rendering."""
        env_cfg = super().modify_env_cfg(env_cfg)
        env_cfg.scene.env_spacing = self.ENV_SPACING
        return env_cfg

    def _check_success(self, env):
        """Cube is 'lifted' if its world z exceeds SUCCESS_HEIGHT_Z.

        Identical to LiftObj's check but with a configurable threshold
        rather than the hardcoded 0.965 (robocasa kitchen frame).
        """
        if self.context.execute_mode == ExecuteMode.TRAIN:
            return torch.tensor([False], device=env.device).repeat(env.num_envs)
        object_height = env.scene["object"].data.root_pos_w[:, 2]
        return object_height >= self.SUCCESS_HEIGHT_Z

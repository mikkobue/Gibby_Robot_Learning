# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from . import mdp

# Re-export key configs for easier imports
from .gibby_cfg import GIBBY_CFG  # noqa: F401
from .gibby_env_cfg import GibbyClimbEnvCfg  # noqa: F401

##
# Register Gym environments.
##


gym.register(
    id="Template-Isaac-Monkee-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_monkee_env_cfg:IsaacMonkeeEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Monkee tree climbing task (Gibby)
gym.register(
    id="Monkee-GibbyClimb-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gibby_env_cfg:GibbyClimbEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
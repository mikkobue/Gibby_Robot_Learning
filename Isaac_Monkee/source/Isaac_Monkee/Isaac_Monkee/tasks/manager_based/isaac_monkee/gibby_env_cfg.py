import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from . import mdp
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .gibby_cfg import GIBBY_CFG


@configclass
class GibbySceneCfg(InteractiveSceneCfg):
    num_envs: int = 1024
    env_spacing: float = 3.0

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0)
    )

    # The 0.5m radius tree (cylinder)
    tree = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tree",
        spawn=sim_utils.CylinderCfg(
            radius=0.5, height=5.0,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
            # Warm-start friction to avoid early contact explosions; can be ramped later
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.5)),
    )

    robot = GIBBY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class GibbyObservationsCfg(ObservationGroupCfg):
    joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
    joint_vel = ObservationTermCfg(func=mdp.joint_vel)
    root_pos = ObservationTermCfg(func=mdp.root_pos_w)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class GibbyRewardsCfg:
    # Primary objective: maximize height
    climb_height = RewardTermCfg(func=mdp.root_height, weight=20.0)
    # Shaping: encourage upward motion only (no reward for going down)
    climb_progress = RewardTermCfg(func=mdp.upward_velocity, weight=2.0)
    # Regularization
    penalize_effort = RewardTermCfg(func=mdp.joint_torques_l2, weight=-0.005)
    penalize_jerky_motion = RewardTermCfg(func=mdp.joint_acc_l2, weight=-0.001)


@configclass
class GibbyTerminationsCfg:
    fell_off = TerminationTermCfg(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)


@configclass
class GibbyActionsCfg:
    arm_action = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*_joint.*"], scale=0.6)
    screw_action = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=[".*_adhesion.*"], scale=2.5)


@configclass
class GibbyClimbEnvCfg(ManagerBasedRLEnvCfg):
    decimation: int = 4
    episode_length_s: float = 20.0

    def __post_init__(self):
        self.viewer.eye = (0.0, 5.0, 5.0)
        self.sim.dt = 0.01
        self.sim.render_interval = 4

        self.scene = GibbySceneCfg(num_envs=1024, env_spacing=3.0)
        self.observations = {"policy": GibbyObservationsCfg()}
        self.actions = GibbyActionsCfg()
        self.rewards = GibbyRewardsCfg()
        self.terminations = GibbyTerminationsCfg()

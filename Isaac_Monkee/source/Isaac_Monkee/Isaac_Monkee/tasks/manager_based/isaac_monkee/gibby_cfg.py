import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# Articulation configuration for the Monkee (Gibby) robot.
GIBBY_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/ubuntu/Monkee/Gibby.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            # Increase solver iterations for stability under high-friction contact
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.6, 0.0, 0.5),
        # Flip 180° from current (-90° -> +90° yaw around +Z)
        # Quaternion order is (w, x, y, z)
        rot=(0.70710678, 0.0, 0.0, 0.70710678),
        joint_pos={
            ".*_joint.*": 0.0,
            ".*_adhesion.*": 0.0,
        },
    ),
    actuators={
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_joint.*"],
            stiffness=80.0,
            damping=4.0,
            effort_limit=100.0,
            velocity_limit=5.0,
        ),
        "screws": ImplicitActuatorCfg(
            joint_names_expr=[".*_adhesion.*"],
            stiffness=0.0,
            damping=10.0,
            effort_limit=50.0,
            velocity_limit=20.0,
        ),
    },
)

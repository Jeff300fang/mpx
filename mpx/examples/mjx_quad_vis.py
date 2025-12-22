import jax.numpy as jnp
import jax
import mujoco

# -----------------------------
# JAX configuration
# -----------------------------
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import numpy as np
from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.mujoco.visual import render_sphere, render_vector

import mpx.utils.mpc_wrapper as mpc_wrapper
import mpx.config.config_aliengo as config

from timeit import default_timer as timer


# -----------------------------
# Define robot and scene parameters
# -----------------------------
robot_name = "aliengo"
scene_name = "random_boxes"

robot_feet_geom_names = dict(FR="FR", FL="FL", RR="RR", RL="RL")

mpc_frequency = config.mpc_frequency
state_observables_names = tuple(QuadrupedEnv.ALL_OBS)


# -----------------------------
# Initialize environment
# -----------------------------
sim_frequency = 200.0
env = QuadrupedEnv(
    robot=robot_name,
    scene=scene_name,
    sim_dt=1 / sim_frequency,
    ref_base_lin_vel=0.0,
    ground_friction_coeff=0.7,
    base_vel_command_type="human",
    state_obs_names=state_observables_names,
)

obs = env.reset(random=False)

# MPC
mpc = mpc_wrapper.MPCControllerWrapper(config)

# Set initial configuration
env.mjData.qpos = jnp.concatenate([config.p0, config.quat0, config.q0])

# -----------------------------
# Create viewer ONCE
# -----------------------------
env.render()

# 🔴 DISABLE CAMERA TRACKING COMPLETELY 🔴
# Monkey-patch the method that forces lookat every frame
env._update_camera_target = lambda cam, target: None

# Force free camera mode
env.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
env.viewer.cam.trackbodyid = -1
env.viewer.cam.fixedcamid = -1


# -----------------------------
# Simulation state
# -----------------------------
counter = 0
tau = jnp.zeros(config.n_joints)
q = config.q0.copy()
dq = jnp.zeros(config.n_joints)

delay = int(0.007 * sim_frequency)
print("Delay:", delay)

mpc.robot_height = config.robot_height
mpc.reset(env.mjData.qpos.copy(), env.mjData.qvel.copy())


# -----------------------------
# Main loop
# -----------------------------
while env.viewer.is_running():
    qpos = env.mjData.qpos.copy()
    qvel = env.mjData.qvel.copy()

    if (counter % (sim_frequency / mpc_frequency) == 0) or counter == 0:
        ref_base_lin_vel = env._ref_base_lin_vel_H
        ref_base_ang_vel = np.array([0.0, 0.0, env._ref_base_ang_yaw_dot])

        input_vec = np.array(
            [
                ref_base_lin_vel[0],
                ref_base_lin_vel[1],
                ref_base_lin_vel[2],
                ref_base_ang_vel[0],
                ref_base_ang_vel[1],
                ref_base_ang_vel[2],
                config.robot_height,
            ]
        )

        contact_temp, _ = env.feet_contact_state()
        contact = np.array(
            [contact_temp[robot_feet_geom_names[leg]] for leg in ["FL", "FR", "RL", "RR"]]
        )

        if counter != 0:
            for _ in range(delay):
                qpos = env.mjData.qpos.copy()
                qvel = env.mjData.qvel.copy()

                tau_fb = (
                    10.0 * (q - qpos[7 : 7 + config.n_joints])
                    - 2.0 * (qvel[6 : 6 + config.n_joints])
                )

                env.step(action=tau + tau_fb)
                counter += 1

        start = timer()
        tau, q, dq = mpc.run(qpos, qvel, input_vec, contact)
        print("Time taken for MPC:", timer() - start)

    tau_fb = (
        10.0 * (q - qpos[7 : 7 + config.n_joints])
        - 2.0 * (qvel[6 : 6 + config.n_joints])
    )

    env.step(action=tau + tau_fb)
    counter += 1

    # Render WITHOUT camera snapping
    env.render()

    # Optional: re-assert free camera (harmless, ultra-safe)
    env.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    env.viewer.cam.trackbodyid = -1
    env.viewer.cam.fixedcamid = -1


env.close()

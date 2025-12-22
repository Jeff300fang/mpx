
import jax.numpy as jnp
import jax
import mujoco
# Update JAX configuration
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
 
import numpy as np
from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.mujoco.visual import render_sphere, render_vector
 
import mpx.utils.mpc_wrapper as mpc_wrapper
import mpx.config.config_aliengo as config

from timeit import default_timer as timer
from mpx.utils.render_obstacles import render_static_vertical_cylinder
# Set GPU device for JAX
# gpu_device = jax.devices('gpu')[0]
# jax.default_device(gpu_device)

# Define robot and scene parameters
robot_name = "aliengo"   # "aliengo", "mini_cheetah", "go2", "hyqreal", ...
scene_name = "flat"
robot_feet_geom_names = dict(FR='FR',FL='FL', RR='RR' , RL='RL')
robot_leg_joints = dict(FR=['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', ],
                        FL=['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', ],
                        RR=['RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', ],
                        RL=['RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'])
mpc_frequency = config.mpc_frequency
state_observables_names = tuple(QuadrupedEnv.ALL_OBS)  # return all available state observables
 
# Initialize simulation environment
sim_frequency = 100.0
env = QuadrupedEnv(robot=robot_name,
                   scene=scene_name,
                   sim_dt = 1/sim_frequency,  # Simulation time step [s]
                   ref_base_lin_vel=0.0, # Constant magnitude of reference base linear velocity [m/s]
                   ground_friction_coeff=0.7,  # pass a float for a fixed value
                   base_vel_command_type="human",  # "forward", "random", "forward+rotate", "human"
                   state_obs_names=state_observables_names,  # Desired quantities in the 'state'
                   )
obs = env.reset(random=False)


# --------- Define Constraints ---------
def render_obstacles(centers, radii):
    for (cx, cy), radius in zip(centers, radii):
        cyl_xy = np.array([cx, cy])   # x, y
        cyl_height = 1.0
        cyl_radius = radius
        cyl_z_base = 0.0                # sits on ground
        cyl_color = np.array([0.6, 0.6, 0.6, 1.0])

        static_cyl_id = -1
        static_cyl_id = render_static_vertical_cylinder(
            env.viewer,
            center_xy=cyl_xy,
            height=cyl_height,
            radius=(cyl_radius - 0.33), # Obstacles are inflated to capture size of quadruped
            z_base=cyl_z_base,
            color=cyl_color,
            geom_id=static_cyl_id,
        )

def random_circles(key, K, radius=0.43, dtype=jnp.float32):
    """
    Uniformly sample K circle centers:
      x ~ U[1, 7], y ~ U[-3, 3]
    All circles share the same radius.

    Returns:
      centers: (K, 2)
      radii:   (K,)
    """
    key_x, key_y = jax.random.split(key, 2)
    x = jax.random.uniform(key_x, (K,), minval=1.0, maxval=7.0, dtype=dtype)
    y = jax.random.uniform(key_y, (K,), minval=-1.0, maxval=1.0, dtype=dtype)
    centers = jnp.stack([x, y], axis=1)
    radii = jnp.full((K,), radius, dtype=dtype)
    return centers, radii

# Example usage
key = jax.random.PRNGKey(1)
centers, radii = random_circles(key, K=10, radius=0.43)

# centers = jnp.array([[2.0, 0.1],
#                      [4.0, 0.15]], dtype=jnp.float32)

# radii = jnp.array([0.43, 0.43], dtype=jnp.float32)
# --------------------------------------


# Define the MPC wrapper
mpc = mpc_wrapper.MPCControllerWrapper(config, centers, radii)
env.mjData.qpos = jnp.concatenate([config.p0, config.quat0,config.q0])
env.render()
ids = []
# for i in range(8):
#      ids.append(render_vector(env.viewer,
#               np.zeros(3),
#               np.zeros(3),
#               0.1,
#               np.array([1, 0, 0, 1])))
counter = 0
# Main simulation loop
tau = jnp.zeros(config.n_joints)
tau_old = jnp.zeros(config.n_joints)
delay = int(0.007*sim_frequency)
print('Delay: ',delay)

q = config.q0.copy()
dq = jnp.zeros(config.n_joints)
mpc_time = 0
mpc.robot_height = config.robot_height
mpc.reset(env.mjData.qpos.copy(),env.mjData.qvel.copy())

while env.viewer.is_running():
 
    qpos = env.mjData.qpos.copy()
    qvel = env.mjData.qvel.copy()
    if (counter % (sim_frequency / mpc_frequency) == 0 or counter == 0):
    
 
        ref_base_lin_vel = env._ref_base_lin_vel_H
        ref_base_ang_vel =  np.array([0., 0., env._ref_base_ang_yaw_dot])
 
        
        input = np.array([ref_base_lin_vel[0],ref_base_lin_vel[1],ref_base_lin_vel[2],
                           ref_base_ang_vel[0],ref_base_ang_vel[1],ref_base_ang_vel[2],
                           config.robot_height])
        
        contact_temp, _ = env.feet_contact_state()
        
        contact = np.array([contact_temp[robot_feet_geom_names[leg]] for leg in ['FL','FR','RL','RR']])

        if counter != 0:
            for i in range(delay):
                qpos = env.mjData.qpos.copy()
                qvel = env.mjData.qvel.copy()
                # tau_fb = K@(x-np.concatenate([qpos,qvel]))

                tau_fb = 10*(q-qpos[7:7+config.n_joints]) -2*(qvel[6:6+config.n_joints])
                state, reward, is_terminated, is_truncated, info = env.step(action=tau + tau_fb)
                counter += 1
        start = timer()
        tau, q, dq = mpc.run(qpos,qvel,input,contact, centers, radii)   
        stop = timer()
        print("Time taken for MPC: ", stop-start)   
        render_obstacles(centers, radii)
        # for i in range(4):
        #     render_sphere(env.viewer,
        #                   collision_point[3*i:3*i+3],
        #                   0.2,
        #                   np.array([1, 0, 0, 0.5]),
        #                   ids[i])

    tau_fb = 10*(q-qpos[7:7+config.n_joints])-2*(qvel[6:6+config.n_joints])
    state, reward, is_terminated, is_truncated, info = env.step(action= tau + tau_fb)

    # time.sleep(0.1)
    counter += 1
    env.render()
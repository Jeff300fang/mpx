import os
import sys

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, "..")))

jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


import mpx.config.config_h1 as config
from mpx.utils.models import h1_wb_dynamics

def main():
    xml_path = os.path.join(dir_path, "../data/unitree_h1/mjx_scene_h1_walk.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    mjx_model = mjx.put_model(model)
    n_joints = config.n_joints
    dt = config.dt
    contact_id = []
    for name in config.contact_frame:
        contact_id.append(
            mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        )

    body_id = []
    for name in config.body_name:
        body_id.append(
            mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_BODY, name)
        )

    qpos0 = jnp.concatenate([config.p0, config.quat0, config.q0])
    qvel0 = jnp.zeros(6 + n_joints)

    dummy_feet = jnp.zeros(12)
    dummy_grf = jnp.zeros(12)

    x = jnp.concatenate([qpos0, qvel0, dummy_feet, dummy_grf])
    u = jnp.zeros(n_joints)

    T = 20
    parameter = jnp.ones((T, 4))
    t = 0

    x_next = h1_wb_dynamics(
        model=model,
        mjx_model=mjx_model,
        contact_id=contact_id,
        body_id=body_id,
        n_joints=n_joints,
        dt=dt,
        x=x,
        u=u,
        t=t,
        parameter=parameter,
    )

    print("x shape:", x.shape)
    print("x_next shape:", x_next.shape)
    print("base pos next:", x_next[:3])
    print("base quat next:", x_next[3:7])
    print("joint pos next:", x_next[7:7+n_joints])

if __name__ == "__main__":
    main()
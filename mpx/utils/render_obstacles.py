import jax
import mujoco
# Update JAX configuration
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
 
import numpy as np

def render_static_vertical_cylinder(
    viewer,
    center_xy: np.ndarray,
    height: float,
    radius: float,
    z_base: float = 0.0,
    color: np.ndarray = np.array([0.7, 0.7, 0.7, 1.0]),
    geom_id: int = -1,
):
    """
    Render a static vertical cylinder aligned with world +Z.

    Args:
        center_xy: (2,) array, (x, y) location
        height: cylinder height
        radius: cylinder radius
        z_base: z coordinate of the bottom of the cylinder
        color: RGBA
        geom_id: reuse id
    """
    if viewer is None:
        return -1

    if geom_id < 0 or geom_id is None:
        viewer.user_scn.ngeom += 1
        geom_id = viewer.user_scn.ngeom - 1

    geom = viewer.user_scn.geoms[geom_id]

    # World-aligned vertical cylinder
    pos = np.array([
        center_xy[0],
        center_xy[1],
        z_base + height / 2.0,  # MuJoCo expects center position
    ])

    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=np.array([radius, height / 2.0, radius]),
        pos=pos,
        mat=np.eye(3).reshape(9),  # identity => vertical
        rgba=color,
    )

    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    geom.segid = -1
    geom.objid = -1
    return geom_id

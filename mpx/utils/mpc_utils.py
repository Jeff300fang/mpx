import jax
from jax import numpy as jnp
from functools import partial
from mujoco.mjx._src import math
from jax.scipy.spatial.transform import Rotation
from dataclasses import dataclass

# def timer_run(duty_factor,step_freq, leg_time, dt):
#     # Extract relevant fields
#     # Update timer
#     leg_time = leg_time + dt * step_freq
#     leg_time = jnp.where(leg_time > 1, leg_time - 1, leg_time)
#     contact = jnp.where(leg_time < duty_factor, 1, 0)

#     return contact, leg_time
# def terrain_orientation(liftoff_pos,Ryaw):

#     # Calculate the vectors between the legs
#     vec_front_back = (liftoff_pos[:3] + liftoff_pos[3:6] - liftoff_pos[6:9] - liftoff_pos[9:12])/2
#     # vec_left_right = (liftoff_pos[:3] + liftoff_pos[6:9] - liftoff_pos[3:6] - liftoff_pos[9:12])/2
#     #DO NOT ADJUST THE ROLL
#     vec_left_right = Ryaw@jnp.array([0,1,0])
#     # Compute the normal vector to the plane
#     normal_vector = jnp.cross(vec_front_back, vec_left_right)

#     # Normalize the vectors
#     vec_front_back = vec_front_back / math.norm(vec_front_back)
#     vec_left_right = vec_left_right / math.norm(vec_left_right)
#     normal_vector = normal_vector / math.norm(normal_vector)

#     # Create the rotation matrix
#     rotation_matrix = Rotation.from_matrix(jnp.stack([vec_front_back, vec_left_right, normal_vector], axis=1))

#     # Convert the rotation matrix to a quaternion
#     quat = rotation_matrix.as_quat()

#     return jnp.roll(quat,1)

# @partial(jax.jit, static_argnums=(0,1,2,3,4,5))
# def reference_generator(use_terrain_estimator,N,dt,n_joints,n_contact,mass,foot0,q0,t_timer, x, foot, input, duty_factor, step_freq,step_height,liftoff,contact,clearence_speed):
#     p = x[:3]
#     quat = x[3:7]
#     # q = x[7:7+n_joints]
#     dp = x[7+n_joints:10+n_joints]
#     # omega = x[10+n_joints:13+n_joints]
#     # dq = x[13+n_joints:13+2*n_joints]
#     yaw = jnp.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 1 - 2*(quat[2]*quat[2] + quat[3]*quat[3]))
#     Ryaw = jnp.array([[jnp.cos(yaw), -jnp.sin(yaw), 0],[jnp.sin(yaw), jnp.cos(yaw), 0],[0, 0, 1]])
#     proprio_height = input[6] + jnp.sum(liftoff[2::3])/n_contact
#     p = jnp.array([p[0], p[1], proprio_height])
#     if use_terrain_estimator:
#         quat_ref = jnp.tile(terrain_orientation(liftoff,Ryaw), (N+1, 1))
#     else:
#         quat_ref = jnp.tile(jnp.array([1, 0, 0, 0]), (N+1, 1))
#     q_ref = jnp.tile(q0, (N+1, 1))
#     contact_sequence = jnp.zeros(((N+1), n_contact))
#     pitch = jnp.arcsin(2 * (quat_ref[0,0] * quat_ref[0,2] - quat_ref[0,3] * quat_ref[0,1]))
#     Rpitch = jnp.array([[jnp.cos(pitch), 0, jnp.sin(pitch)], [0, 1, 0], [-jnp.sin(pitch), 0, jnp.cos(pitch)]])
    
#     ref_lin_vel = Ryaw@Rpitch@input[:3]
#     ref_ang_vel = input[3:6]
#     p_ref_x = jnp.arange(N+1) * dt * ref_lin_vel[0] + p[0]
#     p_ref_y = jnp.arange(N+1) * dt * ref_lin_vel[1] + p[1]
#     p_ref_z = jnp.ones(N+1) * proprio_height
#     p_ref = jnp.stack([p_ref_x, p_ref_y, p_ref_z], axis=1)
#     dp_ref = jnp.tile(ref_lin_vel, (N+1, 1))
#     omega_ref = jnp.tile(ref_ang_vel, (N+1, 1))
#     foot_ref = jnp.tile(foot, (N+1, 1))
#     foot0_projected = jnp.tile(p, n_contact) + foot0 @ jax.scipy.linalg.block_diag(*([Ryaw] * n_contact)).T
#     grf_ref = jnp.zeros((N+1, 3*n_contact))

#     #Estimate Early contact
#     des_contact, current_timer = timer_run(duty_factor, step_freq, t_timer, dt)
#     early_contact = jnp.where(jnp.logical_and(jnp.logical_and(des_contact==0,contact==1),current_timer > 0.5 + 0.5*duty_factor),1,0)    

#     def foot_fn(t,carry):

#         timer_seq, contact_sequence,new_foot,liftoff_x,liftoff_y,liftoff_z,grf_new = carry

#         new_foot_x = new_foot[t-1,::3]
#         new_foot_y = new_foot[t-1,1::3]
#         new_foot_z = new_foot[t-1,2::3]

#         new_contact_sequence, new_t = timer_run(duty_factor, step_freq, timer_seq[t-1,:], dt)

#         contact_sequence = contact_sequence.at[t,:].set(new_contact_sequence)
#         timer_seq = timer_seq.at[t,:].set(new_t)

#         liftoff_x = jnp.where(jnp.logical_and(jnp.logical_not(contact_sequence[t,:]),contact_sequence[t-1,:]),new_foot_x,liftoff_x)
#         liftoff_y = jnp.where(jnp.logical_and(jnp.logical_not(contact_sequence[t,:]),contact_sequence[t-1,:]),new_foot_y,liftoff_y)
#         liftoff_z = jnp.where(jnp.logical_and(jnp.logical_not(contact_sequence[t,:]),contact_sequence[t-1,:]),new_foot_z,liftoff_z)

#         def calc_foothold(direction):
#             f1 = 0.5*ref_lin_vel[direction]*duty_factor/step_freq
#             f2 = jnp.sqrt(input[6]/9.81)*(dp[direction]-ref_lin_vel[direction])
#             f = f1 + f2 + foot0_projected[direction::3]
#             return f

#         foothold_x = calc_foothold(0)
#         foothold_y = calc_foothold(1)

#         def cubic_splineXY(current_foot, foothold,initial_velocity,val):
#             a0 = current_foot
#             a1 = initial_velocity
#             a2 = 3*(foothold - current_foot) - 2*initial_velocity
#             a3 = initial_velocity - 2*(foothold - current_foot)
#             return a0 + a1*val + a2*val**2 + a3*val**3

#         def cubic_splineZ(current_foot, foothold, step_height,val):
            
#             initial_speed = 0.7

#             a = 16*step_height - 8*foothold - 8*current_foot - 2*initial_speed
#             b = 5*initial_speed + 14*foothold + 18*current_foot - 32*step_height
#             c = 16*step_height - 5*foothold - 11*current_foot - 4*initial_speed
#             d = initial_speed
#             e = current_foot
#             return a*val**4 + b*val**3 + c*val**2 + d*val + e
        
#         initial_speed = - ref_lin_vel / (jnp.linalg.norm(ref_lin_vel) + 1e-6) * clearence_speed

#         new_foot_x = jnp.where(jnp.logical_or(new_contact_sequence>0,early_contact==1), new_foot[t-1,::3], cubic_splineXY(liftoff_x, foothold_x,initial_speed[0],(new_t-duty_factor)/(1-duty_factor)))
#         new_foot_y = jnp.where(jnp.logical_or(new_contact_sequence>0,early_contact==1), new_foot[t-1,1::3], cubic_splineXY(liftoff_y, foothold_y,initial_speed[1],(new_t-duty_factor)/(1-duty_factor)))
#         new_foot_z = jnp.where(jnp.logical_or(new_contact_sequence>0,early_contact==1), new_foot[t-1,2::3], cubic_splineZ(liftoff_z,liftoff_z,liftoff_z + step_height,(new_t-duty_factor)/(1-duty_factor)))

#         new_foot = new_foot.at[t,::3].set(new_foot_x)
#         new_foot = new_foot.at[t,1::3].set(new_foot_y)
#         new_foot = new_foot.at[t,2::3].set(new_foot_z)

#         grf_new = grf_new.at[t,2::3].set((new_contact_sequence*mass*9.81/(jnp.sum(new_contact_sequence)+1e-5)))

#         return (timer_seq, contact_sequence,new_foot,liftoff_x,liftoff_y,liftoff_z,grf_new)

#     liftoff_x = liftoff[::3]
#     liftoff_y = liftoff[1::3]
#     liftoff_z = liftoff[2::3]
#     timer_sequence_in = jnp.tile(t_timer, (N+1, 1))
#     init_carry = (timer_sequence_in, contact_sequence,foot_ref,liftoff_x,liftoff_y,liftoff_z,grf_ref)
#     timer_sequence, contact_sequence,foot_ref, liftoff_x,liftoff_y,liftoff_z,grf_ref = jax.lax.fori_loop(0,N+1,foot_fn, init_carry)

#     liftoff = liftoff.at[::3].set(liftoff_x)
#     liftoff = liftoff.at[1::3].set(liftoff_y)
#     liftoff = liftoff.at[2::3].set(liftoff_z)

#     return jnp.concatenate([p_ref, quat_ref, q_ref, dp_ref, omega_ref, foot_ref, contact_sequence,grf_ref], axis=1),jnp.concatenate([contact_sequence], axis=1), liftoff
# --- add to mpx/utils/mpc_utils.py (e.g. right below reference_generator) ---
import jax
import jax.numpy as jnp
from functools import partial

# ----------------------------
# Utilities you already had
# ----------------------------
def timer_run(duty_factor, step_freq, leg_time, dt):
    leg_time = leg_time + dt * step_freq
    leg_time = jnp.where(leg_time > 1.0, leg_time - 1.0, leg_time)
    contact = jnp.where(leg_time < duty_factor, 1.0, 0.0)
    return contact, leg_time


# ----------------------------
# Slice helpers
# ----------------------------
def _state_slices(n_joints: int, n_contact: int, grf_as_state: bool):
    """
    Matches YOUR x0 / X_in state layout:
      x = [qpos, qvel, foot, (grf if grf_as_state)]
      qpos = 7 + n_joints
      qvel = 6 + n_joints
      foot = 3*n_contact
      grf  = 3*n_contact (optional)
    """
    nq = 7 + n_joints
    nv = 6 + n_joints

    i_qpos0, i_qpos1 = 0, nq
    i_qvel0, i_qvel1 = nq, nq + nv
    i_foot0, i_foot1 = i_qvel1, i_qvel1 + 3 * n_contact

    if grf_as_state:
        i_grf0, i_grf1 = i_foot1, i_foot1 + 3 * n_contact
    else:
        i_grf0, i_grf1 = -1, -1

    return (i_qpos0, i_qpos1, i_qvel0, i_qvel1, i_foot0, i_foot1, i_grf0, i_grf1)


def _ref_slices(n_joints: int, n_contact: int):
    """
    Matches YOUR solver reference layout (your concatenation):
      [p(3), quat(4), q(nj), dp(3), omega(3), foot(3*nc), contact(nc), grf(3*nc)]
    """
    i_p0, i_p1   = 0, 3
    i_qt0, i_qt1 = 3, 7
    i_q0, i_q1   = 7, 7 + n_joints
    i_dp0, i_dp1 = i_q1, i_q1 + 3
    i_w0, i_w1   = i_dp1, i_dp1 + 3
    i_f0, i_f1   = i_w1, i_w1 + 3 * n_contact
    i_c0, i_c1   = i_f1, i_f1 + n_contact
    i_g0, i_g1   = i_c1, i_c1 + 3 * n_contact
    return (i_p0, i_p1, i_qt0, i_qt1, i_q0, i_q1, i_dp0, i_dp1, i_w0, i_w1, i_f0, i_f1, i_c0, i_c1, i_g0, i_g1)



def _yaw_R_from_quat_wxyz(quat_wxyz: jnp.ndarray):
    """
    quat_wxyz: (..., 4) in (w,x,y,z)
    returns:
      yaw: (...,)
      Ryaw: (..., 3,3)
    """
    w, x, y, z = quat_wxyz[..., 0], quat_wxyz[..., 1], quat_wxyz[..., 2], quat_wxyz[..., 3]
    yaw = jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cy, sy = jnp.cos(yaw), jnp.sin(yaw)
    Ryaw = jnp.stack(
        [
            jnp.stack([cy, -sy, jnp.zeros_like(cy)], axis=-1),
            jnp.stack([sy,  cy, jnp.zeros_like(cy)], axis=-1),
            jnp.stack([jnp.zeros_like(cy), jnp.zeros_like(cy), jnp.ones_like(cy)], axis=-1),
        ],
        axis=-2,
    )
    return yaw, Ryaw


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5, 17))
def reference_generator_from_Xin_full(
    use_terrain_estimator: bool,
    N: int,
    dt: float,
    n_joints: int,
    n_contact: int,
    mass: float,
    foot0: jnp.ndarray,         # (3*n_contact,) nominal foot offsets in BASE frame
    q0: jnp.ndarray,            # (n_joints,)
    t_timer: jnp.ndarray,       # (n_contact,)
    X_in: jnp.ndarray,          # (N+1, n_state) layout: [qpos, qvel, foot, (optional grf)]
    input: jnp.ndarray,         # (7,) [v_ref(3), w_ref(3), height]
    duty_factor: float,
    step_freq: float,
    step_height: float,
    liftoff: jnp.ndarray,       # (3*n_contact,) world-frame liftoff positions
    contact_meas: jnp.ndarray,  # (n_contact,) measured contact (0/1)
    clearence_speed: float,
    grf_as_state: bool = False,
):
    """
    Use X_in only for base path (p(t), yaw(t)). Regenerate:
      - timer-based contact_sequence
      - foot_ref(t) via swing splines to touchdown targets that move with the base

    Touchdown target for leg i at time t:
      target_xy = (p_ref[t] + Ryaw[t]@foot0_leg)[xy] + capture_xy
      target_z  = (liftoff_z + step_height profile) for swing; stance holds z
    """
    # ----------------------------
    # Slice X_in according to assumed layout: [qpos, qvel, foot, (optional grf)]
    # ----------------------------
    nq = 7 + n_joints
    nv = 6 + n_joints
    i_qpos0, i_qpos1 = 0, nq
    i_qvel0, i_qvel1 = nq, nq + nv
    i_foot0, i_foot1 = i_qvel1, i_qvel1 + 3 * n_contact

    # grf slice exists but we do not use it for the reference
    if grf_as_state:
        i_grf0, i_grf1 = i_foot1, i_foot1 + 3 * n_contact
    else:
        i_grf0, i_grf1 = -1, -1

    qpos_traj = X_in[:, i_qpos0:i_qpos1]          # (N+1, nq)
    qvel_traj = X_in[:, i_qvel0:i_qvel1]          # (N+1, nv)

    p_traj = qpos_traj[:, 0:3]                    # (N+1,3)
    quat_traj = qpos_traj[:, 3:7]                 # (N+1,4) (wxyz)
    quat_traj = quat_traj / (jnp.linalg.norm(quat_traj, axis=-1, keepdims=True) + 1e-12)

    # time-varying yaw matrices from X_in
    _, Ryaw_traj = _yaw_R_from_quat_wxyz(quat_traj)  # (N+1,3,3)

    # dp for capture term: follow your original (use "current" dp, not time-varying)
    dp0 = qvel_traj[0, 0:3]                       # (3,)

    # height heuristic (your original)
    proprio_height = input[6] + jnp.sum(liftoff[2::3]) / n_contact

    # ----------------------------
    # Base position reference: follow X_in laterally, override z
    # ----------------------------
    p_ref = p_traj.at[:, 2].set(proprio_height)   # (N+1,3)

    # ----------------------------
    # Orientation reference: terrain estimator optional (kept same as your original)
    # (If you want yaw-following quats, switch quat_ref=quat_traj.)
    # ----------------------------
    if use_terrain_estimator:
        # use yaw at t=0 for estimator frame (matches your earlier approach)
        Ryaw0 = Ryaw_traj[0]
        quat0_ref = terrain_orientation(liftoff, Ryaw0)   # (wxyz)
    else:
        quat0_ref = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=X_in.dtype)
    quat_ref = jnp.tile(quat0_ref, (N + 1, 1))

    # pitch compensation (your original)
    pitch = jnp.arcsin(2.0 * (quat_ref[0, 0] * quat_ref[0, 2] - quat_ref[0, 3] * quat_ref[0, 1]))
    Rpitch = jnp.array(
        [
            [jnp.cos(pitch), 0.0, jnp.sin(pitch)],
            [0.0,            1.0, 0.0],
            [-jnp.sin(pitch), 0.0, jnp.cos(pitch)],
        ],
        dtype=X_in.dtype,
    )

    # reference velocities
    # Use yaw(t=0) in the velocity mapping like your original (keeps it simple/stable)
    ref_lin_vel = (Ryaw_traj[0] @ Rpitch) @ input[:3]
    ref_ang_vel = input[3:6]

    dp_ref = jnp.tile(ref_lin_vel, (N + 1, 1))
    omega_ref = jnp.tile(ref_ang_vel, (N + 1, 1))
    q_ref = jnp.tile(q0, (N + 1, 1))

    # ----------------------------
    # Foot + contact + GRF refs
    # ----------------------------
    foot_ref = jnp.tile(X_in[0, i_foot0:i_foot1], (N + 1, 1))  # start at current
    contact_sequence = jnp.zeros((N + 1, n_contact), dtype=X_in.dtype)
    grf_ref = jnp.zeros((N + 1, 3 * n_contact), dtype=X_in.dtype)

    timer_sequence = jnp.tile(t_timer, (N + 1, 1))

    # early contact (kept as your original: constant over horizon)
    des_contact0, current_timer0 = timer_run(duty_factor, step_freq, t_timer, dt)
    early_contact = jnp.where(
        jnp.logical_and(
            jnp.logical_and(des_contact0 == 0, contact_meas == 1),
            current_timer0 > 0.5 + 0.5 * duty_factor,
        ),
        1.0,
        0.0,
    ).astype(X_in.dtype)

    # liftoff trackers
    liftoff_x = liftoff[::3]
    liftoff_y = liftoff[1::3]
    liftoff_z = liftoff[2::3]

    # clearance speed direction (your original)
    initial_speed = -ref_lin_vel / (jnp.linalg.norm(ref_lin_vel) + 1e-6) * clearence_speed

    # constants: nominal offsets
    foot0_mat = foot0.reshape(n_contact, 3)  # (nc,3)

    def cubic_splineXY(current_foot, foothold, initial_velocity, val):
        a0 = current_foot
        a1 = initial_velocity
        a2 = 3.0 * (foothold - current_foot) - 2.0 * initial_velocity
        a3 = initial_velocity - 2.0 * (foothold - current_foot)
        return a0 + a1 * val + a2 * val**2 + a3 * val**3

    def cubic_splineZ(current_foot, foothold, step_h, val):
        initial_speed_z = 0.7
        a = 16 * step_h - 8 * foothold - 8 * current_foot - 2 * initial_speed_z
        b = 5 * initial_speed_z + 14 * foothold + 18 * current_foot - 32 * step_h
        c = 16 * step_h - 5 * foothold - 11 * current_foot - 4 * initial_speed_z
        d = initial_speed_z
        e = current_foot
        return a * val**4 + b * val**3 + c * val**2 + d * val + e

    def foot_fn(t, carry):
        timer_seq, contact_seq, new_foot, lo_x, lo_y, lo_z, grf_new = carry

        prev_fx = new_foot[t - 1, ::3]
        prev_fy = new_foot[t - 1, 1::3]
        prev_fz = new_foot[t - 1, 2::3]

        # advance timers/contacts
        new_contact, new_t = timer_run(duty_factor, step_freq, timer_seq[t - 1, :], dt)
        new_contact = new_contact.astype(X_in.dtype)

        contact_seq = contact_seq.at[t, :].set(new_contact)
        timer_seq = timer_seq.at[t, :].set(new_t)

        # liftoff event: stance->swing
        lo_event = jnp.logical_and(contact_seq[t - 1, :] > 0.5, contact_seq[t, :] < 0.5)
        lo_x = jnp.where(lo_event, prev_fx, lo_x)
        lo_y = jnp.where(lo_event, prev_fy, lo_y)
        lo_z = jnp.where(lo_event, prev_fz, lo_z)

        # ----------------------------
        # Touchdown target: move with base path from X_in at time t, using yaw(t)
        # ----------------------------
        p_t = p_ref[t]                 # (3,)
        Ryaw_t = Ryaw_traj[t]          # (3,3)

        # nominal stance anchor in world
        foot0_world = (foot0_mat @ Ryaw_t.T) + p_t[None, :]     # (nc,3)

        # capture-style adjustment (same spirit as your original calc_foothold)
        f1_x = 0.5 * ref_lin_vel[0] * duty_factor / step_freq
        f1_y = 0.5 * ref_lin_vel[1] * duty_factor / step_freq
        f2_x = jnp.sqrt(input[6] / 9.81) * (dp0[0] - ref_lin_vel[0])
        f2_y = jnp.sqrt(input[6] / 9.81) * (dp0[1] - ref_lin_vel[1])

        foothold_x = foot0_world[:, 0] + f1_x + f2_x
        foothold_y = foot0_world[:, 1] + f1_y + f2_y

        # swing phase in [0,1]
        swing_phase = (new_t - duty_factor) / (1.0 - duty_factor + 1e-12)

        # stance mask (hold during stance OR early-contact latch)
        stance = jnp.logical_or(new_contact > 0.5, early_contact > 0.5)

        fx = jnp.where(stance, prev_fx, cubic_splineXY(lo_x, foothold_x, initial_speed[0], swing_phase))
        fy = jnp.where(stance, prev_fy, cubic_splineXY(lo_y, foothold_y, initial_speed[1], swing_phase))
        fz = jnp.where(stance, prev_fz, cubic_splineZ(lo_z, lo_z, lo_z + step_height, swing_phase))

        new_foot = new_foot.at[t, ::3].set(fx)
        new_foot = new_foot.at[t, 1::3].set(fy)
        new_foot = new_foot.at[t, 2::3].set(fz)

        # vertical GRF allocation (unchanged)
        grf_new = grf_new.at[t, 2::3].set(new_contact * mass * 9.81 / (jnp.sum(new_contact) + 1e-5))

        return (timer_seq, contact_seq, new_foot, lo_x, lo_y, lo_z, grf_new)

    init_carry = (timer_sequence, contact_sequence, foot_ref, liftoff_x, liftoff_y, liftoff_z, grf_ref)

    # start at t=1 because we index t-1
    timer_sequence, contact_sequence, foot_ref, liftoff_x, liftoff_y, liftoff_z, grf_ref = \
        jax.lax.fori_loop(1, N + 1, foot_fn, init_carry)

    liftoff_out = liftoff.at[::3].set(liftoff_x).at[1::3].set(liftoff_y).at[2::3].set(liftoff_z)

    X_ref = jnp.concatenate(
        [p_ref, quat_ref, q_ref, dp_ref, omega_ref, foot_ref, contact_sequence, grf_ref],
        axis=1,
    )

    return X_ref, contact_sequence, liftoff_out



def timer_run(duty_factor, step_freq, leg_time, dt):
    # Update timer
    leg_time = leg_time + dt * step_freq
    leg_time = jnp.where(leg_time > 1, leg_time - 1, leg_time)
    contact = jnp.where(leg_time < duty_factor, 1, 0)
    return contact, leg_time


def smooth_interval(z, z_min, z_max, k=10.0):
    return 0.5 * (jnp.tanh(k * (z - z_min)) - jnp.tanh(k * (z - z_max)))


def gate_2d(px, py, x_min=-3.0, x_max=3.0, y_min=-3.0, y_max=3.0, k=10.0):
    return smooth_interval(px, x_min, x_max, k) * smooth_interval(py, y_min, y_max, k)


def terrain_orientation(liftoff_pos, Ryaw):
    """
    liftoff_pos: (3*n_contact,) world-frame liftoff positions (x,y,z) per foot
    Ryaw:        (3,3) yaw rotation matrix
    Returns quaternion in (w,x,y,z) order (consistent with your jnp.roll usage).
    """
    vec_front_back = (liftoff_pos[:3] + liftoff_pos[3:6] - liftoff_pos[6:9] - liftoff_pos[9:12]) / 2.0
    vec_left_right = Ryaw @ jnp.array([0.0, 1.0, 0.0])  # do not adjust roll per your comment
    normal_vector = jnp.cross(vec_front_back, vec_left_right)

    vec_front_back = vec_front_back / (math.norm(vec_front_back) + 1e-12)
    vec_left_right = vec_left_right / (math.norm(vec_left_right) + 1e-12)
    normal_vector = normal_vector / (math.norm(normal_vector) + 1e-12)

    rotation_matrix = Rotation.from_matrix(jnp.stack([vec_front_back, vec_left_right, normal_vector], axis=1))
    quat_xyzw = rotation_matrix.as_quat()  # returns (x,y,z,w) in SciPy convention
    quat_wxyz = jnp.roll(quat_xyzw, 1)     # -> (w,x,y,z)
    return quat_wxyz


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5))
def reference_generator_unlocked(
    use_terrain_estimator,
    N,
    dt,
    n_joints,
    n_contact,
    mass,
    foot0,          # (3*n_contact,) nominal foot offsets in BASE frame (your convention)
    q0,             # (n_joints,)
    t_timer,        # (n_contact,)
    x,              # full state, must contain [p(3), quat(4), q(?), ... base vel ...]
    foot,           # (3*n_contact,) current foot positions in WORLD frame (your convention)
    input,          # (7,) [v_ref(3), w_ref(3), height]
    duty_factor,
    step_freq,
    step_height,
    liftoff,        # (3*n_contact,) world-frame liftoff positions per foot
    contact,        # (n_contact,) current measured contact (0/1)
    clearence_speed
):
    """
    Returns:
      X_ref: (N+1, n_ref) where n_ref matches your expected concatenation:
        [p_ref(3), quat_ref(4), q_ref(n_joints), dp_ref(3), omega_ref(3),
         foot_ref(3*n_contact), contact_sequence(n_contact), grf_ref(3*n_contact)]
      contact_out: (N+1, n_contact) contact sequence
      liftoff_out: (3*n_contact,) updated liftoff positions
    """

    # -----------------------
    # Parse current state
    # -----------------------
    p = x[:3]
    quat = x[3:7]

    # IMPORTANT: You had dp = x[7+n_joints:10+n_joints]
    # Keep the same indexing you used (assumes your state layout places base lin vel there).
    dp = x[7 + n_joints : 10 + n_joints]

    # yaw from quaternion (w,x,y,z)
    yaw = jnp.arctan2(
        2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
        1.0 - 2.0 * (quat[2] * quat[2] + quat[3] * quat[3]),
    )
    Ryaw = jnp.array(
        [
            [jnp.cos(yaw), -jnp.sin(yaw), 0.0],
            [jnp.sin(yaw),  jnp.cos(yaw), 0.0],
            [0.0,           0.0,          1.0],
        ]
    )

    # Height heuristic (your original logic)
    proprio_height = input[6] + jnp.sum(liftoff[2::3]) / n_contact
    p = jnp.array([p[0], p[1], proprio_height])

    # Orientation reference (terrain estimator optional)
    if use_terrain_estimator:
        quat0_ref = terrain_orientation(liftoff, Ryaw)      # (4,)
    else:
        quat0_ref = jnp.array([1.0, 0.0, 0.0, 0.0])
    quat_ref = jnp.tile(quat0_ref, (N + 1, 1))

    # Pitch compensation (your original)
    pitch = jnp.arcsin(2.0 * (quat_ref[0, 0] * quat_ref[0, 2] - quat_ref[0, 3] * quat_ref[0, 1]))
    Rpitch = jnp.array(
        [
            [jnp.cos(pitch), 0.0, jnp.sin(pitch)],
            [0.0,            1.0, 0.0],
            [-jnp.sin(pitch), 0.0, jnp.cos(pitch)],
        ]
    )

    # Reference velocities
    ref_lin_vel = Ryaw @ Rpitch @ input[:3]
    ref_ang_vel = input[3:6]

    # Base position reference
    p_ref_x = jnp.arange(N + 1) * dt * ref_lin_vel[0] + p[0]
    p_ref_y = jnp.arange(N + 1) * dt * ref_lin_vel[1] + p[1]
    p_ref_z = jnp.ones(N + 1) * proprio_height
    p_ref = jnp.stack([p_ref_x, p_ref_y, p_ref_z], axis=1)

    dp_ref = jnp.tile(ref_lin_vel, (N + 1, 1))
    omega_ref = jnp.tile(ref_ang_vel, (N + 1, 1))
    q_ref = jnp.tile(q0, (N + 1, 1))

    # Initialize foot reference with current foot WORLD positions
    foot_ref = jnp.tile(foot, (N + 1, 1))

    # GRF reference
    grf_ref = jnp.zeros((N + 1, 3 * n_contact))

    # Contact sequence and timer sequence buffers
    contact_sequence = jnp.zeros((N + 1, n_contact))
    timer_sequence_in = jnp.tile(t_timer, (N + 1, 1))

    # Early contact (NOTE: kept as in your original; if you want it time-varying, move it into the loop)
    des_contact, current_timer = timer_run(duty_factor, step_freq, t_timer, dt)
    early_contact = jnp.where(
        jnp.logical_and(
            jnp.logical_and(des_contact == 0, contact == 1),
            current_timer > 0.5 + 0.5 * duty_factor,
        ),
        1,
        0,
    )

    # Liftoff trackers (per-foot)
    liftoff_x = liftoff[::3]
    liftoff_y = liftoff[1::3]
    liftoff_z = liftoff[2::3]

    # Precompute constants used in swing splines
    initial_speed = -ref_lin_vel / (jnp.linalg.norm(ref_lin_vel) + 1e-6) * clearence_speed

    def foot_fn(t, carry):
        timer_seq, contact_seq, new_foot, lo_x, lo_y, lo_z, grf_new = carry

        # previous foot world positions
        prev_fx = new_foot[t - 1, ::3]
        prev_fy = new_foot[t - 1, 1::3]
        prev_fz = new_foot[t - 1, 2::3]

        # advance timers / contact
        new_contact, new_t = timer_run(duty_factor, step_freq, timer_seq[t - 1, :], dt)
        contact_seq = contact_seq.at[t, :].set(new_contact)
        timer_seq = timer_seq.at[t, :].set(new_t)

        # detect liftoff (stance->swing transition)
        lo_x = jnp.where(jnp.logical_and(jnp.logical_not(contact_seq[t, :]), contact_seq[t - 1, :]), prev_fx, lo_x)
        lo_y = jnp.where(jnp.logical_and(jnp.logical_not(contact_seq[t, :]), contact_seq[t - 1, :]), prev_fy, lo_y)
        lo_z = jnp.where(jnp.logical_and(jnp.logical_not(contact_seq[t, :]), contact_seq[t - 1, :]), prev_fz, lo_z)

        # ------------- KEY FIX: time-varying nominal foothold anchor -------------
        # Use p_ref[t] so the touchdown target advances with the moving base.
        p_t = p_ref[t]  # (3,)

        # foot0: (3*n_contact,) nominal offsets in BASE frame
        foot0_mat = foot0.reshape(n_contact, 3)            # (n_contact, 3)
        foot0_world = (foot0_mat @ Ryaw.T) + p_t           # (n_contact, 3)
        foot0_projected_t = foot0_world.reshape(-1)        # (3*n_contact,)

        def calc_foothold(direction):
            f1 = 0.5 * ref_lin_vel[direction] * duty_factor / step_freq
            f2 = jnp.sqrt(input[6] / 9.81) * (dp[direction] - ref_lin_vel[direction])
            return foot0_projected_t[direction::3] + f1 + f2

        foothold_x = calc_foothold(0)
        foothold_y = calc_foothold(1)
        # -----------------------------------------------------------------------

        def cubic_splineXY(current_foot, foothold, initial_velocity, val):
            a0 = current_foot
            a1 = initial_velocity
            a2 = 3.0 * (foothold - current_foot) - 2.0 * initial_velocity
            a3 = initial_velocity - 2.0 * (foothold - current_foot)
            return a0 + a1 * val + a2 * val**2 + a3 * val**3

        def cubic_splineZ(current_foot, foothold, step_h, val):
            initial_speed_z = 0.7
            a = 16 * step_h - 8 * foothold - 8 * current_foot - 2 * initial_speed_z
            b = 5 * initial_speed_z + 14 * foothold + 18 * current_foot - 32 * step_h
            c = 16 * step_h - 5 * foothold - 11 * current_foot - 4 * initial_speed_z
            d = initial_speed_z
            e = current_foot
            return a * val**4 + b * val**3 + c * val**2 + d * val + e

        # phase in swing [0,1]
        swing_phase = (new_t - duty_factor) / (1.0 - duty_factor + 1e-12)

        # stance: hold foot fixed in world; swing: move to foothold
        fx = jnp.where(
            jnp.logical_or(new_contact > 0, early_contact == 1),
            prev_fx,
            cubic_splineXY(lo_x, foothold_x, initial_speed[0], swing_phase),
        )
        fy = jnp.where(
            jnp.logical_or(new_contact > 0, early_contact == 1),
            prev_fy,
            cubic_splineXY(lo_y, foothold_y, initial_speed[1], swing_phase),
        )
        fz = jnp.where(
            jnp.logical_or(new_contact > 0, early_contact == 1),
            prev_fz,
            cubic_splineZ(lo_z, lo_z, lo_z + step_height, swing_phase),
        )

        new_foot = new_foot.at[t, ::3].set(fx)
        new_foot = new_foot.at[t, 1::3].set(fy)
        new_foot = new_foot.at[t, 2::3].set(fz)

        # simple vertical GRF allocation in stance
        grf_new = grf_new.at[t, 2::3].set(new_contact * mass * 9.81 / (jnp.sum(new_contact) + 1e-5))

        return (timer_seq, contact_seq, new_foot, lo_x, lo_y, lo_z, grf_new)

    init_carry = (timer_sequence_in, contact_sequence, foot_ref, liftoff_x, liftoff_y, liftoff_z, grf_ref)

    # IMPORTANT: start at t=1 because foot_fn indexes t-1
    timer_sequence, contact_sequence, foot_ref, liftoff_x, liftoff_y, liftoff_z, grf_ref = \
        jax.lax.fori_loop(1, N + 1, foot_fn, init_carry)

    liftoff = liftoff.at[::3].set(liftoff_x)
    liftoff = liftoff.at[1::3].set(liftoff_y)
    liftoff = liftoff.at[2::3].set(liftoff_z)

    X_ref = jnp.concatenate(
        [p_ref, quat_ref, q_ref, dp_ref, omega_ref, foot_ref, contact_sequence, grf_ref],
        axis=1
    )

    return X_ref, contact_sequence, liftoff


@partial(jax.jit, static_argnums=(0,1,2,3))
def reference_generator_srbd(use_terrain_estimator,N,dt,n_contact,mass,foot0,t_timer, x, foot, input, duty_factor, step_freq,step_height,liftoff,contact,clearence_speed):
    p = x[:3]
    quat = x[3:7]
    dp = x[7:10]
    yaw = jnp.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 1 - 2*(quat[2]*quat[2] + quat[3]*quat[3]))
    Ryaw = jnp.array([[jnp.cos(yaw), -jnp.sin(yaw), 0],[jnp.sin(yaw), jnp.cos(yaw), 0],[0, 0, 1]])
    proprio_height = input[6] + jnp.sum(liftoff[2::3])/n_contact
    p = jnp.array([p[0], p[1], proprio_height])
    if use_terrain_estimator:
        quat_ref = jnp.tile(terrain_orientation(liftoff,Ryaw), (N+1, 1))
    else:
        quat_ref = jnp.tile(jnp.array([1, 0, 0, 0]), (N+1, 1))
    contact_sequence = jnp.zeros(((N+1), n_contact))
    pitch = jnp.arcsin(2 * (quat_ref[0,0] * quat_ref[0,2] - quat_ref[0,3] * quat_ref[0,1]))
    Rpitch = jnp.array([[jnp.cos(pitch), 0, jnp.sin(pitch)], [0, 1, 0], [-jnp.sin(pitch), 0, jnp.cos(pitch)]])
    ref_lin_vel = Ryaw@Rpitch@input[:3]
    ref_ang_vel = input[3:6]
    p_ref_x = jnp.arange(N+1) * dt * ref_lin_vel[0] + p[0]
    p_ref_y = jnp.arange(N+1) * dt * ref_lin_vel[1] + p[1]
    p_ref_z = jnp.ones(N+1) * proprio_height
    p_ref = jnp.stack([p_ref_x, p_ref_y, p_ref_z], axis=1)
    dp_ref = jnp.tile(ref_lin_vel, (N+1, 1))
    omega_ref = jnp.tile(ref_ang_vel, (N+1, 1))
    foot_ref = jnp.tile(foot, (N+1, 1))
    foot_ref_dot = jnp.zeros(((N+1), 3*n_contact))
    foot0_projected = jnp.tile(p, n_contact) + foot0 @ jax.scipy.linalg.block_diag(*([Ryaw] * n_contact)).T
    grf_ref = jnp.zeros((N+1, 3*n_contact))

    #Estimate Early contact
    des_contact, current_timer = timer_run(duty_factor, step_freq, t_timer, dt)
    early_contact = jnp.where(jnp.logical_and(jnp.logical_and(des_contact==0,contact==1),current_timer > 0.5 + 0.5*duty_factor),1,0)
    
    def foot_fn(t,carry):

        new_t, contact_sequence,new_foot,new_foot_dot,liftoff_x,liftoff_y,liftoff_z,grf_new = carry

        new_foot_x = new_foot[t-1,::3]
        new_foot_y = new_foot[t-1,1::3]
        new_foot_z = new_foot[t-1,2::3]

        new_contact_sequence, new_t = timer_run(duty_factor, step_freq, new_t, dt)

        contact_sequence = contact_sequence.at[t,:].set(new_contact_sequence)

        liftoff_x = jnp.where(jnp.logical_and(jnp.logical_not(contact_sequence[t,:]),contact_sequence[t-1,:]),new_foot_x,liftoff_x)
        liftoff_y = jnp.where(jnp.logical_and(jnp.logical_not(contact_sequence[t,:]),contact_sequence[t-1,:]),new_foot_y,liftoff_y)
        liftoff_z = jnp.where(jnp.logical_and(jnp.logical_not(contact_sequence[t,:]),contact_sequence[t-1,:]),new_foot_z,liftoff_z)

        def calc_foothold(direction):
            f1 = 0.5*ref_lin_vel[direction]*duty_factor/step_freq
            f2 = jnp.sqrt(input[6]/9.81)*(dp[direction]-ref_lin_vel[direction])
            f = f1 + f2 + foot0_projected[direction::3]
            return f

        foothold_x = calc_foothold(0)
        foothold_y = calc_foothold(1)

        def cubic_splineXY(current_foot, foothold,initial_velocity,val):
            a0 = current_foot
            a1 = initial_velocity
            a2 = 3*(foothold - current_foot) - 2*initial_velocity
            a3 = initial_velocity - 2*(foothold - current_foot)
            return a0 + a1*val + a2*val**2 + a3*val**3

        def cubic_splineZ(current_foot, foothold, step_height,val):
            
            initial_speed = 0.7

            a = 16*step_height - 8*foothold - 8*current_foot - 2*initial_speed
            b = 5*initial_speed + 14*foothold + 18*current_foot - 32*step_height
            c = 16*step_height - 5*foothold - 11*current_foot - 4*initial_speed
            d = initial_speed
            e = current_foot
            return a*val**4 + b*val**3 + c*val**2 + d*val + e

        def cubic_splineXY_dot(current_foot, foothold,initial_velocity,val):
            a1 = initial_velocity
            a2 = 3*(foothold - current_foot) - 2*initial_velocity
            a3 = initial_velocity - 2*(foothold - current_foot)
            return 2*a2*val + 3*a3*val**2 + a1

        def cubic_splineZ_dot(current_foot, foothold, step_height,val):
            
            initial_speed = 0.7
            a = 16*step_height - 8*foothold - 8*current_foot - 2*initial_speed
            b = 5*initial_speed + 14*foothold + 18*current_foot - 32*step_height
            c = 16*step_height - 5*foothold - 11*current_foot - 4*initial_speed
            d = initial_speed
            return 4*a*val**3 + 3*b*val**2 + 2*c*val + d

        new_foot_x = jnp.where(jnp.logical_or(new_contact_sequence>0,early_contact==1), new_foot[t-1,::3], cubic_splineXY(liftoff_x, foothold_x,initial_speed[0],(new_t-duty_factor)/(1-duty_factor)))
        new_foot_y = jnp.where(jnp.logical_or(new_contact_sequence>0,early_contact==1), new_foot[t-1,1::3], cubic_splineXY(liftoff_y, foothold_y,initial_speed[1],(new_t-duty_factor)/(1-duty_factor)))
        new_foot_z = jnp.where(jnp.logical_or(new_contact_sequence>0,early_contact==1), new_foot[t-1,2::3], cubic_splineZ(liftoff_z,liftoff_z,liftoff_z + step_height,(new_t-duty_factor)/(1-duty_factor)))

        new_foot = new_foot.at[t,::3].set(new_foot_x)
        new_foot = new_foot.at[t,1::3].set(new_foot_y)
        new_foot = new_foot.at[t,2::3].set(new_foot_z)

        new_foot_dot = new_foot_dot.at[t,::3].set(jnp.where(jnp.logical_or(new_contact_sequence>0,early_contact==1), 0, cubic_splineXY_dot(liftoff_x, foothold_x,initial_speed[0],(new_t-duty_factor)/(1-duty_factor))))
        new_foot_dot = new_foot_dot.at[t,1::3].set(jnp.where(jnp.logical_or(new_contact_sequence>0,early_contact==1), 0, cubic_splineXY_dot(liftoff_y, foothold_y,initial_speed[1],(new_t-duty_factor)/(1-duty_factor))))
        new_foot_dot = new_foot_dot.at[t,2::3].set(jnp.where(jnp.logical_or(new_contact_sequence>0,early_contact==1), 0, cubic_splineZ_dot(liftoff_z,liftoff_z,liftoff_z + step_height,(new_t-duty_factor)/(1-duty_factor))))

        # new_foot_ddot = new_foot_ddot.at[t,::3].set(jnp.where(new_contact_sequence>0, 0, cubic_splineXY_ddot(liftoff_x, foothold_x,(new_t-duty_factor)/(1-duty_factor))))
        # new_foot_ddot = new_foot_ddot.at[t,1::3].set(jnp.where(new_contact_sequence>0, 0, cubic_splineXY_ddot(liftoff_y, foothold_y,(new_t-duty_factor)/(1-duty_factor))))
        # new_foot_ddot = new_foot_ddot.at[t,2::3].set(jnp.where(new_contact_sequence>0, 0, cubic_splineZ_ddot(liftoff_z,liftoff_z,liftoff_z + step_height,(new_t-duty_factor)/(1-duty_factor))))

        grf_new = grf_new.at[t,2::3].set((new_contact_sequence*mass*9.81/(jnp.sum(new_contact_sequence)+1e-5)))

        return (new_t, contact_sequence,new_foot,new_foot_dot,liftoff_x,liftoff_y,liftoff_z,grf_new)

    liftoff_x = liftoff[::3]
    liftoff_y = liftoff[1::3]
    liftoff_z = liftoff[2::3]

    initial_speed = - ref_lin_vel / (jnp.linalg.norm(ref_lin_vel) + 1e-6) * clearence_speed

    init_carry = (t_timer, contact_sequence,foot_ref,foot_ref_dot,liftoff_x,liftoff_y,liftoff_z,grf_ref)
    _, contact_sequence,foot_ref,foot_ref_dot, liftoff_x,liftoff_y,liftoff_z,grf_ref = jax.lax.fori_loop(0,N+1,foot_fn, init_carry)

    liftoff = liftoff.at[::3].set(liftoff_x)
    liftoff = liftoff.at[1::3].set(liftoff_y)
    liftoff = liftoff.at[2::3].set(liftoff_z)

    return jnp.concatenate([p_ref, quat_ref, dp_ref, omega_ref,contact_sequence], axis=1),jnp.concatenate([ contact_sequence,foot_ref], axis=1), liftoff , foot_ref_dot


@partial(jax.jit, static_argnums=(0,1,2,3,4,5))
def reference_generator(use_terrain_estimator,N,dt,n_joints,n_contact,mass,foot0,q0,t_timer, x, foot, input, duty_factor, step_freq,step_height,liftoff,contact,clearence_speed):
    p = x[:3]
    quat = x[3:7]
    # q = x[7:7+n_joints]
    dp = x[7+n_joints:10+n_joints]
    # omega = x[10+n_joints:13+n_joints]
    # dq = x[13+n_joints:13+2*n_joints]
    yaw = jnp.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 1 - 2*(quat[2]*quat[2] + quat[3]*quat[3]))
    Ryaw = jnp.array([[jnp.cos(yaw), -jnp.sin(yaw), 0],[jnp.sin(yaw), jnp.cos(yaw), 0],[0, 0, 1]])
    proprio_height = input[6] + jnp.sum(liftoff[2::3])/n_contact
    p = jnp.array([p[0], p[1], proprio_height])
    if use_terrain_estimator:
        quat_ref = jnp.tile(terrain_orientation(liftoff,Ryaw), (N+1, 1))
    else:
        quat_ref = jnp.tile(jnp.array([1, 0, 0, 0]), (N+1, 1))
    q_ref = jnp.tile(q0, (N+1, 1))
    contact_sequence = jnp.zeros(((N+1), n_contact))
    pitch = jnp.arcsin(2 * (quat_ref[0,0] * quat_ref[0,2] - quat_ref[0,3] * quat_ref[0,1]))
    Rpitch = jnp.array([[jnp.cos(pitch), 0, jnp.sin(pitch)], [0, 1, 0], [-jnp.sin(pitch), 0, jnp.cos(pitch)]])
    
    ref_lin_vel = Ryaw@Rpitch@input[:3]
    ref_ang_vel = input[3:6]
    p_ref_x = jnp.arange(N+1) * dt * ref_lin_vel[0] + p[0]
    p_ref_y = jnp.arange(N+1) * dt * ref_lin_vel[1] + p[1]
    p_ref_z = jnp.ones(N+1) * proprio_height
    p_ref = jnp.stack([p_ref_x, p_ref_y, p_ref_z], axis=1)
    dp_ref = jnp.tile(ref_lin_vel, (N+1, 1))
    omega_ref = jnp.tile(ref_ang_vel, (N+1, 1))
    foot_ref = jnp.tile(foot, (N+1, 1))
    foot0_projected = jnp.tile(p, n_contact) + foot0 @ jax.scipy.linalg.block_diag(*([Ryaw] * n_contact)).T
    grf_ref = jnp.zeros((N+1, 3*n_contact))

    #Estimate Early contact
    des_contact, current_timer = timer_run(duty_factor, step_freq, t_timer, dt)
    early_contact = jnp.where(jnp.logical_and(jnp.logical_and(des_contact==0,contact==1),current_timer > 0.5 + 0.5*duty_factor),1,0)    

    def foot_fn(t,carry):

        timer_seq, contact_sequence,new_foot,liftoff_x,liftoff_y,liftoff_z,grf_new = carry

        new_foot_x = new_foot[t-1,::3]
        new_foot_y = new_foot[t-1,1::3]
        new_foot_z = new_foot[t-1,2::3]

        new_contact_sequence, new_t = timer_run(duty_factor, step_freq, timer_seq[t-1,:], dt)

        contact_sequence = contact_sequence.at[t,:].set(new_contact_sequence)
        timer_seq = timer_seq.at[t,:].set(new_t)

        liftoff_x = jnp.where(jnp.logical_and(jnp.logical_not(contact_sequence[t,:]),contact_sequence[t-1,:]),new_foot_x,liftoff_x)
        liftoff_y = jnp.where(jnp.logical_and(jnp.logical_not(contact_sequence[t,:]),contact_sequence[t-1,:]),new_foot_y,liftoff_y)
        liftoff_z = jnp.where(jnp.logical_and(jnp.logical_not(contact_sequence[t,:]),contact_sequence[t-1,:]),new_foot_z,liftoff_z)

        def calc_foothold(direction):
            f1 = 0.5*ref_lin_vel[direction]*duty_factor/step_freq
            f2 = jnp.sqrt(input[6]/9.81)*(dp[direction]-ref_lin_vel[direction])
            f = f1 + f2 + foot0_projected[direction::3]
            return f

        foothold_x = calc_foothold(0)
        foothold_y = calc_foothold(1)
        y_min = 0.10   # meters (tune: 0.08–0.14)
        y_max = 0.22   # meters (tune: 0.18–0.28)

        # ordering is [FL, FR, RL, RR]
        is_left = jnp.array([True, False, True, False])
        is_right = jnp.logical_not(is_left)

        foothold_y = jnp.where(
            is_left,
            jnp.clip(foothold_y, p[1] + y_min, p[1] + y_max),
            jnp.clip(foothold_y, p[1] - y_max, p[1] - y_min),
)

        def cubic_splineXY(current_foot, foothold,initial_velocity,val):
            a0 = current_foot
            a1 = initial_velocity
            a2 = 3*(foothold - current_foot) - 2*initial_velocity
            a3 = initial_velocity - 2*(foothold - current_foot)
            return a0 + a1*val + a2*val**2 + a3*val**3

        def cubic_splineZ(current_foot, foothold, step_height,val):
            
            initial_speed = 0.7

            a = 16*step_height - 8*foothold - 8*current_foot - 2*initial_speed
            b = 5*initial_speed + 14*foothold + 18*current_foot - 32*step_height
            c = 16*step_height - 5*foothold - 11*current_foot - 4*initial_speed
            d = initial_speed
            e = current_foot
            return a*val**4 + b*val**3 + c*val**2 + d*val + e
        
        initial_speed = - ref_lin_vel / (jnp.linalg.norm(ref_lin_vel) + 1e-6) * clearence_speed

        new_foot_x = jnp.where(jnp.logical_or(new_contact_sequence>0,early_contact==1), new_foot[t-1,::3], cubic_splineXY(liftoff_x, foothold_x,initial_speed[0],(new_t-duty_factor)/(1-duty_factor)))
        new_foot_y = jnp.where(jnp.logical_or(new_contact_sequence>0,early_contact==1), new_foot[t-1,1::3], cubic_splineXY(liftoff_y, foothold_y,initial_speed[1],(new_t-duty_factor)/(1-duty_factor)))
        new_foot_z = jnp.where(jnp.logical_or(new_contact_sequence>0,early_contact==1), new_foot[t-1,2::3], cubic_splineZ(liftoff_z,liftoff_z,liftoff_z + step_height,(new_t-duty_factor)/(1-duty_factor)))

        new_foot = new_foot.at[t,::3].set(new_foot_x)
        new_foot = new_foot.at[t,1::3].set(new_foot_y)
        new_foot = new_foot.at[t,2::3].set(new_foot_z)

        grf_new = grf_new.at[t,2::3].set((new_contact_sequence*mass*9.81/(jnp.sum(new_contact_sequence)+1e-5)))

        return (timer_seq, contact_sequence,new_foot,liftoff_x,liftoff_y,liftoff_z,grf_new)

    liftoff_x = liftoff[::3]
    liftoff_y = liftoff[1::3]
    liftoff_z = liftoff[2::3]
    timer_sequence_in = jnp.tile(t_timer, (N+1, 1))
    init_carry = (timer_sequence_in, contact_sequence,foot_ref,liftoff_x,liftoff_y,liftoff_z,grf_ref)
    timer_sequence, contact_sequence,foot_ref, liftoff_x,liftoff_y,liftoff_z,grf_ref = jax.lax.fori_loop(0,N+1,foot_fn, init_carry)

    liftoff = liftoff.at[::3].set(liftoff_x)
    liftoff = liftoff.at[1::3].set(liftoff_y)
    liftoff = liftoff.at[2::3].set(liftoff_z)

    return jnp.concatenate([p_ref, quat_ref, q_ref, dp_ref, omega_ref, foot_ref, contact_sequence,grf_ref], axis=1),jnp.concatenate([contact_sequence], axis=1), liftoff

import mujoco
from mujoco import mjx

@partial(jax.jit, static_argnums=(0))
def whole_body_interface(model, mjx_model, contact_id, body_id,sim_frequency,Kp,Kd,qpos,qvel,grf,foot_ref,foot_ref_dot,contact):

    mjx_data = mjx.make_data(model)
    # Update the position and velocity in the data object
    mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
    # Perform forward kinematics and dynamics computations
    mjx_data = mjx.fwd_position(mjx_model, mjx_data)
    mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)

    # Extract the mass matrix and bias forces
    M = mjx_data.qM
    D = mjx_data.qfrc_bias

    # Get the positions of the contact points on the legs
    FL_leg = mjx_data.geom_xpos[contact_id[0]]
    FR_leg = mjx_data.geom_xpos[contact_id[1]]
    RL_leg = mjx_data.geom_xpos[contact_id[2]]
    RR_leg = mjx_data.geom_xpos[contact_id[3]]

    # Compute the Jacobians for each leg
    J_FL, _ = mjx.jac(mjx_model, mjx_data, FL_leg, body_id[0])
    J_FR, _ = mjx.jac(mjx_model, mjx_data, FR_leg, body_id[1])
    J_RL, _ = mjx.jac(mjx_model, mjx_data, RL_leg, body_id[2])
    J_RR, _ = mjx.jac(mjx_model, mjx_data, RR_leg, body_id[3])

    # Concatenate the Jacobians into a single matrix
    J = jnp.concatenate([J_FL, J_FR, J_RL, J_RR], axis=1)
    # Concatenate the positions of the legs into a single vector
    current_leg = jnp.concatenate([FL_leg, FR_leg, RL_leg, RR_leg], axis=0)
    current_leg_dot = J.T @ mjx_data.qvel
    cartesian_space_action = Kp@(foot_ref-current_leg) + Kd@(foot_ref_dot-current_leg_dot)
    tau_fb_lin = D[6:] + (M @ jnp.linalg.pinv(J.T) @ (cartesian_space_action))[6:]
    tau_mpc = -(J@grf)[6:]
    tau_PD = (J @ cartesian_space_action)[6:]
    contact_mask = jnp.array([contact[0],contact[0],contact[0],contact[1],contact[1],contact[1],contact[2],contact[2],contact[2],contact[3],contact[3],contact[3]])
    tau = tau_mpc*contact_mask + (1-contact_mask)*(tau_fb_lin) 

    return tau , J

@partial(jax.jit, static_argnums=(0,1,2,3))
def reference_barell_roll(N,dt,n_joints,n_contact,foot0,q0):
    t1 = 0.2
    t2 = 0.2
    t3 = 0.3
    t4 = 0.1
    z_start = 0.4
    z_land = 0.28
    v_lateral = -0.25/(t2+t3)
    v0 = (z_land - z_start + 0.5*9.81*t3*t3)/t3 
    total_roll_time = t2+t3+t4
    roll_speed = 2*3.14/total_roll_time
    def z_position(t):
        return z_start - 0.5*9.81*t**2 + v0*t
    def z_speed(t):
        return -9.81*t + v0
    acc = v0/t2
    print("v0", v0)
    print("acc", acc)
    #first part full stance 0.1s
    n1 = int(t1/dt)
    p1 = jnp.tile(jnp.array([0,0,0.33]), (n1, 1))
    p1 = p1.at[:,1].set(jnp.arange(n1)*dt*(v_lateral))
    dp1 = jnp.tile(jnp.array([0,v_lateral,0]), (n1, 1))
    contact1 = jnp.tile(jnp.array([1,1,1,1]), (n1, 1))
    quat1 = jnp.tile(jnp.array([1, 0, 0, 0]), (n1, 1))
    omega1 = jnp.tile(jnp.array([0, 0, 0]), (n1, 1))
    #second part lateral support 0.2s
    n2 = int(t2/dt)
    p2 = jnp.tile(jnp.array([0,p1[-1,1],0.33]), (n2, 1))
    p2 = p2.at[:,2].set(0.5*jnp.arange(n2)*dt*jnp.arange(n2)*dt*acc + 0.33)
    p2 = p2.at[:,1].set(jnp.arange(n2)*dt*(v_lateral))
    dp2 = jnp.tile(jnp.array([0,v_lateral,0]), (n2, 1))
    dp2 = dp2.at[:,2].set(jnp.arange(n2)*dt*acc)
    contact2 = jnp.tile(jnp.array([0,1,0,1]), (n2, 1))
    # for i in range(n2):
    #     p2 = p2.at[i,2].set(z_position(i*dt))
    #     dp2 = dp2.at[i,2].set(z_speed(i*dt))
    #third part flying phase 0.4s
    n3 = int(t3/dt)
    p3 = jnp.tile(jnp.array([0,p2[-1,1],p2[-1,2]]), (n3, 1))
    p3 = p3.at[:,1].set(jnp.arange(n3)*dt*(v_lateral))
    dp3 = jnp.tile(jnp.array([0,v_lateral,0]), (n3, 1))
    for i in range(n3):
        p3 = p3.at[i,2].set(z_position(i*dt))
        dp3 = dp3.at[i,2].set(z_speed(i*dt))
    def fn(t,carry):
        quat_new = math.quat_integrate(carry[t-1,:], jnp.array([roll_speed,0,0]), dt)
        carry_new = carry.at[t,:].set(quat_new)
        return carry_new
    
    
    contact3 = jnp.tile(jnp.array([0,0,0,0]), (n3, 1))
    #fourth part full stance 0.2s
    n4 = int(t4/dt)
    p4 = jnp.tile(jnp.array([0,p3[-1,1],z_land]), (n4, 1))
    dp4 = jnp.tile(jnp.array([0,0,0]), (n4, 1))
    quat5 = jnp.tile(jnp.array([1, 0, 0, 0]), (n4, 1))
    omega5 = jnp.tile(jnp.array([0, 0, 0]), (n4, 1))
    contact4 = jnp.tile(jnp.array([1,1,1,1]), (n4, 1))

    init_carry = jnp.tile(jnp.array([1.0, 0.0, 0, 0]), (n2+n3+n4, 1))
    quat234 = jax.lax.fori_loop(1, n2+n3+n4, fn, init_carry)
    omega234 = jnp.tile(jnp.array([roll_speed, 0, 0]), (n2+n3+n4, 1))

    n5 = N - (n1+n2+n3+n4)

    p5 = jnp.tile(jnp.array([0,p4[-1,1],z_land]), (n5, 1))
    dp5 = jnp.tile(jnp.array([0,0,0]), (n5, 1))
    quat5 = jnp.tile(jnp.array([1, 0, 0, 0]), (n5, 1))
    omega5 = jnp.tile(jnp.array([0, 0, 0]), (n5, 1))
    contact5 = jnp.tile(jnp.array([1,1,1,1]), (n5, 1))

    p_ref = jnp.concatenate([p1, p2, p3, p4,p5], axis=0)
    quat_ref = jnp.concatenate([quat1,quat234,quat5], axis=0)
    q_ref = jnp.tile(q0, (n1+n2+n3+n4+n5, 1))
    dp_ref = jnp.concatenate([dp1, dp2, dp3, dp4,dp5], axis=0)
    omega_ref = jnp.concatenate([omega1,omega234,omega5], axis=0)
    foot_ref = jnp.tile(foot0, (n1+n2+n3+n4+n5, 1)) + jnp.tile(p_ref, n_contact)
    foot_ref = foot_ref.at[:,2::3].set(jnp.zeros((n1+n2+n3+n4+n5, n_contact)))
    contact_sequence = jnp.concatenate([contact1, contact2, contact3, contact4,contact5], axis=0)

    grf_ref = jnp.zeros((N, 3*n_contact))

    return jnp.concatenate([p_ref, quat_ref, q_ref, dp_ref, omega_ref, foot_ref, contact_sequence, grf_ref], axis=1), jnp.concatenate([contact_sequence, foot_ref], axis=1)

def circle_pose_constraint(X, U, obstacle):
    # X: (T+1, n), U: (T, m)
    # Example 1: state stays inside a circle of radius r (convex)
    r = obstacle.radius
    pos = X[:, :2]
    g_circle = jnp.sum(pos**2, axis=-1) - r**2          # (T+1,), <= 0

    # Example 2: control effort magnitude constraint ||u_t||^2 <= umax^2
    umax = 0.5
    g_u = jnp.sum(U**2, axis=-1) - umax**2              # (T,), <= 0

    return jnp.concatenate([g_circle.reshape(-1), g_u.reshape(-1)], axis=0)

def outside_circle_constraints(x, u, t, centers, radii):
    """
    Stage constraint at time t:
      g_t[k] = r_k^2 - ||x_pos - c_k||^2 <= 0  (outside circle)
    x: (n,)
    u: (m,)  (unused)
    t: scalar (unused)
    centers: (K, 2)
    radii: (K,)
    returns: (K,)
    """
    pos = x[0:2]                     # (2,)
    diff = pos[None, :] - centers    # (K, 2)
    dist2 = jnp.sum(diff * diff, axis=-1)     # (K,)
    g_t = radii**2 - dist2                     # (K,)
    return g_t

def combine_constraints(*funcs):
    """
    Combine multiple g_i(x,u,t) functions into one by concatenation.
    Each func must return a 1D array.
    """
    def constraints(x, u, t):
        parts = [f(x, u, t) for f in funcs]
        return jnp.concatenate(parts, axis=0)
    return constraints
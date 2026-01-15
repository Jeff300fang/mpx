data = np.load("mpc_data/go2_mpc_rollout.npz")

X = data["X"]
U = data["U"]
V = data["V"]
Phi_x = data["Phi_x"]
Phi_u = data["Phi_u"]
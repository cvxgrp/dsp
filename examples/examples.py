# def test_robust_ols():
# np.random.seed(0)
# N = 100
# alpha = 0.05
# k = int(alpha * N)
# theta = np.array([0.5, 2])
# eps = np.random.normal(0, 0.05, N)
# A = np.concatenate([np.ones((N, 1)), np.random.normal(0, 1, (N, 1))], axis=1)
# observations = A @ theta + eps
# largest_inds = np.argpartition(observations, -k)[-k:]
# observations[largest_inds] -= 10

# wgts = cp.Variable(N, nonneg=True)
# theta_hat = cp.Variable(2)

# loss = cp.square(observations - A @ theta_hat)

# # OLS
# cp.Problem(cp.Minimize(cp.sum(loss))).solve(solver=cp.SCS)
# ols_vals = theta_hat.value

# # DMCP approach
# constraints_dmcp = [cp.sum(wgts) == N - k, wgts <= 1]
# prob_dmcp = cp.Problem(cp.Minimize(cp.sum(cp.multiply(wgts, loss))), constraints_dmcp)
# assert dmcp.is_dmcp(prob_dmcp)
# prob_dmcp.solve(method='bcd')
# cooperative_vals = theta_hat.value

# # DSPP approach
# X_constraints = []
# Y_constraints = [cp.sum(wgts) == N, wgts <= 2, wgts >= 0.5]

# K = K_repr_y_Fx(loss, wgts)
# min_prob = cp.Problem(*minimax_to_min(K, X_constraints, Y_constraints))
# min_prob.solve(solver=cp.SCS)
# assert min_prob.status == cp.OPTIMAL
# adversarial_vals = theta_hat.value

# assert ols_vals[0] < adversarial_vals[0]
# assert ols_vals[1] > adversarial_vals[1]

# import matplotlib.pyplot as plt
# plt.figure()
# plt.scatter(A[:, 1], observations)
# plt.plot(A[:, 1], A @ adversarial_vals, label='Adversarial', c='red')
# plt.plot(A[:, 1], A @ cooperative_vals, label='Cooperative', c='green')
# plt.plot(A[:, 1], A @ ols_vals, label='OLS', color='orange')
# plt.legend()
# plt.show()

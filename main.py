# https://zhuanlan.zhihu.com/p/48876718
# https://www.bilibili.com/video/BV1dV411B7ME?share_source=copy_web
# DR_CAN例子的python实现
import numpy as np
import matplotlib.pyplot as plt


def gaussian_distribution_generator(var):
    return np.random.normal(loc=0.0, scale=var, size=None)


# 状态转移矩阵，上一时刻的状态转移到当前时刻
A = np.array([[1, 1],
              [0, 1]])

# 过程噪声协方差矩阵Q，p(w)~N(0,Q)，噪声来自真实世界中的不确定性
Q = np.array([[0.1, 0],
              [0, 0.1]])

# 观测噪声协方差矩阵R，p(v)~N(0,R)
R = np.array([[1, 0],
              [0, 1]])

# 状态观测矩阵
H = np.array([[1, 0],
              [0, 1]])

# 控制输入矩阵B
B = None
# 初始位置与速度
X0 = np.array([[0],
               [1]])

# 状态估计协方差矩阵P初始化
P = np.array([[1, 0],
              [0, 1]])

if __name__ == "__main__":
    # ---------------------------初始化-------------------------
    X_true = np.array(X0)  # 真实状态初始化
    X_posterior = np.array(X0)
    P_posterior = np.array(P)

    speed_true = []
    position_true = []

    speed_measure = []
    position_measure = []

    speed_prior_est = []
    position_prior_est = []

    speed_posterior_est = []
    position_posterior_est = []

    for i in range(30):
        # -----------------------生成真实值----------------------
        # 生成过程噪声
        w = np.array([[gaussian_distribution_generator(Q[0, 0])],
                      [gaussian_distribution_generator(Q[1, 1])]])
        X_true = np.dot(A, X_true) + w  # 得到当前时刻状态
        speed_true.append(X_true[1, 0])
        position_true.append(X_true[0, 0])
        # -----------------------生成观测值----------------------
        # 生成观测噪声
        v = np.array([[gaussian_distribution_generator(R[0, 0])],
                      [gaussian_distribution_generator(R[1, 1])]])

        Z_measure = np.dot(H, X_true) + v  # 生成观测值,H为单位阵E
        position_measure.append(Z_measure[0, 0])
        speed_measure.append(Z_measure[1, 0])
        # ----------------------进行先验估计---------------------
        X_prior = np.dot(A, X_posterior)
        position_prior_est.append(X_prior[0, 0])
        speed_prior_est.append(X_prior[1, 0])
        # 计算状态估计协方差矩阵P
        P_prior_1 = np.dot(A, P_posterior)
        P_prior = np.dot(P_prior_1, A.T) + Q
        # ----------------------计算卡尔曼增益,用numpy一步一步计算Prior and posterior
        k1 = np.dot(P_prior, H.T)
        k2 = np.dot(np.dot(H, P_prior), H.T) + R
        K = np.dot(k1, np.linalg.inv(k2))
        # ---------------------后验估计------------
        X_posterior_1 = Z_measure - np.dot(H, X_prior)
        X_posterior = X_prior + np.dot(K, X_posterior_1)
        position_posterior_est.append(X_posterior[0, 0])
        speed_posterior_est.append(X_posterior[1, 0])
        # 更新状态估计协方差矩阵P
        P_posterior_1 = np.eye(2) - np.dot(K, H)
        P_posterior = np.dot(P_posterior_1, P_prior)

       

    # 可视化显示
    if True:
        fig, axs = plt.subplots(1,2)
        axs[0].plot(speed_true, "-", label="speed_true", linewidth=1)  # Plot some data on the axes.
        axs[0].plot(speed_measure, "-", label="speed_measure", linewidth=1)  # Plot some data on the axes.
        axs[0].plot(speed_prior_est, "-", label="speed_prior_est", linewidth=1)  # Plot some data on the axes.
        axs[0].plot(speed_posterior_est, "-", label="speed_posterior_est", linewidth=1)  # Plot some data on the axes.
        axs[0].set_title("speed")
        axs[0].set_xlabel('k')  # Add an x-label to the axes.
        axs[0].legend()  # Add a legend.

        axs[1].plot(position_true, "-", label="position_true", linewidth=1)  # Plot some data on the axes.
        axs[1].plot(position_measure, "-", label="position_measure", linewidth=1)  # Plot some data on the axes.
        axs[1].plot(position_prior_est, "-", label="position_prior_est", linewidth=1)  # Plot some data on the axes.
        axs[1].plot(position_posterior_est, "-", label="position_posterior_est", linewidth=1)  # Plot some data on the axes.
        axs[1].set_title("position")
        axs[1].set_xlabel('k')  # Add an x-label to the axes.
        axs[1].legend()  # Add a legend.

        plt.show()

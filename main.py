# Example of Kalman filter used for estimation of ballistic missile trajectory

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class KalmanFilter:

    def __init__(self, A, B, M, H, Q, R):
        self.A = A
        self.B = B
        self.M = M
        self.H = H

        n, m = np.shape(A)
        self.n = n

        o, p = np.shape(H)
        self.o = o

        self.x = []
        # measurments
        self.z = []

        # noise covariances
        self.Q = Q  # process covariance
        self.R = R  # measurement covariance

        self.P_apri = []  # priori covariance estimate
        self.P_apost = []  # posteriori covariance estimate
        self.S = []  # output priori covariance estimate
        self.K = []  # Kalman gain

        self.xhat_apri = []  # a priori state estimate
        self.xhat_apost = []  # a posteriori state estimate

        # observation error
        self.ztilde = []

    def simulate(self, delT, t_max, x0, u, w, v, labels):
        self.delT = delT
        self.t_max = t_max
        self.t = np.arange(0, t_max, delT)  # time vector

        self.x0 = x0
        self.x.append(self.x0)
        self.z.append(np.dot(self.H, self.x0))

        self.xhat_apri.append(x0)
        self.xhat_apost.append(x0)

        self.P_apost.append(np.zeros((self.n, self.n)))
        self.P_apri.append(np.zeros((self.n, self.n)))
        self.S.append(np.zeros((self.n, self.n)))
        self.K.append(np.zeros((self.n, self.n)))

        self.ztilde.append(np.zeros([1, self.o]))

        # control inputs and noise realizations
        self.u = u
        self.w = w
        self.v = v

        i = 0
        while (i < len(self.t) - 1):
            # dynamics
            self.x.append(self.A @ self.x[i] + self.B @ self.u[i, :] + self.M @ self.w[i, :])
            # measurment
            self.z.append(self.H @ self.x[i + 1] + self.v[i + 1, :])

            # Kalman Filter - Prediction Step
            self.P_apri.append(self.A @ self.P_apost[i] @ self.A.T + self.Q)
            self.xhat_apri.append(self.A @ self.xhat_apost[i] + self.B @ self.u[i, :])

            # Kalman Filter - Update Step
            self.ztilde.append(self.z[i + 1] - self.H @ self.xhat_apri[i + 1])

            self.S.append(self.H @ self.P_apri[i + 1] @ self.H.T + R)

            self.K.append(self.P_apri[i + 1] @ self.H.T @ np.linalg.inv(self.S[i + 1]))

            self.xhat_apost.append(self.xhat_apri[i + 1] + self.K[i + 1] @ self.ztilde[i + 1])

            P_gain = np.eye(self.n) - self.K[i + 1] @ self.H

            self.P_apost.append(P_gain @ self.P_apri[i + 1] @ P_gain.T + self.K[i + 1] @ self.R @ self.K[i + 1].T)

            # update iteration counter
            i += 1

        # Position and velocity in last time moment
        end_pos_x = self.x[-1][0] / 20
        end_pos_y = self.x[-1][2] / 20
        end_pos = (end_pos_x ** 2 + end_pos_y ** 2) ** (1/2)
        print("Дальность в конечной точке: ", end_pos)
        end_velocity_x = self.x[-1][1] / 20
        end_velocity_y = self.x[-1][3] / 20
        end_velocity = ((end_velocity_x - self.x[0][1]) ** 2 + (end_velocity_y - self.x[0][3]) ** 2) ** (1/2)
        print("Скорость в конечной точке: ", end_velocity)

        # store data in dataframe
        # convert data to dataframe
        self.df_x = pd.DataFrame(np.vstack(self.x), index=self.t, columns=labels[0])
        self.df_z = pd.DataFrame(np.vstack(self.z), index=self.t, columns=labels[1])
        self.df_xhat_apri = pd.DataFrame(np.vstack(self.xhat_apri), index=self.t, columns=labels[2])

        # self.df = self.df_x.append([self.df_z, self.df_xhat_apri], sort=False)
        # FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version.
        # Use pandas.concat instead.
        self.df = pd.concat([self.df_x, self.df_z, self.df_xhat_apri], sort=False)

        return self.df

######################################################################################################################


def setup_missile_dynamics(delT):
    # dynamics
    A = np.zeros((4, 4))
    A[0, 0] = 1
    A[0, 1] = delT
    A[1, 1] = 1
    # A[1, 2] = delT
    A[2, 2] = 1
    A[2, 3] = delT
    A[3, 3] = 1

    B = np.zeros((4, 2))
    B[0, 0] = 0.5 * delT ** 2
    B[0, 1] = 0.5 * delT ** 2
    B[1, 0] = delT
    B[1, 1] = delT
    B[2, 0] = 0.5 * delT ** 2
    B[2, 1] = 0.5 * delT ** 2
    B[3, 0] = delT
    B[3, 1] = delT

    H = np.zeros((2, 4))
    H[0, 0] = 1
    H[1, 2] = 1

    return A, B, H

######################################################################################################################


def plot_results(df1, df2, df3):
    # X position
    # 1st missile
    plt.plot(df1.index, df1['x1'] / 20, label="x Первая цель")
    plt.plot(df1.index, df1['z1'] / 20, label="z1 Первая цель")
    plt.plot(df1.index, df1['xhat1'] / 20, label="x hat Первая цель")
    # 2nd missile
    plt.plot(df2.index, df2['x1'] / 20, label="x Вторая цель")
    plt.plot(df2.index, df2['z1'] / 20, label="z1 Вторая цель")
    plt.plot(df2.index, df2['xhat1'] / 20, label="x hat Вторая цель")
    # 3rd missile
    plt.plot(df3.index, df3['x1'] / 20, label="x Третья цель")
    plt.plot(df3.index, df3['z1'] / 20, label="z1 Третья цель")
    plt.plot(df3.index, df3['xhat1'] / 20, label="x hat Третья цель")
    plt.legend()
    plt.ylabel('Дальность')
    plt.xlabel('Время')
    plt.title('Дальность по X')
    # plt.ylabel('position')
    # plt.xlabel('time')
    # plt.title('X position')
    plt.show()

    # Y Position
    # 1st missile
    plt.plot(df1.index, df1['x2'] / 20, label="y Первая цель")
    plt.plot(df1.index, df1['z2'] / 20, label="z2 Первая цель")
    plt.plot(df1.index, df1['xhat2'] / 20, label="y hat Первая цель")
    # 2nd missile
    plt.plot(df2.index, df2['x2'] / 20, label="y Вторая цель")
    plt.plot(df2.index, df2['z2'] / 20, label="z2 Вторая цель")
    plt.plot(df2.index, df2['xhat2'] / 20, label="y hat Вторая цель")
    # 3rd missile
    plt.plot(df3.index, df3['x2'] / 20, label="y Третья цель")
    plt.plot(df3.index, df3['z2'] / 20, label="z2 Третья цель")
    plt.plot(df3.index, df3['xhat2'] / 20, label="y hat Третья цель")
    plt.legend()
    plt.ylabel('Дальность')
    plt.xlabel('Время')
    plt.title('Дальность по Y')
    # plt.ylabel('position')
    # plt.xlabel('time')
    # plt.title('Y position')
    plt.show()

    # X velocity
    # 1st missile
    plt.plot(df1.index, df1['x1_dot'] / 20, label="x1_dot Первая цель")
    plt.plot(df1.index, df1['xhat1_dot'] / 20, label="xhat1_dot Первая цель")
    # 2nd missile
    plt.plot(df2.index, df2['x1_dot'] / 20, label="x1_dot Вторая цель")
    plt.plot(df2.index, df2['xhat1_dot'] / 20, label="xhat1_dot Вторая цель")
    # 3rd missile
    plt.plot(df3.index, df3['x1_dot'] / 20, label="x1_dot Третья цель")
    plt.plot(df3.index, df3['xhat1_dot'] / 20, label="xhat1_dot Третья цель")
    plt.legend()
    plt.ylabel('Скорость')
    plt.xlabel('Время')
    plt.title('Скорость по X')
    # plt.ylabel('speed')
    # plt.xlabel('time')
    # plt.title('X velocity')
    plt.show()

    # Y velocity
    # 1st missile
    plt.plot(df1.index, df1['x2_dot'] / 20, label="x2_dot Первая цель")
    plt.plot(df1.index, df1['xhat2_dot'] / 20, label="xhat2_dot Первая цель")
    # 2nd missile
    plt.plot(df2.index, df2['x2_dot'] / 20, label="x2_dot Вторая цель")
    plt.plot(df2.index, df2['xhat2_dot'] / 20, label="xhat2_dot Вторая цель")
    # 3rd missile
    plt.plot(df3.index, df3['x2_dot'] / 20, label="x2_dot Третья цель")
    plt.plot(df3.index, df3['xhat2_dot'] / 20, label="xhat2_dot Третья цель")
    plt.legend()
    plt.ylabel('Скорость')
    plt.xlabel('Время')
    plt.title('Скорость по Y')
    # plt.ylabel('position')
    # plt.xlabel('time')
    # plt.title('Y velocity')
    plt.show()

######################################################################################################################


def data_frame_transform(df):
    df_x = df[['x1', 'x1_dot', 'x2', 'x2_dot']].dropna()
    df_z = df[['z1', 'z2']].dropna()
    df_xhat = df[['xhat1', 'xhat1_dot', 'xhat2', 'xhat2_dot']].dropna()
    df_x_z = df_x.join(df_z, rsuffix='_right')
    df_x_z_xhat = df_x_z.join(df_xhat, rsuffix='_right')
    return df_x_z_xhat

######################################################################################################################


def secondary_plot_results(df, title_suffix):
    # X position difference
    plt.plot(df['xhat1'] - df['x1'])
    plt.ylabel('Дальность')
    plt.xlabel('Время')
    plt.title('Разница значений дальности по X ' + title_suffix)
    # plt.ylabel('position')
    # plt.xlabel('time')
    # plt.title('X position difference ' + title_suffix)
    plt.show()

    # Y position difference
    plt.plot(df['xhat2'] - df['x2'])
    plt.ylabel('Дальность')
    plt.xlabel('Время')
    plt.title('Разница значений дальности по Y ' + title_suffix)
    # plt.ylabel('position')
    # plt.xlabel('time')
    # plt.title('Y position difference ' + title_suffix)
    plt.show()

    # X velocity
    plt.plot(df.index, df['xhat1_dot'] - df['x1_dot'])
    plt.ylabel('Скорость')
    plt.xlabel('Время')
    plt.title('Разница значений скорости по X ' + title_suffix)
    # plt.ylabel('velocity')
    # plt.xlabel('time')
    # plt.title('X velocity difference ' + title_suffix)
    plt.show()

    # Y velocity
    plt.plot(df.index, df['xhat2_dot'] - df['x2_dot'])
    plt.ylabel('Скорость')
    plt.xlabel('Время')
    plt.title('Разница значений скорости по Y ' + title_suffix)
    # plt.ylabel('velocity')
    # plt.xlabel('time')
    # plt.title('Y velocity difference ' + title_suffix)
    plt.show()

######################################################################################################################


if __name__ == "__main__":
    print("Simulation 1 Started")

    # Simulation Parameters
    delT = 0.008  # timestep
    t_max = 2  # time simulation stops (seconds)
    t = np.arange(0, t_max, delT)
    g = 242  # m/s^2
    mass = 22  # kg

    # initial conditions
    x0 = np.zeros(4)
    init_angle = 45
    init_angle_radians = np.radians(init_angle)
    init_velocity = 47

    x0[0] = 0
    x0[1] = init_velocity * np.cos(init_angle_radians)
    x0[2] = 0  # initial height
    x0[3] = init_velocity * np.sin(init_angle_radians)

    # process noise: disturbance forces, normal distributions
    sig_px = 10
    sig_py = 1

    # measurement noise
    sig_mx = 300
    sig_my = 300

    Q = np.diag([sig_px, sig_px, sig_py, sig_py])
    R = np.diag([sig_mx, sig_my])

    A, B, H = setup_missile_dynamics(delT)

    # process and measurement noise
    w_x = np.random.normal(0, sig_px, (len(t), 1))
    w_y = np.random.normal(0, sig_py, (len(t), 1))
    w = np.hstack((w_x, w_y))

    v = np.hstack((np.random.normal(0, sig_mx, (len(t), 1)), np.random.normal(0, sig_my, (len(t), 1))))

    # input (only input is acceleration due to gravity)
    u = np.hstack((np.zeros((len(t), 1)), np.ones((len(t), 1)) * g * mass))

    # setup Kalman Filter
    kalman = KalmanFilter(A, B, B, H, Q, R)

    # dataframe column labels
    state_labels = ['x1', 'x1_dot', 'x2', 'x2_dot']
    measurement_labels = ['z1', 'z2']
    apriori_state_labels = ['xhat1', 'xhat1_dot', 'xhat2', 'xhat2_dot']
    labels = [state_labels, measurement_labels, apriori_state_labels]

    # run the simulation
    df1 = kalman.simulate(delT, t_max, x0, u, w, v, labels)

    print("Simulation 1 Complete")

    ##################################################################################################################

    # 2nd missile
    print("Simulation 2 Started")

    # Simulation Parameters
    delT = 0.008  # timestep
    t_max = 2  # time simulation stops (seconds)
    t = np.arange(0, t_max, delT)
    g = 450  # m/s^2
    mass = 22  # kg

    # initial conditions
    x0 = np.zeros(4)
    init_angle = 45
    init_angle_radians = np.radians(init_angle)
    init_velocity = 49

    x0[0] = 0
    x0[1] = init_velocity * np.cos(init_angle_radians)
    x0[2] = 0  # initial height
    x0[3] = init_velocity * np.sin(init_angle_radians)

    # process noise: disturbance forces, normal distributions
    sig_px = 10
    sig_py = 1

    # measurement noise
    sig_mx = 300
    sig_my = 300

    Q = np.diag([sig_px, sig_px, sig_py, sig_py])
    R = np.diag([sig_mx, sig_my])

    A, B, H = setup_missile_dynamics(delT)

    # process and measurement noise
    w_x = np.random.normal(0, sig_px, (len(t), 1))
    w_y = np.random.normal(0, sig_py, (len(t), 1))
    w = np.hstack((w_x, w_y))

    v = np.hstack((np.random.normal(0, sig_mx, (len(t), 1)), np.random.normal(0, sig_my, (len(t), 1))))

    # input (only input is acceleration due to gravity)
    u = np.hstack((np.zeros((len(t), 1)), np.ones((len(t), 1)) * g * mass))

    # setup Kalman Filter
    kalman = KalmanFilter(A, B, B, H, Q, R)

    # dataframe column labels
    state_labels = ['x1', 'x1_dot', 'x2', 'x2_dot']
    measurement_labels = ['z1', 'z2']
    apriori_state_labels = ['xhat1', 'xhat1_dot', 'xhat2', 'xhat2_dot']
    labels = [state_labels, measurement_labels, apriori_state_labels]

    # run the simulation
    df2 = kalman.simulate(delT, t_max, x0, u, w, v, labels)

    ##################################################################################################################

    # 3rd missile
    print("Simulation 3 Started")

    # Simulation Parameters
    delT = 0.008  # timestep
    t_max = 2  # time simulation stops (seconds)
    t = np.arange(0, t_max, delT)
    g = 331  # m/s^2
    mass = 22  # kg

    # initial conditions
    x0 = np.zeros(4)
    init_angle = 45
    init_angle_radians = np.radians(init_angle)
    init_velocity = 42

    x0[0] = 0
    x0[1] = init_velocity * np.cos(init_angle_radians)
    x0[2] = 0  # initial height
    x0[3] = init_velocity * np.sin(init_angle_radians)

    # process noise: disturbance forces, normal distributions
    sig_px = 10
    sig_py = 1

    # measurement noise
    sig_mx = 300
    sig_my = 300

    Q = np.diag([sig_px, sig_px, sig_py, sig_py])
    R = np.diag([sig_mx, sig_my])

    A, B, H = setup_missile_dynamics(delT)

    # process and measurement noise
    w_x = np.random.normal(0, sig_px, (len(t), 1))
    w_y = np.random.normal(0, sig_py, (len(t), 1))
    w = np.hstack((w_x, w_y))

    v = np.hstack((np.random.normal(0, sig_mx, (len(t), 1)), np.random.normal(0, sig_my, (len(t), 1))))

    # input (only input is acceleration due to gravity)
    u = np.hstack((np.zeros((len(t), 1)), np.ones((len(t), 1)) * g * mass))

    # setup Kalman Filter
    kalman = KalmanFilter(A, B, B, H, Q, R)

    # dataframe column labels
    state_labels = ['x1', 'x1_dot', 'x2', 'x2_dot']
    measurement_labels = ['z1', 'z2']
    apriori_state_labels = ['xhat1', 'xhat1_dot', 'xhat2', 'xhat2_dot']
    labels = [state_labels, measurement_labels, apriori_state_labels]

    # run the simulation
    df3 = kalman.simulate(delT, t_max, x0, u, w, v, labels)

    print("Simulation 3 Complete")

    ##################################################################################################################

    plot_results(df1, df2, df3)

    df1 = data_frame_transform(df1)
    # secondary_plot_results(df1, "(1st missile)")
    secondary_plot_results(df1, "(Первая цель)")
    df2 = data_frame_transform(df2)
    # secondary_plot_results(df2, "(2nd missile)")
    secondary_plot_results(df2, "(Вторая цель)")
    df3 = data_frame_transform(df3)
    # secondary_plot_results(df3, "(3rd missile)")
    secondary_plot_results(df3, "(Третья цель)")

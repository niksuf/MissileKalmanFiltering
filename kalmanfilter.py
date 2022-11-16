import numpy as np
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
        while i < len(self.t) - 1:
            # dynamics
            self.x.append(self.A @ self.x[i] + self.B @ self.u[i, :] + self.M @ self.w[i, :])
            # measurment
            self.z.append(self.H @ self.x[i + 1] + self.v[i + 1, :])

            # Kalman Filter - Prediction Step
            self.P_apri.append(self.A @ self.P_apost[i] @ self.A.T + self.Q)
            self.xhat_apri.append(self.A @ self.xhat_apost[i] + self.B @ self.u[i, :])

            # Kalman Filter - Update Step
            self.ztilde.append(self.z[i + 1] - self.H @ self.xhat_apri[i + 1])

            self.S.append(self.H @ self.P_apri[i + 1] @ self.H.T + self.R)

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
        print("Position in last point: ", end_pos)
        end_velocity_x = self.x[-1][1] / 20
        end_velocity_y = self.x[-1][3] / 20
        end_velocity = ((end_velocity_x - self.x[0][1]) ** 2 + (end_velocity_y - self.x[0][3]) ** 2) ** (1/2)
        print("Velocity in last point: ", end_velocity)

        # store data in dataframe
        # convert data to dataframe
        self.df_x = pd.DataFrame(np.vstack(self.x), index=self.t, columns=labels[0])
        self.df_z = pd.DataFrame(np.vstack(self.z), index=self.t, columns=labels[1])
        self.df_xhat_apri = pd.DataFrame(np.vstack(self.xhat_apri), index=self.t, columns=labels[2])

        self.df = pd.concat([self.df_x, self.df_z, self.df_xhat_apri], sort=False)

        return self.df

######################################################################################################################


def setup_missile_dynamics(delT):
    # dynamics
    A = np.zeros((4, 4))
    A[0, 0] = 1
    A[0, 1] = delT
    A[1, 1] = 1
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

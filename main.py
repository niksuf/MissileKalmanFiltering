import numpy as np

from kalmanfilter import KalmanFilter, setup_missile_dynamics
from graphics import plot_results, data_frame_transform, secondary_plot_results


def main():
    print("Simulation 1 Started")

    # Simulation Parameters
    first_missile_t_min = 0
    first_missile_delT = 0.008  # timestep
    first_missile_t_max = 2.4  # time simulation stops (seconds)
    first_missile_t = np.arange(0, first_missile_t_max, first_missile_delT)
    first_missile_g = 242  # m/s^2
    first_missile_mass = 22  # kg

    # initial conditions
    first_missile_x0 = np.zeros(4)
    init_angle = 45
    init_angle_radians = np.radians(init_angle)
    init_velocity = 47

    first_missile_x0[0] = 0
    first_missile_x0[1] = init_velocity * np.cos(init_angle_radians)
    first_missile_x0[2] = 0  # initial height
    first_missile_x0[3] = init_velocity * np.sin(init_angle_radians)

    # process noise: disturbance forces, normal distributions
    first_missile_sig_px = 10
    first_missile_sig_py = 1

    # measurement noise
    first_missile_sig_mx = 300
    first_missile_sig_my = 300

    first_missile_Q = np.diag([first_missile_sig_px, first_missile_sig_px, first_missile_sig_py, first_missile_sig_py])
    first_missile_R = np.diag([first_missile_sig_mx, first_missile_sig_my])

    first_missile_A, first_missile_B, first_missile_H = setup_missile_dynamics(first_missile_delT)

    # process and measurement noise
    first_missile_w_x = np.random.normal(0, first_missile_sig_px, (len(first_missile_t), 1))
    first_missile_w_y = np.random.normal(0, first_missile_sig_py, (len(first_missile_t), 1))
    first_missile_w = np.hstack((first_missile_w_x, first_missile_w_y))

    first_missile_v = np.hstack((np.random.normal(0, first_missile_sig_mx, (len(first_missile_t), 1)),
                                 np.random.normal(0, first_missile_sig_my, (len(first_missile_t), 1))))

    # input (only input is acceleration due to gravity)
    first_missile_u = np.hstack((np.zeros((len(first_missile_t), 1)), np.ones((len(first_missile_t), 1)) *
                                 first_missile_g * first_missile_mass))

    # setup Kalman Filter
    first_missile_kalman = KalmanFilter(first_missile_A, first_missile_B, first_missile_B,
                                        first_missile_H, first_missile_Q, first_missile_R)

    # dataframe column labels
    state_labels = ['x1', 'x1_dot', 'x2', 'x2_dot']
    measurement_labels = ['z1', 'z2']
    apriori_state_labels = ['xhat1', 'xhat1_dot', 'xhat2', 'xhat2_dot']
    first_missile_labels = [state_labels, measurement_labels, apriori_state_labels]

    # run the simulation
    df1 = first_missile_kalman.simulate(first_missile_t_min, first_missile_delT,
                                        first_missile_t_max, first_missile_x0, first_missile_u,
                                        first_missile_w, first_missile_v, first_missile_labels)

    print("Simulation 1 Complete")

    ##################################################################################################################

    # 2nd missile
    print("Simulation 2 Started")

    # Simulation Parameters
    t_min = 0.1
    delT = 0.008  # timestep
    t_max = 1.9  # time simulation stops (seconds)
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
    df2 = kalman.simulate(t_min, delT, t_max, x0, u, w, v, labels)

    print("Simulation 2 Complete")

    ##################################################################################################################

    # 3rd missile
    print("Simulation 3 Started")

    # Simulation Parameters
    t_min = 0.2
    delT = 0.008  # timestep
    t_max = 2.3  # time simulation stops (seconds)
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
    df3 = kalman.simulate(t_min, delT, t_max, x0, u, w, v, labels)

    print("Simulation 3 Complete")

    ##################################################################################################################

    plot_results(df1, df2, df3)

    df1 = data_frame_transform(df1)
    secondary_plot_results(df1, "(1st missile)")
    df2 = data_frame_transform(df2)
    secondary_plot_results(df2, "(2nd missile)")
    df3 = data_frame_transform(df3)
    secondary_plot_results(df3, "(3rd missile)")


if __name__ == "__main__":
    main()

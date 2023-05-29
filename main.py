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
    first_missile_init_angle = 45
    first_missile_init_angle_radians = np.radians(first_missile_init_angle)
    first_missile_init_velocity = 47

    first_missile_x0[0] = 0
    first_missile_x0[1] = first_missile_init_velocity * np.cos(first_missile_init_angle_radians)
    first_missile_x0[2] = 0  # initial height
    first_missile_x0[3] = first_missile_init_velocity * np.sin(first_missile_init_angle_radians)

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
    first_missile_state_labels = ['x1', 'x1_dot', 'x2', 'x2_dot']
    first_missile_measurement_labels = ['z1', 'z2']
    first_missile_apriori_state_labels = ['xhat1', 'xhat1_dot', 'xhat2', 'xhat2_dot']
    first_missile_labels = [first_missile_state_labels, first_missile_measurement_labels,
                            first_missile_apriori_state_labels]

    # run the simulation
    df1 = first_missile_kalman.simulate(first_missile_t_min, first_missile_delT,
                                        first_missile_t_max, first_missile_x0, first_missile_u,
                                        first_missile_w, first_missile_v, first_missile_labels)

    print("Simulation 1 Complete")

    ##################################################################################################################

    # 2nd missile
    print("Simulation 2 Started")

    # Simulation Parameters
    second_missile_t_min = 0.1
    second_missile_delT = 0.008  # timestep
    second_missile_t_max = 1.9  # time simulation stops (seconds)
    second_missile_t = np.arange(0, second_missile_t_max, second_missile_delT)
    second_missile_g = 450  # m/s^2
    second_missile_mass = 22  # kg

    # initial conditions
    second_missile_x0 = np.zeros(4)
    second_missile_init_angle = 45
    second_missile_init_angle_radians = np.radians(second_missile_init_angle)
    second_missile_init_velocity = 49

    second_missile_x0[0] = 0
    second_missile_x0[1] = second_missile_init_velocity * np.cos(second_missile_init_angle_radians)
    second_missile_x0[2] = 0  # initial height
    second_missile_x0[3] = second_missile_init_velocity * np.sin(second_missile_init_angle_radians)

    # process noise: disturbance forces, normal distributions
    second_missile_sig_px = 10
    second_missile_sig_py = 1

    # measurement noise
    second_missile_sig_mx = 300
    second_missile_sig_my = 300

    second_missile_Q = np.diag([second_missile_sig_px, second_missile_sig_px, second_missile_sig_py, second_missile_sig_py])
    second_missile_R = np.diag([second_missile_sig_mx, second_missile_sig_my])

    second_missile_A, second_missile_B, second_missile_H = setup_missile_dynamics(second_missile_delT)

    # process and measurement noise
    second_missile_w_x = np.random.normal(0, second_missile_sig_px, (len(second_missile_t), 1))
    second_missile_w_y = np.random.normal(0, second_missile_sig_py, (len(second_missile_t), 1))
    second_missile_w = np.hstack((second_missile_w_x, second_missile_w_y))

    second_missile_v = np.hstack((np.random.normal(0, second_missile_sig_mx, (len(second_missile_t), 1)),
                   np.random.normal(0, second_missile_sig_my, (len(second_missile_t), 1))))

    # input (only input is acceleration due to gravity)
    second_missile_u = np.hstack((np.zeros((len(second_missile_t), 1)),
                                  np.ones((len(second_missile_t), 1)) * second_missile_g * second_missile_mass))

    # setup Kalman Filter
    second_missile_kalman = KalmanFilter(second_missile_A, second_missile_B, second_missile_B,
                                         second_missile_H, second_missile_Q, second_missile_R)

    # dataframe column labels
    second_missile_state_labels = ['x1', 'x1_dot', 'x2', 'x2_dot']
    second_missile_measurement_labels = ['z1', 'z2']
    second_missile_apriori_state_labels = ['xhat1', 'xhat1_dot', 'xhat2', 'xhat2_dot']
    second_missile_labels = [second_missile_state_labels, second_missile_measurement_labels,
                             second_missile_apriori_state_labels]

    # run the simulation
    df2 = second_missile_kalman.simulate(second_missile_t_min, second_missile_delT, second_missile_t_max,
                                         second_missile_x0, second_missile_u, second_missile_w,
                                         second_missile_v, second_missile_labels)

    print("Simulation 2 Complete")

    ##################################################################################################################

    # 3rd missile
    print("Simulation 3 Started")

    # Simulation Parameters
    third_missile_t_min = 0.2
    third_missile_delT = 0.008  # timestep
    third_missile_t_max = 2.3  # time simulation stops (seconds)
    third_missile_t = np.arange(0, third_missile_t_max, third_missile_delT)
    third_missile_g = 331  # m/s^2
    third_missile_mass = 22  # kg

    # initial conditions
    third_missile_x0 = np.zeros(4)
    third_missile_init_angle = 45
    third_missile_init_angle_radians = np.radians(third_missile_init_angle)
    third_missile_init_velocity = 42

    third_missile_x0[0] = 0
    third_missile_x0[1] = third_missile_init_velocity * np.cos(third_missile_init_angle_radians)
    third_missile_x0[2] = 0  # initial height
    third_missile_x0[3] = third_missile_init_velocity * np.sin(third_missile_init_angle_radians)

    # process noise: disturbance forces, normal distributions
    third_missile_sig_px = 10
    third_missile_sig_py = 1

    # measurement noise
    third_missile_sig_mx = 300
    third_missile_sig_my = 300

    third_missile_Q = np.diag([third_missile_sig_px, third_missile_sig_px, third_missile_sig_py, third_missile_sig_py])
    third_missile_R = np.diag([third_missile_sig_mx, third_missile_sig_my])

    third_missile_A, third_missile_B, third_missile_H = setup_missile_dynamics(third_missile_delT)

    # process and measurement noise
    third_missile_w_x = np.random.normal(0, third_missile_sig_px, (len(third_missile_t), 1))
    third_missile_w_y = np.random.normal(0, third_missile_sig_py, (len(third_missile_t), 1))
    third_missile_w = np.hstack((third_missile_w_x, third_missile_w_y))

    third_missile_v = np.hstack((np.random.normal(0, third_missile_sig_mx, (len(third_missile_t), 1)),
                                 np.random.normal(0, third_missile_sig_my, (len(third_missile_t), 1))))

    # input (only input is acceleration due to gravity)
    third_missile_u = np.hstack((np.zeros((len(third_missile_t), 1)),
                                 np.ones((len(third_missile_t), 1)) * third_missile_g * third_missile_mass))

    # setup Kalman Filter
    third_missile_kalman = KalmanFilter(third_missile_A, third_missile_B, third_missile_B,
                                        third_missile_H, third_missile_Q, third_missile_R)

    # dataframe column labels
    third_missile_state_labels = ['x1', 'x1_dot', 'x2', 'x2_dot']
    third_missile_measurement_labels = ['z1', 'z2']
    third_missile_apriori_state_labels = ['xhat1', 'xhat1_dot', 'xhat2', 'xhat2_dot']
    third_missile_labels = [third_missile_state_labels, third_missile_measurement_labels,
                            third_missile_apriori_state_labels]

    # run the simulation
    df3 = third_missile_kalman.simulate(third_missile_t_min, third_missile_delT, third_missile_t_max,
                                        third_missile_x0, third_missile_u, third_missile_w,
                                        third_missile_v, third_missile_labels)

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

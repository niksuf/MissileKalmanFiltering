import numpy as np

from kalmanfilter import KalmanFilter, setup_missile_dynamics
from graphics import plot_results, data_frame_transform, secondary_plot_results


def simulate_missile(t_min, delT, t_max, g, mass, init_angle, init_velocity):
    t = np.arange(0, t_max, delT)
    
    x0 = np.zeros(4)
    init_angle_radians = np.radians(init_angle)
    
    x0[0] = 0
    x0[1] = init_velocity * np.cos(init_angle_radians)
    x0[2] = 0
    x0[3] = init_velocity * np.sin(init_angle_radians)
    
    sig_px = 10
    sig_py = 1
    
    sig_mx = 300
    sig_my = 300
    
    Q = np.diag([sig_px, sig_px, sig_py, sig_py])
    R = np.diag([sig_mx, sig_my])
    
    A, B, H = setup_missile_dynamics(delT)
    
    w_x = np.random.normal(0, sig_px, (len(t), 1))
    w_y = np.random.normal(0, sig_py, (len(t), 1))
    w = np.hstack((w_x, w_y))
    
    v = np.hstack((np.random.normal(0, sig_mx, (len(t), 1)), np.random.normal(0, sig_my, (len(t), 1))))
    
    u = np.hstack((np.zeros((len(t), 1)), np.ones((len(t), 1)) * g * mass))
    
    kalman = KalmanFilter(A, B, B, H, Q, R)
    
    state_labels = ['x1', 'x1_dot', 'x2', 'x2_dot']
    measurement_labels = ['z1', 'z2']
    apriori_state_labels = ['xhat1', 'xhat1_dot', 'xhat2', 'xhat2_dot']
    labels = [state_labels, measurement_labels, apriori_state_labels]
    
    df = kalman.simulate(t_min, delT, t_max, x0, u, w, v, labels)
    
    return df


def main():
    simulations = [
        {"t_min": 0, "delT": 0.008, "t_max": 2.4, "g": 242, "mass": 22, "init_angle": 45, "init_velocity": 47},
        {"t_min": 0.1, "delT": 0.008, "t_max": 1.9, "g": 450, "mass": 22, "init_angle": 45, "init_velocity": 49},
        {"t_min": 0.2, "delT": 0.008, "t_max": 2.3, "g": 331, "mass": 22, "init_angle": 45, "init_velocity": 42}
    ]

    
    dfs = []
    
    for i, sim in enumerate(simulations, start=1):
        print(f"Simulation {i} Started")
        df = simulate_missile(**sim)
        dfs.append(df)
        print(f"Simulation {i} Complete")
    
    plot_results(*dfs)
    
    for i, df in enumerate(dfs, start=1):
        df = data_frame_transform(df)
        secondary_plot_results(df, f"({i} missile)")

if __name__ == "__main__":
    main()

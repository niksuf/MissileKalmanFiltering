import matplotlib.pyplot as plt


def plot_results(*dfs):
    categories = [('X position', ['x1', 'z1', 'xhat1']), ('Y position', ['x2', 'z2', 'xhat2']),
                  ('X velocity', ['x1_dot', 'xhat1_dot']), ('Y velocity', ['x2_dot', 'xhat2_dot'])]
    
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(12, 8))

    for i, (title, cols) in enumerate(categories):
        for j, df in enumerate(dfs):
            for col in cols:
                axs[j][i].plot(df.index, df[col] / 20, label=f"{col.split('_')[0]} {' '.join(col.split('_')[1:])} Missile {j+1}")

            axs[j][i].set_title(title)
            axs[j][i].legend()
            axs[j][i].set_ylabel('position' if 'position' in title.lower() else 'speed')
            axs[j][i].set_xlabel('time')
            axs[j][i].grid(True)

    plt.tight_layout()
    plt.show()


def data_frame_transform(df):
    df_x = df[['x1', 'x1_dot', 'x2', 'x2_dot']].dropna()
    df_z = df[['z1', 'z2']].dropna()
    df_xhat = df[['xhat1', 'xhat1_dot', 'xhat2', 'xhat2_dot']].dropna()
    df_x_z = df_x.join(df_z, rsuffix='_right')
    df_x_z_xhat = df_x_z.join(df_xhat, rsuffix='_right')
    return df_x_z_xhat


def plot_difference(ax, data_x, data_y, ylabel, xlabel, title):
    ax.plot(data_x, data_y)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(True)


def secondary_plot_results(df, title_suffix):
    fig, axs = plt.subplots(nrows=2, ncols=2)

    plot_difference(axs[0][0], df.index, df['xhat1'] - df['x1'], 'position', 'time', 'X position difference ' + title_suffix)
    plot_difference(axs[0][1], df.index, df['xhat2'] - df['x2'], 'position', 'time', 'Y position difference ' + title_suffix)
    plot_difference(axs[1][0], df.index, df['xhat1_dot'] - df['x1_dot'], 'velocity', 'time', 'X velocity difference ' + title_suffix)
    plot_difference(axs[1][1], df.index, df['xhat2_dot'] - df['x2_dot'], 'velocity', 'time', 'Y velocity difference ' + title_suffix)

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt


def plot_results(df1, df2, df3):
    fig, axs = plt.subplots(nrows=3, ncols=4)

    # X position
    # 1st missile
    axs[0][0].plot(df1.index, df1['x1'] / 20, label="x First missile")
    axs[0][0].plot(df1.index, df1['z1'] / 20, label="z1 First missile")
    axs[0][0].plot(df1.index, df1['xhat1'] / 20, label="x hat First missile")

    # 2nd missile
    axs[1][0].plot(df2.index, df2['x1'] / 20, label="x Second missile")
    axs[1][0].plot(df2.index, df2['z1'] / 20, label="z1 Second missile")
    axs[1][0].plot(df2.index, df2['xhat1'] / 20, label="x hat Second missile")
    # 3rd missile
    axs[2][0].plot(df3.index, df3['x1'] / 20, label="x Third missile")
    axs[2][0].plot(df3.index, df3['z1'] / 20, label="z1 Third missile")
    axs[2][0].plot(df3.index, df3['xhat1'] / 20, label="x hat Third missile")
    # labels and legend
    axs[0][0].set_title('X position')

    # Y Position
    # 1st missile
    axs[0][1].plot(df1.index, df1['x2'] / 20, label="y First missile")
    axs[0][1].plot(df1.index, df1['z2'] / 20, label="z2 First missile")
    axs[0][1].plot(df1.index, df1['xhat2'] / 20, label="y hat First missile")
    # 2nd missile
    axs[1][1].plot(df2.index, df2['x2'] / 20, label="y Second missile")
    axs[1][1].plot(df2.index, df2['z2'] / 20, label="z2 Second missile")
    axs[1][1].plot(df2.index, df2['xhat2'] / 20, label="y hat Second missile")
    # 3rd missile
    axs[2][1].plot(df3.index, df3['x2'] / 20, label="y Third missile")
    axs[2][1].plot(df3.index, df3['z2'] / 20, label="z2 Third missile")
    axs[2][1].plot(df3.index, df3['xhat2'] / 20, label="y hat Third missile")
    # labels and legend
    axs[0][1].set_title('Y position')
    for i in range(3):
        for j in range(2):
            axs[i][j].legend()
            axs[i][j].set_ylabel('position')
            axs[i][j].set_xlabel('time')
            axs[i][j].grid(True)

    # X velocity
    # 1st missile
    axs[0][2].plot(df1.index, df1['x1_dot'] / 20, label="x1_dot First missile")
    axs[0][2].plot(df1.index, df1['xhat1_dot'] / 20, label="xhat1_dot First missile")
    # 2nd missile
    axs[1][2].plot(df2.index, df2['x1_dot'] / 20, label="x1_dot Second missile")
    axs[1][2].plot(df2.index, df2['xhat1_dot'] / 20, label="xhat1_dot Second missile")
    # 3rd missile
    axs[2][2].plot(df3.index, df3['x1_dot'] / 20, label="x1_dot Third missile")
    axs[2][2].plot(df3.index, df3['xhat1_dot'] / 20, label="xhat1_dot Third missile")
    # labels and legend
    axs[0][2].set_title('X velocity')

    # Y velocity
    # 1st missile
    axs[0][3].plot(df1.index, df1['x2_dot'] / 20, label="x2_dot First missile")
    axs[0][3].plot(df1.index, df1['xhat2_dot'] / 20, label="xhat2_dot First missile")
    # 2nd missile
    axs[1][3].plot(df2.index, df2['x2_dot'] / 20, label="x2_dot Second missile")
    axs[1][3].plot(df2.index, df2['xhat2_dot'] / 20, label="xhat2_dot Second missile")
    # 3rd missile
    axs[2][3].plot(df3.index, df3['x2_dot'] / 20, label="x2_dot Third missile")
    axs[2][3].plot(df3.index, df3['xhat2_dot'] / 20, label="xhat2_dot Third missile")
    # labels and legend
    axs[0][3].set_title('Y velocity')
    for i in range(3):
        for j in range(2, 4):
            axs[i][j].legend()
            axs[i][j].set_ylabel('speed')
            axs[i][j].set_xlabel('time')
            axs[i][j].grid(True)

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
    fig, axs = plt.subplots(nrows=2, ncols=2)

    # X position difference
    axs[0][0].plot(df['xhat1'] - df['x1'])
    axs[0][0].set_title('X position difference ' + title_suffix)

    # Y position difference
    axs[0][1].plot(df['xhat2'] - df['x2'])
    axs[0][1].set_title('Y position difference ' + title_suffix)

    for i in range(2):
        axs[0][i].set_ylabel('position')
        axs[0][i].set_xlabel('time')
        axs[0][i].grid(True)

    # X velocity
    axs[1][0].plot(df.index, df['xhat1_dot'] - df['x1_dot'])
    axs[1][0].set_title('X velocity difference ' + title_suffix)

    # Y velocity
    axs[1][1].plot(df.index, df['xhat2_dot'] - df['x2_dot'])
    axs[1][1].set_title('Y velocity difference ' + title_suffix)

    for i in range(2):
        axs[1][i].set_ylabel('velocity')
        axs[1][i].set_xlabel('time')
        axs[1][i].grid(True)

    plt.show()

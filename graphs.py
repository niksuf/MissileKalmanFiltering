import matplotlib.pyplot as plt


def plot_results(df1, df2, df3):
    # X position
    # 1st missile
    plt.plot(df1.index, df1['x1'] / 20, label="x First missile")
    plt.plot(df1.index, df1['z1'] / 20, label="z1 First missile")
    plt.plot(df1.index, df1['xhat1'] / 20, label="x hat First missile")
    # 2nd missile
    plt.plot(df2.index, df2['x1'] / 20, label="x Second missile")
    plt.plot(df2.index, df2['z1'] / 20, label="z1 Second missile")
    plt.plot(df2.index, df2['xhat1'] / 20, label="x hat Second missile")
    # 3rd missile
    plt.plot(df3.index, df3['x1'] / 20, label="x Third missile")
    plt.plot(df3.index, df3['z1'] / 20, label="z1 Third missile")
    plt.plot(df3.index, df3['xhat1'] / 20, label="x hat Third missile")
    plt.legend()
    plt.ylabel('position')
    plt.xlabel('time')
    plt.title('X position')
    plt.show()

    # Y Position
    # 1st missile
    plt.plot(df1.index, df1['x2'] / 20, label="y First missile")
    plt.plot(df1.index, df1['z2'] / 20, label="z2 First missile")
    plt.plot(df1.index, df1['xhat2'] / 20, label="y hat First missile")
    # 2nd missile
    plt.plot(df2.index, df2['x2'] / 20, label="y Second missile")
    plt.plot(df2.index, df2['z2'] / 20, label="z2 Second missile")
    plt.plot(df2.index, df2['xhat2'] / 20, label="y hat Second missile")
    # 3rd missile
    plt.plot(df3.index, df3['x2'] / 20, label="y Third missile")
    plt.plot(df3.index, df3['z2'] / 20, label="z2 Third missile")
    plt.plot(df3.index, df3['xhat2'] / 20, label="y hat Third missile")
    plt.legend()
    plt.ylabel('position')
    plt.xlabel('time')
    plt.title('Y position')
    plt.show()

    # X velocity
    # 1st missile
    plt.plot(df1.index, df1['x1_dot'] / 20, label="x1_dot First missile")
    plt.plot(df1.index, df1['xhat1_dot'] / 20, label="xhat1_dot First missile")
    # 2nd missile
    plt.plot(df2.index, df2['x1_dot'] / 20, label="x1_dot Second missile")
    plt.plot(df2.index, df2['xhat1_dot'] / 20, label="xhat1_dot Second missile")
    # 3rd missile
    plt.plot(df3.index, df3['x1_dot'] / 20, label="x1_dot Third missile")
    plt.plot(df3.index, df3['xhat1_dot'] / 20, label="xhat1_dot Third missile")
    plt.legend()
    plt.ylabel('speed')
    plt.xlabel('time')
    plt.title('X velocity')
    plt.show()

    # Y velocity
    # 1st missile
    plt.plot(df1.index, df1['x2_dot'] / 20, label="x2_dot First missile")
    plt.plot(df1.index, df1['xhat2_dot'] / 20, label="xhat2_dot First missile")
    # 2nd missile
    plt.plot(df2.index, df2['x2_dot'] / 20, label="x2_dot Second missile")
    plt.plot(df2.index, df2['xhat2_dot'] / 20, label="xhat2_dot Second missile")
    # 3rd missile
    plt.plot(df3.index, df3['x2_dot'] / 20, label="x2_dot Third missile")
    plt.plot(df3.index, df3['xhat2_dot'] / 20, label="xhat2_dot Third missile")
    plt.legend()
    plt.ylabel('position')
    plt.xlabel('time')
    plt.title('Y velocity')
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
    plt.ylabel('position')
    plt.xlabel('time')
    plt.title('X position difference ' + title_suffix)
    plt.show()

    # Y position difference
    plt.plot(df['xhat2'] - df['x2'])
    plt.ylabel('position')
    plt.xlabel('time')
    plt.title('Y position difference ' + title_suffix)
    plt.show()

    # X velocity
    plt.plot(df.index, df['xhat1_dot'] - df['x1_dot'])
    plt.ylabel('velocity')
    plt.xlabel('time')
    plt.title('X velocity difference ' + title_suffix)
    plt.show()

    # Y velocity
    plt.plot(df.index, df['xhat2_dot'] - df['x2_dot'])
    plt.ylabel('velocity')
    plt.xlabel('time')
    plt.title('Y velocity difference ' + title_suffix)
    plt.show()

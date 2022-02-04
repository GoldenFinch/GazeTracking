import numpy as np
import matplotlib.pyplot as plt


def angle_calculation_avg(vector1, vector2):
    absolute_vector1 = np.sqrt(np.sum(vector1 * vector1, axis=1))
    absolute_vector2 = np.sqrt(np.sum(vector2 * vector2, axis=1))
    cos_theta = np.sum(vector1 * vector2, axis=1) / (absolute_vector1 * absolute_vector2)
    cos_theta = np.clip(cos_theta, -1, 1)
    theta = np.degrees(np.arccos(cos_theta))

    return np.sum(theta) / len(vector1)


# print(angle_calculation_avg(np.array([[0,0,1],[0,1,0]]), np.array([[0,1,0],[0, -1,0]])))


def paint1():
    fig, ax = plt.subplots(figsize=(8,8),dpi=80)
    ax.bar(np.arange(4), [1.12, 0.83, 0.76, 0.71], tick_label=['0','4','6','8'], color=['#66806A', '#B4C6A6', '#FFC286', '#FFF1AF'])
    ax.set_ylabel('angle mean error', size = 18)
    ax.set_xlabel('number of calibration parameters(N)', size =18)
    ax.text(0-0.12,1.12+0.02,1.12, va='center',fontsize=14)
    ax.text(1 - 0.12, 0.83 + 0.02, 0.83, va='center', fontsize=14)
    ax.text(2 - 0.12, 0.76 + 0.02, 0.76, va='center', fontsize=14)
    ax.text(3 - 0.12, 0.71 + 0.02, 0.71, va='center', fontsize=14)
    plt.show()


def paint2():
    fig, ax = plt.subplots(figsize=(8,8),dpi=80)

    ax.bar(np.arange(3),[3.31,3.09,2.99],tick_label=['4','6','8'],color=['#66806A', '#B4C6A6', '#FFC286'], label='Sample = 9')
    ax.set_ylabel('angle mean error', size=18)
    ax.set_xlabel('number of calibration parameters(N)', size=18)
    ax.set_title("Number of calibration samples = 9",size=18)
    ax.text(0 - 0.12, 3.31 + 0.08, 3.31, va='center', fontsize=14)
    ax.text(1 - 0.12, 3.09 + 0.08, 3.09, va='center', fontsize=14)
    ax.text(2 - 0.12, 2.99 + 0.08, 2.99, va='center', fontsize=14)
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x = np.linspace(0, 6, 11) + np.random.randn(11)
x = np.sort(x)
y = x ** 2 + 2 + np.random.randn(11)


def compute_gradient(m_current, b_current, learning_rate):
    N = len(x)  # 数据的长度
    m_gradient = 0.0
    b_gradient = 0.0
    for i in range(N):
        m_gradient += -(2 / N) * x[i] * (y[i] - (m_current * x[i] + b_current))
        b_gradient += -(2 / N) * (y[i] - (m_current * x[i] + b_current))

    new_m = m_current - (learning_rate * m_gradient)
    new_b = b_current - (learning_rate * b_gradient)
    return new_m, new_b


def optimizer():
    w = 0
    b = 0

    for i in range(1000):
        w, b = compute_gradient(w, b, 0.02)
        # if i % 50 == 0:
            # plt.plot(x, x * w + b, 'b-')
            # plt.pause(0.5)
    y_pre = x * w + b
    print(w, b)
    return y_pre


plt.plot(x, y, 'ro')
plt.plot(x, optimizer(), 'b-')
#optimizer()
plt.show()


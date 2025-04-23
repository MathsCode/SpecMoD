import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf


def relu(x):
    return np.maximum(0, x)


def gelu(x):
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))


def silu(x):
    return x / (1 + np.exp(-x))


x = np.linspace(-5, 5, 1000)


relu_y = relu(x)
gelu_y = gelu(x)
silu_y = silu(x)


plt.rcParams['figure.dpi'] = 300
plt.figure(figsize=(6, 3)) 
plt.plot(x, relu_y,label='ReLU',linewidth=2)
plt.plot(x, gelu_y, label='GeLU',linewidth=2)
plt.plot(x, silu_y, label='SiLU',linewidth=2)


plt.legend()
plt.xticks([])
plt.yticks([])
# plt.title('Activation Functions_ReLU')
# plt.xlabel('x')
# plt.xticks(rotation=45)
# plt.ylabel('y')


# plt.grid(True)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(False)

plt.savefig('activation_functions_silu.svg', bbox_inches='tight')
    
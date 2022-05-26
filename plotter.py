import matplotlib.pyplot as plt
import numpy as np

folder = 'results/mnist_lenet5'
experiment = 'adgd_0.02_1_'
file = 'ta.npy'

data = np.load(folder+'/'+experiment+file)
print(data)

plt.figure(1)
plt.plot(data)

plt.show()

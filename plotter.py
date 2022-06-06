import matplotlib.pyplot as plt
import numpy as np

folder = 'results/mnist_Lenet5_seed_'
experiment = 'adgd_0.02_1_'

file = 'ta.npy'

data = None
print(data)

for r in range(10):
    print(np.load(folder+str(r)+'/'+experiment+file))
    
    if data is not None:
        data = np.append(data, np.load(folder+str(r)+'/'+experiment+file))
    else:
        data = np.load(folder+str(r)+'/'+experiment+file)

data = np.reshape(data, (10,-1))
print()
print(data)
print(data.shape)

mean = np.mean(data,axis=0)
std = np.std(data,axis=0)

x = [i for i in range(len(mean))]
plt.figure()
plt.fill_between(x,mean-std, mean+std, color='r',alpha=0.325)
#plt.plot(x,mean, color='r', linewidth=2)
#plt.plot(x,mean-std, color='r', linewidth=1)
#plt.plot(x,mean+std, color='r', linewidth=1)
plt.plot(x,mean, color='r', linewidth=1)


plt.savefig("test.png")

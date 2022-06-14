import matplotlib.pyplot as plt
import numpy as np

opt = "sgdm"
data_set = "cifar10"
folder = 'results_'+opt+'/'+data_set+'/lenet5_seed_'
experiment = opt+'_0.02_1_'

file = 'ta.npy'
n_seed = 5

data = None


for r in range(n_seed):
    print(np.load(folder+str(r)+'/'+experiment+file))
    
    if data is not None:
        data = np.append(data, np.load(folder+str(r)+'/'+experiment+file))
    else:
        data = np.load(folder+str(r)+'/'+experiment+file)

data = np.reshape(data, (n_seed,-1))
print()
print(data)
print(data.shape)

plt.figure()
for i in range(n_seed):
    plt.plot(data[i])
mean = np.mean(data,axis=0)
std = np.std(data,axis=0)

x = [i for i in range(len(mean))]

plt.figure()
plt.fill_between(x,mean-std, mean+std, color='r',alpha=0.325)
plt.plot(x,mean, color='r', linewidth=1)
plt.plot(x,mean-std, color='r', linewidth=0.5)
plt.plot(x,mean+std, color='r', linewidth=0.5)
#plt.plot(x,mean, color='r', linewidth=1)


#plt.savefig("test.png")
#plt.savefig("test.pdf", format='pdf')

plt.show()
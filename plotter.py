import matplotlib.pyplot as plt
import numpy as np

opt = 'adgd'
dataset = 'emnist'

#folder = 'results_test/mnist/mnist_Lenet5_seed_'
#folder = 'results_' + opt + '/fmnist/lenet5_seed_'
#folder = 'results_' + opt + '/' + dataset + '/lenet5_seed_'


experiment = opt + '_0.1_1.0'

folder = 'results/' + dataset + '/'+ experiment + '/lenet5_seed_'

file = '_ta.npy'
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

mean = np.mean(data,axis=0)
std = np.std(data,axis=0)

x = [i for i in range(len(mean))]

plt.figure()
plt.fill_between(x,mean-std, mean+std, color='r',alpha=0.325)
plt.plot(x,mean, color='r', linewidth=1)
plt.plot(x,mean-std, color='r', linewidth=0.5)
plt.plot(x,mean+std, color='r', linewidth=0.5)
#plt.plot(x,mean, color='r', linewidth=1)


plt.savefig("test_" + dataset + "_" + experiment + ".png")
plt.savefig("test.pdf", format='pdf')


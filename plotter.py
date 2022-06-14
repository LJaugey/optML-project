import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

opt = 'adgd'
dataset = 'emnist'

#folder = 'results_test/mnist/mnist_Lenet5_seed_'
#folder = 'results_' + opt + '/fmnist/lenet5_seed_'
#folder = 'results_' + opt + '/' + dataset + '/lenet5_seed_'


datasets = ['mnist', 'fmnist', 'emnist', 'cifar10']
experiments = ['adam_0.01', 'adgd_0.1_1.0', 'adgd_0.02_1.0', 'adgd_0.02_2.0', 'sgdm_0.01_0.9']

# test accuracy
# step size
# train loss

parent_folder = 'results/'

n_seed = 5

for dataset in datasets:
    rootdir = parent_folder + dataset + '/'

    for experiment in experiments:
        for seed in range(n_seed):
            if data is not None:
                data = np.append(data, np.load(rootdir + experiment + '/' + str(r)+'/'+experiment+file))
            else:
                data = np.load(folder+str(r)+'/'+experiment+file)

            

    path_to_data = []

    for path in Path(rootdir).iterdir():
    if path.is_dir():
        path_to_data.append(path)

    





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


plt.savefig("test_" + dataset + "_" + experiment + ".png")
plt.savefig("test.pdf", format='pdf')

plt.show()

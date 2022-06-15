import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

save_folder = 'plots/'

# directory of data is organized in the following way :
# results / dataset / experiment / lenet5_seed_i / 

# data in directory :
# a.npy == train accuracy
# ta.npy == test accuracy
# gn.npy == gradient norms
# itr.npy == iterations train
# ite.npy == iterations test
# l.npy == train losses
# tl.npy == test losses
# lr.npy = learning rate

def getDataExperiment(experiment, dir):
    # we load the test accuracy, train loss and learning rates
    # we then process it to obtain the std and mean of these data
    # load the test accuracy
    ta = np.load(dir + experiment + '/lenet5_seed_0/ta.npy')
    l = np.load(dir + experiment + '/lenet5_seed_0/l.npy')
    lr = np.load(dir + experiment + '/lenet5_seed_0/lr.npy')

    print(ta.shape[0])
    print(l.shape[0])
    print(lr.shape[0])
    n_seed = 5
    tas = np.zeros((n_seed, ta.shape[0]))
    ls = np.zeros((n_seed, l.shape[0]))
    lrs = np.zeros((n_seed, lr.shape[0]))
    # store all data for this experiment in the data variable
    for r in range(n_seed):
        # load the test accuracy
        tas[r, :] = np.load(dir + experiment + '/lenet5_seed_' + str(r)+'/ta.npy')
        # load the train loss
        ls[r, :] = np.load(dir + experiment + '/lenet5_seed_' + str(r)+'/l.npy')
        # load the learning rate
        lrs[r, :] = np.load(dir + experiment + '/lenet5_seed_' + str(r)+'/lr.npy')

    # compute mean and std across the seed dimension
    mean = [np.mean(tas, axis=0), np.mean(ls, axis=0), np.mean(lrs, axis=0)]
    std = [np.std(tas, axis=0), np.std(ls, axis=0), np.std(lrs, axis=0)]

    return mean, std

# load all the data
datasets = ['mnist', 'fmnist', 'emnist', 'cifar10']
experiments = ['adgd_0.1_1.0', 'adgd_0.02_1.0', 'adgd_0.02_2.0', 'adam_0.01', 'sgdm_0.01_0.9']
mean_stds_dict = dict()

parent_folder = 'results/'
for dataset in datasets:
    rootdir = parent_folder + dataset + '/'
    mean_stds_dict[dataset] = dict()

    for ind, experiment in enumerate(experiments):
        mean, std = getDataExperiment(experiment, rootdir)
        mean_stds_dict[dataset][experiment] = [mean, std]

    
# for each dataset, compare the three experiments of adgd
epochs = np.linspace(1.5, 100, 200)

for ind, dataset in enumerate(datasets):
    colors = ['r', 'b', 'g']
    # plot test accuracies
    plt.figure()
    for i in range(3):
        
        mean = mean_stds_dict[dataset][experiments[i]][0][0]
        std = mean_stds_dict[dataset][experiments[i]][1][0]
        epochs = np.arange(1, mean.shape[0]+1) / mean.shape[0] * 100
        plt.fill_between(epochs,mean-std, mean+std, color=colors[i],alpha=0.325)
        plt.plot(epochs, mean, color=colors[i], linewidth=1)

    plt.xlabel('Epochs')
    plt.ylabel('Test accuracy')
    plt.title('Comparison adgd')

    plt.savefig(save_folder + dataset + '/' + 'ta_adgd.png')

    # plot losses
    plt.figure()
    for i in range(3):
        
        mean = mean_stds_dict[dataset][experiments[i]][0][1]
        std = mean_stds_dict[dataset][experiments[i]][1][1]
        epochs = np.arange(1, mean.shape[0]+1) / mean.shape[0] * 100
        plt.fill_between(epochs,mean-std, mean+std, color=colors[i],alpha=0.325)
        plt.plot(epochs, mean, color=colors[i], linewidth=1)
        # plt.plot(mean-std, color=colors[i], linewidth=0.5)
        # plt.plot(mean+std, color=colors[i], linewidth=0.5)

    plt.xlabel('Epochs')
    plt.ylabel('Training loss')
    plt.title('Comparison adgd')
    plt.savefig(save_folder + dataset + '/' + 'l_adgd.png')

    plt.figure()
    for i in range(3):
        
        mean = mean_stds_dict[dataset][experiments[i]][0][2]
        std = mean_stds_dict[dataset][experiments[i]][1][2]
        epochs = np.arange(1, mean.shape[0]+1) / mean.shape[0] * 100
        plt.fill_between(epochs,mean-std, mean+std, color=colors[i],alpha=0.325)
        plt.plot(epochs, mean, color=colors[i], linewidth=1)
        # plt.plot(mean-std, color=colors[i], linewidth=0.5)
        # plt.plot(mean+std, color=colors[i], linewidth=0.5)

    plt.xlabel('Epochs')
    plt.ylabel('Learning rates')
    plt.title('Comparison adgd')
    plt.yscale('log')
    plt.savefig(save_folder + dataset + '/' + 'lr_adgd.png')


# for each dataset, compare the adgd with adam and sgdm
epochs = np.linspace(1.5, 100, 200)
indices_exp = [0, 3, 4]

for ind, dataset in enumerate(datasets):
    colors = ['r', 'b', 'g']
    # plot test accuracies
    plt.figure()
    for index, i in enumerate(indices_exp):
        
        mean = mean_stds_dict[dataset][experiments[i]][0][0]
        std = mean_stds_dict[dataset][experiments[i]][1][0]
        epochs = np.arange(1, mean.shape[0]+1) / mean.shape[0] * 100
        plt.fill_between(epochs,mean-std, mean+std, color=colors[index],alpha=0.325)
        plt.plot(epochs, mean, color=colors[index], linewidth=1)

    plt.xlabel('Epochs')
    plt.ylabel('Test accuracy')
    plt.title('Comparison adgd')

    plt.savefig(save_folder + dataset + '/' + 'ta_comp.png')

    # plot losses
    plt.figure()
    for index, i in enumerate(indices_exp):
        
        mean = mean_stds_dict[dataset][experiments[i]][0][1]
        std = mean_stds_dict[dataset][experiments[i]][1][1]
        epochs = np.arange(1, mean.shape[0]+1) / mean.shape[0] * 100
        plt.fill_between(epochs,mean-std, mean+std, color=colors[index],alpha=0.325)
        plt.plot(epochs, mean, color=colors[index], linewidth=1)
        # plt.plot(mean-std, color=colors[i], linewidth=0.5)
        # plt.plot(mean+std, color=colors[i], linewidth=0.5)

    plt.xlabel('Epochs')
    plt.ylabel('Training loss')
    plt.title('Comparison adgd')
    plt.savefig(save_folder + dataset + '/' + 'l_comp.png')

    plt.figure()
    for index, i in enumerate(indices_exp):
        
        mean = mean_stds_dict[dataset][experiments[i]][0][2]
        std = mean_stds_dict[dataset][experiments[i]][1][2]
        epochs = np.arange(1, mean.shape[0]+1) / mean.shape[0] * 100
        plt.fill_between(epochs,mean-std, mean+std, color=colors[index],alpha=0.325)
        plt.plot(epochs, mean, color=colors[index], linewidth=1)
        # plt.plot(mean-std, color=colors[i], linewidth=0.5)
        # plt.plot(mean+std, color=colors[i], linewidth=0.5)

    plt.xlabel('Epochs')
    plt.ylabel('Learning rates')
    plt.title('Comparison adgd')
    plt.yscale('log')
    plt.savefig(save_folder + dataset + '/' + 'lr_comp.png')

# plt.savefig("test_" + dataset + "_" + experiment + ".png")
# plt.savefig("test.pdf", format='pdf')

# plt.show()

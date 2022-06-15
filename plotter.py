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

# Use LaTeX font
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern Roman"]})


# load all the data
datasets = ['mnist', 'fmnist', 'emnist', 'cifar10']
experiments = ['adgd_0.1_1.0', 'adgd_0.02_1.0', 'adgd_0.02_2.0', 'adam_0.01', 'sgdm_0.01_0.9']
leg_exp = [r'AdGD: $\gamma=0.1, \alpha=1.0$', r'AdGD: $\gamma=0.02 , \alpha=1.0$', r'AdGD: $\gamma=0.02 , \alpha=0.5$', r'Adam: $\lambda=0.01$', r'SGDm: $\lambda=0.01 , \beta=0.9$']
mean_stds_dict = dict()

parent_folder = 'results/'
for dataset in datasets:
    rootdir = parent_folder + dataset + '/'
    mean_stds_dict[dataset] = dict()

    for ind, experiment in enumerate(experiments):
        mean, std = getDataExperiment(experiment, rootdir)
        mean_stds_dict[dataset][experiment] = [mean, std]

    
# for each dataset, compare the three experiments of adgd

for ind, dataset in enumerate(datasets):
    colors = ['r', 'b', 'g']
    # plot test accuracies
    fig = plt.figure(figsize=(4,3))
    plt.grid(which='major')
    plt.grid(which='minor', linestyle=':', linewidth=0.5)
    for i in range(3):
        
        mean = mean_stds_dict[dataset][experiments[i]][0][0]
        std = mean_stds_dict[dataset][experiments[i]][1][0]
        epochs = np.arange(1, mean.shape[0]+1) / mean.shape[0] * 100
        plt.fill_between(epochs,mean-std, mean+std, color=colors[i],alpha=0.325, label=leg_exp[i])
        plt.plot(epochs, mean, color=colors[i], linewidth=1, label=leg_exp[i])

    plt.xlabel('Epochs')
    plt.ylabel('Test accuracy')
    plt.title('Comparison AdGD')
    handles, l = fig.get_axes()[0].get_legend_handles_labels()
    plt.legend([(handles[i],handles[i+1]) for i in range(0,len(handles),2)],[leg_exp[k] for k in range(3)])
    plt.subplots_adjust(bottom=0.14)
    plt.savefig(save_folder + dataset + '/' + 'ta_adgd.pdf')
    
    # plot losses
    fig = plt.figure(figsize=(4,3))
    plt.grid(which='major')
    plt.grid(which='minor', linestyle=':', linewidth=0.5)
    for i in range(3):
        
        mean = mean_stds_dict[dataset][experiments[i]][0][1]
        std = mean_stds_dict[dataset][experiments[i]][1][1]
        epochs = np.arange(1, mean.shape[0]+1) / mean.shape[0] * 100
        plt.fill_between(epochs,mean-std, mean+std, color=colors[i],alpha=0.325, label=leg_exp[i])
        plt.plot(epochs, mean, color=colors[i], linewidth=1, label=leg_exp[i])
        
        

    plt.xlabel('Epochs')
    plt.ylabel('Training loss')
    plt.title('Comparison AdGD')
    handles, l = fig.get_axes()[0].get_legend_handles_labels()
    plt.legend([(handles[i],handles[i+1]) for i in range(0,len(handles),2)],[leg_exp[k] for k in range(3)])
    
    if ind == 2: plt.subplots_adjust(bottom=0.14, left=0.15)
    else: plt.subplots_adjust(bottom=0.14)
    plt.savefig(save_folder + dataset + '/' + 'l_adgd.pdf')

    fig = plt.figure(figsize=(4,3))
    plt.grid(which='major')
    plt.grid(which='minor', linestyle=':', linewidth=0.5)
    for i in range(3):
        
        mean = mean_stds_dict[dataset][experiments[i]][0][2]
        std = mean_stds_dict[dataset][experiments[i]][1][2]
        epochs = np.arange(1, mean.shape[0]+1) / mean.shape[0] * 100
        plt.fill_between(epochs,mean-std, mean+std, color=colors[i],alpha=0.325, label=leg_exp[i])
        plt.plot(epochs, mean, color=colors[i], linewidth=1, label=leg_exp[i])
        
        

    plt.xlabel('Epochs')
    plt.ylabel('Learning rates')
    plt.title('Comparison AdGD')
    plt.yscale('log')
    handles, l = fig.get_axes()[0].get_legend_handles_labels()
    plt.legend([(handles[i],handles[i+1]) for i in range(0,len(handles),2)],[leg_exp[k] for k in range(3)])
    plt.subplots_adjust(bottom=0.14, left=0.15)
    plt.savefig(save_folder + dataset + '/' + 'lr_adgd.pdf')


# for each dataset, compare the adgd with adam and sgdm
epochs = np.linspace(1.5, 100, 200)
indices_exp = [0, 3, 4]

for ind, dataset in enumerate(datasets):
    colors = ['r', 'b', 'g']
    # plot test accuracies
    
    fig = plt.figure(figsize=(4,3))
    plt.grid(which='major')
    plt.grid(which='minor', linestyle=':', linewidth=0.5)
    
    for index, i in enumerate(indices_exp):
        
        mean = mean_stds_dict[dataset][experiments[i]][0][0]
        std = mean_stds_dict[dataset][experiments[i]][1][0]
        epochs = np.arange(1, mean.shape[0]+1) / mean.shape[0] * 100
        plt.fill_between(epochs,mean-std, mean+std, color=colors[index],alpha=0.325, label=leg_exp[i])
        plt.plot(epochs, mean, color=colors[index], linewidth=1, label=leg_exp[i])

    plt.xlabel('Epochs')
    plt.ylabel('Test accuracy')
    plt.title('Comparison')
    handles, l = fig.get_axes()[0].get_legend_handles_labels()
    plt.legend([(handles[i],handles[i+1]) for i in range(0,len(handles),2)],[leg_exp[k] for k in indices_exp])
    plt.subplots_adjust(bottom=0.14)

    plt.savefig(save_folder + dataset + '/' + 'ta_comp.pdf')
    
    # plot losses
    fig = plt.figure(figsize=(4,3))
    plt.grid(which='major')
    plt.grid(which='minor', linestyle=':', linewidth=0.5)
    
    for index, i in enumerate(indices_exp):
        
        mean = mean_stds_dict[dataset][experiments[i]][0][1]
        std = mean_stds_dict[dataset][experiments[i]][1][1]
        epochs = np.arange(1, mean.shape[0]+1) / mean.shape[0] * 100
        plt.fill_between(epochs,mean-std, mean+std, color=colors[index],alpha=0.325, label=leg_exp[i])
        plt.plot(epochs, mean, color=colors[index], linewidth=1, label=leg_exp[i])
        
        

    plt.xlabel('Epochs')
    plt.ylabel('Training loss')
    plt.title('Comparison')
    handles, l = fig.get_axes()[0].get_legend_handles_labels()
    plt.legend([(handles[i],handles[i+1]) for i in range(0,len(handles),2)],[leg_exp[k] for k in indices_exp])
    plt.subplots_adjust(bottom=0.14)
    plt.savefig(save_folder + dataset + '/' + 'l_comp.pdf')
    
    # plot learning rates
    fig = plt.figure(figsize=(4,3))
    plt.grid(which='major')
    plt.grid(which='minor', linestyle=':', linewidth=0.5)
    
    for index, i in enumerate(indices_exp):
        
        mean = mean_stds_dict[dataset][experiments[i]][0][2]
        std = mean_stds_dict[dataset][experiments[i]][1][2]
        epochs = np.arange(1, mean.shape[0]+1) / mean.shape[0] * 100
        plt.fill_between(epochs,mean-std, mean+std, color=colors[index],alpha=0.325, label=leg_exp[i])
        plt.plot(epochs, mean, color=colors[index], linewidth=1, label=leg_exp[i])
        
        

    plt.xlabel('Epochs')
    plt.ylabel('Learning rates')
    plt.title('Comparison')
    plt.yscale('log')
    handles, l = fig.get_axes()[0].get_legend_handles_labels()
    plt.legend([(handles[i],handles[i+1]) for i in range(0,len(handles),2)],[leg_exp[k] for k in indices_exp])
    plt.subplots_adjust(bottom=0.14, left=0.15)
    plt.savefig(save_folder + dataset + '/' + 'lr_comp.pdf')
    
plt.show()

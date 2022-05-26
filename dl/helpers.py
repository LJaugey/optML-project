import torch
import dlc_practical_prologue as getdata
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


#=======================DATASET=======================#
class data_set(Dataset):
    def __init__(self, x, y, number):

        self.x = x
        self.y = y
        self.number = number

    def __len__(self):
        return len(self.y)


    def __getitem__(self, id):

        sample={"image":self.x[id] , "labels":self.y[id], "number":self.number[id]}

        return sample


#=======================GET_DATA_LOADER=======================#
def Get_data_loaders(batch_size=16, augment=True):
    
    # Generate data
    train_x, train_y, train_num, test_x, test_y, test_num = getdata.generate_pair_sets(1000)

    # Normalization
    std_train, mean_train = torch.std_mean(train_x)
    std_test, mean_test = torch.std_mean(test_x)

    train_set = (train_x-mean_train)/std_train
    test_set = (test_x-mean_test)/std_test

    if augment:
        # augment data (N x 2 x 14 x 14) -> (2N x 2 x 14 x 14)
        dims = train_set.size()
        augmented_train_set = torch.empty((2*dims[0], dims[1], dims[2], dims[3]))
        augmented_train_y = torch.empty(2*dims[0])
        augmented_train_num = torch.empty((2*dims[0], dims[1]))

        # switch the images
        augmented_train_set[:dims[0], :, :, :] = train_set
        augmented_train_set[dims[0]:, 0, :, :] = train_set[:, 1, :, :]
        augmented_train_set[dims[0]:, 1, :, :] = train_set[:, 0, :, :]

        # switch the labels, except when the two images are the same
        augmented_train_y[:dims[0]] = train_y
        augmented_train_y[dims[0]:] = 1-train_y
        augmented_train_y[dims[0]:][train_num[:, 0]==train_num[:, 1]] = 1

        augmented_train_num[:dims[0], :] = train_num
        augmented_train_num[dims[0]:, 0] = train_num[:, 1]
        augmented_train_num[dims[0]:, 1] = train_num[:, 0]
        
        train_set = data_set(augmented_train_set, augmented_train_y, augmented_train_num)
    else:
        train_set = data_set(train_set, train_y, train_num)

    test_set = data_set(test_set, test_y, test_num)

    # Create data_loaders
    loader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    loader_test = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

    return loader_train, loader_test

#=======================Loss=======================#
def AuxLoss(criterion, alpha, output, target, output_class, output_class2, target_classes):
    # compute MSEloss of output/target
    bce_output = F.binary_cross_entropy(output, target)

    class_output = criterion(output_class, target_classes[:,0]) # penalize if prediction of 1 image is false
    class_output2 = criterion(output_class2, target_classes[:,1]) # penalize if prediction of 2 image is false
    
    return bce_output + alpha * (class_output + class_output2)


#=======================TEST=======================#
def Test(model, loader_test, flatten=False, aux=False):

    model.eval()
    true = 0
    false = 0
    for batch in loader_test:

        if flatten:
            train_x = batch["image"].flatten(start_dim=1, end_dim=-1)   #flatten in all dimensions (except over the batch)
        else:
            train_x = batch["image"]
        
        train_y = batch["labels"]

        main_output = 0

        if aux:
            _, _, main_output = model(train_x)
        else:
            main_output = model(train_x)
        
        prediction = main_output.squeeze()

        prediction[prediction>=0.5] = 1
        prediction[prediction<0.5] = 0

        true += torch.sum(prediction==train_y)
        false+= torch.sum(prediction!=train_y)


    total = true + false
    

    return float(true), float(false)

#=======================TRAIN=======================#
def Train(model, criterion, loader_train, optimizer, epoch, flatten=False, aux=False, alpha=0):

    model.train()

    epoch_count = 0
    for _ in range(epoch):
        for batch in loader_train:
  
            if flatten:
                train_x = batch["image"].flatten(start_dim=1, end_dim=-1)   #flatten in all dimensions (except over the batch)
            else:
                train_x = batch["image"]
            
            train_classes = batch["number"].type(torch.LongTensor)  # size n x 2

            train_y = batch["labels"].type(torch.FloatTensor) # output size n

            if aux:
                aux_output, aux_output2, main_output = model(train_x)
                prediction = main_output.squeeze()

                # Compute loss
                loss = AuxLoss(criterion, alpha, prediction, train_y, aux_output, aux_output2, train_classes)

            else:
                # Evaluate the network (forward pass)
                main_output = model(train_x)
                prediction = main_output.squeeze()

                # Compute loss
                loss = criterion(prediction, train_y)

            # Compute the gradient
            optimizer.zero_grad()
            model.zero_grad()
            loss.backward()

            # Update the parameters of the model with a gradient step
            optimizer.step()
            
        epoch_count+=1
        

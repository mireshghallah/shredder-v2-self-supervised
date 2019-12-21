# Email fmireshg@eng.ucsd.edu in case of any questions
import torch
import torch.nn as nn
import torch.nn.functional as F
from lenet import LeNet5
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
#from tqdm import tqdm, trange
import math
import csv
import numpy as np
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from lenet import LeNet5
from lenet_5 import LeNet5_5
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
#from tqdm import tqdm, trange
import math
import csv
import numpy as np
import itertools
import matplotlib.pyplot as plt
import time
import scipy.stats as st

BATCH_SIZE = 256
BATCH_TEST_SIZE = 1024
data_train = MNIST('./data/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]))
data_test = MNIST('./data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()]))
data_train_loader = DataLoader(data_train, batch_size = BATCH_SIZE , shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test,  batch_size = BATCH_TEST_SIZE, num_workers=8)

TRAIN_SIZE = len(data_train_loader.dataset)
TEST_SIZE = len(data_test_loader.dataset)
NUM_BATCHES = len(data_train_loader)
NUM_TEST_BATCHES = len(data_test_loader)

CLASSES = 10
TRAIN_EPOCHS = 10
SAMPLES = 2
TEST_SAMPLES = 10
SAMPLE = True
LR = 0.001
COEF = 0.1


noises = np.load("self-super-std20-nonsen-27ep-noise-2.npy")
weights = np.load("self-super-std20-nonsen-27ep-weight-2.npy")




data_test_loader_2 = DataLoader(data_test,  batch_size = 1, num_workers=0)



model_original = LeNet5_5()
model_original.load_state_dict(torch.load("LeNet-5-saved"))

criterion = nn.NLLLoss()

conv_layers = []
fc_layers=[]
for i, layer in enumerate(model_original.convnet):
    if isinstance(layer, nn.Conv2d):
        if ((i is not 0 )):
            conv_layers.append(i)
conv_layers.append(len(model_original.convnet))
for i, layer in enumerate(model_original.fc):
    if isinstance(layer, nn.Linear):
        fc_layers.append(i)
        
print (conv_layers, fc_layers)


conv_shapes=[]
for cnt2, (data, target) in enumerate(data_test_loader):
    for cnt,i in enumerate(conv_layers):
        #newmodel = torch.nn.Sequential(*(list(model_test.features)[0:i]))
        newmodel_original =  torch.nn.Sequential(*(list(model_original.convnet)[0:i]))
        
        output_original = newmodel_original(data)
        conv_shapes.append(output_original.shape[1:])

    if (cnt2==0):
        break
    




size = noises.shape[0]
print(size, "SIZE")

class NoisyActivation(nn.Module):
    def __init__(self, activation_size):
        super(NoisyActivation, self).__init__()
        index_memory = np.random.randint(1,size)
        

        weight = weights[index_memory]
        params = st.norm.fit( weight)

        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        dist = st.norm(loc, scale)
        weight_norm = dist.rvs(weight.shape[0]*weight.shape[1])
        weight_flatten = weight.flatten()

        sorted_index_norm= np.argsort(weight_norm)
        sorted_index  = np.argsort(weight_flatten)
        #print(sorted_index.shape,"sorted")

        #print(sorted_index)

        weight_flatten[sorted_index] = weight_norm[sorted_index_norm]
        expanded = weight_flatten.reshape(weight.shape[0],weight.shape[1],1)

        weight = torch.Tensor(expanded)
        #print(weight.shape)
        self.noise = torch.Tensor((noises[index_memory]))
        self.weight= torch.Tensor((weight))
      

    def forward(self, input):
        
        
        #print(self.noise.shape, "noise shape")

        return input*self.weight + self.noise
    
    
    


class LeNet_syn(nn.Module):

    def __init__(self, model_features, model_classifier, conv_layers, conv_shapes, index ):
        super(LeNet_syn, self).__init__()
        
        self.model_pt1 =  torch.nn.Sequential(*(list(model_features)[0:conv_layers[index]]))
        self.intermed = NoisyActivation(conv_shapes[index])
        self.model_pt2 =  torch.nn.Sequential(*(list(model_features)[conv_layers[index]:]))
        self.model_pt3 = model_classifier
        for child in itertools.chain(self.model_pt1,self.model_pt2,self.model_pt3):
            for param in child.parameters():
                param.requires_grad = False
            if isinstance(child, nn.modules.batchnorm._BatchNorm):
                child.eval()
                child.affine = False
                child.track_running_stats = False


    def forward(self, img):
        x = self.model_pt1(img)
        x = self.intermed (x)
        x = self.model_pt2(x)
        x = x.view(x.size(0), -1)
        x = self.model_pt3(x)

        return x


model_original = LeNet5_5()
model_original.load_state_dict(torch.load("LeNet-5-saved"))
model_syn = LeNet_syn(model_original.convnet, model_original.fc ,conv_layers, conv_shapes, 2)





##########################
class NoisyActivation_private(nn.Module):
    def __init__(self, activation_size, noise, weight):
        super(NoisyActivation_private, self).__init__()
        self.noise = noise
        self.weight = weight
        
    def forward(self, input):

        return input*self.weight + self.noise



class LeNet_syn_private(nn.Module):

    def __init__(self, model_features, model_classifier, conv_layers, conv_shapes, index , noise, weight):
        super(LeNet_syn_private, self).__init__()
        
        self.model_pt1 =  torch.nn.Sequential(*(list(model_features)[0:conv_layers[index]]))
        self.intermed = NoisyActivation_private(conv_shapes[index], noise, weight)
        self.model_pt2 =  torch.nn.Sequential(*(list(model_features)[conv_layers[index]:]))
        self.model_pt3 = model_classifier
        for child in itertools.chain(self.model_pt1,self.model_pt2,self.model_pt3):
            for param in child.parameters():
                param.requires_grad = False
            if isinstance(child, nn.modules.batchnorm._BatchNorm):
                child.eval()
                child.affine = False
                child.track_running_stats = False


    def forward(self, img):
        x = self.model_pt1(img)
        x = self.intermed (x)
        x = self.model_pt2(x)
        x = x.view(x.size(0), -1)
        x = self.model_pt3(x)

        return x
    




#########################

def validate_private(model_syn_private):
    start = time.perf_counter()
    model_syn_private.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(data_test_loader):
        output = model_syn_private(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
    end=time.perf_counter()

    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))

    return float(total_correct) / len(data_test)


def validate_above_5(model_syn, index):
    model_syn.eval()
    total_correct = 0
    avg_loss = 0.0
    SNR =[]
    for i, (images, labels) in enumerate(data_test_loader):
        output = model_syn(images)
        labels = (labels > 5).long()
        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

        output_original = model_original.convnet[0:conv_layers[index]](data)
        output_syn = model_syn.model_pt1(data)
        output_syn= model_syn.intermed(output_syn)

       
        SNR.append((((output_original**2).mean())/((output_original-output_syn).var())).item())
    print("Avg SNR" , sum(SNR)/len(SNR))
    avg_loss /= len(data_test)
    print('Test above 5 Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))
    return  float(total_correct) / len(data_test), sum(SNR)/len(SNR)



total_correct = 0
avg_loss = 0.0
for i, (images, labels) in enumerate(data_test_loader_2):
        break



imgs =np.reshape(np.squeeze(images.detach().numpy()), (1,-1))
lbls = np.reshape(np.squeeze(labels.detach().numpy()), (1,-1))
activation_noise = np.expand_dims( model_syn.intermed.noise.detach().numpy(),axis=0)
activation_original = np.expand_dims( model_syn.intermed.noise.detach().numpy(),axis=0)




for i, (images, labels) in enumerate(data_test_loader_2):
    model_original = LeNet5_5()
    model_original.load_state_dict(torch.load("LeNet-5-saved"))
    
    model_original_private = LeNet5()
    model_original_private.load_state_dict(torch.load("LeNet-saved"))

    criterion = nn.NLLLoss()

    acc =0

    while(acc < 0.1):
        model_syn = LeNet_syn(model_original.convnet, model_original.fc ,conv_layers, conv_shapes, 2)
        acc, SNR = validate_above_5 (model_syn,2)
        model_syn_private = LeNet_syn_private(model_original_private.convnet, model_original_private.fc ,conv_layers, conv_shapes, 2, model_syn.intermed.noise, model_syn.intermed.weight)
        acc_adv = validate_private(model_syn_private)
        print(acc_adv, "accuracy adversary")
        print (acc, "accuracy")

        if (acc > 0.1):
            print("test and save")
   
            output_original = model_original.convnet[0:conv_layers[2]](images)
            output_syn = model_syn.model_pt1(images)
            output_syn= model_syn.intermed(output_syn)


            with open('mutual_info-self-memory-uniform-std20-ep27-new.csv','a') as fd:
                writer = csv.writer(fd)
                writer.writerow([SNR, acc, acc_adv])

           
            activation_noise=np.concatenate((activation_noise, output_syn.detach().numpy()))
            activation_original=np.concatenate((activation_original,output_original.detach().numpy()),axis=0)
            
            np.save("noisy-activation-mutual_info-self-memory-uniform-std20-ep27-new", activation_noise)
            np.save("original-activation-mutual_info-self-memory-uniform-std20-ep27-new", activation_original)
            
            imgs=np.concatenate((imgs,np.reshape(np.squeeze(images.detach().numpy()), (1,-1)) ))
            np.save("original-image-mutual_info-self-memory-uniform-std20-ep27-new", imgs)
            lbls=np.concatenate((lbls,np.reshape(np.squeeze(labels.detach().numpy()), (1,-1)) ))
            np.save("original-labels-mutual_info-self-memory-uniform-std20-ep27-new", lbls)


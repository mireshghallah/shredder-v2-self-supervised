#Email fmireshg@eng.ucsd.edu fatamehsadat Mireshghallah in case of questions
import torch
import torch.nn as nn
import torch.nn.functional as F
#from lenet import LeNet5
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
from lenet import LeNet5
import scipy.stats as st


# In[2]:


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")





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

data_train_loader_2 = list(DataLoader(data_train, batch_size = BATCH_SIZE , shuffle=True, num_workers=0))
data_test_loader_2 = list(DataLoader(data_test,  batch_size = BATCH_TEST_SIZE, num_workers=0))

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


# # Load 5 model

# In[4]:


model_original = LeNet5_5()
model_original.load_state_dict(torch.load("LeNet-5-saved"))

criterion = nn.NLLLoss()


# # Test 5 Model

# In[5]:



model_original.eval()
total_correct = 0
avg_loss = 0.0
for i, (images, labels) in enumerate(data_test_loader):
    output = model_original(images)
    labels = (labels > 5).long()
    avg_loss += criterion(output, labels).sum()
    pred = output.detach().max(1)[1]
    total_correct += pred.eq(labels.view_as(pred)).sum()
end=time.perf_counter()

avg_loss /= len(data_test)
print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))




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


# # Extract conv

# In[8]:


conv_shapes=[]
for cnt2, (data, target) in enumerate(data_test_loader):
    for cnt,i in enumerate(conv_layers):
        #newmodel = torch.nn.Sequential(*(list(model_test.features)[0:i]))
        newmodel_original =  torch.nn.Sequential(*(list(model_original.convnet)[0:i]))
        
        output_original = newmodel_original(data)
        conv_shapes.append(output_original.shape[1:])
        print (output_original.shape[1:])
    if (cnt2==0):
        break
    


# # build new model

# In[9]:


class NoisyActivation(nn.Module):
    def __init__(self, activation_size):
        super(NoisyActivation, self).__init__()
        
        m =torch.distributions.laplace.Laplace(loc = 0.0, scale = 20.0, validate_args=None)
        self.noise = nn.Parameter(m.rsample(activation_size))
        #self.noise = nn.Parameter(torch.Tensor(activation_size).normal_(loc=0.0, scale=12.0))
        self.weight = nn.Parameter(torch.Tensor(activation_size))
        nn.init.xavier_normal_(self.weight)
        
    def forward(self, input):

        return input*self.weight + self.noise


# In[10]:



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
    


    



model_syn = LeNet_syn(model_original.convnet, model_original.fc ,conv_layers, conv_shapes, 2)


# In[12]:





total_correct = 0
avg_loss = 0.0
for i, (images, labels) in enumerate(data_test_loader):
    output = model_syn(images)
    labels = (labels > 5).long()
    avg_loss += criterion(output, labels).sum()
    pred = output.detach().max(1)[1]
    total_correct += pred.eq(labels.view_as(pred)).sum()
end=time.perf_counter()

avg_loss /= len(data_test)
print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))




optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_syn.parameters()), lr=0.001, weight_decay=-0.01)
weights_noise =np.expand_dims( model_syn.intermed.noise.detach().numpy(),axis=0)
weights_weight =np.expand_dims( model_syn.intermed.weight.detach().numpy(),axis=0)
criterion = nn.NLLLoss()





size = len(data_train_loader_2)


size2 = len(data_test_loader)




def train_above_5(net, optimizer, epoch):
    net.train()
    avg_loss = 0
    total_correct=0
    for i, (images, labels) in enumerate(data_train_loader):
        if (i == size-1):
            continue
            
            
        net.zero_grad()
        b_size = images.shape[0]
        #print(b_size)
        index_rand = np.random.randint(1,size-1)
        (images_2,labels_2) = data_train_loader_2[index_rand]
        #print("img size", images_2.shape[0], index_rand)
        
        ####################################  self sup
        labels = (labels > 5).long()
        labels_2 = (labels_2 > 5).long()
        
        output_main = net.model_pt1(images)
        output_main = net.intermed(output_main)
        
        output_rand = net.model_pt1(images_2)
        output_rand = net.intermed(output_rand)
        
        distance = abs(output_rand-output_main)
        distance = torch.sum(distance, dim = 1)
        distance = distance.squeeze()
        pos = (labels == labels_2)
        neg = (labels != labels_2)
        pos = torch.sum(distance * pos.type(torch.FloatTensor))
        neg = torch.sum(distance * neg.type(torch.FloatTensor))
        #print(output_rand.shape)
        
        output = net(images)

        
        all = pos+ neg
        ######################################## calculate distribution distance 
        #params = st.norm.fit(model_syn.intermed.weight.detach().numpy())
        #arg = params[:-2]
        #loc = params[-2]
        #scale = params[-1]
        #dist = st.norm(loc, scale)
        #y, x = np.histogram(model_syn.intermed.weight.detach().numpy(), bins=200, density=True)
        #x = (x + np.roll(x, -1))[:-1] / 2.0
        
        #pdf = st.laplace.pdf(x, loc=loc, scale=scale, *arg)
        #sse = np.sum(np.power(y - pdf, 2.0))
        #print(sse, "SSE")
        
        ########################################
        
        #print(pos, "pos")
        #print(neg,"neg")
        #print ((pos - neg)/all , "norm")
        loss = criterion(output, labels) + 1.01*(pos - neg)/all  #+ 0.01*sse
        
        #if ()
        avg_loss += loss 
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        loss.backward()
        optimizer.step()
        #print("here")
    avg_loss /= len(data_train)
    print('Train Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_train)))
    return float(float(total_correct) / len(data_train))


# In[107]:


def validate(model_syn, index):
    model_syn.eval()
    total_correct = 0
    avg_loss = 0.0
    SNR =[]
    for i, (images, labels) in enumerate(data_test_loader):
        output = model_syn(images)
        avg_loss += criterion_5(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

        output_original = model_original.convnet[0:conv_layers[index]](data)
        output_syn = model_syn.model_pt1(data)
        output_syn= model_syn.intermed(output_syn)

        
        SNR.append((((output_original**2).mean())/((output_original-output_syn).var())).item())

    print("Avg SNR", sum(SNR)/len(SNR))
    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))
    return  float(total_correct) / len(data_test), sum(SNR)/len(SNR)


# In[108]:


def validate_above_5(model_syn, index):
    model_syn.eval()
    total_correct = 0
    avg_loss = 0.0
    SNR =[]
    pos_s = 0
    neg_s =0
    norm_s =0
    for i, (images, labels) in enumerate(data_test_loader):

        
        index_rand = np.random.randint(1,size2-1)
        if (i == size2-1 and index_rand != size2-1):
            b_size = labels.shape[0]
            (images_2,labels_2) = data_test_loader_2[index_rand]
            (images_2,labels_2) = ((images_2[0:b_size],labels_2[0:b_size]) )
        
        else:
            (images_2,labels_2) = data_test_loader_2[index_rand]
        #print("img size", images_2.shape[0], index_rand)
        labels = (labels > 5).long()
        labels_2 = (labels_2 > 5).long()
        
        output_main = model_syn.model_pt1(images)
        output_main = model_syn.intermed(output_main)
        
        output_rand = model_syn.model_pt1(images_2)
        output_rand = model_syn.intermed(output_rand)
        
        distance = abs(output_rand-output_main)
        distance = torch.sum(distance, dim = 1)
        distance = distance.squeeze()
        pos = (labels == labels_2)
        neg = (labels != labels_2)
        pos = torch.sum(distance * pos.type(torch.FloatTensor))
        neg = torch.sum(distance * neg.type(torch.FloatTensor))
        #print(output_rand.shape)
        pos_s += pos
        neg_s += neg
      
        #######################################
        output = model_syn(images)

        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

        output_original = model_original.convnet[0:conv_layers[index]](data)
        output_syn = model_syn.model_pt1(data)
        output_syn= model_syn.intermed(output_syn)


        SNR.append((((output_original**2).mean())/((output_original-output_syn).var())).item())
    print("AVG SNR", sum(SNR)/len(SNR) )
    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))
    return  float(total_correct) / len(data_test), sum(SNR)/len(SNR)


# In[109]:


#print (list(synthesized.sequential_list[0].parameters()))
deltas = []

for run in range (1000):
    wd =0
    lr = 0.001
    model_original = LeNet5_5()
    model_original.load_state_dict(torch.load("LeNet-5-saved"))
    model_syn = LeNet_syn(model_original.convnet, model_original.fc ,conv_layers, conv_shapes, 2)
    model_syn.eval()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_syn.parameters()), lr=lr, weight_decay=wd)
    acc_shadow = 100.0

    for epoch in range(40):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_syn.parameters()), lr=lr, weight_decay=wd)
        print (epoch)
        acc, SNR =validate_above_5 (model_syn,2)
        
        if (acc > 0.95 and epoch>27):
            #break
            print ("saved")
            with open('self-super-std20-nonsen-27ep-2-new.csv','a') as fd:
                    writer = csv.writer(fd)
                    writer.writerow([SNR, acc,epoch])
            print (weights_noise.shape)
            print (np.expand_dims(model_syn.intermed.noise.detach().numpy(),axis=0).shape)
            
            weights_noise=np.concatenate((weights_noise,np.expand_dims(model_syn.intermed.noise.detach().numpy(),axis=0)),axis=0)
            weights_weight=np.concatenate((weights_weight, np.expand_dims(model_syn.intermed.weight.detach().numpy(),axis=0)),axis=0)

     
            np.save("self-super-std20-nonsen-27ep-noise-2-new", weights_noise)
            np.save("self-super-std20-nonsen-27ep-weight-2-new", weights_weight)
            
            break
        train_above_5(model_syn, optimizer, epoch)
        







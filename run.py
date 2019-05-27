import torch
import argparse
import math
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
from torch.nn.init import *
#import pandas as pd
from torch.autograd import Variable
#import nibabel as nib
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import random
import pydicom as dicom
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle

import pylab
parser=argparse.ArgumentParser()
parser.add_argument('--txt_path',default='/home/liubin/data3.txt',help='')
parser.add_argument('--train_batchsize',default=1,help='')
parser.add_argument('--epochs',default=20,help='')
parser.add_argument('--channel_size',default=166,help='')
parser.add_argument('--imagesize',default=256,help='')
parser.add_argument('--lr',default=0.0001,help='learning rate')
opt=parser.parse_args()

class MyDataset(Dataset):
    def __init__(self,txt_path=opt.txt_path, transform=None, target_transform=None):
        f=open(txt_path,'r')
        imgs=[]
        for line in f:
            line=line.rstrip()
            word=line.split()
            image_name='/home/liubin/ADNI_data/'+word[0]+'.dcm'
            imgs.append((image_name, float(word[1]), int(word[2]), word[0])) # image path, time, delta, iamge id
            #print(word)
            #print(imgs)
        self.imgs=imgs
        self.transform=transform
        self.target_transform=target_transform
        '''
        image_path,survive_time,censor=self.imgs[0]
        image_data=nib.load(image_path).get_data()
        plt.figure()
        plt.subplot(1,2,1)
        
        plt.imshow(image_data[:,:,85],cmap='gray')
        print('###################3')
        plt.show()
        '''
    def __getitem__(self, idx):
        image_path, survive_time, censor, image_id=self.imgs[idx]
        ds=dicom.read_file(image_path)
        #print('The image path is:{}'.format(image_path))
        #plt.imshow(image_data[:,:,85],cmap='gray')
        #print("features:{}".format(ds.dir("pat")))
        #print("PatientName:{}".format(ds.PatientName))
        image_data=ds.pixel_array
        #print('The size(0) is:{}'.format(image_data.shape[0]))
#        pylab.imshow(image_data,cmap=pylab.cm.bone)
#        pylab.show()
        if self.transform is not None:
           image_data=self.transform(image_data)
        #print('The type of image_data is:{}'.format(type(image_data)))
        #print('The type of survive_time is:{}'.format(type (survive_time)))
        image_data=image_data[:,:,np.newaxis]
        #print('The image_data shape is:{}'.format(image_data.shape))

        image_data=function(image_data,256,256)
        image_data=image_data.reshape(1,256,256)
        #print('The afterward_deal image_data shape is:{}'.format(image_data.shape))
        # image_data=image_data.astype(np.int16)
        # print('7777777')
        image_data = image_data.astype('float32')
        #image_data=torch.FloatTensor(image_data)
        image_data = torch.from_numpy(image_data)
        # print('888888888')
        #print(image_data.size())

        survive_time=torch.FloatTensor([survive_time])
        censor=torch.FloatTensor([censor])
        #print(censor.size())
        return image_data, survive_time, censor, image_id
    def __len__(self):
        return len(self.imgs)

My_trainset=MyDataset('/home/liubin/train.txt',None,None)
My_valset=MyDataset('/home/liubin/valid.txt',None,None)
My_testset=MyDataset('/home/liubin/test.txt',None,None)
fullSet=MyDataset('/home/liubin/full.txt',None,None)

batchSize = 50
trainloader=DataLoader(My_trainset,batch_size=batchSize,shuffle=True,num_workers=10)
valloader=DataLoader(My_valset,batch_size=4,shuffle=True,num_workers=1)
testloader = DataLoader(My_testset,batch_size=27,shuffle=True,num_workers=1)
#print('##################44444444444444444#####33{}'.format(My_trainset))
fullSetloader=DataLoader(fullSet, batch_size=1, shuffle=True, num_workers=10)



      
def S(x):
    x = np.abs(x)
    if 0 <= x < 1:
        return 1 - 2 * x * x + x * x * x
    if 1 <= x < 2:
        return 4 - 8 * x + 5 * x * x - x * x * x
    else:
        return 0
def function(img,m,n):#The bicubic interpolation upsampling
    height,width,channels=img.shape
    #print('The height is:{}'.format(height))
    #print('The width is:{}'.format(width))
    #print('The channels is:{}'.format(channels))
    emptyImage=np.zeros((m,n,channels),np.uint16)
    sh=m/height
    sw=n/width
    for i in range(m):
        for j in range(n):
            x = i/sh
            y = j/sw
            p=(i+0.0)/sh-x
            q=(j+0.0)/sw-y
            x=int(x)-2
            y=int(y)-2
            A = np.array([
                [S(1 + p), S(p), S(1 - p), S(2 - p)]
            ])
            if x>=m-3:
                m-1
            if y>=n-3:
                n-1
            if x>=1 and x<=(m-3) and y>=1 and y<=(n-3):
                B = np.array([
                    [img[x-1, y-1], img[x-1, y],
                     img[x-1, y+1],
                     img[x-1, y+1]],
                    [img[x, y-1], img[x, y],
                     img[x, y+1], img[x, y+2]],
                    [img[x+1, y-1], img[x+1, y],
                     img[x+1, y+1], img[x+1, y+2]],
                    [img[x+2, y-1], img[x+2, y],
                     img[x+2, y+1], img[x+2, y+1]],
 
                    ])
                C = np.array([
                    [S(1 + q)],
                    [S(q)],
                    [S(1 - q)],
                    [S(2 - q)]
                ])
                blue = np.dot(np.dot(A, B[:, :, 0]), C)[0, 0]
                #green = np.dot(np.dot(A, B[:, :, 1]), C)[0, 0]
                #red = np.dot(np.dot(A, B[:, :, 2]), C)[0, 0]
 
                # ajust the value to be in [0,255]
                def adjust(value):
                    if value > 255:
                        value = 255
                    elif value < 0:
                        value = 0
                    return value
 
                blue = adjust(blue)
                #green = adjust(green)
                #red = adjust(red)
                emptyImage[i, j] = np.array([blue], dtype=np.uint16)
 
 
    return emptyImage


      

class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        identity = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
 
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out += identity
        out = self.relu(out)
 
        return out
 
 
class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        identity = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out += identity
        out = self.relu(out)
 
        return out
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
 
 
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
 
class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        identity = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out += identity
        out = self.relu(out)
 
        return out

class ResNet(nn.Module):
 
    def __init__(self, block, layers, num_classes=1, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)
 
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
 
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
 
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
 
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
 
        x = self.avgpool(x)
        x_fea = x.view(x.size(0), -1)
        x = self.fc(x_fea)
 
        return x, x_fea
def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


class Concordance_Loss(torch.nn.Module):
    
    def __init__(self):
        super(Concordance_Loss, self).__init__()
        
    def forward(self, event_indicator, event_time, estimate):#indicator:the censor event_time:the survival time estimate:the resnet output
        #event_indicator, event_time, estimate are all 1-dimension torch tensors 
        event_time=torch.exp(event_time)
        event_time=torch.squeeze(event_time)
        #print('the dim of event_time size:', event_time.size())
        #print('event_time is:', event_time)
        n_samples = event_time.size(0) # if event_time is not 1-dim, we have to modify the index here!!!
        timeInOrder, order = event_time.sort()
        #print('The event_time.sort() is:{}'.format(event_time.sort()))
        
        tied_time = torch.tensor(0)
        
        comparable = {}
        for i in range(n_samples - 1):
            inext = i + 1
            j = inext
            time_i = event_time[order[i]]
            while j < n_samples and event_time[order[j]] == time_i:
                j += 1
            
            if event_indicator[order[i]]: # uncensored
                mask = torch.zeros(n_samples).byte() # define boolen tensor
                mask[inext:] = 1
                
                if j - i > 1:
                    # event times are tied, need to check for coinciding events
                    event_at_same_time = event_indicator[order[inext:j]]
                    '''
                    print('The inext  is:{}'.format(inext))
                    print('The j is :{}'.format(j))
                    print('The event_at_same_time is:{}'.format(event_at_same_time))
                    '''
                    event_at_same_time=event_at_same_time.int()
                    '''
                    event_at_same_time=event_at_same_time.cpu().detach().numpy()
                    event_at_same_time=event_at_same_time.astype(np.int)
                    event_at_same_time=np.logical_not(event_at_same_time)
                    event_at_same_time=event_at_same_time.astype(np.int)
                    event_at_same_time=torch.from_numpy(event_at_same_time).cuda()
                    event_at_same_time=torch.squeeze(event_at_same_time)
                    event_at_same_time=torch.squeeze(event_at_same_time)
                    event_at_same_time=Variable(event_at_same_time)
                    '''
                    event_at_same_time=torch.squeeze(event_at_same_time)
                    event_at_same_time=torch.squeeze(event_at_same_time)
                    mask[inext:j] = event_at_same_time^1  #logical not in torch, notInd = indtensor^1
                    tied_time =tied_time+event_at_same_time.sum()
                comparable[i] = mask   
            elif j - i > 1:
                # events at same time are comparable if at least one of them is positive
                mask = torch.zeros(n_samples).byte()
                mask_=event_indicator[order[inext:j]]
                mask_=torch.squeeze(mask_)
                mask_=torch.squeeze(mask_)
                '''
                print('The inext  is:{}'.format(inext))
                print('The j is :{}'.format(j))
                print('The event_indicator is:{}'.format(event_indicator))
                print('The mask[inext:j] is:{}'.format(mask[inext:j]))
                print('The order[inext:j] is:{}'.format(order[inext:j]))
                print('The make_ is:{}'.format(mask_))
                '''
                mask[inext:j] = mask_
                comparable[i] = mask
        concordant = discordant = tied_risk = torch.Tensor([0])
        con=tie=torch.Tensor([0])
        '''
        print('1The disconcordant is:{}'.format(discordant))
        print('1The concordant is:{}'.format(concordant))
        print('1The tied_risk is:{}'.format(tied_risk))
        '''
        concordant=Variable(concordant,requires_grad=True)
        discordant=Variable(discordant,requires_grad=True)
        tied_risk=Variable(tied_risk,requires_grad=True)
        
        for ind, mask in comparable.items():
            #print('The mask is:{}'.format(mask))
            est_i = estimate[order[ind]]
            event_i = event_indicator[order[ind]]

            est = estimate[order[mask]]
            
            if event_i: # delta_i=1
                # an event should have a higher score
                #print('1The est is:{}'.format(est))
                #print('1The est_i is:{}'.format(est_i))
                con=(est<est_i).sum()
                
            else:
                # a non-event should have a lower score
                #print('2The est is :{}'.format(est))
                #print('2The est_i is:{}'.format(est_i))
                con=(est>est_i).sum()
            concordant =concordant+con

            est_minus_esti = torch.abs(est - est_i)
            #print('The est_minus_esti is:{}'.format(est_minus_esti))
            tie=(est_minus_esti <= 1e-8).sum()

            tied_risk =tied_risk+tie
            #print('est_size is:{}'.format(est))
            number=torch.numel(est)
            discordant =discordant+number - con - tie
        loss = (discordant + 0.5 * tied_risk) / (discordant + concordant + tied_risk + 1e-7)
        #loss=torch.div(torch.add(discordant,torch.mul(torch.Tensor([0.5]),tied_risk)),torch.add(torch.add(discordant,concordant),tied_risk))
        #print('The loss in Concordance is:{}'.format(loss))
        #print('The disconcordant is:{}'.format(discordant))
        #print('The concordant is:{}'.format(concordant))
        #print('The tied_risk is:{}'.format(tied_risk))
        #print('The &**********************is:{}'.format((discordant+0.5*tied_risk)/(discordant+concordant+tied_risk)))
        
        
        return 1-loss


os.environ["CUDA_VISIBLE_DEVICES"] = '2'
use_cuda = torch.cuda.is_available()
#print(use_cuda)

model=resnet18(pretrained=False)
#num_ftrs=resnet18.fc.in_features
#resnet18.fc=torch.nn.Linear(num_ftrs,1)
criterion=Concordance_Loss()
optimizer=torch.optim.SGD(model.parameters(),lr=opt.lr,momentum=0.9)
#exp_lr_scheduler=lr_scheduler.StepLR(optimizer,step_size=7,gamma=0.1)
num_epoches=50
model.cuda()




image=torch.FloatTensor(opt.train_batchsize,opt.imagesize,opt.imagesize,opt.channel_size)
indicator=torch.FloatTensor(0)
outputs=torch.FloatTensor(0)
time=torch.FloatTensor(0)
image=Variable(image,requires_grad=True)
indicator=Variable(indicator,requires_grad=True)
outputs=Variable(outputs,requires_grad=True)
val_loss=torch.FloatTensor(0)
loss=torch.FloatTensor(0)
loss=Variable(loss,requires_grad=True)
image=Variable(image,requires_grad=True)
time=Variable(time,requires_grad=True)
val_loss=Variable(val_loss,requires_grad=True)
image=image.cuda()
indicator=indicator.cuda()
outputs=outputs.cuda()
loss=loss.cuda()
val_loss=val_loss.cuda()
time=time.cuda()


validLoss = []
trainLoss = []
for epoch in range(num_epoches):
    #print('get into the loop')
    model.train()
    for i,data in enumerate(trainloader):
    
    #     print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    #     # print(model.state_dict())
    #     print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    #
        image, time, indicator, ims_id = data
    #     print(image.size())
        image=Variable(image.cuda())
        indicator=Variable(indicator.cuda())
        outputs=outputs.cuda()
        loss=loss.cuda()
    #
        time=time.cuda()
    #     #print('The censor is:{}'.format(indicator))
        image=Variable(image,requires_grad=True)
        time=Variable(time,requires_grad=True)
        indicator=Variable(indicator,requires_grad=True)
    #     #print('The size of input is:{}'.format(image.size()))
        image=image.cuda()
    #     #print('The input is:{}'.format(image))
        optimizer.zero_grad()
        outputs, fea_train=model(image)
    #
    #     print('The outputs is:{}'.format(outputs))
    #     print('The survival_time is:{}'.format(time))
    #     print('The censor is:{}'.format(indicator))
    #
        loss=criterion(indicator,time,outputs)
        # f=open(r'PydicomTrain.txt','w')
        print('Epoch:{} i:{} The Loss is:{}'.format(epoch,i,loss))
        # f.close()
        loss.sum().backward()
    #
    #
    #
    #     #During the calculation, only the grad of the leaf node is kept, in order to save memory
    #     '''
    #     print('The grad of indicator is :{}'.format(indicator_.grad))
    #     print('The grad of time is:{}'.format(time_.grad))
    #     print('The grad of outputs is:{}'.format(outputs_.grad))
    #     print('The grad of loss is:{}'.format(loss_.grad))
    #     print('The grad of inputs is:{}'.format(image_.grad))
    #     '''
        optimizer.step()

        trainLoss.append(loss.unsqueeze(0))

    # val_loss=0
    model.eval()
    with torch.no_grad():
        for i,data in enumerate(valloader):
            image, time, indicator, img_id = data
            image=Variable(image.cuda())
            indicator=Variable(indicator.cuda())
            time = Variable(time.cuda())
            # outputs=outputs.cuda() # comment by Bin
            # loss=loss.cuda()
            # val_loss=val_loss.cuda()

            outputs, x=model(image)
            # loss=criterion(indicator,time,outputs)  # comment by Bin
            val_loss = criterion(indicator, time, outputs)
            print('Epoch:{} i:{} The valloss is:{}'.format(epoch, i, val_loss))
            # val_loss=val_loss+loss # comment by Bin
            val_loss += val_loss
            loss_average = val_loss/(i+1)  ############## add by bin
            validLoss.append(loss_average.unsqueeze(0)) ########## add by bin
            print('The average val_loss is:{}', format(loss_average))

# with open('./validLoss.data', 'wb') as pathFile:
#     pickle.dump(validLoss, pathFile)
with open('./validLoss-%s.txt' % batchSize, 'w') as file:
    np.savetxt(file, np.array(validLoss))

with open('./trainLoss-%s.txt' % batchSize, 'w') as file1:
    np.savetxt(file1, np.array(trainLoss))
        
#testing
model.eval()
with torch.no_grad():
    for i, data in enumerate(testloader):
        image, time, indicator, image_id = data
        image = Variable(image.cuda())
        indicator = Variable(indicator.cuda())
        time = Variable(time.cuda())

        predicts, x = model(image)
        test_loss = criterion(indicator, time, predicts)
        print('the testing loss is:', test_loss)

with open('./testingLoss-%s.txt' % batchSize, 'w') as file2:
    np.savetxt(file2, np.array(test_loss))

PATH = './model.tar.gz'
torch.save(model.state_dict(), PATH)


import csv
#feature extracting
model.eval()
with open('./featureMatrix-%s.csv' % batchSize, 'a') as file3:

    with torch.no_grad():
        for i, data in enumerate(fullSetloader):
            image, time, indicator, sample_id = data
            image = Variable(image.cuda())
            indicator = Variable(indicator.cuda())
            time = Variable(time.cuda())

            predicts, feature = model(image)
            x = feature.cpu().numpy()
            line = [sample_id, x.flatten()]

            wr = csv.writer(file3, dialect='excel')
            wr.writerow(line)









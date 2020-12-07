#!/usr/bin/env python
# coding: utf-8

# In[37]:


import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import numpy as np


# In[38]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

batch_size = 256
n_iters = 60000

compose = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
     ])# (x-mean) / std

train_dataset = dsets.MNIST(root='./datasets',
                            train=True, 
                            transform=compose,
                            download=True)

test_dataset = dsets.MNIST(root='./datasets',
                           train=False, 
                           transform=compose)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)
print("train_dataset len ",len(train_dataset))


# In[39]:



def generator_model():
    model = nn.Sequential(
        nn.Linear(100,256),
        nn.LeakyReLU(),
        
        nn.Linear(256,256),
        nn.LeakyReLU(),
        
        nn.Linear(256,28*28*1),
#         nn.Linear(256, 784),
        nn.Tanh()
    )
    return model

def discriminator_model():
    model = nn.Sequential(
        nn.Linear(28*28*1, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 1),
        nn.Sigmoid() 
    )
    return model


# In[40]:


criterion = nn.BCELoss()

def discriminator_loss(real_output, fake_output):
    
    
    real_label = Variable(torch.ones_like(real_output))
    fake_label = Variable(torch.zeros_like(fake_output))
    
#     real_label = Variable(torch.ones(batch_size))
#     fake_label = Variable(torch.zeros(batch_size))
    d_loss_real = criterion(real_output, real_label)
    d_loss_fake = criterion(fake_output, fake_label)
    return d_loss_real + d_loss_fake

def generator_loss(fake_output):
#     real_label = Variable(torch.ones(num_img))
    real_label = Variable(torch.ones_like(fake_output))

    g_loss = criterion(fake_output, real_label)
    return g_loss

gpus = [0]
# gpus = [0, 1, 2]
cuda_gpu = torch.cuda.is_available()
print (cuda_gpu)
    
generator_model = generator_model()
discriminator_model = discriminator_model()

if cuda_gpu == True:
    generator_model = torch.nn.DataParallel(generator_model, device_ids=gpus).cuda()
    discriminator_model = torch.nn.DataParallel(discriminator_model, device_ids=gpus).cuda()


d_optimizer = torch.optim.Adam(discriminator_model.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(generator_model.parameters(), lr=0.0003)


# In[41]:


save_path = "./img2"
if not os.path.exists(save_path):
    os.mkdir(save_path)
 


# In[42]:


noise_dim = 100
def train_step(img):
    num_img = img.size(0) #maybe not batchsize

    img = img.view(num_img, -1)
    
    if(cuda_gpu):
        real_img = Variable(img.cuda())
    else:
        real_img = Variable(img)

    if(cuda_gpu):
        input_noise = Variable(torch.randn(num_img, noise_dim).cuda())
    else:
        input_noise = Variable(torch.randn(num_img, noise_dim))
        
    real_out = discriminator_model(real_img)
    
    fake_img = generator_model(input_noise).detach()
    fake_out = discriminator_model(fake_img)
    d_loss = discriminator_loss(real_out, fake_out)

    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    fake_img = generator_model(input_noise)  
    facke_output = discriminator_model(fake_img)  
    g_loss = generator_loss(facke_output)  

    g_optimizer.zero_grad() 
    g_loss.backward()  
    g_optimizer.step() 
    
#     G = generator_model
#     D = discriminator_model
#     z_dimension = noise_dim
#     num_img = img.size(0)

#     img = img.view(num_img, -1)
#     real_img = Variable(img)
#     real_label = Variable(torch.ones(num_img))
#     fake_label = Variable(torch.zeros(num_img))

#     real_out = D(real_img)
#     d_loss_real = criterion(real_out, real_label)
#     real_scores = real_out

#     z = Variable(torch.randn(num_img, z_dimension))
#     fake_img = G(z).detach()
#     fake_out = D(fake_img)
#     d_loss_fake = criterion(fake_out, fake_label)
#     fake_scores = fake_out
#     d_loss = d_loss_real + d_loss_fake
#     d_optimizer.zero_grad()
#     d_loss.backward()
#     d_optimizer.step()

#     z = Variable(torch.randn(num_img, z_dimension))
#     fake_img = G(z)  
#     output = D(fake_img)  
#     g_loss = criterion(output, real_label)  
#     # bp and optimize
#     g_optimizer.zero_grad() 
#     g_loss.backward()  
#     g_optimizer.step() 

    return fake_img


# In[43]:


if(cuda_gpu):
    seed = Variable(torch.randn(16, noise_dim).cuda())
else:
    seed = Variable(torch.randn(16, noise_dim))


# In[44]:


import matplotlib.pyplot as plt
from torchvision.utils import save_image

def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)  # Clamp函数可以将随机变化的数值限制在一个给定的区间[min, max]内：
    out = out.view(-1, 1, 28, 28)  # view()函数作用是将一个多行的Tensor,拼接成一行
    return out


def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input)
    predictions = predictions.cpu().data
    
    fig = plt.figure(figsize=(4, 4))
    print(predictions.shape)
    
    out = 0.5 * (predictions .data+ 1)
    out = out.clamp(0, 1)
    out =out.view(-1,28, 28, 1)
    print(out.shape)
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        
        plt.imshow((out[i, :, :, 0]), cmap='gray')
        plt.axis('off')

    plt.savefig(save_path+'/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


# In[45]:


def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch,label in dataset:
            
            fake_img = train_step(image_batch)
            fake_images = to_img(fake_img.cpu().data)
            save_image(fake_images, save_path + '/fake_images-{}.png'.format(epoch + 1))
 
            print('.', end='')
        print()

        generate_and_save_images(generator_model,
                             epoch + 1,
                             seed)


    generate_and_save_images(generator_model,
                           epochs,
                           seed)


# In[46]:


for image_batch,label in train_loader:
       print(image_batch.data.size())
       print(label.data.size())
       
       image_batch = Variable(image_batch)
       label = Variable(label)
       
       print(image_batch.data.size())
       print(label.data.size())
       
       break



train(train_loader, num_epochs)


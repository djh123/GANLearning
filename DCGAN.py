import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
imageSize = 64
batch_size = 64
n_iters = 60000

compose = transforms.Compose([
    transforms.Resize(imageSize),
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

ngf = 64
noise_dim = 100
nz = noise_dim
ndf = ngf
nc = 1

def generator_model():
    model = nn.Sequential(
        nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(ngf * 8),
        nn.ReLU(True),
        
        nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 4),
        nn.ReLU(True),
        
        nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 2),
        nn.ReLU(True),
        
        nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf),
        nn.ReLU(True),
        
        nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
        nn.Tanh()
        
    )
    return model

def discriminator_model():
    model = nn.Sequential(
        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
 
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True),
        
        nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        nn.Sigmoid()
    )
    return model


criterion = nn.BCELoss()
def discriminator_loss(real_output, fake_output):   
    real_label = Variable(torch.ones_like(real_output))
    fake_label = Variable(torch.zeros_like(fake_output))
    d_loss_real = criterion(real_output, real_label)
    d_loss_fake = criterion(fake_output, fake_label)
    return d_loss_real + d_loss_fake

def generator_loss(fake_output):
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
save_path = "./DCGAN_out_img"
if not os.path.exists(save_path):
    os.mkdir(save_path)
def train_step(img):
    num_img = img.size(0) #maybe not batchsize

    img = img.view(num_img, 1, imageSize, imageSize)

    if(cuda_gpu):
        real_img = Variable(img.cuda())
    else:
        real_img = Variable(img)

    if(cuda_gpu):
        input_noise = Variable(torch.randn(num_img, noise_dim,1,1).cuda())
    else:
        input_noise = Variable(torch.randn(num_img, noise_dim,1,1))
        
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

    return fake_img

if(cuda_gpu):
    seed = Variable(torch.randn(16, noise_dim, 1, 1).cuda())
else:
    seed = Variable(torch.randn(16, noise_dim,1 ,1))

import matplotlib.pyplot as plt
from torchvision.utils import save_image

def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1) 
    out = out.view(-1, 1, imageSize, imageSize)
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
    out =out.view(-1,imageSize, imageSize, 1)
    print(out.shape)
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        
        plt.imshow((out[i, :, :, 0]), cmap='gray')
        plt.axis('off')

    plt.savefig(save_path+'/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

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
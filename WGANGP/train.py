'''
Training of WGAN(with gradient penalty) network with models imported from model.py
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Discriminator, Generator, weights_init
from utils import gradient_penalty

# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
IMG_CHANNELS = 3
Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_D = 64
FEATURES_G = 64
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(IMG_CHANNELS)], [0.5 for _ in range(IMG_CHANNELS)]),
])

# dataset = datasets.MNIST(root="D:/PyProjects/Datasets/mnist", train=True, transform=transforms, download=False) # IMG_CHANNELS=1
dataset = datasets.ImageFolder(root="D:/PyProjects/Datasets/celeb_dataset", transform=transforms) # IMG_CHANNELS=3
dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

gen = Generator(Z_DIM, IMG_CHANNELS, FEATURES_G).to(device)
critic = Discriminator(IMG_CHANNELS, FEATURES_D).to(device)
weights_init(gen)
weights_init(critic)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)

writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")

step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)

        # Train Discriminator
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
            fake = gen(noise)
            
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)

            # use gradient penalty to get smooth Discriminator
            gp = gradient_penalty(critic, real, fake, device=device)
            # original WGAN loss + gradient penalty
            loss_critic = (-(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP*gp)
            
            critic.zero_grad()
            loss_critic.backward(retain_graph=True) # keep the computing graph to reuse fake
            opt_critic.step()

        # Train Generator--min -E[critic(gen(z))]
        output = critic(fake).reshape(-1)
        loss_gen = -torch.mean(output)

        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # print to tensorboard
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                        Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}") # loss here is meaningful

            with torch.no_grad():
                fake = gen(fixed_noise)

                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real_celeb_wgangp", img_grid_real, global_step=step)
                writer_fake.add_image("Fake_celeb_wgangp", img_grid_fake, global_step=step)
            
            step += 1

writer_fake.close()
writer_real.close()

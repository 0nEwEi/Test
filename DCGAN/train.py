'''
Training of DCGAN network with models imported from model.py
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

# Hyperparameters etc. (essential for DCGAN)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
IMG_CHANNELS = 3
Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_D = 64
FEATURES_G = 64

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
disc = Discriminator(IMG_CHANNELS, FEATURES_D).to(device)
weights_init(gen)
weights_init(disc)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

criterion = nn.BCELoss()

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)

writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")

step = 0

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
        fake = gen(noise)

        # Train Discriminator--max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real).reshape(-1) # N x 1 x 1 x 1 to 1 x N
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        # see detach() and clone() in https://blog.csdn.net/qq_37692302/article/details/107459525
        '''
        RuntimeError: Trying to backward through the graph a second time 
        (or directly access saved tensors after they have already been freed). 
        Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). 
        Specify retain_graph=True if you need to backward through the graph a second time or 
        if you need to access saved tensors after calling backward.
        '''
        disc_fake = disc(fake.detach()).reshape(-1) # use detach (if not, see RuntimeError(line 89) above)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        loss_disc = (loss_disc_real + loss_disc_fake) / 2

        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # Train Generator--min log(1 - D(G(z))) <-> max D(G(z))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # print to tensorboard
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                        Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}") # loss here is meaningless

            with torch.no_grad():
                fake = gen(fixed_noise)

                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real_celeb_dcganssss", img_grid_real, global_step=step)
                writer_fake.add_image("Fake_celeb_dcganssss", img_grid_fake, global_step=step)
            
            step += 1

writer_fake.close()
writer_real.close()

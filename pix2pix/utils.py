import os
import torch
from torchvision.utils import save_image

import config

def save_some_examples(gen, data_loader, epoch):
    x, y = next(iter(data_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)

    gen.eval()

    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization
        save_image(y_fake, config.EXAMPLES_DIR + f"/output_{epoch}.png")
        save_image(x * 0.5 + 0.5, config.EXAMPLES_DIR + f"/input_{epoch}.png")
    
    gen.train()

    print("===--Examples saved--===")

def save_checkpoint(model, optimizer, filename):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, os.path.join(config.CHECKPOINTS_DIR, filename))
    print(f"===--Checkpoints saved: {filename})--===")

def load_checkpoint(filename, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(os.path.join(config.CHECKPOINTS_DIR, filename), map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

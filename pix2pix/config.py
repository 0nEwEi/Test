import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "D:/PyProjects/Datasets/archive/maps/maps/train"
VAL_DIR = "D:/PyProjects/Datasets/archive/maps/maps/val"
CHECKPOINTS_DIR = "./checkpoints"
EXAMPLES_DIR = "./examples"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 4 # you just cann't use 8, why?
IMAGE_SIZE = 256
IMG_CHANNELS = 3
L1_LAMBDA = 100
NUM_EPOCHS = 500
LOAD_MODEL = False
SAVE_MODEL = True

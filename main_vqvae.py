import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modelling_dvae import (
    DVAE,
    DVAEConfig,
    EncoderConfig,
    DecoderConfig,
    KLSchedConfig,
    TempSchedConfig,
)
from models.modelling_vqvae import VQVAE
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, FGVCAircraft
from trainer.training_engine_vqvae import train

global_config = DVAEConfig(
    image_size=32,
    hidden_size=256,
    encoder=EncoderConfig(
        input_channels=3, output_channels=512, num_layers=4, num_resnet_blocks=4
    ),
    decoder=DecoderConfig(
        num_layers=4,
        num_resnet_blocks=4,
        input_channels=512,
        output_channels=3,
    ),
    num_embeddings=512,
    embedding_dim=512,
    temperature=0.9,
    learning_rate=1e-3,
    learning_rate_scheduler_min=1e-5,
    kl_div_scheduler=KLSchedConfig(
        start=0.0,
        end=0.3,
        warmup=0.1,
        cooldown=0.2,
    ),
    temperature_scheduler=TempSchedConfig(
        start=0.9, end=0.00625, warmup=0.0, cooldown=0.2
    ),
    batch_size=16,
    max_epochs=50,
)

vqvae_model = VQVAE(config=global_config).to("cuda:0")
loss_fn = nn.MSELoss()
epochs = 60
optimizer = torch.optim.Adam(vqvae_model.parameters(), lr=1e-3)

traindata = FGVCAircraft(
    root="data", split="train", download=True, transform=transforms.Compose([transforms.Resize(size=(128, 128)), transforms.ToTensor()])
)
trainloader = DataLoader(traindata, batch_size=16, shuffle=True, num_workers=8, persistent_workers=True, prefetch_factor=2, pin_memory=True)

train(
    epochs=epochs,
    trainloader=trainloader,
    criterion=loss_fn,
    optimizer=optimizer,
    model=vqvae_model,
)

import pytorch_lightning as pl
import torch

from models.modelling_dvae import (
    DVAE,
    DVAEConfig,
    EncoderConfig,
    DecoderConfig,
    KLSchedConfig,
    TempSchedConfig,
)
from PIL import Image

from torchvision.datasets import FGVCAircraft
from torchvision.transforms import ToTensor, Normalize, Compose, Resize
from torch.utils.data import DataLoader
from trainer.training_engine_dvae import DVAETrainer
from trainer.scheduler import LinearScheduler

global_config = DVAEConfig(
    image_size=224,
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
    learning_rate=5e-4,
    learning_rate_scheduler_min=1e-2,
    kl_div_scheduler=KLSchedConfig(
        start=0.0,
        end=1.0,
        warmup=0.1,
        cooldown=0.2,
    ),
    temperature_scheduler=TempSchedConfig(
        start=0.9, end=0.00625, warmup=0.0, cooldown=0.2
    ),
    batch_size=512,
    max_epochs=50,
)

temp_schd = LinearScheduler(
    start=global_config.temperature_scheduler.start,
    end=global_config.temperature_scheduler.end,
    warmup=global_config.temperature_scheduler.warmup,
    cooldown=global_config.temperature_scheduler.cooldown,
    steps=global_config.max_epochs,
)

kl_schd = LinearScheduler(
    start=global_config.kl_div_scheduler.start,
    end=global_config.kl_div_scheduler.end,
    warmup=global_config.kl_div_scheduler.warmup,
    cooldown=global_config.kl_div_scheduler.cooldown,
    steps=global_config.max_epochs,
)

print(f"The training setting is: {global_config}")

dvae = DVAE(config=global_config)

cifar10_test = FGVCAircraft(
    root="data",
    split="test",
    download=True,
    transform=Compose(
        [
            Resize(size=(128, 128)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
)
cifar10_testloader = DataLoader(cifar10_test, batch_size=1)

dvae = DVAETrainer.load_from_checkpoint(
    "checkpoints/dvae-epoch=03-val_loss=0.52.ckpt",
    dvae_model=dvae,
    dvae_config=global_config,
    temp_scheduler=temp_schd,
    kl_div_weight_scheduler=kl_schd,
)
dvae.eval()

print(dvae.device)

import numpy as np

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

for i, (img, _) in enumerate(cifar10_testloader):

    # infer the image, save the image as original and reconstructed concatenated
    with torch.no_grad():
        img = img.to(dvae.device).float()
        decoded = dvae.test_step(img)

    # save the image
    img = img.squeeze().permute(1, 2, 0)
    img = img.detach().cpu().numpy()
    denormalized = img * std + mean
    img = Image.fromarray((denormalized * 255).astype("uint8"))
    img.save(f"images/original_{i}.png")

    # save the reconstructed image
    decoded = decoded.squeeze().permute(1, 2, 0)
    decoded = decoded.detach().cpu().numpy()
    dec_denorm = decoded * std + mean
    decoded = Image.fromarray((dec_denorm * 255).astype("uint8"))
    decoded.save(f"images/reconstructed_{i}.png")

    if i == 5:
        break

import pytorch_lightning as pl

from models.modelling_dvae import (
    DVAE,
    DVAEConfig,
    EncoderConfig,
    DecoderConfig,
    KLSchedConfig,
    TempSchedConfig,
)
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from trainer.training_engine import DVAETrainer
from trainer.scheduler import LinearScheduler

global_config = DVAEConfig(
    image_size=224,
    hidden_size=64,
    encoder=EncoderConfig(
        input_channels=3, output_channels=512, num_layers=2, num_resnet_blocks=2
    ),
    decoder=DecoderConfig(
        num_layers=2,
        num_resnet_blocks=2,
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
        end=1e-4,
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

dvae_trainer = DVAETrainer(
    dvae_model=dvae,
    dvae_config=global_config,
    temp_scheduler=temp_schd,
    kl_div_weight_scheduler=kl_schd,
)

cifar_train = CIFAR10(root="data", train=True, transform=ToTensor(), download=True)
cifar_test = CIFAR10(root="data", train=False, transform=ToTensor(), download=True)
cifar_loader = DataLoader(
    cifar_train, batch_size=global_config.batch_size, shuffle=True
)
cifar_test_loader = DataLoader(
    cifar_test, batch_size=global_config.batch_size, shuffle=False
)

wandb_logger = WandbLogger(
    project="beit_scratch",
    name="Discrete Variational Autoencoder Tokenizer",
    version=f"DryRun",
)

trainer = pl.Trainer(
    max_epochs=global_config.max_epochs,
    precision=16,
    callbacks=[
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="dvae-{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ],
    logger=wandb_logger,
)

if __name__ == "__main__":
    trainer.fit(
        model=dvae_trainer,
        train_dataloaders=cifar_loader,
        val_dataloaders=cifar_test_loader,
    )

import pytorch_lightning as pl

from models.modelling_dvae import (
    DVAE,
    DVAEConfig,
    EncoderConfig,
    DecoderConfig,
    KLSchedConfig,
    TempSchedConfig,
)
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers import WandbLogger
from torchvision.datasets import FGVCAircraft
from torchvision.transforms import ToTensor, Normalize, Compose, Resize
from torch.utils.data import DataLoader
from trainer.training_engine_dvae import DVAETrainer
from trainer.scheduler import LinearScheduler

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

train = FGVCAircraft(
    root="data",
    split="train",
    transform=Compose(
        [
            Resize(size=(128, 128)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    download=True,
)
test = FGVCAircraft(
    root="data",
    split="test",
    transform=Compose(
        [
            Resize(size=(128, 128)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    download=True,
)
trainloader = DataLoader(
    train, batch_size=global_config.batch_size, shuffle=True, num_workers=8
)
testloader = DataLoader(
    test, batch_size=global_config.batch_size, shuffle=False, num_workers=8
)

wandb_logger = WandbLogger(
    project="beit_scratch",
)

trainer = pl.Trainer(
    max_epochs=global_config.max_epochs,
    precision="16-mixed",
    callbacks=[
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="dvae-{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(monitor="val_loss", patience=5, mode="min"),
    ],
    logger=wandb_logger,
)

if __name__ == "__main__":
    trainer.fit(
        model=dvae_trainer,
        train_dataloaders=trainloader,
        val_dataloaders=testloader,
    )

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
from PIL import Image
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

# vqvae_model = VQVAE(config=global_config)
# vqvae_model.load_state_dict(torch.load("ARTIFACTS/best_model_vqvae_-0.4388.pth"))
# vqvae_model.to("cuda:0")
# vqvae_model.eval()

vqvae_model = torch.load("ARTIFACTS/VQVAE/best_model_vqvae_epoch_12.pth", map_location=torch.device("cuda:0"), weights_only=False)
vqvae_model.to("cuda:0")
vqvae_model.eval()

testdata = FGVCAircraft(
    root="data",
    split="test",
    download=True,
    transform=transforms.Compose(
        [transforms.Resize(size=(128, 128)), transforms.ToTensor()]
    ),
)

testloader = DataLoader(
    testdata,
    batch_size=1,
    shuffle=False,
    num_workers=8,
    persistent_workers=True,
    prefetch_factor=2,
    pin_memory=True,
)

vqvae_model.eval()

for i, data in enumerate(testloader):
    inputs, _ = data
    inputs = inputs.to("cuda:0").float()

    with torch.no_grad():
        recon_data, commitment_loss, encoding_indices = vqvae_model(inputs)

    recon_data = recon_data.squeeze(0).permute(1,2,0).cpu().numpy()
    inputs = inputs.squeeze(0).permute(1,2,0).cpu().numpy()
    
    print(recon_data.shape)

    recon = Image.fromarray((recon_data * 255).astype("uint8"))
    orig = Image.fromarray((inputs * 255).astype("uint8"))

    recon.save(f"recon_{i}.png")
    orig.save(f"orig_{i}.png")

    if i == 0:
        break
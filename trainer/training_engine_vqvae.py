import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.amp import autocast, GradScaler

def train(epochs, trainloader, criterion, optimizer, model):

    best_loss = float("inf")
    patience = 4
    patience_counter = 0
    model_params = {}

    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
            inputs, _ = data
            inputs = inputs.to("cuda:0").float()
            optimizer.zero_grad()

            with autocast(device_type="cuda"):
                recon_data, commitment_loss, encoding_indices = model(inputs)
                recon_loss = criterion(recon_data, inputs)
                loss = recon_loss + commitment_loss
            

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        epoch_loss = running_loss / len(trainloader)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0

            if best_loss not in model_params:
                model_params[best_loss] = model.state_dict()

            torch.save(
            model, f"ARTIFACTS/VQVAE/best_model_vqvae_epoch_{epoch}.pth"
        )

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"There is no improvement since last {patience_counter} epochs.")
                break

        print(f"Epoch {epoch+1} Loss: {epoch_loss}")
    try:
        torch.save(
            model_params[best_loss], f"ARTIFACTS/best_model_vqvae_{best_loss:.4f}.pth"
        )
        print("Finished Training")

    except KeyboardInterrupt:
        torch.save(
            model_params[best_loss], f"ARTIFACTS/best_model_vqvae_{best_loss:.4f}.pth"
        )
        print("Training interrupted. Model saved.")
        raise KeyboardInterrupt

    except Exception as e:
        sorted_dict = sorted(model_params.items())
        sorted_keys = [k for k, _ in sorted_dict]
        best_model = model_params[sorted_keys[0]]
        torch.save(best_model, f"ARTIFACTS/best_model_vqvae.pth")
        print(f"An error occurred: {e}")

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import UNet2DConditionModel, DDPMScheduler
from torch.optim import AdamW
from tqdm import tqdm

# Configuration
config = {
    "image_size": 32,
    "batch_size": 64,
    "epochs": 100,
    "lr": 1e-4,
    "dataset_path": "./dataset",  # Replace with your dataset path
    "model_save_path": "./stable_diffusion_model.pth",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# Data preparation
def prepare_data():
    transform = transforms.Compose([
        transforms.Resize(config["image_size"]),
        transforms.CenterCrop(config["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # Normalize images to [-1, 1]
    ])
    dataset = datasets.ImageFolder(config["dataset_path"], transform=transform)
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

# Initialize model, scheduler, and optimizer
def initialize_model():
    model = UNet2DConditionModel(
        sample_size=config["image_size"],  # Image size
        in_channels=3,  # RGB images
        out_channels=3,  # RGB images
        layers_per_block=2,
        block_out_channels=(64, 128, 256),  # Scaled-down model for 32x32
        attention_resolutions=(8,),  # Attention mechanism
    )
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    return model, noise_scheduler

# Training loop
def train(model, noise_scheduler, dataloader):
    optimizer = AdamW(model.parameters(), lr=config["lr"])
    model.to(config["device"])
    model.train()

    for epoch in range(config["epochs"]):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for images, _ in progress_bar:
            images = images.to(config["device"])
            noise = torch.randn_like(images).to(config["device"])
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (images.shape[0],)).to(config["device"])
            
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
            pred_noise = model(noisy_images, timesteps).sample

            loss = torch.nn.functional.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=loss.item())
        torch.save(model.state_dict(), config["model_save_path"])
    print("Training complete. Model saved.")

if __name__ == "__main__":
    dataloader = prepare_data()
    model, noise_scheduler = initialize_model()
    train(model, noise_scheduler, dataloader)

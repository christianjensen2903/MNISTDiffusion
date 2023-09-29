from pydantic import BaseModel


class ArgsModel(BaseModel):
    batch_size: int = 256
    timesteps: int = 400
    n_feat = 128
    epochs: int = 15
    lr: float = 1e-4
    n_samples: int = 36  # Samples per class after every epoch trained
    log_freq: int = 10
    image_size: int = 32
    drop_prob = 0.1
    betas = (1e-4, 0.02)
    w = 0.5
    n_classes: int = 10
    log_wandb: bool = False
    save_model = False
    save_dir = "./data/diffusion_outputs10/"

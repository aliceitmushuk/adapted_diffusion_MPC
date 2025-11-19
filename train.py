"""
Training script for Diffusion Factor Model
"""

import torch
import numpy as np
from torch.utils.data import TensorDataset
import os
import gc
import argparse
import time

from diffusion_factor_model.diffusion_factor_model import (
    ConditionalTransformer,
    SequentialGaussianDiffusion,
    Trainer,
)
import config.config as config

def train_model(
    data_path,
    seed=None,
    num_samples=None,
    gpu_id=0,
    epochs=None,
    save_timesteps=None,
    sample_window_start=None,
    sample_window_length=None,
):
    """
    Train the diffusion model using a specific data file
    
    Args:
        data_path: Path to the data file to use for training
        seed: Random seed for reproducibility
        num_samples: Number of training samples to use (None = use all)
        gpu_id: GPU ID to use
        epochs: Number of epochs to train (None = use config.EPOCHS)
        save_timesteps: List of specific timesteps to save during sampling for early stopping evaluation
                       (None = use config.SAVE_TIMESTEPS, which defaults to None meaning save only final result)
        sample_window_start: Optional start index (inclusive) for sequential sampling
        sample_window_length: Optional number of sequential entries to generate
    """
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Use config default if save_timesteps not specified
    if save_timesteps is None:
        save_timesteps = config.SAVE_TIMESTEPS
    
    # Set seed and get timestamp for experiment ID
    seed = config.set_seed(seed)
    timestamp = int(time.time())
    
    # Get filename from path for experiment ID
    filename = os.path.basename(data_path)
    data_id = os.path.splitext(filename)[0]
    
    # Create experiment ID
    exp_id = f"{config.EXP_PREFIX}_{data_id}_ts{timestamp}_seed{seed}"
    
    # Load data to determine shape and dimensions
    data_np = np.load(data_path)
    data_shape = data_np.shape
    print(f"Loaded data with shape: {data_shape}, dtype: {data_np.dtype}")
    
    # Limit number of samples if specified
    if num_samples is not None and num_samples < data_shape[0]:
        data_np = data_np[:num_samples]
        print(f"Using {num_samples} samples from the data")
    
    # Ensure data has shape [samples, sequence_length]
    if data_np.ndim == 1:
        data_np = data_np.reshape(1, -1)
        print("Input data reshaped to 2D with batch dimension 1")
    elif data_np.ndim > 2:
        data_np = data_np.reshape(data_np.shape[0], -1)
        print(f"Flattened high-dimensional data to shape: {data_np.shape}")

    data = torch.from_numpy(data_np).float()
    samples, seq_len = data.shape
    print(f"Using sequence data with length {seq_len} and {samples} samples")

    # Create directories for this experiment
    model_dir = os.path.join(config.MODELS_DIR, exp_id)
    sample_dir = os.path.join(config.SAMPLES_DIR, exp_id)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create dataset
    data_mean = data.mean(dim=0, keepdim=True)
    data_std = data.std(dim=0, keepdim=True)
    data_std = torch.where(data_std == 0, torch.ones_like(data_std), data_std)
    normalized_data = (data - data_mean) / data_std
    dataset = TensorDataset(normalized_data)

    # Use epochs from argument or config
    if epochs is None:
        epochs = config.EPOCHS

    # Initialize conditional transformer for sequential diffusion
    model = ConditionalTransformer(
        seq_len=seq_len,
        dim=config.TRANSFORMER_DIM,
        depth=config.TRANSFORMER_LAYERS,
        heads=config.TRANSFORMER_HEADS,
        ff_mult=config.TRANSFORMER_FF_MULT,
        dropout=config.TRANSFORMER_DROPOUT,
    )

    print("Model initialized")

    # Initialize sequential diffusion process
    diffusion = SequentialGaussianDiffusion(
        model,
        seq_len=seq_len,
        timesteps=config.TIMESTEPS,
        objective=config.OBJECTIVE,
        beta_schedule=config.BETA_SCHEDULE,
        auto_normalize=config.AUTO_NORMALIZE
    )
    
    print("Diffusion process initialized")

    # Determine sampling window
    window_start = sample_window_start
    if window_start is None:
        window_start = config.SAMPLE_WINDOW_START
    window_start = max(0, int(window_start))

    window_length = sample_window_length
    if window_length is None:
        window_length = config.SAMPLE_WINDOW_LENGTH
    if window_length is None:
        window_end = seq_len
    else:
        window_end = min(seq_len, window_start + max(1, int(window_length)))

    if window_start >= seq_len:
        raise ValueError(f"Sampling window start {window_start} exceeds sequence length {seq_len}")

    print(f"Sampling window configured to indices [{window_start}, {window_end})")
    
    # Initialize Trainer with custom epochs and optional save_timesteps for early stopping
    trainer = Trainer(
        diffusion,
        dataset,
        train_batch_size=min(config.BATCH_SIZE, len(dataset)),  # Ensure batch size doesn't exceed dataset size
        train_lr=config.LEARNING_RATE,
        train_epochs=epochs,
        adamw_weight_decay=config.WEIGHT_DECAY,
        cosine_scheduler=config.USE_COSINE_SCHEDULER,
        warm_up=config.USE_WARM_UP,
        warmup_iters=config.WARMUP_STEPS,
        T_0=config.COSINE_CYCLE_LENGTH,
        T_mult=config.T_MULT,
        eta_min=config.COSINE_LR_MIN,
        cosine_steps=config.COSINE_STEPS,
        gradient_accumulate_every=config.GRADIENT_ACCUMULATION,
        ema_decay=config.EMA_DECAY,
        split_batches=config.SPLIT_BATCHES,
        save_and_sample_every=config.SAVE_INTERVAL,
        results_folder=model_dir,
        param_path="",
        amp=config.USE_AMP,
        save_timesteps=save_timesteps,  # Pass save_timesteps for early stopping evaluation
    )
    
    print("Trainer initialized")
    
    # Train model
    print(f"Starting training for {epochs} epochs...")
    trainer.train()
    
    # Generate samples
    print("Generating samples...")
    sample_batches = config.SAMPLE_BATCHES
    samples_per_batch = config.SAMPLES_PER_BATCH
    
    config.set_seed(seed)  # Reset seed for reproducibility
    
    for i in range(sample_batches):
        # Pass save_timesteps parameter to sample method for early stopping evaluation
        progress_desc = f"Sampling batch {i+1}/{sample_batches}"
        samples = diffusion.sample(
            batch_size=samples_per_batch,
            save_timesteps=save_timesteps,
            start_idx=window_start,
            end_idx=window_end,
            show_progress=True,
            progress_desc=progress_desc,
        )
        if samples.dim() == 3:
            # (batch, snapshots, seq_len)
            scaled = samples * data_std.unsqueeze(1) + data_mean.unsqueeze(1)
        else:
            scaled = samples * data_std + data_mean
        samples = scaled.reshape(scaled.size(0), -1).cpu().numpy()
        
        sample_file = os.path.join(sample_dir, f"sample_batch{i+1}.npy")
        np.save(sample_file, samples)
        
        # Clean up to prevent memory issues
        del samples
        gc.collect()
    
    # Clean up
    del trainer, model, diffusion, data, dataset
    gc.collect()
    
    print(f"Training and sampling complete for {exp_id}")
    print(f"Models saved to: {model_dir}")
    print(f"Samples saved to: {sample_dir}")
    
    return model_dir, sample_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train diffusion factor model on specific data file")
    parser.add_argument("--data_path", type=str, required=True, 
                      help="Path to the data file for training")
    parser.add_argument("--seed", type=int, default=None, 
                      help="Random seed")
    parser.add_argument("--num_samples", type=int, default=None, 
                      help="Number of training samples (None = use all)")
    parser.add_argument("--gpu", type=int, default=0, 
                      help="GPU ID")
    parser.add_argument("--epochs", type=int, default=None, 
                      help="Number of epochs to train (None = use config value)")
    parser.add_argument("--save_timesteps", type=int, nargs='+', default=None,
                      help="Specific timesteps to save during sampling for early stopping evaluation (e.g., --save_timesteps 100 200 500)")
    parser.add_argument("--sample_window_start", type=int, default=None,
                      help="Start index (inclusive) for sequential sampling window")
    parser.add_argument("--sample_window_length", type=int, default=None,
                      help="Number of indices to generate in the sampling window")
    
    args = parser.parse_args()
    
    train_model(
        args.data_path,
        args.seed,
        args.num_samples,
        args.gpu,
        args.epochs,
        args.save_timesteps,
        args.sample_window_start,
        args.sample_window_length,
    )

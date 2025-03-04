#!/usr/bin/env python
"""
QFNN Quick Test Script
----------------------
This script demonstrates the entire QFNN pipeline in a single, easy-to-follow process:
1. Creates a minimal QFNN model
2. Trains it on a tiny dataset
3. Generates text with the model
4. Visualizes the model's internal mechanisms
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import shutil
import time

# Import QFNN components
from qfnn_implementation import QFNN, QFNNConfig
from qfnn_visualizations import QFNNVisualizer

# Create output directory
OUTPUT_DIR = "qfnn_test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{OUTPUT_DIR}/test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("qfnn_test")

def print_section(title):
    """Print a section divider to make the output more readable"""
    width = 80
    print("\n" + "=" * width)
    print(f"{title.center(width)}")
    print("=" * width + "\n")

def create_toy_dataset(vocab_size=100, seq_length=32, num_samples=100):
    """Create a toy dataset for quick testing"""
    print_section("Creating Toy Dataset")
    
    # Create random data
    X = torch.randint(0, vocab_size, (num_samples, seq_length))
    
    # Create targets (shifted input)
    y = torch.cat([X[:, 1:], torch.randint(0, vocab_size, (num_samples, 1))], dim=1)
    
    # Create dataloaders
    train_size = int(0.8 * num_samples)
    train_dataset = TensorDataset(X[:train_size], y[:train_size])
    val_dataset = TensorDataset(X[train_size:], y[train_size:])
    
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    logger.info(f"Created dataset with {num_samples} samples, {train_size} for training")
    logger.info(f"Vocabulary size: {vocab_size}, Sequence length: {seq_length}")
    
    # Create a simple tokenizer for this toy dataset
    class ToyTokenizer:
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size
        
        def convert_ids_to_tokens(self, ids):
            return [f"TOK{id}" for id in ids]
    
        def decode(self, ids, skip_special_tokens=True):
            return " ".join([f"TOK{id}" for id in ids])
    
        def encode(self, text, return_tensors=None):
            # Just map random tokens for testing
            ids = torch.randint(0, self.vocab_size, (1, 10))
            if return_tensors == 'pt':
                return ids
            return ids[0].tolist()
    
    tokenizer = ToyTokenizer(vocab_size)
    
    return train_dataloader, val_dataloader, tokenizer

def create_and_test_model(train_dataloader, val_dataloader, tokenizer):
    """Create a QFNN model and test its forward pass"""
    print_section("Creating and Testing QFNN Model")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set up configuration
    config = QFNNConfig(
        vocab_size=100,          # Small vocabulary for testing
        hidden_dim=64,           # Small hidden dimension
        num_layers=2,            # Just 2 layers for quick testing  
        max_seq_length=32,       # Short sequences
        epsilon=1e-5,            # Small constant for numerical stability
        threshold=0.05,          # Threshold for attention sparsity
        alpha=0.5,               # Weight for orthogonal component
        dt_scale=0.1,            # Time step scale
        beta=10.0,               # Skip connection sensitivity
        r_min=0.5,               # Minimum radius
        r_max=2.0,               # Maximum radius
        radius_rate=0.01,        # Radius adjustment rate
        learning_rate=0.001,     # Learning rate
        use_hebbian=True,        # Use Hebbian learning
        device=device
    )
    
    # Create model
    model = QFNN(config).to(device)
    
    # Print model summary
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {num_params} parameters")
    
    # Test a forward pass
    for input_ids, target_ids in train_dataloader:
        input_ids = input_ids.to(device)
        
        # Track time for forward pass
        start_time = time.time()
        logits = model(input_ids)
        forward_time = time.time() - start_time
        
        logger.info(f"Forward pass shape: {logits.shape}, took {forward_time:.4f}s")
        
        # Test with attention patterns
        _, attention_patterns = model(input_ids, return_attention=True)
        logger.info(f"Number of attention layers: {len(attention_patterns)}")
        logger.info(f"Attention shape: {attention_patterns[0].shape}")
        
        # Just test the first batch
        break
    
    return model, config

def mini_train(model, train_dataloader, val_dataloader, config, epochs=2):
    """Train the model for a few epochs"""
    print_section("Mini-training the Model")
    
    device = config.device
    
    # Use either Hebbian learning or standard backprop
    if config.use_hebbian:
        logger.info("Using Hebbian learning")
        optimizer = None
    else:
        logger.info("Using standard backpropagation")
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Train for a few epochs
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Training loop
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for input_ids, target_ids in pbar:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            if config.use_hebbian:
                # Hebbian learning (no backprop)
                with torch.no_grad():
                    # Forward pass
                    logits = model(input_ids)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                    
                    # Get hidden states
                    states = model.token_representation(input_ids)
                    for layer in model.layers:
                        states, _ = layer(states)
                    hidden = model.projection(states)
                    
                    # Calculate error
                    outputs = model.output_layer(hidden)
                    probs = F.softmax(outputs, dim=-1)
                    target_onehot = F.one_hot(target_ids, config.vocab_size).float()
                    error = target_onehot - probs
                    
                    # Hebbian update
                    delta_w = torch.einsum('bnv,bnd->vd', error, hidden) * config.learning_rate
                    model.output_layer.output.weight.data += delta_w
            else:
                # Standard backpropagation
                optimizer.zero_grad()
                logits = model(input_ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{epochs} - Avg loss: {avg_loss:.4f}")
        
        # Quick validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for input_ids, target_ids in val_dataloader:
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                
                logits = model(input_ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        logger.info(f"Validation loss: {avg_val_loss:.4f}")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__
    }, f"{OUTPUT_DIR}/mini_trained_model.pt")
    
    logger.info(f"Model saved to {OUTPUT_DIR}/mini_trained_model.pt")
    
    return model

def test_generation(model, tokenizer, config):
    """Test text generation with the model"""
    print_section("Testing Text Generation")
    
    device = config.device
    model.eval()
    
    # Create a simple prompt
    prompt = "TOK5 TOK10 TOK15"
    logger.info(f"Generation prompt: {prompt}")
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate text
    generated_ids = input_ids[0].tolist()
    
    # Track attention for visualization
    all_attention_patterns = []
    
    # Generate a few tokens
    max_new_tokens = 10
    logger.info(f"Generating {max_new_tokens} new tokens...")
    
    with torch.no_grad():
        for _ in tqdm(range(max_new_tokens)):
            # Forward pass with attention
            logits, attention_patterns = model(input_ids, return_attention=True)
            all_attention_patterns.append([a.cpu() for a in attention_patterns])
            
            # Get next token (simple greedy decoding for test)
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
            
            # Add to generated sequence
            generated_ids.append(next_token)
            
            # Update input for next iteration
            input_ids = torch.tensor([generated_ids]).to(device)
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_ids)
    logger.info(f"Generated text: {generated_text}")
    
    # Save attention patterns for later visualization
    return generated_ids, all_attention_patterns

def visualize_model(model, tokenizer, generated_ids, attention_patterns):
    """Create visualizations of the model's internal state"""
    print_section("Creating Visualizations")
    
    device = model.config.device
    vis_dir = f"{OUTPUT_DIR}/visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create visualizer
    visualizer = QFNNVisualizer(model, tokenizer, vis_dir)
    
    # 1. Visualize token embeddings
    logger.info("Visualizing token embeddings...")
    visualizer.visualize_token_embeddings()
    
    # 2. Visualize attention patterns
    logger.info("Visualizing attention patterns...")
    input_tensor = torch.tensor([generated_ids]).to(device)
    # In your visualize_model function
    max_tokens = min(len(generated_ids), attention_patterns[-1][0].shape[1])
    visualizer.visualize_attention_matrices(input_tensor[:, :max_tokens], attention_patterns[-1])
        
    # 3. Visualize phase coherence
    logger.info("Visualizing phase coherence...")
    visualizer.visualize_phase_coherence(input_tensor)
    
    # 4. Visualize token state evolution
    logger.info("Visualizing token state evolution...")
    # Only use first 10 tokens for clarity
    short_input = torch.tensor([generated_ids[:10]]).to(device)
    visualizer.visualize_state_evolution(short_input)
    
    logger.info(f"Visualizations saved to {vis_dir}")

def run_full_test():
    """Run a complete test of the QFNN system"""
    print_section("QFNN QUICK TEST")
    start_time = time.time()
    
    # Step 1: Create dataset
    train_dataloader, val_dataloader, tokenizer = create_toy_dataset()
    
    # Step 2: Create and test model
    model, config = create_and_test_model(train_dataloader, val_dataloader, tokenizer)
    
    # Step 3: Train model
    model = mini_train(model, train_dataloader, val_dataloader, config, epochs=2)
    
    # Step 4: Test generation
    generated_ids, attention_patterns = test_generation(model, tokenizer, config)
    
    # Step 5: Create visualizations
    visualize_model(model, tokenizer, generated_ids, attention_patterns)
    
    total_time = time.time() - start_time
    print_section(f"TEST COMPLETE IN {total_time:.2f}s")
    logger.info(f"All test artifacts saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_full_test()
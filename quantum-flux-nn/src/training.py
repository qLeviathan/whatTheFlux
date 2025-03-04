import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import math
import os
import json
import argparse
from tqdm import tqdm
import time
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
import logging
import matplotlib.pyplot as plt

# Import QFNN model and visualizer
from qfnn_implementation import QFNN, QFNNConfig
from qfnn_visualizations import QFNNVisualizer


def setup_logging(log_dir='logs'):
    """Set up logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/training.log"),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('qfnn_training')


class WikiTextDataModule:
    """Data module for WikiText corpus"""
    def __init__(self, config, tokenizer_name='bert-base-uncased', batch_size=4, context_length=1024):
        self.config = config
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.context_length = context_length
        self.tokenizer = None
        self.dataset = None
        self.tokenized_dataset = None
    
    def prepare_data(self):
        """Load and prepare the dataset"""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        
        # Load dataset
        self.dataset = load_dataset('wikitext', 'wikitext-103-v1')
        
        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(examples['text'], truncation=False, padding=False)
        
        self.tokenized_dataset = self.dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=['text']
        )
    
    def create_dataloaders(self):
        """Create training and validation dataloaders"""
        # Create input/target pairs for language modeling
        def group_texts(examples):
            # Concatenate all texts
            concatenated = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated[list(concatenated.keys())[0]])
            
            # Drop small remainder
            total_length = (total_length // self.context_length) * self.context_length
            
            # Reshape and create input/target pairs
            result = {
                k: [t[i:i+self.context_length] for i in range(0, total_length, self.context_length)]
                for k, t in concatenated.items()
            }
            
            # Create target by shifting input
            result["labels"] = [
                ids[1:] + [self.tokenizer.pad_token_id] for ids in result["input_ids"]
            ]
            
            return result
        
        # Process train and validation sets
        train_grouped = self.tokenized_dataset['train'].map(
            group_texts,
            batched=True
        )
        
        val_grouped = self.tokenized_dataset['validation'].map(
            group_texts,
            batched=True
        )
        
        # Create PyTorch datasets
        class LMDataset(Dataset):
            def __init__(self, dataset):
                self.dataset = dataset
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                input_ids = torch.tensor(self.dataset[idx]['input_ids'])
                labels = torch.tensor(self.dataset[idx]['labels'])
                return input_ids, labels
        
        train_dataset = LMDataset(train_grouped)
        val_dataset = LMDataset(val_grouped)
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4
        )
        
        return train_dataloader, val_dataloader


class HebbianTrainer:
    """Trainer with Hebbian learning (wire together, fire together)"""
    def __init__(self, model, config, logger):
        self.model = model
        self.config = config
        self.logger = logger
        self.learning_rate = config.learning_rate
        self.device = config.device
        
        # Track training metrics
        self.training_stats = {
            'epoch_losses': [],
            'val_losses': [],
            'train_perplexities': [],
            'val_perplexities': [],
            'sparsity_by_epoch': [],
            'radius_stats_by_epoch': []
        }
    
    def train_step(self, input_ids, target_ids):
        """Perform one training step with Hebbian learning"""
        # Forward pass
        logits = self.model(input_ids)
        
        # Calculate cross-entropy loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        
        # Update output layer weights using Hebbian rule
        with torch.no_grad():
            # Get hidden states before output layer
            states = self.model.token_representation(input_ids)
            
            # Process through QFNN layers
            attention_patterns = []
            for layer in self.model.layers:
                states, attention = layer(states)
                attention_patterns.append(attention)
            
            # Get final hidden states
            hidden = self.model.projection(states)
            
            # Calculate predictions from current weights
            outputs = self.model.output_layer(hidden)
            
            # Calculate error
            probs = F.softmax(outputs, dim=-1)
            target_onehot = F.one_hot(target_ids, self.config.vocab_size).float()
            error = target_onehot - probs
            
            # Hebbian update (neurons that fire together, wire together)
            # Scale by learning rate
            delta_weights = torch.einsum('bnv,bnd->vd', error, hidden) * self.learning_rate
            
            # Apply update
            self.model.output_layer.output.weight.data += delta_weights
        
        return logits, loss, attention_patterns
    
    def evaluate(self, dataloader, max_batches=None):
        """Evaluate model on dataloader"""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        attention_patterns_list = []
        
        with torch.no_grad():
            for i, (input_ids, target_ids) in enumerate(tqdm(dataloader, desc="Evaluating")):
                if max_batches and i >= max_batches:
                    break
                
                # Move to device
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                # Forward pass
                logits, attention_patterns = self.model(input_ids, return_attention=True)
                
                # Calculate loss
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                                       target_ids.view(-1), 
                                       reduction='sum')
                
                total_loss += loss.item()
                total_tokens += target_ids.numel()
                
                # Store attention patterns for first few batches
                if len(attention_patterns_list) < 5:
                    attention_patterns_list.append([a.cpu() for a in attention_patterns])
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return avg_loss, perplexity, attention_patterns_list
    
    def train(self, train_dataloader, val_dataloader, epochs, output_dir, 
              eval_every=1000, visualize_every=5000, save_every=10000):
        """Train the model"""
        os.makedirs(output_dir, exist_ok=True)
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Training loop
        global_step = 0
        best_val_loss = float('inf')
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Training epoch
            self.model.train()
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for input_ids, target_ids in pbar:
                # Move to device
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                # Training step
                _, loss, attention_patterns = self.train_step(input_ids, target_ids)
                
                # Update statistics
                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'ppl': f"{math.exp(loss.item()):.2f}"
                })
                
                # Periodic evaluation
                if global_step % eval_every == 0:
                    val_loss, val_ppl, _ = self.evaluate(val_dataloader, max_batches=100)
                    self.logger.info(f"Step {global_step}: Val loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save_checkpoint(os.path.join(output_dir, 'best_model.pt'))
                        self.logger.info(f"New best model saved with Val loss: {val_loss:.4f}")
                    
                    # Return to training mode
                    self.model.train()
                
                # Periodic visualization
                if global_step % visualize_every == 0:
                    step_vis_dir = os.path.join(vis_dir, f'step_{global_step}')
                    os.makedirs(step_vis_dir, exist_ok=True)
                    
                    visualizer = QFNNVisualizer(self.model, train_dataloader.dataset.dataset.tokenizer, step_vis_dir)
                    visualizer.visualize_token_embeddings()
                    
                    if len(attention_patterns) > 0:
                        visualizer.visualize_attention_matrices(input_ids, attention_patterns)
                    
                    self.logger.info(f"Visualizations saved to {step_vis_dir}")
                
                # Periodic model saving
                if global_step % save_every == 0:
                    self.save_checkpoint(os.path.join(output_dir, f'model_step_{global_step}.pt'))
            
            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches
            epoch_ppl = math.exp(avg_epoch_loss)
            
            self.training_stats['epoch_losses'].append(avg_epoch_loss)
            self.training_stats['train_perplexities'].append(epoch_ppl)
            
            # Full validation
            val_loss, val_ppl, val_attention = self.evaluate(val_dataloader)
            self.training_stats['val_losses'].append(val_loss)
            self.training_stats['val_perplexities'].append(val_ppl)
            
            # Calculate sparsity metrics
            sparsity = self.calculate_attention_sparsity(val_attention)
            self.training_stats['sparsity_by_epoch'].append(sparsity)
            
            # Calculate radius statistics
            radius_stats = self.calculate_radius_stats()
            self.training_stats['radius_stats_by_epoch'].append(radius_stats)
            
            # Log progress
            self.logger.info(f"Epoch {epoch+1}/{epochs} completed in {time.time() - start_time:.2f}s")
            self.logger.info(f"Train loss: {avg_epoch_loss:.4f}, Train PPL: {epoch_ppl:.2f}")
            self.logger.info(f"Val loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")
            self.logger.info(f"Attention sparsity: {sparsity['avg_sparsity']:.2%}")
            self.logger.info(f"Radius stats: min={radius_stats['min']:.2f}, max={radius_stats['max']:.2f}, mean={radius_stats['mean']:.2f}")
            
            # Save checkpoint
            self.save_checkpoint(os.path.join(output_dir, f'model_epoch_{epoch+1}.pt'))
            
            # Reset timer for next epoch
            start_time = time.time()
        
        # Save final model
        self.save_checkpoint(os.path.join(output_dir, 'final_model.pt'))
        
        # Save training statistics
        with open(os.path.join(output_dir, 'training_stats.json'), 'w') as f:
            json.dump(self.training_stats, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
        
        # Create training curves
        self.plot_training_curves(output_dir)
        
        return self.training_stats
    
    def calculate_attention_sparsity(self, attention_patterns_list):
        """Calculate sparsity metrics from attention patterns"""
        layer_sparsity = []
        
        for attention_patterns in attention_patterns_list:
            for layer_idx, attention in enumerate(attention_patterns):
                nonzero = torch.count_nonzero(attention).item()
                total = torch.numel(attention)
                sparsity = 1.0 - (nonzero / total)
                
                layer_sparsity.append({
                    'layer': layer_idx,
                    'sparsity': sparsity
                })
        
        # Calculate average sparsity per layer
        layer_avg = {}
        for item in layer_sparsity:
            layer = item['layer']
            if layer not in layer_avg:
                layer_avg[layer] = []
            layer_avg[layer].append(item['sparsity'])
        
        layer_avg = {layer: sum(values) / len(values) for layer, values in layer_avg.items()}
        
        # Overall average
        avg_sparsity = sum(layer_avg.values()) / len(layer_avg)
        
        return {
            'by_layer': layer_avg,
            'avg_sparsity': avg_sparsity
        }
    
    def calculate_radius_stats(self):
        """Calculate radius statistics for token embeddings"""
        # Get embedding weights
        weights = self.model.token_representation.embeddings.weight.data.cpu()
        
        # Calculate radii
        radii = torch.norm(weights, dim=1)
        
        return {
            'min': radii.min().item(),
            'max': radii.max().item(),
            'mean': radii.mean().item(),
            'std': radii.std().item(),
            'median': radii.median().item(),
            'histogram': torch.histc(radii, bins=10, min=self.config.r_min, max=self.config.r_max).tolist()
        }
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__
        }, path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Update config if available
        if 'config' in checkpoint:
            for key, value in checkpoint['config'].items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
    
    def plot_training_curves(self, output_dir):
        """Plot training curves"""
        # Loss curves
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_stats['epoch_losses'], label='Train Loss')
        plt.plot(self.training_stats['val_losses'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=300)
        plt.close()
        
        # Perplexity curves
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_stats['train_perplexities'], label='Train PPL')
        plt.plot(self.training_stats['val_perplexities'], label='Validation PPL')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.title('Training and Validation Perplexity')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'perplexity_curves.png'), dpi=300)
        plt.close()
        
        # Attention sparsity
        if self.training_stats['sparsity_by_epoch']:
            plt.figure(figsize=(10, 6))
            sparsity_values = [stats['avg_sparsity'] for stats in self.training_stats['sparsity_by_epoch']]
            plt.plot(sparsity_values)
            plt.xlabel('Epoch')
            plt.ylabel('Attention Sparsity')
            plt.title('Average Attention Sparsity Over Training')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'sparsity_curve.png'), dpi=300)
            plt.close()
        
        # Radius statistics
        if self.training_stats['radius_stats_by_epoch']:
            plt.figure(figsize=(10, 6))
            min_r = [stats['min'] for stats in self.training_stats['radius_stats_by_epoch']]
            max_r = [stats['max'] for stats in self.training_stats['radius_stats_by_epoch']]
            mean_r = [stats['mean'] for stats in self.training_stats['radius_stats_by_epoch']]
            
            plt.plot(min_r, label='Min Radius')
            plt.plot(max_r, label='Max Radius')
            plt.plot(mean_r, label='Mean Radius')
            plt.xlabel('Epoch')
            plt.ylabel('Radius')
            plt.title('Token Radius Statistics Over Training')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'radius_stats.png'), dpi=300)
            plt.close()


def train_wikitext(config, output_dir='output', epochs=5, batch_size=8, context_length=1024):
    """Train QFNN on WikiText dataset"""
    # Setup logging
    logger = setup_logging(os.path.join(output_dir, 'logs'))
    logger.info(f"Training QFNN on WikiText with config: {config.__dict__}")
    
    # Create data module
    data_module = WikiTextDataModule(config, batch_size=batch_size, context_length=context_length)
    data_module.prepare_data()
    train_dataloader, val_dataloader = data_module.create_dataloaders()
    
    logger.info(f"Data prepared: {len(train_dataloader)} training batches, {len(val_dataloader)} validation batches")
    
    # Create model
    model = QFNN(config).to(config.device)
    
    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model created with {total_params} total parameters ({trainable_params} trainable)")
    
    # Create trainer
    trainer = HebbianTrainer(model, config, logger)
    
    # Train model
    logger.info("Starting training")
    stats = trainer.train(train_dataloader, val_dataloader, epochs, output_dir)
    
    logger.info("Training complete")
    return model, stats, data_module.tokenizer


def main():
    """Main function for QFNN training"""
    parser = argparse.ArgumentParser(description='Train Quantum Flux Neural Network')
    
    # Training parameters
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--context_length', type=int, default=1024, help='Context window length')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    
    # Model parameters
    parser.add_argument('--vocab_size', type=int, default=30522, help='Vocabulary size')
    parser.add_argument('--hidden_dim', type=int, default=768, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of QFNN layers')
    parser.add_argument('--epsilon', type=float, default=1e-5, help='Epsilon for attention stability')
    parser.add_argument('--threshold', type=float, default=0.01, help='Threshold for attention sparsity')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for orthogonal component')
    parser.add_argument('--dt_scale', type=float, default=0.1, help='Time step scale')
    parser.add_argument('--r_min', type=float, default=0.5, help='Minimum radius')
    parser.add_argument('--r_max', type=float, default=2.0, help='Maximum radius')
    parser.add_argument('--radius_rate', type=float, default=0.01, help='Radius adjustment rate')
    parser.add_argument('--use_hebbian', action='store_true', help='Use Hebbian learning')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to train on')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create config
    config = QFNNConfig(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        max_seq_length=args.context_length,
        epsilon=args.epsilon,
        threshold=args.threshold,
        alpha=args.alpha,
        dt_scale=args.dt_scale,
        r_min=args.r_min,
        r_max=args.r_max,
        radius_rate=args.radius_rate,
        learning_rate=args.learning_rate,
        use_hebbian=args.use_hebbian,
        device=args.device
    )
    
    # Train model
    model, stats, tokenizer = train_wikitext(
        config,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        context_length=args.context_length
    )
    
    # Final visualization
    visualizer = QFNNVisualizer(model, tokenizer, os.path.join(args.output_dir, 'final_visualizations'))
    visualizer.visualize_token_embeddings()
    
    print(f"Training completed and results saved to {args.output_dir}")


if __name__ == "__main__":
    main()ok=
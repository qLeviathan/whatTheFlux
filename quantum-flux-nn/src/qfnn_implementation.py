import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import json
import os


class QFNNConfig:
    """Configuration for Quantum Flux Neural Network"""
    def __init__(self, 
                 vocab_size=30522,           # Standard BERT vocabulary size
                 hidden_dim=768,             # Projection dimension
                 num_layers=6,               # Number of QFNN layers
                 max_seq_length=4096,        # Maximum sequence length
                 epsilon=1e-5,               # Small constant for stability in attention
                 threshold=0.01,             # Threshold for attention sparsity
                 alpha=0.5,                  # Weight for orthogonal component
                 dt_scale=0.1,               # Time step scale for state evolution
                 beta=10.0,                  # Skip connection sensitivity
                 r_min=0.5,                  # Minimum radius value
                 r_max=2.0,                  # Maximum radius value
                 radius_rate=0.01,           # Rate of radius adjustment
                 learning_rate=0.001,        # Learning rate for optimizer
                 use_hebbian=True,           # Whether to use Hebbian learning
                 device='cuda'):             # Device to run on
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        self.epsilon = epsilon
        self.threshold = threshold
        self.alpha = alpha
        self.dt_scale = dt_scale
        self.beta = beta
        self.r_min = r_min
        self.r_max = r_max
        self.radius_rate = radius_rate
        self.learning_rate = learning_rate
        self.use_hebbian = use_hebbian
        self.device = device


class QuantumTokenRepresentation(nn.Module):
    """Maps token indices to 2D quantum states on a cylinder"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize token embeddings in cylindrical space (r*cos(θ), r*sin(θ))
        self.embeddings = nn.Embedding(config.vocab_size, 2)
        
        # Initialize with random angles and radii based on vocabulary position
        with torch.no_grad():
            # Random angles (θ) from 0 to 2π - semantic meaning
            thetas = torch.rand(config.vocab_size) * 2 * math.pi
            
            # Radii scaled by token index (importance proxy) - more frequent tokens
            # typically appear earlier in vocabulary
            # Scale from r_min to r_max based on position in vocabulary
            vocab_positions = torch.arange(config.vocab_size).float()
            normalized_positions = vocab_positions / config.vocab_size
            radii = config.r_min + (config.r_max - config.r_min) * (1.0 - normalized_positions)
            
            # Convert to Cartesian coordinates
            self.embeddings.weight.data[:, 0] = radii * torch.cos(thetas)  # x = r*cos(θ)
            self.embeddings.weight.data[:, 1] = radii * torch.sin(thetas)  # y = r*sin(θ)
    
    def forward(self, token_ids):
        # Map token IDs to 2D quantum states
        # Input: [B, N], Output: [B, N, 2]
        return self.embeddings(token_ids)


class QuantumFluxAttention(nn.Module):
    """Implements quantum geometric attention mechanism"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.epsilon = config.epsilon
        self.threshold = config.threshold
        self.alpha = config.alpha
    
    def forward(self, states):
        """
        Compute attention based on geometric relationships between states
        Input: states [B, N, 2]
        Output: attention [B, N, N], context [B, N, 2]
        """
        # Direct geometric similarity between states
        # S_direct[i,j] = r_i * r_j * cos(θ_i - θ_j)
        direct_sim = torch.einsum('bik,bjk->bij', states, states)  # [B, N, N]
        
        # Orthogonal transformation (in-place for memory efficiency)
        # For (x,y) = (r*cos(θ), r*sin(θ)), orthogonal = (r*sin(θ), -r*cos(θ))
        states_orth = torch.zeros_like(states)
        states_orth[:, :, 0] = states[:, :, 1]      # r*sin(θ)
        states_orth[:, :, 1] = -states[:, :, 0]     # -r*cos(θ)
        
        # Orthogonal similarity - captures phase relationships
        # S_ortho[i,j] = r_i * r_j * sin(θ_i - θ_j)
        ortho_sim = torch.einsum('bik,bjk->bij', states, states_orth)  # [B, N, N]
        
        # Combined similarity with weighting
        similarity = direct_sim + self.alpha * ortho_sim  # [B, N, N]
        
        # Inverse perturbation gating for natural sparsity
        # Smaller values for strong interactions, larger for weak ones
        gate = 1.0 / (self.epsilon + torch.abs(similarity))  # [B, N, N]
        
        # Apply threshold for hard sparsity
        mask = gate > self.threshold  # [B, N, N]
        
        # Final sparse attention matrix
        attention = similarity * mask * gate  # [B, N, N]
        
        # Compute context vectors via attention
        context = torch.einsum('bij,bjk->bik', attention, states)  # [B, N, 2]
        
        return attention, context


class StateEvolution(nn.Module):
    """Evolves quantum states using Heun-Euler integration"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dt = config.dt_scale
        self.beta = config.beta
    
    def forward(self, states, attention_fn):
        """
        Evolve states using Heun-Euler method
        Input: states [B, N, 2], attention_fn (callable)
        Output: new_states [B, N, 2]
        """
        # First step of Heun-Euler method
        attention, context = attention_fn(states)
        k1 = self.dt * context  # [B, N, 2]
        
        # Intermediate state
        mid_states = states + k1  # [B, N, 2]
        
        # Second step
        _, mid_context = attention_fn(mid_states)
        k2 = self.dt * mid_context  # [B, N, 2]
        
        # Final update
        evolved_states = states + 0.5 * (k1 + k2)  # [B, N, 2]
        
        # Quantum tunneling via skip connections
        # Larger time steps → more original information preserved
        skip_weight = torch.sigmoid(self.beta * (self.dt - 1.0))
        new_states = skip_weight * states + (1 - skip_weight) * evolved_states  # [B, N, 2]
        
        return new_states, attention


class RadiusAdjustment(nn.Module):
    """Adjusts token radii based on attention patterns"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.r_min = config.r_min
        self.r_max = config.r_max
        self.rate = config.radius_rate
    
    def forward(self, states, attention):
        """
        Adjust radii based on attention strength
        Input: states [B, N, 2], attention [B, N, N]
        Output: new_states [B, N, 2]
        """
        # Calculate current radii
        radii = torch.norm(states, dim=2, keepdim=True)  # [B, N, 1]
        
        # Calculate total attention for each token
        attention_sum = torch.sum(attention, dim=2, keepdim=True)  # [B, N, 1]
        
        # Adjust radii based on attention (mean-reverting process)
        delta_r = attention_sum * self.rate
        new_radii = radii + delta_r  # [B, N, 1]
        
        # Clamp to prevent extreme values
        new_radii = torch.clamp(new_radii, self.r_min, self.r_max)  # [B, N, 1]
        
        # Scale original states to new radii (preserving angles)
        scale_factor = new_radii / (radii + 1e-8)  # [B, N, 1]
        new_states = states * scale_factor  # [B, N, 2]
        
        return new_states


class QFNNLayer(nn.Module):
    """Full Quantum Flux Neural Network layer"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention = QuantumFluxAttention(config)
        self.evolution = StateEvolution(config)
        self.radius_adjustment = RadiusAdjustment(config)
    
    def forward(self, states):
        """
        Process states through one complete QFNN layer
        Input: states [B, N, 2]
        Output: new_states [B, N, 2], attention [B, N, N]
        """
        # Evolve states using quantum flux attention
        evolved_states, attention = self.evolution(states, self.attention)
        
        # Adjust radii based on attention patterns
        new_states = self.radius_adjustment(evolved_states, attention)
        
        return new_states, attention


class ProjectionLayer(nn.Module):
    """Projects 2D quantum states to higher dimensions"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Simple linear projection from 2D space to hidden_dim
        self.projection = nn.Linear(2, config.hidden_dim)
    
    def forward(self, states):
        """
        Project 2D quantum states to higher-dimensional space
        Input: states [B, N, 2]
        Output: hidden [B, N, D]
        """
        return self.projection(states)  # [B, N, D]


class OutputLayer(nn.Module):
    """Maps hidden states to vocabulary distribution"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Linear layer from hidden_dim to vocab_size
        self.output = nn.Linear(config.hidden_dim, config.vocab_size)
    
    def forward(self, hidden):
        """
        Generate logits for next token prediction
        Input: hidden [B, N, D]
        Output: logits [B, N, V]
        """
        return self.output(hidden)  # [B, N, V]


class QFNN(nn.Module):
    """Complete Quantum Flux Neural Network model"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token representation in 2D quantum space
        self.token_representation = QuantumTokenRepresentation(config)
        
        # QFNN layers
        self.layers = nn.ModuleList([QFNNLayer(config) for _ in range(config.num_layers)])
        
        # Projection to higher dimensions
        self.projection = ProjectionLayer(config)
        
        # Output layer
        self.output_layer = OutputLayer(config)
        
        # Store attention patterns for analysis
        self.attention_patterns = []
    
    def forward(self, input_ids, return_attention=False):
        """
        Process input tokens through the QFNN
        Input: input_ids [B, N]
        Output: logits [B, N, V], (optionally) attention patterns
        """
        # Map tokens to quantum states
        states = self.token_representation(input_ids)  # [B, N, 2]
        
        # Clear previous attention patterns
        self.attention_patterns = []
        
        # Process through QFNN layers
        for layer in self.layers:
            states, attention = layer(states)  # [B, N, 2], [B, N, N]
            
            # Store attention for visualization/analysis
            if return_attention:
                self.attention_patterns.append(attention.detach())
        
        # Project to higher dimensions
        hidden = self.projection(states)  # [B, N, D]
        
        # Generate output logits
        logits = self.output_layer(hidden)  # [B, N, V]
        
        if return_attention:
            return logits, self.attention_patterns
        else:
            return logits


class HebbianTrainer:
    """Implements Hebbian learning without backpropagation"""
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.learning_rate = config.learning_rate
    
    def train_step(self, input_ids, target_ids):
        """
        Perform one training step using Hebbian learning
        Input: input_ids [B, N], target_ids [B, N]
        Output: logits [B, N, V]
        """
        # Forward pass
        logits = self.model(input_ids)  # [B, N, V]
        
        # Calculate cross-entropy loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        
        # Update output layer weights using Hebbian rule
        # "Neurons that fire together, wire together"
        with torch.no_grad():
            # Get hidden states before output layer
            states = self.model.token_representation(input_ids)
            for layer in self.model.layers:
                states, _ = layer(states)
            hidden = self.model.projection(states)  # [B, N, D]
            
            # Calculate current outputs
            outputs = self.model.output_layer(hidden)  # [B, N, V]
            
            # Convert targets to one-hot
            batch_size, seq_len = target_ids.size()
            target_onehot = F.one_hot(target_ids, self.config.vocab_size).float()  # [B, N, V]
            
            # Calculate error
            error = target_onehot - F.softmax(outputs, dim=-1)  # [B, N, V]
            
            # Compute delta weights using outer product of error and activations
            # Average across batch and sequence dimensions
            delta_w = torch.einsum('bnv,bnd->vd', error, hidden) * self.learning_rate
            
            # Apply update
            self.model.output_layer.output.weight.data += delta_w
        
        return logits, loss


class WikiTextDataModule:
    """Loads and processes WikiText data for QFNN training"""
    def __init__(self, config, batch_size=4, context_length=1024):
        self.config = config
        self.batch_size = batch_size
        self.context_length = context_length
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Load dataset
        self.dataset = load_dataset('wikitext', 'wikitext-103-v1')
    
    def prepare_data(self):
        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(examples['text'], truncation=False, padding=False)
        
        self.tokenized_dataset = self.dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=['text']
        )
    
    def create_dataloader(self, split='train'):
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
        
        grouped_dataset = self.tokenized_dataset[split].map(
            group_texts,
            batched=True
        )
        
        # Create PyTorch dataset
        class LMDataset(Dataset):
            def __init__(self, dataset):
                self.dataset = dataset
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                input_ids = torch.tensor(self.dataset[idx]['input_ids'])
                labels = torch.tensor(self.dataset[idx]['labels'])
                return input_ids, labels
        
        dataset = LMDataset(grouped_dataset)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(split == 'train'),
            pin_memory=True,
            num_workers=4
        )
        
        return dataloader


# Visualization functions
def visualize_token_states(model, tokenizer, n_tokens=1000):
    """Visualize token embeddings in the cylindrical space"""
    # Get embedding weights
    weights = model.token_representation.embeddings.weight.data[:n_tokens].cpu().numpy()
    
    # Extract x, y components
    x, y = weights[:, 0], weights[:, 1]
    
    # Calculate radius and angle
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Scatter plot in Cartesian coordinates
    scatter = ax1.scatter(x, y, c=r, cmap='viridis', alpha=0.7)
    ax1.set_xlabel('x = r*cos(θ)')
    ax1.set_ylabel('y = r*sin(θ)')
    ax1.set_title('Token Embeddings in 2D Space')
    ax1.grid(True)
    ax1.set_aspect('equal')
    fig.colorbar(scatter, ax=ax1, label='Radius')
    
    # Polar plot
    ax2 = plt.subplot(122, projection='polar')
    scatter = ax2.scatter(theta, r, c=r, cmap='viridis', alpha=0.7)
    ax2.set_title('Token Embeddings in Polar Coordinates')
    ax2.grid(True)
    
    # Add token labels for some interesting points
    top_r_indices = np.argsort(r)[-10:]
    for idx in top_r_indices:
        token = tokenizer.convert_ids_to_tokens([idx])[0]
        ax1.annotate(token, (x[idx], y[idx]))
        ax2.annotate(token, (theta[idx], r[idx]))
    
    plt.tight_layout()
    return fig


def visualize_attention_patterns(attention_patterns, sequence, tokenizer):
    """Visualize attention patterns between tokens"""
    # Only use the first batch
    attention = attention_patterns[0][0].cpu().numpy()  # [N, N]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot attention heatmap
    im = ax.imshow(attention, cmap='Blues')
    
    # Add colorbar
    fig.colorbar(im, ax=ax, label='Attention Weight')
    
    # Get token strings
    tokens = tokenizer.convert_ids_to_tokens(sequence[0])
    
    # Add labels
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticklabels(tokens)
    
    ax.set_title('Attention Pattern')
    ax.set_xlabel('Token (target)')
    ax.set_ylabel('Token (source)')
    
    # Adjust layout
    plt.tight_layout()
    return fig


def analyze_attention_sparsity(model, dataloader, num_samples=5):
    """Analyze sparsity of attention matrices"""
    model.eval()
    sparsity_stats = []
    
    with torch.no_grad():
        for i, (input_ids, _) in enumerate(dataloader):
            if i >= num_samples:
                break
            
            # Forward pass with attention patterns
            _, attention_patterns = model(input_ids.to(model.config.device), return_attention=True)
            
            # Calculate sparsity for each layer
            for layer_idx, attention in enumerate(attention_patterns):
                nonzero = torch.count_nonzero(attention)
                total = torch.numel(attention)
                sparsity = 1.0 - (nonzero / total).item()
                
                sparsity_stats.append({
                    'sample': i,
                    'layer': layer_idx,
                    'sparsity': sparsity,
                    'active_connections': nonzero.item(),
                    'total_connections': total
                })
    
    return sparsity_stats


def train_qfnn(model, dataloader, eval_dataloader=None, epochs=1, config=None):
    """Train the QFNN model"""
    if config.use_hebbian:
        trainer = HebbianTrainer(model, config)
        optimizer = None
    else:
        trainer = None
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for input_ids, target_ids in pbar:
            # Move to device
            input_ids = input_ids.to(config.device)
            target_ids = target_ids.to(config.device)
            
            if config.use_hebbian:
                # Hebbian learning
                _, loss = trainer.train_step(input_ids, target_ids)
            else:
                # Standard backprop
                optimizer.zero_grad()
                logits = model(input_ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / num_batches:.4f}",
                'ppl': f"{math.exp(total_loss / num_batches):.2f}"
            })
        
        # Evaluation
        if eval_dataloader is not None:
            eval_loss = evaluate_qfnn(model, eval_dataloader, config)
            print(f"Epoch {epoch+1}/{epochs} - Train loss: {total_loss / num_batches:.4f}, "
                  f"Train PPL: {math.exp(total_loss / num_batches):.2f}, "
                  f"Eval loss: {eval_loss:.4f}, Eval PPL: {math.exp(eval_loss):.2f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} - Train loss: {total_loss / num_batches:.4f}, "
                  f"Train PPL: {math.exp(total_loss / num_batches):.2f}")
    
    return model


def evaluate_qfnn(model, dataloader, config):
    """Evaluate QFNN model on test data"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for input_ids, target_ids in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            input_ids = input_ids.to(config.device)
            target_ids = target_ids.to(config.device)
            
            # Forward pass
            logits = model(input_ids)
            
            # Calculate loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                                   target_ids.view(-1), 
                                   reduction='sum')
            
            total_loss += loss.item()
            total_tokens += target_ids.numel()
    
    return total_loss / total_tokens


def main():
    # Configuration
    config = QFNNConfig(
        vocab_size=30522,      # BERT vocabulary size
        hidden_dim=768,        # Projection dimension
        num_layers=6,          # Number of QFNN layers
        max_seq_length=1024,   # Maximum sequence length
        epsilon=1e-5,          # Small constant for stability
        threshold=0.01,        # Attention sparsity threshold
        alpha=0.5,             # Weight for orthogonal component
        dt_scale=0.1,          # Time step scale
        beta=10.0,             # Skip connection sensitivity
        r_min=0.5,             # Minimum radius
        r_max=2.0,             # Maximum radius
        radius_rate=0.01,      # Radius adjustment rate
        learning_rate=0.001,   # Learning rate
        use_hebbian=True,      # Use Hebbian learning
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Create model
    model = QFNN(config).to(config.device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create data module
    data_module = WikiTextDataModule(config, batch_size=4, context_length=1024)
    data_module.prepare_data()
    train_dataloader = data_module.create_dataloader('train')
    val_dataloader = data_module.create_dataloader('validation')
    
    # Train model
    model = train_qfnn(model, train_dataloader, val_dataloader, epochs=3, config=config)
    
    # Save model
    os.makedirs('output', exist_ok=True)
    torch.save(model.state_dict(), 'output/qfnn_model.pt')
    
    # Visualize token states
    fig = visualize_token_states(model, data_module.tokenizer)
    fig.savefig('output/token_states.png')
    
    # Analyze attention sparsity
    sparsity_stats = analyze_attention_sparsity(model, val_dataloader)
    with open('output/sparsity_stats.json', 'w') as f:
        json.dump(sparsity_stats, f, indent=2)
    
    print("Training and analysis complete!")


if __name__ == "__main__":
    main()

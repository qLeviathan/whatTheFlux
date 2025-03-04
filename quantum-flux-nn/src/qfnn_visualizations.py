import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import math
import os
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


class QFNNVisualizer:
    """Comprehensive visualization toolkit for QFNN models"""
    
    def __init__(self, model, tokenizer, output_dir='visualizations'):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def visualize_all(self, dataloader, num_samples=5):
        """Run all visualizations and save results"""
        # Visualize token embeddings
        self.visualize_token_embeddings()
        
        # Process some data through the model
        samples = []
        attention_patterns = []
        
        with torch.no_grad():
            for i, (input_ids, _) in enumerate(dataloader):
                if i >= num_samples:
                    break
                    
                input_ids = input_ids.to(self.model.config.device)
                _, attn = self.model(input_ids, return_attention=True)
                
                samples.append(input_ids.cpu())
                attention_patterns.append([a.cpu() for a in attn])
        
        # Visualize attention patterns
        self.visualize_attention_matrices(samples[0], attention_patterns[0])
        
        # Visualize sparsity patterns
        self.analyze_attention_sparsity(attention_patterns)
        
        # Visualize phase coherence
        self.visualize_phase_coherence(samples[0])
        
        # Visualize token state evolution
        self.visualize_state_evolution(samples[0])
    
    def visualize_token_embeddings(self):
        """Visualize token embeddings in 2D quantum space"""
        # Get embedding weights
        weights = self.model.token_representation.embeddings.weight.data.cpu().numpy()
        
        # Plot for top N most frequent tokens
        n_tokens = min(1000, weights.shape[0])
        
        # Extract x, y components
        x, y = weights[:n_tokens, 0], weights[:n_tokens, 1]
        
        # Calculate radius and angle
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Scatter plot in Cartesian coordinates - colored by vocabulary index
        scatter = ax1.scatter(x, y, c=np.arange(n_tokens), cmap='viridis', alpha=0.7)
        ax1.set_xlabel('x = r*cos(θ)', fontsize=12)
        ax1.set_ylabel('y = r*sin(θ)', fontsize=12)
        ax1.set_title('Token Embeddings in Cartesian Space', fontsize=14)
        ax1.grid(True)
        ax1.set_aspect('equal')
        fig.colorbar(scatter, ax=ax1, label='Vocabulary Index')
        
        # Polar plot - colored by vocabulary index
        ax2 = plt.subplot(122, projection='polar')
        scatter2 = ax2.scatter(theta, r, c=np.arange(n_tokens), cmap='viridis', alpha=0.7)
        ax2.set_title('Token Embeddings in Polar Coordinates', fontsize=14)
        ax2.grid(True)
        fig.colorbar(scatter2, ax=ax2, label='Vocabulary Index')
        
        # Add token labels for some interesting points
        # Most frequent tokens (smallest indices)
        frequent_tokens = np.arange(10)
        # Tokens with largest radii
        large_r_indices = np.argsort(r)[-10:]
        
        # Add labels
        for idx in np.concatenate([frequent_tokens, large_r_indices]):
            if idx < n_tokens:
                token = self.tokenizer.convert_ids_to_tokens([idx])[0]
                ax1.annotate(token, (x[idx], y[idx]), fontsize=8)
                ax2.annotate(token, (theta[idx], r[idx]), fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/token_embeddings.png", dpi=300)
        plt.close()
        
        # Save 3D visualization with radius, angle, and index
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Vocabulary index as z-axis
        z = np.arange(n_tokens)
        
        sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=10, alpha=0.7)
        
        ax.set_xlabel('x = r*cos(θ)', fontsize=12)
        ax.set_ylabel('y = r*sin(θ)', fontsize=12)
        ax.set_zlabel('Vocabulary Index', fontsize=12)
        ax.set_title('Token Embeddings in 3D Space', fontsize=14)
        
        plt.colorbar(sc, ax=ax, label='Vocabulary Index')
        plt.savefig(f"{self.output_dir}/token_embeddings_3d.png", dpi=300)
        plt.close()
        
        # Return for immediate display if needed
        return fig
    
    def visualize_attention_matrices(self, input_ids, attention_patterns):
        """Visualize attention patterns across layers"""
        # Get token strings for the first sequence
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        seq_len = len(tokens)
        
        # Limit to a reasonable number of tokens for visualization
        max_tokens = min(50, seq_len)
        tokens = tokens[:max_tokens]
        
        # Create a figure with subplots for each layer
        num_layers = len(attention_patterns)
        fig, axes = plt.subplots(1, num_layers, figsize=(num_layers * 5, 5))
        
        if num_layers == 1:
            axes = [axes]
        
        # Plot attention matrix for each layer
        for i, (ax, attention) in enumerate(zip(axes, attention_patterns)):
            # Extract attention for first batch, limit to max_tokens
            attn_matrix = attention[0, :max_tokens, :max_tokens].cpu().numpy()
            
            # Create heatmap
            im = ax.imshow(attn_matrix, cmap='Blues')
            
            # Add labels
            if i == 0:  # Only add y-labels for first subplot
                ax.set_yticks(np.arange(len(tokens)))
                ax.set_yticklabels(tokens)
            else:
                ax.set_yticks([])
            
            # Add x-labels with rotation for all subplots
            ax.set_xticks(np.arange(len(tokens)))
            ax.set_xticklabels(tokens, rotation=90)
            
            ax.set_title(f'Layer {i+1}')
        
        # Add colorbar
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Attention Weight')
        
        # Add overall title
        fig.suptitle('Attention Patterns Across Layers', fontsize=16)
        
        # Save figure
        plt.savefig(f"{self.output_dir}/attention_matrices.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Visualize attention flow
        self._visualize_attention_flow(tokens, attention_patterns[0][0, :max_tokens, :max_tokens].cpu().numpy())
        
        return fig
    
    def _visualize_attention_flow(self, tokens, attention_matrix):
        """Visualize attention flow between tokens"""
        # Create directed graph from attention matrix
        import networkx as nx
        
        # Create graph
        G = nx.DiGraph()
    
        # Add nodes
        for i, token in enumerate(tokens):
            G.add_node(i, label=token)
        
        # Add edges with attention weights
        max_idx = min(len(tokens), attention_matrix.shape[0], attention_matrix.shape[1])
        for i in range(max_idx):
            for j in range(max_idx):
                weight = attention_matrix[i, j]
                if weight > 0.01:  # Only add edges with significant attention
                    G.add_edge(i, j, weight=weight)
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Generate positions
        pos = nx.spring_layout(G, seed=42)
        
        # Get edge weights for width and color
        edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
        nx.draw_networkx_labels(G, pos, labels={i: token for i, token in enumerate(tokens)})
        edges = nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color=edge_weights,
                                      edge_cmap=plt.cm.Blues, arrows=True, 
                                      arrowsize=10, arrowstyle='->')
        
        # Add colorbar - FIX: Create a ScalarMappable with proper normalization
        # Get the current figure and axes
        fig = plt.gcf()
        ax = plt.gca()

        # Add colorbar with explicit axes reference
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, 
                                norm=plt.Normalize(vmin=min(edge_weights) if edge_weights else 0, 
                                                    vmax=max(edge_weights) if edge_weights else 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Attention Weight')
        
        plt.title('Attention Flow Between Tokens')
        plt.axis('off')
        
        # Save figure
        plt.savefig(f"{self.output_dir}/attention_flow.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_attention_sparsity(self, attention_patterns_list):
        """Analyze sparsity patterns across layers and samples"""
        # Collect sparsity data
        sparsity_data = []
        
        for sample_idx, attention_patterns in enumerate(attention_patterns_list):
            for layer_idx, attention in enumerate(attention_patterns):
                # Calculate sparsity
                nonzero = torch.count_nonzero(attention).item()
                total = torch.numel(attention)
                sparsity = 1.0 - (nonzero / total)
                
                # Calculate entropy (measure of concentration)
                attn_flat = attention.flatten(1)
                attn_norm = attn_flat / (torch.sum(attn_flat, dim=1, keepdim=True) + 1e-10)
                entropy = -torch.sum(attn_norm * torch.log2(attn_norm + 1e-10), dim=1).mean().item()
                
                # Calculate average superposition probability
                avg_superposition = torch.mean(attention[attention > 0]).item()
                
                sparsity_data.append({
                    'Sample': sample_idx,
                    'Layer': layer_idx + 1,
                    'Sparsity': sparsity,
                    'Entropy': entropy,
                    'Avg Superposition': avg_superposition,
                    'Active Connections': nonzero,
                    'Total Connections': total
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(sparsity_data)
        
        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Sparsity by layer
        sns.barplot(x='Layer', y='Sparsity', data=df, ax=ax1)
        ax1.set_title('Attention Sparsity by Layer')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Sparsity (1 - nonzero/total)')
        ax1.set_ylim(0, 1)
        
        # Entropy by layer
        sns.barplot(x='Layer', y='Entropy', data=df, ax=ax2)
        ax2.set_title('Attention Entropy by Layer')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Entropy (bits)')
        
        # Average superposition probability by layer
        sns.barplot(x='Layer', y='Avg Superposition', data=df, ax=ax3)
        ax3.set_title('Average Superposition Probability by Layer')
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('Superposition Probability')
        ax3.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/attention_sparsity.png", dpi=300)
        plt.close()
        
        # Relationship between sparsity and superposition
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Sparsity', y='Avg Superposition', hue='Layer', data=df, s=100)
        plt.title('Relationship Between Attention Sparsity and Superposition')
        plt.xlabel('Sparsity (1 - nonzero/total)')
        plt.ylabel('Average Superposition Probability')
        plt.savefig(f"{self.output_dir}/sparsity_superposition.png", dpi=300)
        plt.close()
        
        return df
    
    def visualize_phase_coherence(self, input_ids):
        """Visualize phase coherence between tokens"""
        # Process input through model to get states
        self.model.eval()
        
        # States at each layer
        states_by_layer = []
        
        # Forward pass to get states
        with torch.no_grad():
            # Initial states
            input_ids = input_ids.to(self.model.config.device)
            states = self.model.token_representation(input_ids)
            states_by_layer.append(states.cpu())
            
            # States after each layer
            for layer in self.model.layers:
                states, _ = layer(states)
                states_by_layer.append(states.cpu())
        
        # Extract angles and radii for first sequence
        angles_by_layer = []
        radii_by_layer = []
        
        for states in states_by_layer:
            # Extract for first sequence
            state = states[0]  # [N, 2]
            
            # Calculate angles and radii
            angles = torch.atan2(state[:, 1], state[:, 0]).cpu().numpy()  # [N]
            radii = torch.norm(state, dim=1).cpu().numpy()  # [N]
            
            angles_by_layer.append(angles)
            radii_by_layer.append(radii)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot angles progression
        for i, angles in enumerate(angles_by_layer):
            # Limit to reasonable sequence length
            max_tokens = min(50, len(angles))
            ax1.plot(np.arange(max_tokens), angles[:max_tokens], 
                    label=f'Layer {i}' if i > 0 else 'Input')
        
        ax1.set_xlabel('Token Position')
        ax1.set_ylabel('Phase Angle (radians)')
        ax1.set_title('Phase Angle Progression Across Layers')
        ax1.legend()
        ax1.grid(True)
        
        # Plot radii progression
        for i, radii in enumerate(radii_by_layer):
            # Limit to reasonable sequence length
            max_tokens = min(50, len(radii))
            ax2.plot(np.arange(max_tokens), radii[:max_tokens], 
                    label=f'Layer {i}' if i > 0 else 'Input')
        
        ax2.set_xlabel('Token Position')
        ax2.set_ylabel('Radius')
        ax2.set_title('Radius Progression Across Layers')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/phase_coherence.png", dpi=300)
        plt.close()
        
        # Calculate phase differences
        phase_diffs = []
        
        for angles in angles_by_layer:
            # Calculate differences between adjacent tokens
            diffs = np.diff(angles)
            # Normalize to [-π, π]
            diffs = np.arctan2(np.sin(diffs), np.cos(diffs))
            phase_diffs.append(diffs)
        
        # Create histogram of phase differences
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, diffs in enumerate(phase_diffs):
            sns.kdeplot(diffs, label=f'Layer {i}' if i > 0 else 'Input', ax=ax)
        
        ax.set_xlabel('Phase Difference Between Adjacent Tokens (radians)')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Phase Differences')
        ax.legend()
        ax.grid(True)
        
        plt.savefig(f"{self.output_dir}/phase_differences.png", dpi=300)
        plt.close()
        
        return fig
    
    def visualize_state_evolution(self, input_ids):
        """Visualize how token states evolve through layers"""
        # Process input through model to get states
        self.model.eval()
        
        # States at each layer
        states_by_layer = []
        
        # Forward pass to get states
        with torch.no_grad():
            # Initial states
            input_ids = input_ids.to(self.model.config.device)
            states = self.model.token_representation(input_ids)
            states_by_layer.append(states.cpu())
            
            # States after each layer
            for layer in self.model.layers:
                states, _ = layer(states)
                states_by_layer.append(states.cpu())
        
        # Get token strings for the first sequence
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        # Limit to a reasonable number of tokens
        max_tokens = min(10, len(tokens))
        tokens = tokens[:max_tokens]
        
        # Plot state evolution for each token
        for i, token in enumerate(tokens[:max_tokens]):
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Extract states for this token across layers
            x_values = []
            y_values = []
            
            for states in states_by_layer:
                # Get state for first sequence, token i
                x_values.append(states[0, i, 0].item())
                y_values.append(states[0, i, 1].item())
            
            # Plot evolution path
            ax.plot(x_values, y_values, 'o-', linewidth=2, markersize=8)
            
            # Add arrows to show direction
            for j in range(len(x_values) - 1):
                ax.annotate('', 
                           xy=(x_values[j+1], y_values[j+1]),
                           xytext=(x_values[j], y_values[j]),
                           arrowprops=dict(arrowstyle='->', lw=1.5, color='red'))
                
            # Add layer labels
            for j, (x, y) in enumerate(zip(x_values, y_values)):
                ax.annotate(f'L{j}' if j > 0 else 'In', 
                           (x, y),
                           textcoords="offset points",
                           xytext=(0, 10),
                           ha='center')
            
            # Add unit circle for reference
            circle = plt.Circle((0, 0), 1.0, fill=False, edgecolor='gray', linestyle='--')
            ax.add_patch(circle)
            
            # Equal aspect ratio
            ax.set_aspect('equal')
            
            # Add labels and title
            ax.set_xlabel('x = r*cos(θ)')
            ax.set_ylabel('y = r*sin(θ)')
            ax.set_title(f'State Evolution for Token "{token}"')
            ax.grid(True)
            
            # Save figure
            plt.savefig(f"{self.output_dir}/state_evolution_{i}_{token}.png", dpi=300)
            plt.close()
        
        # Visualize state evolution in 3D (layer, x, y)
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create colormap for tokens
        colors = plt.cm.tab10(np.linspace(0, 1, max_tokens))
        
        for i, token in enumerate(tokens[:max_tokens]):
            # Extract states for this token across layers
            x_values = []
            y_values = []
            z_values = []  # layer number
            
            for j, states in enumerate(states_by_layer):
                # Get state for first sequence, token i
                x_values.append(states[0, i, 0].item())
                y_values.append(states[0, i, 1].item())
                z_values.append(j)
            
            # Plot 3D path
            ax.plot(x_values, y_values, z_values, 'o-', linewidth=2, markersize=6, 
                   label=token, color=colors[i])
        
        # Add labels
        ax.set_xlabel('x = r*cos(θ)')
        ax.set_ylabel('y = r*sin(θ)')
        ax.set_zlabel('Layer')
        ax.set_title('3D State Evolution Through Layers')
        
        # Add legend
        ax.legend()
        
        # Save figure
        plt.savefig(f"{self.output_dir}/state_evolution_3d.png", dpi=300)
        plt.close()
        
        return fig


def visualize_model(model, tokenizer, dataloader, output_dir='visualizations'):
    """Helper function to run all visualizations"""
    visualizer = QFNNVisualizer(model, tokenizer, output_dir)
    visualizer.visualize_all(dataloader)
    print(f"Visualizations saved to {output_dir}")
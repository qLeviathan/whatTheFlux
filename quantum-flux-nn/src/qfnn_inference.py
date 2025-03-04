import torch
import torch.nn.functional as F
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from tqdm import tqdm

# Import QFNN model and visualizer
from qfnn_implementation import QFNN, QFNNConfig
from qfnn_visualizations import QFNNVisualizer


class QFNNInference:
    """Inference and analysis toolkit for QFNN models"""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Ensure model is in eval mode
        self.model.eval()
    
    def generate(self, prompt, max_length=50, temperature=1.0, top_k=50, top_p=0.9, 
                 repetition_penalty=1.2, return_attention=False):
        """Generate text using the QFNN model"""
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Track generated token ids
        generated_ids = input_ids[0].tolist()
        
        # Track attention patterns if requested
        all_attention_patterns = []
        
        # Generate tokens
        with torch.no_grad():
            for _ in tqdm(range(max_length), desc="Generating"):
                # Ensure we don't exceed model's max sequence length
                if len(generated_ids) >= self.model.config.max_seq_length:
                    input_ids = input_ids[:, -self.model.config.max_seq_length:]
                
                # Forward pass
                if return_attention:
                    outputs, attention_patterns = self.model(input_ids, return_attention=True)
                    all_attention_patterns.append([a.cpu() for a in attention_patterns])
                else:
                    outputs = self.model(input_ids)
                
                # Get logits for next token
                next_token_logits = outputs[0, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply repetition penalty
                for id in set(generated_ids):
                    next_token_logits[id] /= repetition_penalty
                
                # Filter with top-k
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Filter with top-p (nucleus sampling)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                # Add the token to the generated sequence
                generated_ids.append(next_token)
                
                # Update input_ids for next iteration
                input_ids = torch.tensor([generated_ids]).to(self.device)
                
                # Stop if we hit the end of text token
                if next_token == self.tokenizer.eos_token_id:
                    break
        
        # Decode the generated tokens
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        if return_attention:
            return generated_text, all_attention_patterns
        else:
            return generated_text
    
    def analyze_text(self, text, output_dir='inference_analysis'):
        """Analyze a text sample through the model"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Tokenize text
        input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
        token_strings = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        
        # Process through model
        with torch.no_grad():
            _, attention_patterns = self.model(input_ids, return_attention=True)
        
        # Create visualizer
        visualizer = QFNNVisualizer(self.model, self.tokenizer, output_dir)
        
        # Visualize attention patterns
        visualizer.visualize_attention_matrices(input_ids, attention_patterns)
        
        # Visualize phase coherence
        visualizer.visualize_phase_coherence(input_ids)
        
        # Visualize token state evolution
        visualizer.visualize_state_evolution(input_ids)
        
        # Analyze token radii changes through layers
        self._analyze_token_radii_progression(input_ids, token_strings, output_dir)
        
        # Analyze orthogonal relationships
        self._analyze_orthogonal_relationships(input_ids, token_strings, output_dir)
        
        print(f"Analysis complete. Results saved to {output_dir}")
        
        return {
            'attention_patterns': attention_patterns,
            'token_strings': token_strings
        }
    
    def _analyze_token_radii_progression(self, input_ids, token_strings, output_dir):
        """Analyze how token radii change through layers"""
        # Get states at each layer
        states_by_layer = []
        
        with torch.no_grad():
            # Initial states
            states = self.model.token_representation(input_ids)
            states_by_layer.append(states.cpu())
            
            # Process through layers
            for layer in self.model.layers:
                states, _ = layer(states)
                states_by_layer.append(states.cpu())
        
        # Extract radii for first sequence
        radii_by_layer = []
        for states in states_by_layer:
            # Calculate radii
            radii = torch.norm(states[0], dim=1).numpy()  # [N]
            radii_by_layer.append(radii)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Number of tokens to show (limit for readability)
        max_tokens = min(20, len(token_strings))
        
        # Plot radius progression for each token
        x = np.arange(len(radii_by_layer))
        for i in range(max_tokens):
            token_radii = [radii[i] for radii in radii_by_layer]
            ax.plot(x, token_radii, 'o-', label=token_strings[i])
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Radius')
        ax.set_title('Token Radius Progression Through Layers')
        ax.set_xticks(x)
        ax.set_xticklabels(['Input'] + [f'Layer {i+1}' for i in range(len(radii_by_layer)-1)])
        ax.grid(True)
        
        # Add legend with token labels
        if max_tokens <= 10:
            ax.legend()
        else:
            # For many tokens, put legend outside
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/token_radius_progression.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate radius change statistics
        radius_changes = []
        for i in range(1, len(radii_by_layer)):
            change = radii_by_layer[i] - radii_by_layer[i-1]
            radius_changes.append(change)
        
        # Create heatmap of radius changes
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Convert to array for heatmap
        changes_array = np.array(radius_changes)[:, :max_tokens]
        
        # Create heatmap
        im = ax.imshow(changes_array.T, cmap='RdBu_r', aspect='auto')
        
        # Add labels
        ax.set_xlabel('Layer Transition')
        ax.set_ylabel('Token')
        ax.set_title('Radius Changes Between Layers')
        
        ax.set_xticks(np.arange(len(radius_changes)))
        ax.set_xticklabels([f'L{i}→L{i+1}' for i in range(len(radius_changes))])
        
        ax.set_yticks(np.arange(max_tokens))
        ax.set_yticklabels(token_strings[:max_tokens])
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Radius Change')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/radius_changes_heatmap.png", dpi=300)
        plt.close()
    
    def _analyze_orthogonal_relationships(self, input_ids, token_strings, output_dir):
        """Analyze orthogonal relationships between tokens"""
        # Get final states after all layers
        with torch.no_grad():
            # Process through model
            states = self.model.token_representation(input_ids)
            
            for layer in self.model.layers:
                states, _ = layer(states)
        
        # Extract states for first sequence
        token_states = states[0].cpu().numpy()  # [N, 2]
        
        # Calculate orthogonal states
        orthogonal_states = np.zeros_like(token_states)
        orthogonal_states[:, 0] = token_states[:, 1]      # r*sin(θ)
        orthogonal_states[:, 1] = -token_states[:, 0]     # -r*cos(θ)
        
        # Number of tokens to show (limit for readability)
        max_tokens = min(20, len(token_strings))
        
        # Calculate direct and orthogonal similarities
        direct_sim = np.zeros((max_tokens, max_tokens))
        ortho_sim = np.zeros((max_tokens, max_tokens))
        
        for i in range(max_tokens):
            for j in range(max_tokens):
                # Direct similarity = dot product
                direct_sim[i, j] = np.dot(token_states[i], token_states[j])
                
                # Orthogonal similarity = dot product with orthogonal state
                ortho_sim[i, j] = np.dot(token_states[i], orthogonal_states[j])
        
        # Calculate phase distance
        phase_distance = np.sqrt(direct_sim**2 + ortho_sim**2 + 1e-8)
        
        # Calculate superposition probability
        max_distance = np.max(phase_distance)
        superposition_prob = 1.0 - (phase_distance / (max_distance + 1e-8))
        
        # Create similarity heatmaps
        fig, ax_grid = plt.subplots(2, 2, figsize=(18, 16))
        axs = ax_grid.flatten()
        
        # Direct similarity heatmap
        im1 = axs[0].imshow(direct_sim, cmap='Blues')
        axs[0].set_title('Direct Token Similarities (Real Component)')
        axs[0].set_xlabel('Token')
        axs[0].set_ylabel('Token')
        
        # Add token labels
        axs[0].set_xticks(np.arange(max_tokens))
        axs[0].set_yticks(np.arange(max_tokens))
        axs[0].set_xticklabels(token_strings[:max_tokens], rotation=90)
        axs[0].set_yticklabels(token_strings[:max_tokens])
        
        # Add colorbar
        plt.colorbar(im1, ax=axs[0], label='Direct Similarity')
        
        # Orthogonal similarity heatmap
        im2 = axs[1].imshow(ortho_sim, cmap='RdBu_r', vmin=-np.max(np.abs(ortho_sim)), vmax=np.max(np.abs(ortho_sim)))
        axs[1].set_title('Orthogonal Token Relationships (Complex Component)')
        axs[1].set_xlabel('Token')
        axs[1].set_ylabel('Token')
        
        # Add token labels
        axs[1].set_xticks(np.arange(max_tokens))
        axs[1].set_yticks(np.arange(max_tokens))
        axs[1].set_xticklabels(token_strings[:max_tokens], rotation=90)
        axs[1].set_yticklabels(token_strings[:max_tokens])
        
        # Add colorbar
        plt.colorbar(im2, ax=axs[1], label='Orthogonal Similarity')
        
        # Phase distance heatmap
        im3 = axs[2].imshow(phase_distance, cmap='viridis')
        axs[2].set_title('Phase Distance Between Tokens')
        axs[2].set_xlabel('Token')
        axs[2].set_ylabel('Token')
        
        # Add token labels
        axs[2].set_xticks(np.arange(max_tokens))
        axs[2].set_yticks(np.arange(max_tokens))
        axs[2].set_xticklabels(token_strings[:max_tokens], rotation=90)
        axs[2].set_yticklabels(token_strings[:max_tokens])
        
        # Add colorbar
        plt.colorbar(im3, ax=axs[2], label='Phase Distance')
        
        # Superposition probability heatmap
        im4 = axs[3].imshow(superposition_prob, cmap='plasma')
        axs[3].set_title('Superposition Probability Between Tokens')
        axs[3].set_xlabel('Token')
        axs[3].set_ylabel('Token')
        
        # Add token labels
        axs[3].set_xticks(np.arange(max_tokens))
        axs[3].set_yticks(np.arange(max_tokens))
        axs[3].set_xticklabels(token_strings[:max_tokens], rotation=90)
        axs[3].set_yticklabels(token_strings[:max_tokens])
        
        # Add colorbar
        plt.colorbar(im4, ax=axs[3], label='Superposition Probability')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/token_relationships.png", dpi=300)
        plt.close()
        
        # Calculate combined similarity with superposition weighting
        combined_sim = (direct_sim + self.model.config.alpha * ortho_sim) * superposition_prob
        
        # Apply threshold for sparsity
        combined_sim[superposition_prob < self.model.config.threshold] = 0
        
        # Create network graph visualization
        self._create_token_network_graph(combined_sim, token_strings[:max_tokens], output_dir)
    
    def _create_token_network_graph(self, similarity_matrix, token_labels, output_dir):
        """Create network graph visualization of token relationships"""
        import networkx as nx
        
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes
        for i, token in enumerate(token_labels):
            G.add_node(i, label=token)
        
        # Add edges with weights from similarity matrix
        for i in range(len(token_labels)):
            for j in range(len(token_labels)):
                weight = similarity_matrix[i, j]
                if weight != 0:  # Only add non-zero edges
                    G.add_edge(i, j, weight=weight)
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Generate positions using spring layout
        pos = nx.spring_layout(G, seed=42)
        
        # Get edge weights for width and color
        edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
        # Handle negative weights for coloring
        edge_colors = [w if w > 0 else 0 for w in edge_weights]
        edge_widths = [abs(w) for w in edge_weights]
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
        nx.draw_networkx_labels(G, pos, labels={i: label for i, label in enumerate(token_labels)})
        edges = nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors,
                                      edge_cmap=plt.cm.Blues, arrows=True, 
                                      arrowstyle='->', arrowsize=15)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(0, max(edge_colors)))
        sm.set_array([])
        plt.colorbar(sm, label='Similarity (Edge Weight)')
        
        plt.title('Token Relationship Network')
        plt.axis('off')
        
        # Save figure
        plt.savefig(f"{output_dir}/token_network.png", dpi=300, bbox_inches='tight')
        plt.close()


def load_qfnn_model(model_path, device='cuda'):
    """Load QFNN model from checkpoint"""
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create config
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        config = QFNNConfig(**config_dict)
    else:
        # Default config if not found in checkpoint
        config = QFNNConfig(device=device)
    
    # Create model
    model = QFNN(config).to(device)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Set to eval mode
    model.eval()
    
    return model, config


def main():
    """Main function for QFNN inference"""
    parser = argparse.ArgumentParser(description='Run inference with Quantum Flux Neural Network')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='inference_results', help='Output directory')
    parser.add_argument('--tokenizer', type=str, default='bert-base-uncased', help='Tokenizer to use')
    
    # Generation parameters
    parser.add_argument('--prompt', type=str, default='', help='Prompt for text generation')
    parser.add_argument('--input_file', type=str, default='', help='File containing text for analysis')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k filtering')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p (nucleus) filtering')
    
    # Analysis parameters
    parser.add_argument('--analyze', action='store_true', help='Run analysis on generated or input text')
    
    # Other parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model, config = load_qfnn_model(args.model_path, device=args.device)
    print(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    # Create inference engine
    inference = QFNNInference(model, tokenizer, device=args.device)
    
    # If prompt is provided, generate text
    if args.prompt:
        print(f"Generating text from prompt: {args.prompt}")
        generated_text, attention_patterns = inference.generate(
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            return_attention=args.analyze
        )
        
        print("\nGenerated text:")
        print("-" * 50)
        print(generated_text)
        print("-" * 50)
        
        # Save generated text
        with open(f"{args.output_dir}/generated_text.txt", 'w') as f:
            f.write(generated_text)
        
        # If analyze flag is set, analyze the generated text
        if args.analyze:
            print("Analyzing generated text...")
            inference.analyze_text(generated_text, output_dir=f"{args.output_dir}/analysis")
    
    # If input file is provided, analyze it
    elif args.input_file and args.analyze:
        print(f"Analyzing text from {args.input_file}")
        with open(args.input_file, 'r') as f:
            text = f.read()
        
        # Analyze the text
        inference.analyze_text(text, output_dir=f"{args.output_dir}/analysis")
    
    # If neither prompt nor input file is provided
    else:
        print("Please provide either a prompt for generation or an input file for analysis.")


if __name__ == "__main__":
    main()
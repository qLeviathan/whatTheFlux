Add wandb to trianing 
import wandb

# In your main function:
def main():
    # Parse args...
    
    # Initialize wandb
    wandb.init(project="quantum-flux-nn", name=f"qfnn_{args.num_layers}l_{args.hidden_dim}h", config=vars(args))
    
    # Train model
    model, stats, tokenizer = train_wikitext(
        config,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        context_length=args.context_length
    )
    
    # Close wandb
    wandb.finish()

# In your training loop:
def train(self, train_dataloader, val_dataloader, epochs, output_dir):
    # Setup...
    
    for epoch in range(epochs):
        # Training epoch
        for input_ids, target_ids in pbar:
            # Training step
            _, loss, attention_patterns = self.train_step(input_ids, target_ids)
            
            # Log metrics to wandb
            wandb.log({
                "train_loss": loss.item(),
                "train_ppl": math.exp(loss.item()),
                "global_step": global_step
            })
            
        # Evaluation
        val_loss, val_ppl, _ = self.evaluate(val_dataloader)
        wandb.log({
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "epoch": epoch
        })
        
        # Log visualizations to wandb
        if have_visualizations:
            wandb.log({
                "token_embeddings": wandb.Image(f"{vis_dir}/token_embeddings.png"),
                "attention_matrices": wandb.Image(f"{vis_dir}/attention_matrices.png"),
                "phase_coherence": wandb.Image(f"{vis_dir}/phase_coherence.png")
            })
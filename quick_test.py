#!/usr/bin/env python3
"""
Quick test to validate EEGConformer training
"""

import torch
from models import create_model
from config import *

def test_model():
    print("Testing EEGConformer with optimized parameters...")
    print(f"Embedding dim: {CONFORMER_EMBEDDING_DIM}")
    print(f"Num heads: {CONFORMER_NUM_HEADS}")
    print(f"Num layers: {CONFORMER_NUM_LAYERS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Dropout: {DROPOUT_RATE}")
    
    # Test model creation
    model = create_model(n_channels=32, model_name='EEGConformer')
    print(f"Model created: {type(model)}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Test with different batch sizes
    for batch_size in [8, 16, 32]:
        try:
            x = torch.randn(batch_size, 32, 128).to(device)  # batch, channels, time
            output = model(x)
            print(f"✅ Batch size {batch_size}: {x.shape} -> {output.shape}")
        except Exception as e:
            print(f"❌ Batch size {batch_size} failed: {e}")
    
    # Test training step
    print("\nTesting training step...")
    try:
        model.train()
        x = torch.randn(16, 32, 128).to(device)
        y = torch.randint(0, 2, (16,)).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = torch.nn.CrossEntropyLoss()
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        print(f"✅ Training step successful. Loss: {loss.item():.4f}")
        
        # Test prediction
        model.eval()
        with torch.no_grad():
            pred = model(x)
            accuracy = (pred.argmax(dim=1) == y).float().mean()
            print(f"Random accuracy: {accuracy.item():.4f}")
            
    except Exception as e:
        print(f"❌ Training step failed: {e}")

if __name__ == "__main__":
    test_model()
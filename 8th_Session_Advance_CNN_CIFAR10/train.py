import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from models.custom_net import CustomNet
from utils.data_loader import CIFAR10Albumentation
from utils.train_utils import train, test

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = CustomNet().to(device)
    
    # Print model summary
    print("\nModel Architecture:")
    summary(model, input_size=(3, 32, 32))
    
    # Get dataloaders
    data_loader = CIFAR10Albumentation()
    train_loader, test_loader = data_loader.get_dataloader()
    
    # Set training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.03,  # Initial learning rate
        momentum=0.95,  # Momentum factor
        weight_decay=5e-4  # L2 regularization
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.2,
        patience=2,
        verbose=True
    )
    
    # Training loop
    best_acc = 0
    for epoch in range(1, 41):  # 40 epochs
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, epoch)
        test_loss, test_acc = test(model, device, test_loader, criterion)
        
        # Update learning rate
        scheduler.step(test_acc)
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            
        print(f"Epoch {epoch}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

if __name__ == '__main__':
    main() 
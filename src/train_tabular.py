import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from tabular_model import TabularModel

def train_tabular():
    train_loader, test_loader, in_dim, cols = get_dataloaders('data/tabular/flood_risk_dataset_india.csv', 'data/images', batch_size=32)
    
    model = TabularModel(input_dim=in_dim, hidden_dim=16, output_dim=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for tab_features, _, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(tab_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
        
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for tab_features, _, labels in test_loader:
            outputs = model(tab_features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(f"Tabular Model Test Accuracy: {100 * correct / total:.2f}%")
    
    # Save the model
    torch.save(model.state_dict(), 'tabular_model.pth')
    print("Tabular model saved to tabular_model.pth")

if __name__ == "__main__":
    train_tabular()

import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from image_model import ImageModel

def train_image():
    train_loader, test_loader, _, _ = get_dataloaders('data/tabular/flood_risk_dataset_india.csv', 'data/images', batch_size=32)
    
    model = ImageModel(num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for _, images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
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
        for _, images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(f"Image Model Test Accuracy: {100 * correct / total:.2f}%")
    
    # Save model
    torch.save(model.state_dict(), 'image_model.pth')
    print("Image model saved to image_model.pth")

if __name__ == "__main__":
    train_image()

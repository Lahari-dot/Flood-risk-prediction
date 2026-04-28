
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from fusion_model import FusionModel

def train_fusion():
    train_loader, test_loader, in_dim, cols = get_dataloaders('data/tabular/flood_risk_dataset_india.csv', 'data/images', batch_size=32)
    
    model = FusionModel(tabular_input_dim=in_dim, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for tab_features, images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(tab_features, images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
        
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    from sklearn.metrics import f1_score, precision_score, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for tab_features, images, labels in test_loader:
            outputs = model(tab_features, images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"Fusion Model Test Accuracy: {accuracy:.2f}%")
    print(f"Fusion Model Precision: {precision:.4f}")
    print(f"Fusion Model F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Fusion Model')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Plot Metrics Bar Chart
    metrics = ['Accuracy', 'Precision', 'F1 Score']
    # Scaling to percentage for comparable bar heights
    values = [accuracy, precision * 100, f1 * 100] 
    
    plt.figure(figsize=(8, 5))
    plt.bar(metrics, values, color=['#4c72b0', '#dd8452', '#55a868'])
    plt.title('Fusion Model Performance Metrics')
    plt.ylabel('Score / Percentage (%)')
    plt.ylim(0, 105)
    for i, v in enumerate(values):
        plt.text(i, v + 2, f'{v:.2f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('performance_metrics.png')
    plt.close()
    
    print("Charts saved as 'confusion_matrix.png' and 'performance_metrics.png'")
    
    # Save model
    torch.save(model.state_dict(), 'fusion_model.pth')
    print("Fusion model saved to fusion_model.pth")

if __name__ == "__main__":
    train_fusion()

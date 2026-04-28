import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile

# Ensure truncated images don't crash the data loader
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MultimodalFloodDataset(Dataset):
    def __init__(self, tabular_path, image_dir, transform=None):
        """
        tabular_path: Path to the CSV file
        image_dir: Path to the directory containing images
        transform: PyTorch transforms for the image
        """
        # Load tabular data
        self.tabular_data = pd.read_csv(tabular_path)
        self.image_dir = image_dir
        self.transform = transform
        
        # Preprocessing Tabular Data
        # 1. Handle missing values
        self.tabular_data.fillna(self.tabular_data.mean(numeric_only=True), inplace=True)
        for col in self.tabular_data.select_dtypes(include=['object']):
            self.tabular_data[col].fillna(self.tabular_data[col].mode()[0], inplace=True)
        
        # 2. Extract features and labels
        target_col = 'Flood Occurred' if 'Flood Occurred' in self.tabular_data.columns else 'flood_risk'
        self.labels = self.tabular_data[target_col].values
        
        # Drop id if exists and target
        drop_cols = [col for col in ['id', target_col] if col in self.tabular_data.columns]
        feature_df = self.tabular_data.drop(columns=drop_cols)
        
        # One-hot encode categorical columns if any
        cat_cols = ['Land Cover', 'Soil Type']
        cat_cols = [c for c in cat_cols if c in feature_df.columns]
        if cat_cols:
            feature_df = pd.get_dummies(feature_df, columns=cat_cols)
            
        self.feature_columns = feature_df.columns.tolist()
        self.features = feature_df.values.astype(np.float32)
        
        # 3. Handle Images dynamically
        self.image_paths = []
        for root, dirs, files in os.walk(self.image_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                    self.image_paths.append(os.path.join(root, file))
                    
        if not self.image_paths:
            raise FileNotFoundError(f"No valid images found in {self.image_dir}")
            
        self.image_paths.sort() # Ensure consistent order
        
        # 4. Normalize features
        self.features = (self.features - np.mean(self.features, axis=0)) / (np.std(self.features, axis=0) + 1e-8)
        self.input_dim = self.features.shape[1]

    def __len__(self):
        return len(self.tabular_data)

    def __getitem__(self, idx):
        # 1. Get Tabular Features and Label
        tabular_features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # 2. Get Image from dynamically loaded paths
        img_idx = idx % len(self.image_paths)
        img_path = self.image_paths[img_idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform if none provided
            image = transforms.ToTensor()(image)
            
        return tabular_features, image, label

def get_dataloaders(tabular_path, image_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = MultimodalFloodDataset(tabular_path, image_dir, transform)
    
    # Split into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    # Set seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, dataset.input_dim, dataset.feature_columns

if __name__ == "__main__":
    t_loader, _, in_dim, cols = get_dataloaders('data/tabular/flood_risk_dataset_india.csv', 'data/images')
    tab, img, lbl = next(iter(t_loader))
    print(f"Tabular features shape: {tab.shape}")
    print(f"Input Dim extracted: {in_dim}")
    print(f"Image batch shape: {img.shape}")
    print(f"Labels shape: {lbl.shape}")

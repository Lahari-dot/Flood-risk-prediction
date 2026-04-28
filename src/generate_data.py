import os
import pandas as pd
import numpy as np
from PIL import Image

def generate_tabular_data(num_samples=1000):
    os.makedirs('data/tabular', exist_ok=True)
    np.random.seed(42)
    data = {
        'id': range(num_samples),
        'rainfall_mm': np.random.uniform(50, 500, num_samples),
        'river_water_level_m': np.random.uniform(10, 50, num_samples),
        'soil_moisture_index': np.random.uniform(0.1, 0.9, num_samples),
        'elevation_m': np.random.uniform(0, 1000, num_samples),
        'forest_coverage_pct': np.random.uniform(0, 100, num_samples),
        'flood_risk': np.random.randint(0, 2, num_samples) # 0: No Flood, 1: Flood
    }
    df = pd.DataFrame(data)
    
    # Introduce some missing values (NaN) to demonstrate preprocessing
    # Missing values in 'river_water_level_m'
    mask = np.random.rand(num_samples) < 0.05
    df.loc[mask, 'river_water_level_m'] = np.nan

    df.to_csv('data/tabular/flood_india.csv', index=False)
    print(f"Generated {num_samples} tabular records at data/tabular/flood_india.csv")

def generate_image_data(num_samples=1000, img_size=(64, 64)):
    os.makedirs('data/images', exist_ok=True)
    np.random.seed(42)
    for i in range(num_samples):
        # Generate random noise image simulating satellite radar or optic
        img_array = np.random.randint(0, 255, (img_size[0], img_size[1], 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(f'data/images/img_{i}.jpg')
    print(f"Generated {num_samples} images at data/images/")

if __name__ == '__main__':
    print("Generating synthetic data...")
    generate_tabular_data(1000)
    generate_image_data(1000)
    print("Data generation complete.")

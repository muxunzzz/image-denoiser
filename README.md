# Image Denoiser

A Python-based implementation of image denoising techniques using **Radial Basis Function (RBF) Regression** and **K-Means Clustering** to remove salt-and-pepper noise from images.

## Overview

This project demonstrates the application of machine learning techniques for image denoising. Images corrupted with salt-and-pepper noise are restored using two different approaches:

1. **RBF Regression with K-Means Clustering**: Uses spatial features to cluster clean pixels and reconstructs corrupted regions using radial basis functions.
2. **KNN Weighted Average Regression**: Employs k-nearest neighbors with spatial and color weighting to interpolate missing pixel values.

## Features

- **Salt-and-Pepper Noise Generation**: Add controllable noise to clean images
- **RBF Regression Model**: Custom implementation with L2 regularization
- **K-Means Clustering**: Automatic optimal cluster detection using the elbow method
- **KNN-Based Denoising**: Weighted averaging based on spatial and color proximity
- **Multiple Test Images**: Includes three sample images (cherry blossom, bridge, maple tree)
- **Interactive Jupyter Notebook**: Complete workflow with visualizations

## Algorithms Implemented

### 1. RBF Regression with K-Means Clustering
- Identifies clean pixels (non-corrupted by salt-and-pepper noise)
- Uses K-Means clustering to group clean pixels based on spatial features
- Derives cluster centers and widths for RBF basis functions
- Applies L2-regularized RBF regression to predict values for corrupted pixels

### 2. KNN Weighted Average Regression
- Identifies clean vs. corrupted pixels
- For each corrupted pixel, finds k-nearest clean neighbors
- Computes weights based on spatial distance and color similarity
- Reconstructs pixel values using weighted average of neighbors

## Requirements

```
PIL (Pillow)
matplotlib
numpy
scikit-learn
kneed
scipy
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/muxunzzz/image-denoiser.git
cd image-denoiser
```

2. Install required dependencies:
```bash
pip install pillow matplotlib numpy scikit-learn kneed scipy
```

## Usage

Open and run the Jupyter notebook:

```bash
jupyter notebook "Image Denoiser.ipynb"
```

### Basic Usage Example

```python
from PIL import Image
import numpy as np

# Load an image
img = Image.open('./cherry_blossom.png')
img = np.array(img) / 255.0

# Add salt-and-pepper noise
noisy_img = add_salt_and_pepper_noise(img, salt_prob=0.2, pepper_prob=0.2)

# Denoise using RBF Regression (Method 1)
denoised_img_rbf = denoiser1(noisy_img, l2_coeff=0.1, max_K=30)

# Denoise using KNN Weighted Average (Method 2)
denoised_img_knn = denoiser2(noisy_img, k_neighbors=10, sigma_spatial=3.0, sigma_color=0.2)
```

## Project Structure

```
image-denoiser/
│
├── Image Denoiser.ipynb    # Main Jupyter notebook with implementation
├── cherry_blossom.png      # Sample test image 1
├── bridge.jpg              # Sample test image 2
├── maple_tree.jpg          # Sample test image 3
└── README.md               # This file
```

## How It Works

### RBF Regression Method

1. **Noise Detection**: Identify salt (white) and pepper (black) pixels
2. **Clustering**: Apply K-Means to spatial coordinates of clean pixels
3. **RBF Setup**: Use cluster centers as RBF centers and cluster spread as widths
4. **Training**: Fit RBF weights using L2-regularized least squares
5. **Reconstruction**: Predict RGB values for corrupted pixels

### KNN Weighted Average Method

1. **Noise Detection**: Identify corrupted pixels
2. **Neighbor Search**: Build k-d tree for efficient nearest neighbor queries
3. **Weight Computation**: Calculate weights based on spatial and color distances using Gaussian kernels
4. **Interpolation**: Compute weighted average of k-nearest clean neighbors

## Parameters

### RBF Method (`denoiser1`)
- `l2_coeff`: L2 regularization coefficient (default: 0.1)
- `max_K`: Maximum clusters for elbow method (default: 30)

### KNN Method (`denoiser2`)
- `k_neighbors`: Number of nearest neighbors (default: 10)
- `sigma_spatial`: Spatial distance weighting parameter (default: 3.0)
- `sigma_color`: Color distance weighting parameter (default: 0.2)

## Results

The notebook demonstrates denoising on three different images with 40% total noise (20% salt + 20% pepper). Both methods effectively restore the original images while preserving important features and textures.

## Technical Details

- **Image Format**: RGB images normalized to [0, 1] range
- **Coordinate System**: 2D spatial coordinates (row, column)
- **Noise Model**: Binary salt (white pixels) and pepper (black pixels)
- **RBF Function**: Gaussian RBF with exponential decay

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## Author

Created by [muxunzzz](https://github.com/muxunzzz)

## Acknowledgments

This project demonstrates the practical application of machine learning techniques in computer vision, specifically for noise reduction in digital images.

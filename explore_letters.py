
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Path to the CSV file (update this path)
csv_path = "data/A_Z Handwritten Data.csv"

# Load dataset
data = pd.read_csv(csv_path)

# Separate labels and images
labels = data.iloc[:, 0].values.squeeze()
images = data.iloc[:, 1:].values
idx_shuffle = np.arange(len(labels))
np.random.shuffle(idx_shuffle)
labels = labels[idx_shuffle]
images = images[idx_shuffle]

# Function to convert numeric label to letter
def label_to_letter(label):
    return chr(ord('A') + label)

# Display a few images
num_samples = 9
plt.figure(figsize=(6, 6))

for i in range(num_samples):
    plt.subplot(3, 3, i + 1)
    
    img = images[i].reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.title(label_to_letter(labels[i]))
    plt.axis('off')

plt.tight_layout()
plt.show()
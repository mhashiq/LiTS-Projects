import SimpleITK as sitk
import os
import numpy as np

# Define paths for training and testing datasets
training_dataset_path = r'C:\Users\hasmy065\OneDrive - University of South Australia\Projects\LiTS-Dataset\Training_Batch1\media\nas\01_Datasets\CT\LITS\Training Batch 1'

# Function to analyze dataset
def analyze_dataset(dataset_path):
    class_distribution = {}
    total_files = 0
    total_voxels = 0

    for filename in os.listdir(dataset_path):
        if filename.endswith('.nii') or filename.endswith('.nii.gz'):  # Check for NIfTI files
            total_files += 1
            file_path = os.path.join(dataset_path, filename)
            
            # Load the image using SimpleITK
            image = sitk.ReadImage(file_path)
            
            # Convert to a NumPy array
            image_array = sitk.GetArrayFromImage(image)
            
            # Update total voxel count
            total_voxels += image_array.size
            
            # Flatten the array to count unique intensity values
            unique, counts = np.unique(image_array, return_counts=True)
            
            # Update the class distribution
            for u, c in zip(unique, counts):
                if u in class_distribution:
                    class_distribution[u] += c
                else:
                    class_distribution[u] = c

    # Print dataset description
    print("Dataset Description:")
    print(f"Total Files: {total_files}")
    print(f"Total Voxels: {total_voxels}")
    print(f"Class-wise Distribution:")
    for intensity, count in sorted(class_distribution.items()):
        print(f"  Intensity {intensity}: {count} voxels")

# Analyze the training dataset
analyze_dataset(training_dataset_path)
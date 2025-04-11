import SimpleITK as sitk
import matplotlib.pyplot as plt
import os

# Define paths for training and testing datasets
training_dataset_path = r'C:\Users\hasmy065\OneDrive - University of South Australia\Projects\LiTS-Dataset\Training_Batch1\media\nas\01_Datasets\CT\LITS\Training Batch 1'
testing_dataset_path = r'C:\Users\hasmy065\OneDrive - University of South Australia\Projects\LiTS-Dataset\Testing_Batch1\media\nas\01_Datasets\CT\LITS\Testing Batch 1'

# Choose which dataset to process
dataset_path = training_dataset_path  # Change to testing_dataset_path if needed

# Iterate through all files in the directory
for filename in os.listdir(dataset_path):
    if filename.endswith('.nii') or filename.endswith('.nii.gz'):  # Check for NIfTI files
        file_path = os.path.join(dataset_path, filename)
        
        # Load the image using SimpleITK
        image = sitk.ReadImage(file_path)
        
        # Convert to a NumPy array
        image_array = sitk.GetArrayFromImage(image)
        
        # Visualize the middle slice (change slice index if needed)
        middle_slice = image_array[image_array.shape[0] // 2]
        
        # Plot the middle slice
        plt.imshow(middle_slice, cmap='gray')
        plt.title(f'Middle Slice of {filename}')
        plt.axis('off')
        plt.show()
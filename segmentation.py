import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import label, find_objects

# Define paths for training and testing datasets
training_dataset_path = r'C:\Users\hasmy065\OneDrive - University of South Australia\Projects\LiTS-Dataset\Training_Batch1\media\nas\01_Datasets\CT\LITS\Training Batch 1'

# Function to segment liver using thresholding
def segment_liver(image_array):
    # Adjust thresholds based on the printed intensity range
    liver_mask = (image_array > 100) & (image_array < 300)  # Example adjusted thresholds
    return liver_mask

# Iterate through all files in the directory
for filename in os.listdir(training_dataset_path):
    if filename.endswith('.nii') or filename.endswith('.nii.gz'):  # Check for NIfTI files
        file_path = os.path.join(training_dataset_path, filename)
        
        # Load the image using SimpleITK
        image = sitk.ReadImage(file_path)
        
        # Convert to a NumPy array
        image_array = sitk.GetArrayFromImage(image)
        
        # Segment the liver
        liver_mask = segment_liver(image_array)
        
        # Visualize the original middle slice
        original_middle_slice = image_array[image_array.shape[0] // 2]
        plt.imshow(original_middle_slice, cmap='gray')
        plt.title(f'Original Middle Slice of {filename}')
        plt.axis('off')
        plt.show()
        
        # Visualize the liver mask middle slice
        middle_slice = liver_mask[liver_mask.shape[0] // 2]
        
        # Find connected components and bounding boxes
        labeled_array, num_features = label(middle_slice)
        slices = find_objects(labeled_array)
        
        # Plot the liver mask with bounding boxes
        plt.imshow(middle_slice, cmap='gray')
        for s in slices:
            y_min, y_max = s[0].start, s[0].stop
            x_min, x_max = s[1].start, s[1].stop
            plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                              edgecolor='red', facecolor='none', lw=2))
        plt.title(f'Liver Segmentation with Bounding Box of {filename}')
        plt.axis('off')
        plt.show()
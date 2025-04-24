import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# === Set path to the folder containing all segmentations ===
data_folder = r'C:\Users\hasmy065\OneDrive - University of South Australia\Projects\LiTS-Dataset\Training Dataset'

# === List all segmentation files ===
segmentation_files = sorted([f for f in os.listdir(data_folder) if 'segmentation' in f and (f.endswith('.nii') or f.endswith('.nii.gz'))])

# === Function to analyze liver size from segmentation files ===
def analyze_liver_from_segmentations(segmentation_files, data_folder):
    for i, seg_file in enumerate(segmentation_files):
        seg_path = os.path.join(data_folder, seg_file)

        # Load the segmentation file
        seg_img = nib.load(seg_path)
        segmentation = seg_img.get_fdata()

        # Extract voxel size from the header
        voxel_size = np.prod(seg_img.header.get_zooms())  # Voxel size in mm³

        # Extract the liver region (assuming label 1 corresponds to the liver)
        liver_region = segmentation == 1

        # Calculate liver volume
        liver_volume = np.sum(liver_region) * voxel_size / 1000  # Convert to mL

        # Calculate liver height and width from the segmented region
        liver_coords = np.argwhere(liver_region)
        if liver_coords.size > 0:
            min_row, max_row = liver_coords[:, 0].min(), liver_coords[:, 0].max()
            min_col, max_col = liver_coords[:, 1].min(), liver_coords[:, 1].max()

            # Liver height and width in pixels
            liver_height = max_row - min_row + 1  # Add 1 to include both endpoints
            liver_width = max_col - min_col + 1  # Add 1 to include both endpoints
        else:
            liver_height = 0
            liver_width = 0

        # Display the results visually
        plt.figure(figsize=(10, 6))
        plt.imshow(segmentation[:, :, segmentation.shape[2] // 2] == 1, cmap='jet')

        # Add bounding box and lines for height and width
        if liver_coords.size > 0:
            # Draw the bounding box
            rect = Rectangle((min_col, min_row), liver_width, liver_height,
                             linewidth=2, edgecolor='yellow', facecolor='none')
            plt.gca().add_patch(rect)

            # Add height and width lines
            plt.plot([min_col, max_col], [min_row, min_row], color='cyan', linestyle='--', linewidth=2, label='Width')
            plt.plot([min_col, min_col], [min_row, max_row], color='magenta', linestyle='--', linewidth=2, label='Height')

            # Annotate the height and width
            plt.annotate(f"Height: {liver_height} px", xy=(min_col, (min_row + max_row) // 2),
                         xytext=(min_col - 50, (min_row + max_row) // 2),
                         arrowprops=dict(facecolor='green', shrink=0.05),
                         fontsize=10, color='white')
            plt.annotate(f"Width: {liver_width} px", xy=((min_col + max_col) // 2, min_row),
                         xytext=((min_col + max_col) // 2, min_row - 20),
                         arrowprops=dict(facecolor='green', shrink=0.05),
                         fontsize=10, color='white')

        # Add title with liver size information
        plt.title(f"Segmentation: {seg_file}\nLiver Volume: {liver_volume:.2f} mL\n"
                  f"Liver Height: {liver_height} px, Liver Width: {liver_width} px")
        plt.axis('off')
        plt.show()

# === Run the analysis ===
analyze_liver_from_segmentations(segmentation_files, data_folder)
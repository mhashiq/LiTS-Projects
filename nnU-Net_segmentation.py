import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# === Set path to the folder containing all volumes and segmentations ===
data_folder = r'C:\Users\hasmy065\OneDrive - University of South Australia\Projects\LiTS-Dataset\Training Dataset'

# === Set path to the output folder ===
output_folder = r'C:\Users\hasmy065\OneDrive - University of South Australia\Projects\LiTS-Dataset\Merged_Images'
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

# === List all volume and segmentation files ===
files = sorted([f for f in os.listdir(data_folder) if f.endswith('.nii') or f.endswith('.nii.gz')])

# Separate volume and segmentation files
volume_files = [f for f in files if 'volume' in f]  # e.g., 'volume-0.nii'
seg_files = [f for f in files if 'segmentation' in f]  # e.g., 'segmentation-0.nii']

# Sort them to ensure proper pairing
volume_files.sort()
seg_files.sort()

# === Check if volumes and segmentations match in number ===
if len(volume_files) != len(seg_files):
    raise ValueError("Mismatch in the number of volume and segmentation files!")

# === Function to visualize up to 10 pairs in a single file ===
def visualize_all_pairs(volume_files, seg_files, data_folder):
    num_pairs = min(10, len(volume_files))  # Limit to 10 pairs or fewer if fewer files exist
    fig, axes = plt.subplots(num_pairs, 3, figsize=(18, 6 * num_pairs))
    fig.suptitle("CT Volumes, Segmentation Masks, and Overlays", fontsize=16)

    for i in range(num_pairs):
        vol_file = os.path.join(data_folder, volume_files[i])
        seg_file = os.path.join(data_folder, seg_files[i])

        vol_img = nib.load(vol_file)
        seg_img = nib.load(seg_file)

        volume = vol_img.get_fdata()
        segmentation = seg_img.get_fdata()

        slice_idx = volume.shape[2] // 2  # Choose middle slice for visualization

        # CT Image
        axes[i, 0].imshow(volume[:, :, slice_idx], cmap='gray')
        axes[i, 0].set_title(f'CT Volume {i+1}')
        axes[i, 0].axis('off')

        # Segmentation
        axes[i, 1].imshow(segmentation[:, :, slice_idx], cmap='jet')
        axes[i, 1].set_title(f'Segmentation Mask {i+1}')
        axes[i, 1].axis('off')

        # Overlay (CT + Segmentation)
        axes[i, 2].imshow(volume[:, :, slice_idx], cmap='gray')
        axes[i, 2].imshow(segmentation[:, :, slice_idx], cmap='jet', alpha=0.5)
        axes[i, 2].set_title(f'Overlay {i+1}')
        axes[i, 2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title
    output_file = os.path.join(data_folder, 'top_10_pairs_visualization.png')
    plt.savefig(output_file, dpi=300)  # Save with high resolution
    plt.show()
    print(f"Visualization saved to: {output_file}")

# === Function to export merged images ===
def export_merged_images(volume_files, seg_files, data_folder, output_folder):
    for i in range(len(volume_files)):
        vol_file = os.path.join(data_folder, volume_files[i])
        seg_file = os.path.join(data_folder, seg_files[i])

        vol_img = nib.load(vol_file)
        seg_img = nib.load(seg_file)

        volume = vol_img.get_fdata()
        segmentation = seg_img.get_fdata()

        slice_idx = volume.shape[2] // 2  # Choose middle slice for visualization

        # Create a figure for the merged image
        plt.figure(figsize=(6, 6))
        plt.imshow(volume[:, :, slice_idx], cmap='gray')
        plt.imshow(segmentation[:, :, slice_idx], cmap='jet', alpha=0.5)
        plt.axis('off')  # Remove axes for a clean image

        # Save the merged image
        output_file = os.path.join(output_folder, f'Merged-{i}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0)  # Save without extra padding
        plt.close()  # Close the figure to free memory

    print(f"All merged images have been saved to: {output_folder}")

# === Visualize the first 10 pairs in a single file ===
visualize_all_pairs(volume_files, seg_files, data_folder)

# === Export all merged images ===
export_merged_images(volume_files, seg_files, data_folder, output_folder)


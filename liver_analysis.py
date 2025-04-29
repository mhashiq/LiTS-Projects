import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# === Set path to the folder containing all volumes and segmentations ===
data_folder = r'C:\Users\hasmy065\OneDrive - University of South Australia\Projects\LiTS-Dataset\Training Dataset'

# === Set path to the output folder ===
output_folder = r'C:\Users\hasmy065\OneDrive - University of South Australia\Projects\LiTS-Dataset\Liver_Analysis'
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

# === List all volume and segmentation files ===
volume_files = sorted([f for f in os.listdir(data_folder) if 'volume' in f and (f.endswith('.nii') or f.endswith('.nii.gz'))])
segmentation_files = sorted([f for f in os.listdir(data_folder) if 'segmentation' in f and (f.endswith('.nii') or f.endswith('.nii.gz'))])

# === Match volume and segmentation files ===
if len(volume_files) != len(segmentation_files):
    raise ValueError("Mismatch in the number of volume and segmentation files!")

# === Function to analyze liver size and HU values ===
def analyze_liver(volume_files, seg_files, data_folder, output_folder):
    for i in range(len(volume_files)):
        vol_file = os.path.join(data_folder, volume_files[i])
        seg_file = os.path.join(data_folder, seg_files[i])

        vol_img = nib.load(vol_file)
        seg_img = nib.load(seg_file)

        volume = vol_img.get_fdata()
        segmentation = seg_img.get_fdata()

        # Extract the liver region (assuming label 1 corresponds to the liver)
        liver_region = segmentation == 1
        liver_hu_values = volume[liver_region]

        # Calculate liver size
        voxel_size = np.prod(vol_img.header.get_zooms())  # Voxel size in mm³
        liver_volume = np.sum(liver_region) * voxel_size / 1000  # Convert to mL

        # Calculate HU statistics
        mean_hu = np.mean(liver_hu_values)
        std_hu = np.std(liver_hu_values)

        # Determine if the liver is normal or abnormal based on size
        liver_status = "Normal" if 1200 <= liver_volume <= 1600 else "Abnormal"

        # Diagnosis based on HU values
        if mean_hu < 40:
            diagnosis = "Fatty Liver (Low HU)"
        elif 40 <= mean_hu <= 70:
            diagnosis = "Healthy Liver (Normal HU)"
        else:
            diagnosis = "Iron Overload or Fibrosis (High HU)"

        # Choose the middle slice for visualization
        slice_idx = volume.shape[2] // 2

        # === Visualization ===
        plt.figure(figsize=(18, 6))

        # Panel 1: Segmented Liver Photo
        plt.subplot(1, 4, 1)
        plt.imshow(segmentation[:, :, slice_idx] == 1, cmap='jet')
        plt.title("Segmented Liver")
        plt.axis('off')

        # Panel 2: Merged Photo with Liver Overlay
        plt.subplot(1, 4, 2)
        plt.imshow(volume[:, :, slice_idx], cmap='gray')
        plt.imshow(segmentation[:, :, slice_idx] == 1, cmap='jet', alpha=0.5)
        plt.title(f"CT Slice with Liver Overlay\nLiver Status: {liver_status}")
        plt.axis('off')

        # Panel 3: Liver Size and Status
        plt.subplot(1, 4, 3)
        plt.text(0.5, 0.5, f"Liver Size:\n{liver_volume:.2f} mL\nStatus: {liver_status}", fontsize=14, ha='center', va='center')
        plt.axis('off')

        # Panel 4: Histogram of HU Values with Diagnosis
        plt.subplot(1, 4, 4)
        plt.hist(liver_hu_values, bins=50, color='blue', alpha=0.7)
        plt.title(f"HU Distribution\nMean HU: {mean_hu:.2f}\nDiagnosis: {diagnosis}")
        plt.xlabel("Hounsfield Units (HU)")
        plt.ylabel("Frequency")

        # Show the visualization before saving
        plt.tight_layout()
        plt.show()

        # Save the output image
        output_file = os.path.join(output_folder, f'Liver_Analysis-{i}.png')
        plt.savefig(output_file, dpi=300)
        plt.close()

        print(f"Liver analysis saved to: {output_file}")

# === Run the analysis ===
analyze_liver(volume_files, segmentation_files, data_folder, output_folder)
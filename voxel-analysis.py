import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# === Path to the dataset folder ===
data_folder = r'C:\Users\hasmy065\OneDrive - University of South Australia\Projects\LiTS-Dataset\Training Dataset'

# === List all volume and segmentation files ===
volume_files = sorted([f for f in os.listdir(data_folder) if 'volume' in f and (f.endswith('.nii') or f.endswith('.nii.gz'))])
segmentation_files = sorted([f for f in os.listdir(data_folder) if 'segmentation' in f and (f.endswith('.nii') or f.endswith('.nii.gz'))])

# === Match volume and segmentation files ===
if len(volume_files) != len(segmentation_files):
    print("Warning: The number of volume and segmentation files does not match!")
else:
    print(f"Found {len(volume_files)} datasets.")

# === Automatically select the first dataset ===
dataset_index = 0  # Change this index to select a different dataset (e.g., 1 for the second dataset)

if 0 <= dataset_index < len(volume_files):
    selected_volume = os.path.join(data_folder, volume_files[dataset_index])
    selected_segmentation = os.path.join(data_folder, segmentation_files[dataset_index])
    print(f"\nAutomatically Selected Dataset:\n  Volume: {selected_volume}\n  Segmentation: {selected_segmentation}")
else:
    print("Invalid dataset index. Please check the dataset folder.")

# === Load the selected dataset ===
print("\nLoading the selected dataset...")
ct_scan = nib.load(selected_volume)
segmentation = nib.load(selected_segmentation)

ct_data = ct_scan.get_fdata()
seg_data = segmentation.get_fdata()

# === Analyze a segmented region (e.g., tumor) ===
tumor_region = seg_data == 2  # Assuming label 2 corresponds to the tumor
tumor_hu_values = ct_data[tumor_region]

# === Compute HU statistics ===
mean_hu = np.mean(tumor_hu_values)
std_hu = np.std(tumor_hu_values)
min_hu = np.min(tumor_hu_values)
max_hu = np.max(tumor_hu_values)

print(f"\nTumor HU Statistics:")
print(f"  Mean HU: {mean_hu:.2f}")
print(f"  Standard Deviation: {std_hu:.2f}")
print(f"  Min HU: {min_hu:.2f}")
print(f"  Max HU: {max_hu:.2f}")

# === Calculate tumor volume ===
voxel_size = np.prod(ct_scan.header.get_zooms())  # Voxel size in mm³
tumor_volume = np.sum(tumor_region) * voxel_size / 1000  # Convert to mL
print(f"Tumor Volume: {tumor_volume:.2f} mL")

# === Visualization: CT Slice and Histogram ===
plt.figure(figsize=(12, 6))

# Display the middle slice with segmentation overlay
z_center = ct_data.shape[2] // 2  # Middle slice index
plt.subplot(1, 2, 1)
plt.imshow(ct_data[:, :, z_center], cmap='gray')
plt.imshow(seg_data[:, :, z_center], cmap='jet', alpha=0.5)
plt.title(f"CT Slice with Tumor Overlay\nMean HU: {mean_hu:.2f}, Volume: {tumor_volume:.2f} mL")
plt.axis('off')

# Display the histogram of HU values
plt.subplot(1, 2, 2)
plt.hist(tumor_hu_values, bins=50, color='blue', alpha=0.7)
plt.title("Tumor HU Distribution")
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")

# Save the output image
output_path = os.path.join(data_folder, "tumor_analysis_output.png")
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.show()

print(f"\nAnalysis output saved to: {output_path}")
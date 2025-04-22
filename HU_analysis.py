import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from pyfeats import glcm_features

print("Dependencies loaded successfully!")

# === Path setup ===
data_folder = r'C:\Users\hasmy065\OneDrive - University of South Australia\Projects\LiTS-Dataset\Training Dataset'

# === List all segmentation files ===
seg_files = sorted([f for f in os.listdir(data_folder) if 'segmentation' in f and (f.endswith('.nii') or f.endswith('.nii.gz'))])

# === Texture features using PyFeats ===
def compute_texture_features(region):
    try:
        region_uint8 = region.astype(np.uint8)
        features, labels = glcm_features(region_uint8, distance=1, angle=0, levels=256, symmetric=True, normed=True)

        # Extract selected features
        contrast = features[0]           # Angular Second Moment (ASM)
        dissimilarity = features[1]      # Contrast
        homogeneity = features[4]        # Inverse Difference Moment (Homogeneity)
        energy = features[5]             # Sum of Squares: Variance
        correlation = features[8]        # Correlation

        return contrast, dissimilarity, homogeneity, energy, correlation
    except Exception as e:
        print(f"Error computing texture features: {e}")
        return None, None, None, None, None

# === Analyze HU and lesion features ===
def analyze_hu_and_lesion_features(segmentation_path, volume_path=None):
    try:
        seg_img = nib.load(segmentation_path)
        segmentation = seg_img.get_fdata()

        # Label connected components
        labeled_segmentation, num_features = ndimage.label(segmentation > 0)
        objects = ndimage.find_objects(labeled_segmentation)

        lesion_stats = []

        for i, slice_tuple in enumerate(objects):
            lesion_region = labeled_segmentation[slice_tuple] == (i + 1)

            if volume_path:
                vol_img = nib.load(volume_path)
                volume = vol_img.get_fdata()
                lesion_volume = volume[slice_tuple]
                hu_values = lesion_volume[lesion_region]

                lesion_mean_hu = np.mean(hu_values)
                lesion_std_hu = np.std(hu_values)
                lesion_min_hu = np.min(hu_values)
                lesion_max_hu = np.max(hu_values)

                lesion_stats.append({
                    'mean_hu': lesion_mean_hu,
                    'std_hu': lesion_std_hu,
                    'min_hu': lesion_min_hu,
                    'max_hu': lesion_max_hu
                })

                # Histogram
                plt.figure(figsize=(8, 6))
                plt.hist(hu_values.flatten(), bins=50, color='blue', alpha=0.7)
                plt.title(f'Lesion HU Distribution (Mean HU: {lesion_mean_hu:.2f})')
                plt.xlabel('Hounsfield Units (HU)')
                plt.ylabel('Frequency')
                plt.show()

                # Classification
                if lesion_mean_hu < 0:
                    lesion_type = 'Cyst (Low HU)'
                elif lesion_mean_hu < 50:
                    lesion_type = 'Benign Tumor (Low HU)'
                else:
                    lesion_type = 'Malignant Tumor (High HU)'
                print(f"Lesion Type: {lesion_type}")

            # Texture from the central slice
            lesion_3d = lesion_region.astype(np.uint8)
            z_center = lesion_3d.shape[2] // 2
            lesion_2d = lesion_3d[:, :, z_center]

            contrast, dissimilarity, homogeneity, energy, correlation = compute_texture_features(lesion_2d)

            print(f"Lesion {i + 1} - Texture Features:")
            print(f"  Contrast: {contrast:.2f}")
            print(f"  Dissimilarity: {dissimilarity:.2f}")
            print(f"  Homogeneity: {homogeneity:.2f}")
            print(f"  Energy: {energy:.2f}")
            print(f"  Correlation: {correlation:.2f}")

        return lesion_stats

    except Exception as e:
        print(f"Error analyzing HU and lesion features: {e}")
        return []

# === Analyze HU and lesion features with visualization ===
def analyze_hu_and_lesion_features_with_visualization(segmentation_path, volume_path=None):
    try:
        seg_img = nib.load(segmentation_path)
        segmentation = seg_img.get_fdata()

        # Label connected components
        labeled_segmentation, num_features = ndimage.label(segmentation > 0)
        objects = ndimage.find_objects(labeled_segmentation)

        for i, slice_tuple in enumerate(objects):
            lesion_region = labeled_segmentation[slice_tuple] == (i + 1)

            if volume_path:
                vol_img = nib.load(volume_path)
                volume = vol_img.get_fdata()
                lesion_volume = volume[slice_tuple]
                hu_values = lesion_volume[lesion_region]

                lesion_mean_hu = np.mean(hu_values)
                lesion_std_hu = np.std(hu_values)
                lesion_min_hu = np.min(hu_values)
                lesion_max_hu = np.max(hu_values)

                # === Visualization ===
                plt.figure(figsize=(12, 6))

                # Display the middle slice with segmentation overlay
                z_center = lesion_volume.shape[2] // 2
                plt.subplot(1, 2, 1)
                plt.imshow(volume[:, :, z_center], cmap='gray')
                plt.imshow(segmentation[:, :, z_center], cmap='jet', alpha=0.5)
                plt.title(f"Lesion {i + 1} - Mean HU: {lesion_mean_hu:.2f}")
                plt.axis('off')

                # Display the histogram of HU values
                plt.subplot(1, 2, 2)
                plt.hist(hu_values.flatten(), bins=50, color='blue', alpha=0.7)
                plt.title(f"Lesion {i + 1} - HU Distribution")
                plt.xlabel('Hounsfield Units (HU)')
                plt.ylabel('Frequency')

                # Interpretation of HU values
                if lesion_mean_hu < 0:
                    lesion_type = 'Cyst (Low HU)'
                elif lesion_mean_hu < 50:
                    lesion_type = 'Benign Tumor (Low HU)'
                else:
                    lesion_type = 'Malignant Tumor (High HU)'
                plt.figtext(0.5, 0.01, f"Lesion Type: {lesion_type}", wrap=True, horizontalalignment='center', fontsize=12)

                plt.tight_layout()
                plt.show()

    except Exception as e:
        print(f"Error analyzing HU and lesion features: {e}")

# === Identify tumor images ===
def identify_tumor_images(seg_files, data_folder):
    tumor_files = []
    for seg_file in seg_files:
        seg_path = os.path.join(data_folder, seg_file)
        seg_img = nib.load(seg_path)
        segmentation = seg_img.get_fdata()
        if 2 in np.unique(segmentation):
            tumor_files.append(seg_file)
    return tumor_files

# === Main loop ===
def main():
    tumor_files = identify_tumor_images(seg_files, data_folder)
    print(f"Found {len(tumor_files)} segmentation files with tumors:")
    for tumor_file in tumor_files:
        print(tumor_file)

    for seg_file in seg_files:
        print(f"\nAnalyzing {seg_file}...")
        segmentation_path = os.path.join(data_folder, seg_file)
        volume_path = segmentation_path.replace('segmentation', 'volume')
        analyze_hu_and_lesion_features_with_visualization(segmentation_path, volume_path)

if __name__ == "__main__":
    main()

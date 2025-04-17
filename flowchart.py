import os
import nibabel as nib
import SimpleITK as sitk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from skimage.measure import regionprops
from skimage import filters
from sklearn.preprocessing import StandardScaler

# Step 1: Data Collection
# Assuming CT scans are in NII format and stored in 'data/' directory
nii_files = [f for f in os.listdir('data/') if f.endswith('.nii')]

# Step 2: Preprocessing
def preprocess_ct_scan(file_path):
    # Load NII format file
    img = nib.load(file_path)
    img_data = img.get_fdata()

    # Noise reduction (N4 bias field correction)
    sitk_img = sitk.GetImageFromArray(img_data)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_img = corrector.Execute(sitk_img)
    corrected_img_data = sitk.GetArrayFromImage(corrected_img)

    # Normalize intensity (optional)
    corrected_img_data = (corrected_img_data - corrected_img_data.min()) / (corrected_img_data.max() - corrected_img_data.min())

    return corrected_img_data

# Step 3: HU Measurement (Region of Interest - ROI)
def get_hu_values(ct_img_data):
    # Example ROI placement: Liver parenchyma (manually segmented or using nnU-Net/3D Slicer)
    liver_roi = ct_img_data[50:150, 50:150, 20:40]  # Placeholder for liver ROI
    spleen_roi = ct_img_data[100:200, 100:200, 50:70]  # Placeholder for spleen ROI

    # Calculate mean HU for liver and spleen
    mean_liver_hu = np.mean(liver_roi)
    mean_spleen_hu = np.mean(spleen_roi)

    # Liver-spleen HU difference
    l_s_hu_difference = mean_liver_hu - mean_spleen_hu

    return mean_liver_hu, l_s_hu_difference

# Step 4: Pathological Classification
def classify_pathology(mean_liver_hu, l_s_hu_difference):
    # Define classification criteria (placeholder for thresholds)
    if mean_liver_hu < 40 or l_s_hu_difference <= 5:
        pathology = 'Steatosis'
    elif 45 <= mean_liver_hu <= 55 and l_s_hu_difference > 5:
        pathology = 'Fibrosis'
    elif 50 <= mean_liver_hu <= 55 and l_s_hu_difference > 5:
        pathology = 'Cirrhosis'
    else:
        pathology = 'Normal'

    return pathology

# Step 5: Validation (Optional validation with external data, e.g., biopsy, MRI-PDFF)
def validate_with_external_data(predictions, true_labels):
    fpr, tpr, thresholds = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

# Step 6: Biomarker Development
def develop_biomarker(liver_hu, l_s_hu_diff):
    # Example of simple biomarker development
    biomarker = {
        'Steatosis HU Threshold': liver_hu < 40,
        'Fibrosis HU + APRI/FIB-4 Index': liver_hu > 45 and liver_hu < 55,
        'Radiomics Model Features': 'Entropy, Uniformity'  # Placeholder
    }
    return biomarker

# Step 7: Clinical Integration
def flag_abnormalities_in_reports(biomarkers, ct_scan_file):
    if biomarkers['Steatosis HU Threshold']:
        print(f"Steatosis detected in scan {ct_scan_file}")
    elif biomarkers['Fibrosis HU + APRI/FIB-4 Index']:
        print(f"Fibrosis detected in scan {ct_scan_file}")
    else:
        print(f"Normal scan detected: {ct_scan_file}")

# Example workflow for one file
for nii_file in nii_files:
    file_path = os.path.join('data/', nii_file)

    # Step 2: Preprocessing
    processed_ct_img_data = preprocess_ct_scan(file_path)

    # Step 3: HU Measurement
    mean_liver_hu, l_s_hu_diff = get_hu_values(processed_ct_img_data)

    # Step 4: Pathological Classification
    pathology = classify_pathology(mean_liver_hu, l_s_hu_diff)
    print(f"Pathology for {nii_file}: {pathology}")

    # Step 5: Validation (Dummy True Labels)
    true_labels = [1 if pathology == 'Cirrhosis' else 0]  # Placeholder: 1 for Cirrhosis, 0 for others
    predictions = [1 if pathology == 'Cirrhosis' else 0]
    fpr, tpr, roc_auc = validate_with_external_data(predictions, true_labels)
    print(f"AUC: {roc_auc}")

    # Step 6: Biomarker Development
    biomarkers = develop_biomarker(mean_liver_hu, l_s_hu_diff)
    print(f"Biomarkers: {biomarkers}")

    # Step 7: Clinical Integration
    flag_abnormalities_in_reports(biomarkers, nii_file)

# Output example: You can add a function to save biomarkers or flagged results
def save_results_to_csv(results, filename='biomarker_results.csv'):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
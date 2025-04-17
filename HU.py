from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_opening, binary_closing, label

# Load the grayscale image
image_path = r'C:\Users\hasmy065\OneDrive - University of South Australia\Desktop\img.jpg'
image = Image.open(image_path).convert('L')
gray_image_array = np.array(image)

# Load the ground truth mask (binary image: liver=1, background=0)
mask_path = r'C:\Users\hasmy065\OneDrive - University of South Australia\Desktop\gt.png'
mask = Image.open(mask_path).convert('L')

# Resize the mask to match the dimensions of the grayscale image
mask = mask.resize(image.size, Image.NEAREST)
mask_array = np.array(mask) > 0  # Convert to binary mask (True for liver, False for background)

# Apply morphological operations to clean up the mask
# Step 1: Remove small noise using binary opening
cleaned_mask = binary_opening(mask_array, structure=np.ones((3, 3)))

# Step 2: Fill small holes using binary closing
cleaned_mask = binary_closing(cleaned_mask, structure=np.ones((5, 5)))

# Step 3: Keep the largest connected component (assumes liver is the largest region)
labeled_mask, num_features = label(cleaned_mask)
largest_component = (labeled_mask == np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1)

# Apply the refined mask to the grayscale image
segmented_liver = np.where(largest_component, gray_image_array, 0)

# Plot the original grayscale image
plt.figure(figsize=(8, 6))
plt.imshow(gray_image_array, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')
plt.show()

# Plot the ground truth mask
plt.figure(figsize=(8, 6))
plt.imshow(mask_array, cmap='gray')
plt.title('Original Ground Truth Liver Mask')
plt.axis('off')
plt.show()

# Plot the refined liver mask
plt.figure(figsize=(8, 6))
plt.imshow(largest_component, cmap='gray')
plt.title('Refined Liver Mask')
plt.axis('off')
plt.show()

# Plot the segmented liver
plt.figure(figsize=(8, 6))
plt.imshow(segmented_liver, cmap='gray')
plt.title('Segmented Liver (Using Refined Mask)')
plt.axis('off')
plt.show()

# Convert the grayscale image to a numpy array
gray_image_array = np.array(image)

# Simulate HU values (map pixel intensities to HU range)
# Assuming grayscale values (0-255) map to HU range (-1000 to 1000)
hu_image_array = (gray_image_array / 255.0) * 2000 - 1000

# Segment the liver based on HU range (40 to 70 HU)
liver_mask = (hu_image_array >= 40) & (hu_image_array <= 70)

# Plot the simulated HU image
plt.figure(figsize=(8, 6))
plt.imshow(hu_image_array, cmap='gray', vmin=-1000, vmax=1000)
plt.title('Simulated HU Image')
plt.colorbar(label='Hounsfield Units (HU)')
plt.axis('off')
plt.show()

# Plot the liver segmentation mask
plt.figure(figsize=(8, 6))
plt.imshow(liver_mask, cmap='gray')
plt.title('Liver Segmentation Mask')
plt.axis('off')
plt.show()

# Overlay the liver mask on the original grayscale image
plt.figure(figsize=(8, 6))
plt.imshow(gray_image_array, cmap='gray')  # Use gray_image_array instead of gray_image
plt.imshow(liver_mask, cmap='Reds', alpha=0.5)  # Overlay in red
plt.title('Liver Segmentation Overlay')
plt.axis('off')
plt.show()

# Plot the histogram of grayscale pixel intensities
plt.figure(figsize=(8, 6))
plt.hist(gray_image_array.flatten(), bins=256, range=(0, 255), color='blue', alpha=0.7)
plt.title('Histogram of Grayscale Pixel Intensities')
plt.xlabel('Pixel Intensity (0-255)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Plot the histogram of simulated HU values
plt.figure(figsize=(8, 6))
plt.hist(hu_image_array.flatten(), bins=256, range=(-1000, 1000), color='green', alpha=0.7)
plt.title('Histogram of Simulated HU Values')
plt.xlabel('Hounsfield Units (HU)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Plot the histogram of HU values within the liver mask
masked_hu_values = hu_image_array[mask_array]  # Extract HU values where the mask is True

plt.figure(figsize=(8, 6))
plt.hist(masked_hu_values.flatten(), bins=256, range=(-1000, 1000), color='orange', alpha=0.7)
plt.title('Histogram of HU Values in Liver Mask')
plt.xlabel('Hounsfield Units (HU)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Data for tissue types and their HU ranges
tissues = [
    ("Air", -1000, -1000),  # Air has a fixed value
    ("Fat", -100, -50),
    ("Muscle", 10, 50),
    ("Liver", 40, 70),
    ("Blood (Hemorrhage)", 40, 100),
    ("Bone", 500, 1000),
    ("Tumor (Benign)", 20, 40),
    ("Tumor (Malignant)", 30, 80),
    ("Cyst", 0, 30),
    ("Fatty Liver", -10, 40),
    ("Cirrhosis", 30, 70),
    ("Abscess", 30, 50)
]

# Extracting names, lower and upper HU ranges
names = [tissue[0] for tissue in tissues]
lower_bound = [tissue[1] for tissue in tissues]
upper_bound = [tissue[2] for tissue in tissues]

# Create an array of x-values for plotting
x_vals = np.arange(len(names))

# Plotting the line chart
plt.figure(figsize=(12, 8))

# Plotting the lower and upper bounds as lines
plt.plot(x_vals, lower_bound, label="Lower Bound", marker='o', linestyle='-', color='blue')
plt.plot(x_vals, upper_bound, label="Upper Bound", marker='o', linestyle='-', color='red')

# Adding arrows and annotations for each value
for i, (name, lower, upper) in enumerate(zip(names, lower_bound, upper_bound)):
    # Check if the label is one of the specified tissues
    if name in ["Cyst", "Fatty Liver", "Cirrhosis", "Abscess", "Liver"]:
        annotation_color = 'red'  # Highlight in red
    else:
        annotation_color = 'black'  # Default color

    # Annotating the lower bound (blue line)
    plt.annotate(f'{lower}', xy=(i, lower), xytext=(i-0.2, lower-100), 
                 arrowprops=dict(facecolor='blue', arrowstyle="->"),
                 fontsize=9, color=annotation_color, ha='center', va='center')
    
    # Annotating the upper bound (red line)
    plt.annotate(f'{upper}', xy=(i, upper), xytext=(i+0.2, upper+100),
                 arrowprops=dict(facecolor='red', arrowstyle="->"),
                 fontsize=9, color=annotation_color, ha='center', va='center')

# Add labels, title, and grid
plt.xticks(x_vals, names, rotation=90)  # Set tissue names as x-axis labels
plt.xlabel('Tissues/Diseases')
plt.ylabel('Hounsfield Units (HU)')
plt.title('HU Range for Different Tissues/Diseases')
plt.grid(True)
plt.tight_layout()

# Display legend
plt.legend()

# Show plot
plt.show()

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# Paths
flow_path = "data/data/raw/4DFlow/P103V1/P103V1-LVOT-MOD_ph0001.nii.gz"
cine_path = "data/data/raw/Cine/P103V1/P103V1-LVOT-CINE_ph0001.nii.gz"

# Load images
flow_img = nib.load(flow_path)
cine_img = nib.load(cine_path)

flow_data = flow_img.get_fdata()
cine_data = cine_img.get_fdata()

# Print shapes
print("4D Flow shape:", flow_data.shape)
print("Cine shape:", cine_data.shape)

# Select middle slice
flow_slice = flow_data.shape[2] // 2
cine_slice = cine_data.shape[2] // 2

# Display
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(flow_data[:, :, flow_slice], cmap="gray")
axes[0].set_title(f"4D Flow (slice {flow_slice})")
axes[0].axis("off")

axes[1].imshow(cine_data[:, :, cine_slice], cmap="gray")
axes[1].set_title(f"Cine MRI (slice {cine_slice})")
axes[1].axis("off")

plt.tight_layout()
plt.show()
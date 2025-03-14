import torch

# Create a dummy tensor of shape (3, 1000, 2, 16)
tensor = torch.randn(3, 1000, 2, 16)

# Reshape: Merge the first and third dimensions
reshaped_tensor = tensor.permute(1, 0, 2, 3) #.reshape(1000, 6, 16)

# Print the new shape
print(reshaped_tensor.shape)  # Expected: (1000, 6, 16)

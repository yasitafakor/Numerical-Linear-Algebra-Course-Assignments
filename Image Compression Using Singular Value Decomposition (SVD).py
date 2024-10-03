import numpy as np
from PIL import Image

# Load the image
image_data = np.array(Image.open('khayam.jpg'))

# Normalize the image data
normalized_data = image_data.astype(float) / 255.0

# Set the compression levels
compression_levels = [10, 20, 40, 80, 160]

for k in compression_levels:
    # Create an array of zeros with the same dimensions as the input image
    compressed_matrix = np.zeros_like(normalized_data)

    # Perform SVD on each color channel
    for channel in range(3):
        u, s, v = np.linalg.svd(normalized_data[:, :, channel])

        # Reconstruct the image using k singular values
        compressed_matrix[:, :, channel] = u[:, :k] @ np.diag(s[:k]) @ v[:k, :]

    # Clip pixel values to the range [0, 1]
    compressed_matrix = np.clip(compressed_matrix, 0, 1)

    # Save the compressed image
    compressed_image = Image.fromarray(np.uint8(compressed_matrix * 255))
    compressed_image.save('compressed_k_{}.jpg'.format(k))

    # Display the compressed image
    compressed_image.show()
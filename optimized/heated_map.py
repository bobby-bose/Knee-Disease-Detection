import cv2
import numpy as np

def generate_heatmap(image_path, center, radius):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a blank heatmap image
    heatmap = np.zeros_like(gray, dtype=np.uint8)

    # Generate heatmap
    heatmap = cv2.circle(heatmap, center, radius, (255, 255, 255), -1)

    # Apply colormap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)

    # Adjust blending weights
    result = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)

    return result

# Example usage
image_path = 'test.png'  # Path to your knee X-ray image
center = (250, 250)  # Center coordinates of the heatmap circle
radius = 100  # Radius of the heatmap circle

# Generate heatmap image
heatmap_image = generate_heatmap(image_path, center, radius)

# Display the heatmap image
cv2.imshow('Heatmap Image', heatmap_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

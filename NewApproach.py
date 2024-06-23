import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to load and preprocess images
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image at {image_path} could not be loaded.")
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, image_gray

# Function to perform template matching and return bounding boxes
def template_matching(scene_image_gray, template_image_gray, threshold):
    method = cv2.TM_CCOEFF_NORMED
    result = cv2.matchTemplate(scene_image_gray, template_image_gray, method)
    locations = np.where(result >= threshold)
    locations = list(zip(*locations[::-1]))
    bounding_boxes = []
    for loc in locations:
        start_x, start_y = loc
        end_x = start_x + template_image_gray.shape[1]
        end_y = start_y + template_image_gray.shape[0]
        bounding_boxes.append((start_x, start_y, end_x, end_y))
    return bounding_boxes

# Load and preprocess images
refImagePath = 'dataset/models/ref15.png'
sceneImagePath = 'dataset/scenes/scene6.png'

refImage, refImageGray = load_and_preprocess_image(refImagePath)
sceneImage, sceneImageGray = load_and_preprocess_image(sceneImagePath)

# Display the images
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.title("Template image")
plt.imshow(cv2.cvtColor(refImage, cv2.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.title("Scene image")
plt.imshow(cv2.cvtColor(sceneImage, cv2.COLOR_BGR2RGB))
plt.show()

# Set a threshold for template matching
threshold = 0.5


# Perform template matching
bounding_boxes = template_matching(sceneImageGray, refImageGray, threshold)

# Draw bounding boxes on scene image
for (start_x, start_y, end_x, end_y) in bounding_boxes:
    cv2.rectangle(sceneImage, (start_x, start_y), (end_x, end_y), (0, 255, 0), 3)

# Display the scene image with detected regions
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(sceneImage, cv2.COLOR_BGR2RGB))
plt.title(f"Detected {len(bounding_boxes)} objects in scene")
plt.axis('off')
plt.show()

import tensorflow as tf
from tensorflow import keras

import cv2
import numpy as np
import matplotlib.pyplot as plt
def imagePreprocess(pathOfImage):
  image = cv2.imread(pathOfImage, cv2.IMREAD_COLOR)
  # to read the image from the specified file path using the 'cv2.IMREAD_COLOR' function
  # to clarify each pixel's BGR(default color format used by OpenCV) values
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # to convert the image from BGR to RGB color format which is used by most other libraries like tensorflow and matplotlib
  # thus, the image colors are displayed with correct values of the corresponding image values
  # imageResized = cv2.resize(image, (250,250))  # to resize the image as 250x250 pixels
  # this function is significant for batch processing in machine learning models
  imageNormalized = image.astype(np.float32) / 255.0
  # this code line provides basic representation of the corresponding color values for each pixel
  return imageNormalized

refImagePreprocessing = imagePreprocess('dataset/models/ref15.png')
# to preprocess for a picture using with imagePreprocess function
sceneImagePreprocessing = imagePreprocess('dataset/scenes/scene6.png')

plt.figure(figsize=(40, 10))
# to specify the figure in inches, 
# 40 represents the figure's width, 10 represents the figure's height
plt.subplot(1, 2, 1)
# 1 means that, number of rows in the subplot grid
# 2 means that, number of columns in the subplot grid
# 1 means that, index of the subplot for creating or modifying

plt.title("Template image")
plt.imshow(refImagePreprocessing) # to show the colorful image of the  corresponding normalized pixel values
plt.subplot(1, 2, 2)
plt.title("Scene image")
plt.imshow(sceneImagePreprocessing)
plt.axis('on')
plt.show()



from tensorflow.keras.applications import VGG16
# VGG16(develooped by Visual Geometry Group,
# 16 refers to the number of layers with trainable parameters such as convolutional and fully connected layers)
# which is pre-trained Convolutional Neural Network arhitecture from the keras module
# that architecture consists of 16 layers which include convolutional layers, fully connected layers and pooling layers

from tensorflow.keras.models import Model
# Model is to create a new model based on the pre-trained VGG16 model

from tensorflow.keras.applications.vgg16 import preprocess_input
# 

modelDefault =VGG16(weights = 'imagenet') # to load VGG16 model
  #  weights = 'imagenet' means that, the model should load pre-trained weights
  # imagesnet refers to the large dataset used to train this model
  # contains millions of images related to the many categories
  # the model can leverage the knowledge it gained from the imagenet to be beneficial for various image proessing tasks using these pre-trained weights
  # even dataset is small

featurExtractor = Model(inputs=modelDefault.input, outputs=modelDefault.get_layer('block5_conv3').output)
  # Model() function allows to create a new model from the Keras library
  # inputs = modelDefault.input means that, the input of this new model is the same with the input of the original VGG16 model which is (250,250,3)
  # outputs = modelDefault.get_layer('block5_conv3').output means that, the output of the this new model is the output of the layer named 'block5_conv3' in the VGG16 model
  # this layer is a convolutional layer towards the end of the network, capturing high-level features of the image
  # layers which are closer to the input capture low-level features(like textures and edges)
  # layers which are deeper layers (like 'block5_conv3) capture high-level features(lke objects and shapes)
  # 'block5_conv3' has more information. Therefore, it is useful for image recognition and object detection tasks
  # output means that, output of the 'block5_conv3' layer feature map

def featurExtracting(imageArr, featurExtractor): # imageArr is preprocessed(transformed to RGB color ) and normalized ([0-1] range) image
 
  resizedImage = cv2.resize(imageArr, (224,224))
  # to resize the input image as 224x224 pixels 
  # because VGG16 model expects input images as 224x224 pixel format, therefore, input images should be resized for being compatible with the model 
  preprocessedImage = preprocess_input(resizedImage * 255.0)
  # to perform some preprocessing operations like scales pixel values from the range [0-1] to [0-255]  
  # because VGG16 model was trained with these mean values 

  preprocessedImageBatch = np.expand_dims(preprocessedImage,axis=0) # to transform the shape to (1,250,250,3),  1 indicates that, there is 1 image in this batch
  features = featurExtractor.predict(preprocessedImageBatch)
  # features variable will be a numpy array of the feature map processed by CNN 
  return  features[0] 
  #  remove the batch dimension for returning feature map of the input image 
  # then, the shape will be (height of feature map, width of feature map, number of channels)


featuresTemplate = featurExtracting(refImagePreprocessing, featurExtractor)
featuresScene = featurExtracting(sceneImagePreprocessing, featurExtractor)
# extracted features variables

plt.figure(figsize=(40, 10))
# to specify the figure in inches, 
# 40 represents the figure's width, 10 represents the figure's height

plt.subplot(1, 2, 1)
plt.title("Reference Features (Channel 0)")

plt.imshow(featuresTemplate[:, :, 0], cmap='gray')
# featuresTemplate[:, :, 0] extracts the first channel of the feature map 
# to display the first channel of the template features by grayscale colormap 
print(featuresTemplate.shape)


plt.subplot(1, 2, 2)
plt.title("Scene Features (Channel 0)")
plt.imshow(featuresScene[:, :, 0], cmap='gray')
print(featuresScene.shape)

plt.show()





mapFeaturesTemplate = np.mean(featuresTemplate, axis= -1)
# to transform feature channels to a single channel for template matching simplicity
mapFeaturesScene = np.mean(featuresScene, axis = -1)

templateMatchingScore = cv2.matchTemplate(mapFeaturesScene, mapFeaturesTemplate, cv2.TM_CCOEFF_NORMED)
# templateMatchingScore is an array which keeps data of the similarity scores of the each region in the scene image
# to scale the similarity score of the regions in the scene image to match the template image 

threshold = 0.6
regions = np.where(templateMatchingScore >= threshold)
# to clarify regions where the matching score is exceeding the threshold limit in the scene image 

cornersAndCenterLocListOfDetected = []
for startingPoint in zip(*regions[::-1]):
  # to iterate over the starting points of the detected images 
  heightOfDetected, widthOfDetected = mapFeaturesTemplate.shape
  rightBottomPoint = (startingPoint[0] + widthOfDetected, startingPoint[0] + heightOfDetected)
  centerPoint = ((startingPoint[0] + (widthOfDetected // 2)), (startingPoint[0] + (heightOfDetected // 2)))
  detectedCornersAndCenterLoc = startingPoint, rightBottomPoint, centerPoint
  cornersAndCenterLocListOfDetected.append(detectedCornersAndCenterLoc)

# in this above loop we collected the corresponding locations of bounding boxes and centers of the detected regions 

for detectedCornersAndCenterLoc in cornersAndCenterLocListOfDetected:
  boxes = np.array([detectedCornersAndCenterLoc[0] + detectedCornersAndCenterLoc[1]])
  similarityScores = templateMatchingScore[regions]
  selectedBoxIndices.extend(non_max_suppression(
      boxes=boxes, scores=similarityScores, max_output_size=45, iou_threshold=0.5
  ).numpy())

# in this above code, we are collected indices(this indices are retrieved from the indices of the boxes variable) of 45 boxes which have highest similarity scores
# we also set the intersection over union threshold as 0.5, which means that, we will not collect if the boxes's iou_threshold is higher than 0.5

lastDetectedBoundingBoxes = []
for eachSelectedBoxIndices in selectedBoxIndices:
  lastDetectedBoundingBoxes.append(cornersAndCenterLocListOfDetected[eachSelectedBoxIndices])

# in this above code represents that the last time indicated bounding boxes after the non-max suppression

counterForIndex = 0
for eachBoundingBox in lastDetectedBoundingBoxes:
  counterForIndex = counterForIndex + 1
  print(f"The number of instance is: {eachBoundingBox} {{the width of box: {detectedCornersAndCenterLoc[1][0]-detectedCornersAndCenterLoc[0][0]}px, the height of box: {detectedCornersAndCenterLoc[1][1]-detectedCornersAndCenterLoc[0][1]}px, the center location is: {detectedCornersAndCenterLoc[2]}}}")

# in this above code prints the number of selected detected boxes, width of this boxes, height of this boxes and center location of this boxes

copiedSceneImage = sceneImagePreprocessing.copy()
# to copy preprocessed image to draw rectangles on the copy instead of destroying the original image

for eachBb in lastDetectedBoundingBoxes:
  # to iterate over corners and center locations of the final detected bounding boxes
  cv2.rectangle(copiedSceneImage, eachBb[0], eachBb[1], (0,255,0), 5)
  # to draw rectangle on the copied scene image from the starting point(eachBb[0]), to the end point(eachBb[1])
  # by green color (0,255,0), and thickness value (3)

plt.figure(figsize=(40,40))
# to create a figure with 40x40 inches
plt.imshow(copiedSceneImage)
# to display the expected format of the image
plt.title("The detected products are: ")
plt.show()
# to render the plot
  


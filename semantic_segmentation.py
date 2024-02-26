import pixellib
import numpy as np
from pixellib.semantic import semantic_segmentation
import cv2

# Loading pre-trained DeepLabv3 model
segment_image = semantic_segmentation()
segment_image.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")

# Perform semantic segmentation on the input image
segment_image.segmentAsPascalvoc("test.jpeg", output_image_name="output.jpg")

# Read the output images and labels
image = cv2.imread('output.jpg')
labels = cv2.imread('output.jpg', 0)

# Segment the image based on labels
for label in set(labels.flatten()):
    if label not in [13, 15]:  # 13 for person, 15 for horse
        continue
    lowerb = np.array([label])
    upperb = np.array([label])
    mask = cv2.inRange(labels, lowerb, upperb)
    segmented = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite(f'output_{label}.jpg', segmented)

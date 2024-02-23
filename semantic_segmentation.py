import pixellib
import numpy as np
from pixellib.semantic import semantic_segmentation
import cv2

# 加载预训练的DeepLabv3模型
segment_image = semantic_segmentation()
segment_image.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")

# 对输入图像进行语义分割
segment_image.segmentAsPascalvoc("test.jpeg", output_image_name="output.jpg")

# 读取输出图像和分割标签
image = cv2.imread('output.jpg')
labels = cv2.imread('output.jpg', 0)

# 根据标签分割图像
for label in set(labels.flatten()):
    if label not in [13, 15]:  # 1 for person, 18 for dog
        continue
    lowerb = np.array([label])
    upperb = np.array([label])
    mask = cv2.inRange(labels, lowerb, upperb)
    segmented = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite(f'output_{label}.jpg', segmented)
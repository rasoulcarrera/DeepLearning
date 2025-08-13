import numpy as np
import tensorflow as tf
import cv2
from matplotlib import pyplot as plt

tensor_1rank = tf.constant([1.0, 2.0, 3.0])
tensor0_2rank = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
tensor1_2rank = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)

print('Task 4,5', tf.add(tensor0_2rank, tensor1_2rank), '\n')

# Task 6 convert the image into a rank-4 tensor
img_path_0 = 'albert.jpg'
img_path = 'dog.jpg'
img_src = cv2.imread(img_path_0)
if img_src.shape[2] == 3:
    img = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
else:
    img = cv2.imread(img_src)

plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.show()


max_pix = np.max(img)
img = img.astype(np.float32) / max_pix
tensor_img = tf.convert_to_tensor(img)
back_2_norm = tensor_img.numpy()
cv2.imshow('Normal Image',back_2_norm)
cv2.waitKey(0)
print(tensor_img)

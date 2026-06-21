import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("fruit_ripeness_model.h5")

# Class labels
classes = ['Unripe', 'Ripe', 'Overripe']

# Load image
img_path = "test_image.jpg"

img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predict
prediction = model.predict(img_array)

predicted_class = classes[np.argmax(prediction)]

print("Prediction:", predicted_class)
print("Confidence:", np.max(prediction) * 100, "%")

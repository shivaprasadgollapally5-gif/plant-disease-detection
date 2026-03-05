import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model
model = load_model("model/plant_disease_model.h5")

# Get class names automatically from dataset
class_names = sorted(os.listdir("dataset/train"))

print("Classes:", class_names)

# Image path
img_path = "test_image.jpg"

# Load image
img = image.load_img(img_path, target_size=(224,224))

# Preprocess
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predict
prediction = model.predict(img_array)

predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print("Predicted Disease:", predicted_class)
print("Confidence:", confidence,"%")

print(prediction)

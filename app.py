from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import os
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load trained model
model = load_model("model/plant_disease_model.h5")

with open("model/class_indices.json","r") as f:
    class_indices = json.load(f)

class_names = list(class_indices.keys())



solutions = {

"Rice_Sheath_Blight":"Apply fungicide like validamycin. Avoid dense planting and reduce nitrogen fertilizer.",

"Rice_Tungro":"Remove infected plants and control green leafhoppers using insecticides.",

"Rice_bacterial_leaf_blight":"Use resistant rice varieties and avoid excessive nitrogen fertilizers.",

"Rice_brown_spot":"Apply balanced fertilizers and fungicides. Improve soil nutrition.",

"Rice_healthy":"Your plant is healthy. Maintain proper irrigation and balanced fertilization.",

"Rice_leaf_blast":"Use fungicide like tricyclazole and avoid excessive nitrogen fertilizer.",

"Rice_leaf_scald":"Use disease-free seeds and maintain proper water management.",

"Rice_narrow_brown_spot":"Apply balanced fertilizer and maintain proper field drainage.",

"Rice_neck_Blast":"Use resistant varieties and apply fungicide at early stages.",

"Rice_rice_Hispa":"Use insecticides or neem oil spray to control rice hispa insects.",

"Tomato___Bacterial_spot":"Apply copper-based bactericides and avoid overhead watering.",

"Tomato___Early_blight":"Use fungicides like mancozeb and remove infected leaves.",

"Tomato___Late_blight":"Apply fungicides like chlorothalonil and improve air circulation.",

"Tomato___Leaf_Mold":"Reduce humidity and apply fungicides if needed.",

"Tomato___Septoria_leaf_spot":"Remove infected leaves and apply fungicides like chlorothalonil.",

"Tomato___Spider_mites Two-spotted_spider_mite":"Use neem oil spray or insecticidal soap.",

"Tomato___Target_Spot":"Use resistant varieties and apply fungicides.",

"Tomato___Tomato_Yellow_Leaf_Curl_Virus":"Control whiteflies and remove infected plants.",

"Tomato___Tomato_mosaic_virus":"Remove infected plants and disinfect gardening tools.",

"Tomato___healthy":"Your plant is healthy. Continue proper irrigation and fertilization."

}

pesticides = {

"Rice_Sheath_Blight":"Validamycin 3L",

"Rice_Tungro":"Imidacloprid",

"Rice_bacterial_leaf_blight":"Streptomycin sulphate",

"Rice_brown_spot":"Carbendazim",

"Rice_healthy":"No pesticide needed",

"Rice_leaf_blast":"Tricyclazole",

"Rice_leaf_scald":"Copper oxychloride",

"Rice_narrow_brown_spot":"Mancozeb",

"Rice_neck_Blast":"Tricyclazole",

"Rice_rice_Hispa":"Chlorpyrifos",

"Tomato___Bacterial_spot":"Copper hydroxide",

"Tomato___Early_blight":"Mancozeb",

"Tomato___Late_blight":"Chlorothalonil",

"Tomato___Leaf_Mold":"Azoxystrobin",

"Tomato___Septoria_leaf_spot":"Chlorothalonil",

"Tomato___Spider_mites Two-spotted_spider_mite":"Abamectin",

"Tomato___Target_Spot":"Azoxystrobin",

"Tomato___Tomato_Yellow_Leaf_Curl_Virus":"Imidacloprid",

"Tomato___Tomato_mosaic_virus":"No chemical control",

"Tomato___healthy":"No pesticide needed"
}

precautions = {

"Rice_Sheath_Blight":"Wear gloves and face mask while spraying. Avoid spraying during windy conditions.",

"Rice_Tungro":"Use protective gloves and mask. Wash hands after spraying pesticide.",

"Rice_bacterial_leaf_blight":"Wear protective clothing and avoid inhaling spray mist.",

"Rice_brown_spot":"Use gloves and face mask while applying fungicide.",

"Rice_healthy":"No pesticide required. Maintain proper field hygiene.",

"Rice_leaf_blast":"Wear gloves, mask and goggles while spraying fungicide.",

"Rice_leaf_scald":"Use protective clothing and wash equipment after spraying.",

"Rice_narrow_brown_spot":"Avoid spraying near water sources and wear safety gear.",

"Rice_neck_Blast":"Use gloves and mask. Do not spray during high wind.",

"Rice_rice_Hispa":"Wear gloves and wash hands thoroughly after application.",

"Tomato___Bacterial_spot":"Wear gloves, mask and avoid direct skin contact with pesticide.",

"Tomato___Early_blight":"Use gloves and protective clothing during spraying.",

"Tomato___Late_blight":"Wear mask and goggles while applying fungicide.",

"Tomato___Leaf_Mold":"Use gloves and wash hands after spraying.",

"Tomato___Septoria_leaf_spot":"Wear mask and avoid inhaling spray mist.",

"Tomato___Spider_mites Two-spotted_spider_mite":"Use gloves and avoid contact with eyes.",

"Tomato___Target_Spot":"Wear gloves and protective clothing.",

"Tomato___Tomato_Yellow_Leaf_Curl_Virus":"Wear gloves and wash hands after pesticide use.",

"Tomato___Tomato_mosaic_virus":"Remove infected plants carefully and wash tools.",

"Tomato___healthy":"No precautions needed."
}

# Prediction Function
def predict_disease(img_path):

    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    class_index = np.argmax(prediction)
    confidence = round(np.max(prediction)*100,2)

    if confidence < 60:
        predicted_class = "Invalid Image"
        solution = "Please upload a clear plant leaf image."
        pesticide = "Not applicable"
        precaution = "Not applicable"
        crop = "Unknown"

    else:
        predicted_class = class_names[class_index]

        # Detect crop type
        if "Rice" in predicted_class:
            crop = "Rice"
        elif "Tomato" in predicted_class:
            crop = "Tomato"
        else:
            crop = "Unknown"

        solution = solutions.get(predicted_class,"No solution available")
        pesticide = pesticides.get(predicted_class,"No pesticide recommendation")
        precaution = precautions.get(predicted_class,"Follow general pesticide safety precautions.")

    print("Prediction:", predicted_class)
    print("Confidence:", confidence)
    print("Precaution:", precaution)

    # Determine severity
    if confidence >= 90:
        severity = "Severe Infection"
    elif confidence >= 75:
        severity = "Moderate Infection"
    elif confidence >= 60:
        severity = "Mild Infection"
    else:
        severity = "Invalid Image"

    return predicted_class, confidence, solution, pesticide, severity, crop, precaution

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    solution = None
    pesticide = None
    severity = None
    crop = None
    precaution = None
    image_path = None

    if request.method == 'POST':

        # 1️⃣ Normal File Upload
        if 'image' in request.files and request.files['image'].filename != '':
            file = request.files['image']
            image_path = os.path.join("static", file.filename)
            file.save(image_path)

            prediction, confidence, solution, pesticide, severity, crop, precaution = predict_disease(image_path)

        # 2️⃣ Camera Capture Upload
        elif 'camera_image' in request.files and request.files['camera_image'].filename != '':
            file = request.files['camera_image']
            image_path = os.path.join("static", file.filename)
            file.save(image_path)

            prediction, confidence, solution, pesticide, severity, crop, precaution = predict_disease(image_path)

        # Solution fallback
        if prediction:
            solution = solutions.get(prediction, "No solution available")
            pesticide = pesticides.get(prediction, "No pesticide info available")
            precaution = precautions.get(prediction,"Follow general pesticide safety precautions.")

            if confidence > 90:
                severity = "High Confidence Detection"
            elif confidence > 70:
                severity = "Moderate Confidence Detection"
            else:
                severity = "Low Confidence Detection"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        solution=solution,
        pesticide=pesticide,
        severity=severity,
        crop=crop,
        precaution=precaution,
        image_path=image_path
    )


if __name__ == "__main__":
    app.run(debug=True)

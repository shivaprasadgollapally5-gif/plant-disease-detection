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

class_names = dict((v, k) for k, v in class_indices.items())



solutions = {

# 🌾 WHEAT
"Wheat_Aphid": "Spray water and remove affected parts.",
"Wheat_Black_Rust": "Use resistant varieties and remove infected leaves.",
"Wheat_Blast": "Use certified seeds and proper irrigation.",
"Wheat_Brown_Rust": "Apply fungicide and remove infected plants.",
"Wheat_Common_Root_Rot": "Improve soil drainage and rotate crops.",
"Wheat_Fusarium_Head_Blight": "Use disease-free seeds.",
"Wheat_Healthy": "No disease detected.",
"Wheat_Leaf_Blight": "Remove infected leaves.",
"Wheat_Mildew": "Ensure proper airflow.",
"Wheat_Mite": "Spray water and control pests.",
"Wheat_Septoria": "Remove infected leaves.",
"Wheat_Smut": "Use treated seeds.",
"Wheat_Stem_fly": "Remove infected stems.",
"Wheat_Tan_spot": "Crop rotation and resistant varieties.",
"Wheat_Yellow_Rust": "Use resistant seeds and early treatment.",

# 🌿 COTTON
"Cotton_Aphids": "Wash plants and remove infected leaves.",
"Cotton_Army_worm": "Handpick larvae and use traps.",
"Cotton_Bacterial_Blight": "Use disease-free seeds.",
"Cotton_Healthy": "No disease detected.",
"Cotton_Powdery_Mildew": "Remove infected parts.",
"Cotton_Target_spot": "Maintain proper spacing.",

# 🥭 MANGO
"Mango_Anthracnose": "Prune infected branches.",
"Mango_Bacterial Canker": "Remove infected bark.",
"Mango_Cutting Weevil": "Remove affected fruits.",
"Mango_Die Back": "Cut affected branches.",
"Mango_Gall Midge": "Remove damaged parts.",
"Mango_Healthy": "No disease detected.",
"Mango_Powdery Mildew": "Remove infected flowers.",
"Mango_Sooty Mould": "Clean leaves properly.",

# 🌾 RICE
"Rice_Sheath_Blight": "Reduce plant density.",
"Rice_Tungro": "Remove infected plants.",
"Rice_bacterial_leaf_blight": "Use resistant varieties.",
"Rice_brown_spot": "Apply proper fertilizers.",
"Rice_healthy": "No disease detected.",
"Rice_leaf_blast": "Use resistant seeds.",
"Rice_leaf_scald": "Remove infected leaves.",
"Rice_narrow_brown_spot": "Improve soil nutrients.",
"Rice_neck_Blast": "Use fungicide early.",
"Rice_rice_Hispa": "Remove insects manually.",

# 🍅 TOMATO
"Tomato___Bacterial_spot": "Remove infected leaves.",
"Tomato___Early_blight": "Use crop rotation.",
"Tomato___Late_blight": "Remove infected plants.",
"Tomato___Leaf_Mold": "Improve ventilation.",
"Tomato___Septoria_leaf_spot": "Remove infected leaves.",
"Tomato___Spider_mites Two-spotted_spider_mite": "Wash leaves regularly.",
"Tomato___Target_Spot": "Avoid wet leaves.",
"Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies.",
"Tomato___Tomato_mosaic_virus": "Remove infected plants.",
"Tomato___healthy": "No disease detected."
}

pesticides = {

# WHEAT
"Wheat_Aphid": "Imidacloprid",
"Wheat_Black_Rust": "Propiconazole",
"Wheat_Blast": "Tricyclazole",
"Wheat_Brown_Rust": "Mancozeb",
"Wheat_Common_Root_Rot": "Carbendazim",
"Wheat_Fusarium_Head_Blight": "Tebuconazole",
"Wheat_Healthy": "Not required",
"Wheat_Leaf_Blight": "Chlorothalonil",
"Wheat_Mildew": "Sulfur fungicide",
"Wheat_Mite": "Dicofol",
"Wheat_Septoria": "Azoxystrobin",
"Wheat_Smut": "Seed treatment fungicide",
"Wheat_Stem_fly": "Chlorpyrifos",
"Wheat_Tan_spot": "Propiconazole",
"Wheat_Yellow_Rust": "Mancozeb",

# COTTON
"Cotton_Aphids": "Imidacloprid",
"Cotton_Army_worm": "Spinosad",
"Cotton_Bacterial_Blight": "Copper oxychloride",
"Cotton_Healthy": "Not required",
"Cotton_Powdery_Mildew": "Sulfur",
"Cotton_Target_spot": "Mancozeb",

# MANGO
"Mango_Anthracnose": "Carbendazim",
"Mango_Bacterial Canker": "Copper fungicide",
"Mango_Cutting Weevil": "Lambda-cyhalothrin",
"Mango_Die Back": "Bordeaux mixture",
"Mango_Gall Midge": "Dimethoate",
"Mango_Healthy": "Not required",
"Mango_Powdery Mildew": "Sulfur",
"Mango_Sooty Mould": "Neem oil",

# RICE
"Rice_Sheath_Blight": "Validamycin",
"Rice_Tungro": "Control vectors",
"Rice_bacterial_leaf_blight": "Copper oxychloride",
"Rice_brown_spot": "Mancozeb",
"Rice_healthy": "Not required",
"Rice_leaf_blast": "Tricyclazole",
"Rice_leaf_scald": "Carbendazim",
"Rice_narrow_brown_spot": "Mancozeb",
"Rice_neck_Blast": "Tricyclazole",
"Rice_rice_Hispa": "Chlorpyrifos",

# TOMATO
"Tomato___Bacterial_spot": "Copper fungicide",
"Tomato___Early_blight": "Mancozeb",
"Tomato___Late_blight": "Metalaxyl",
"Tomato___Leaf_Mold": "Chlorothalonil",
"Tomato___Septoria_leaf_spot": "Mancozeb",
"Tomato___Spider_mites Two-spotted_spider_mite": "Neem oil",
"Tomato___Target_Spot": "Azoxystrobin",
"Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Imidacloprid",
"Tomato___Tomato_mosaic_virus": "No cure",
"Tomato___healthy": "Not required"
}

precautions = {

# WHEAT
"Wheat_Aphid": "Avoid excess nitrogen.",
"Wheat_Black_Rust": "Maintain field hygiene.",
"Wheat_Blast": "Avoid water stress.",
"Wheat_Brown_Rust": "Early detection.",
"Wheat_Common_Root_Rot": "Ensure drainage.",
"Wheat_Fusarium_Head_Blight": "Avoid humidity.",
"Wheat_Healthy": "Maintain regular care.",
"Wheat_Leaf_Blight": "Proper spacing.",
"Wheat_Mildew": "Ensure airflow.",
"Wheat_Mite": "Monitor regularly.",
"Wheat_Septoria": "Avoid wet leaves.",
"Wheat_Smut": "Use certified seeds.",
"Wheat_Stem_fly": "Timely sowing.",
"Wheat_Tan_spot": "Crop rotation.",
"Wheat_Yellow_Rust": "Early spraying.",

# COTTON
"Cotton_Aphids": "Avoid overwatering.",
"Cotton_Army_worm": "Monitor fields regularly.",
"Cotton_Bacterial_Blight": "Avoid water splash.",
"Cotton_Healthy": "Maintain plant health.",
"Cotton_Powdery_Mildew": "Avoid humidity.",
"Cotton_Target_spot": "Proper spacing.",

# MANGO
"Mango_Anthracnose": "Avoid moisture.",
"Mango_Bacterial Canker": "Avoid injury to plants.",
"Mango_Cutting Weevil": "Clean fallen fruits.",
"Mango_Die Back": "Regular pruning.",
"Mango_Gall Midge": "Remove debris.",
"Mango_Healthy": "Regular monitoring.",
"Mango_Powdery Mildew": "Ensure airflow.",
"Mango_Sooty Mould": "Control insects.",

# RICE
"Rice_Sheath_Blight": "Avoid dense planting.",
"Rice_Tungro": "Control insects.",
"Rice_bacterial_leaf_blight": "Use clean seeds.",
"Rice_brown_spot": "Balanced fertilizer.",
"Rice_healthy": "Maintain proper care.",
"Rice_leaf_blast": "Avoid excess nitrogen.",
"Rice_leaf_scald": "Monitor regularly.",
"Rice_narrow_brown_spot": "Improve soil health.",
"Rice_neck_Blast": "Timely spraying.",
"Rice_rice_Hispa": "Manual removal.",

# TOMATO
"Tomato___Bacterial_spot": "Avoid wet leaves.",
"Tomato___Early_blight": "Crop rotation.",
"Tomato___Late_blight": "Avoid humidity.",
"Tomato___Leaf_Mold": "Proper ventilation.",
"Tomato___Septoria_leaf_spot": "Remove debris.",
"Tomato___Spider_mites Two-spotted_spider_mite": "Regular cleaning.",
"Tomato___Target_Spot": "Avoid overcrowding.",
"Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control insects.",
"Tomato___Tomato_mosaic_virus": "Avoid tool contamination.",
"Tomato___healthy": "Maintain plant health."
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
    predicted_class = class_names[class_index]

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
        elif "Wheat" in predicted_class:
            crop = "Wheat"
        elif "Cotton" in predicted_class:
            crop = "Cotton"
        elif "Mango" in predicted_class:
            crop = "Mango"
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
    app.run(host="0.0.0.0", port=10000)

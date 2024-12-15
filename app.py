from fastapi import FastAPI, UploadFile, Form, HTTPException
import json
import face_recognition
import numpy as np
import cv2
from typing import Optional, Union, List, Dict
import pickle

# Load the pre-trained model
with open("dt_model.pkl", "rb") as model_file:
    logit_model = pickle.load(model_file)

app = FastAPI()

def create_input_array(sym_array):
    initial_list = [1,1,0]
    for element in array_example.ravel():
        if element == 1:
            initial_list += [1,1,0]
        else:
            initial_list += [0,0,1]
    final_array = np.array(initial_list).reshape(1, -1)

    return final_array

def make_prediction(input_array, model):
    """
    Make department prediction based on input array.
    
    Args:
    input_array (np.ndarray): Input features array
    model (object): Trained machine learning model
    
    Returns:
    str: Predicted department
    """
    # Ensure input is 2D array
    if input_array.ndim == 1:
        input_array = input_array.reshape(1, -1)
    
    # Get prediction probabilities
    predicted_array = model.predict_proba(input_array)
    input_array = create_input_array(input_array)
    
    emergency_value = predicted_array[0, 0]
    medicine_value = predicted_array[0, 1]
    surgery_value = predicted_array[0, 2]
    
    # Determine department based on highest probability
    if emergency_value >= medicine_value and emergency_value >= surgery_value:
        return 'Emergency Room'
    elif medicine_value >= surgery_value:
        return 'Medicine Department'
    else:
        return 'Surgery Department'

def encode_faces(face_image):
    """Encode faces in an image, returning the first encoding or None."""
    encoding = face_recognition.face_encodings(np.array(face_image))
    return encoding[0] if encoding else None

def face_distance_to_conf(face_distance, face_match_threshold=0.8):
    """Convert face distance to confidence percentage."""
    range_val = (1.0 - face_match_threshold) if face_distance > face_match_threshold else face_match_threshold
    linear_val = 1.0 - (face_distance / (range_val * 2.0))
    return linear_val + ((1.0 - linear_val) * pow((linear_val - 0.5) * 2, 0.2)) if face_distance <= face_match_threshold else linear_val

@app.post("/recognize")
async def recognize_patient(
    test_image: Optional[UploadFile] = None,
    hn: Optional[int] = Form(None)
):
    """Recognize a patient based on an image or HN."""
    # Validate input
    if not hn and not test_image:
        raise HTTPException(status_code=400, detail="Either HN or image must be provided")

    # Load patient data
    with open("patient.json", "r", encoding="utf-8") as file:
        patient_list = json.load(file)

    # Matching patients
    matching_patients = []

    # Match by HN
    if hn:
        matching_hn_patients = [
            patient for patient in patient_list 
            if patient["hn"] == hn 
        ]
        matching_patients.extend(matching_hn_patients)

    # Match by face recognition
    if test_image:
        # Load the uploaded image
        contents = await test_image.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is not None:
            # Detect faces in the uploaded image
            face_locations = face_recognition.face_locations(frame)
            if face_locations:
                face_encoding = encode_faces(frame)
                if face_encoding is not None:
                    # Find matching patients by face
                    for patient in patient_list:
                        # Skip if patient already matched by HN
                        if patient in matching_patients:
                            continue
                        
                        # Check if patient has an appointment
                        if not patient.get("Appointment_Date", False):
                            continue

                        # Load patient image for face comparison
                        image = cv2.imread(patient["image"])
                        patient_encoding = encode_faces(image)
                        
                        if patient_encoding is not None:
                            distance = face_recognition.face_distance([patient_encoding], face_encoding)[0]
                            confidence = face_distance_to_conf(distance) * 100
                            is_match = face_recognition.compare_faces([patient_encoding], face_encoding, tolerance=0.5)[0]

                            if is_match and confidence > 90:
                                patient_match = patient.copy()
                                patient_match['confidence'] = round(confidence, 2)
                                matching_patients.append(patient_match)

    # Prepare response
    if matching_patients:
        return {
            "status": "Match Found",
            "patients": matching_patients
        }
    else:
        return {"status": "No Match Found"}

@app.post("/predict")
async def predict_patient(
    data: Union[List[Dict], Dict]
):
    """
    Predict patient department based on input data.
    
    Accepts either a single dictionary or a list of dictionaries.
    Returns department predictions.
    
    Expected input format:
    {
        "features": [list of feature values matching model input]
    }
    """
    # Ensure input is a list
    if isinstance(data, dict):
        data = [data]
    
    # Validate input data
    if not data:
        raise HTTPException(status_code=400, detail="No input data provided")
    
    # Prepare features for prediction
    try:
        # Extract features from input
        features = np.array([
            item.get('features', []) for item in data
        ])
        
        # Validate feature input
        if features.shape[1] == 0:
            raise ValueError("No features provided in input")
        
        # Make predictions
        predictions = [
            make_prediction(np.array(feature), logit_model) 
            for feature in features
        ]
        
        return {
            "status": "Prediction Complete",
            "predictions": predictions
        }
    
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction error: {str(e)}")

# Optional: Add a prediction validation endpoint
@app.post("/validate_prediction_input")
async def validate_prediction_input(
    data: Union[List[Dict], Dict]
):
    """
    Validate the input format for prediction without making a prediction.
    """
    # Ensure input is a list
    if isinstance(data, dict):
        data = [data]
    
    # Validate input data
    if not data:
        raise HTTPException(status_code=400, detail="No input data provided")
    
    # Check that all dictionaries have the same keys
    if len(set(tuple(d.keys()) for d in data)) > 1:
        raise HTTPException(status_code=400, detail="All input dictionaries must have the same keys")
    
    return {
        "status": "Input Valid",
        "input_count": len(data)
    }
from fastapi import FastAPI, UploadFile, Form, HTTPException
import json
import face_recognition
import numpy as np
import cv2
from typing import Optional, Union, List, Dict
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pickle
import openai


# Load the pre-trained model
with open("dt_model.pkl", "rb") as model_file:
    logit_model = pickle.load(model_file)

app = FastAPI()

# OpenAI client setup
client = openai.Client(api_key="")

# Middleware for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[""],  # Allow all origins
    allow_credentials=True,
    allow_methods=[""],     # Allow all HTTP methods
    allow_headers=["*"],     # Allow all headers
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Columns for input formatting
columns_to_keep = [
    'Itching', 'Skin rash', 'Continuous sneezing', 'Shivering', 'Chills', 'Joint pain', 'Stomach pain', 
    'Vomiting', 'Fatigue', 'Anxiety', 'Cold hands and feets', 'Restlessness', 'Lethargy', 'Cough', 
    'High fever', 'Breathlessness', 'Sweating', 'Headache', 'Nausea', 'Back pain', 'Abdominal pain', 
    'Diarrhoea', 'Mild fever', 'Throat irritation', 'Redness of eyes', 'Sinus pressure', 'Runny nose', 
    'Chest pain', 'Fast heart rate', 'Neck pain', 'Dizziness', 'Bruising', 'Internal itching', 'Prognosis'
]

# System prompt for LLM
system_prompt = """
You are just an intern doctor. You will be provided with a patient’s symptom list.
Your task is to summarize the patient’s symptoms and suggest possible diagnoses that can be presented to the doctor.
Follow the breakdown structure and example given below to guide your response.

Breakdown Symptom:
- Itching (Yes): The patient is experiencing itching, which could indicate an allergic reaction, skin condition, or an infectious disease.
- Skin Rash (Yes): The presence of a skin rash suggests a visible dermatological issue. Combined with itching, it might point towards conditions like hives, eczema, or a systemic infection.
- Fatigue (Yes): The patient is feeling physically tired or lacking energy, which can result from an ongoing infection, chronic conditions, or other systemic causes.
- Lethargy (Yes): Similar to fatigue, lethargy implies a lack of enthusiasm or mental alertness, often associated with illnesses or underlying health issues.

Possible Conditions:
- Viral Infections: Conditions like chickenpox, dengue, or measles could explain a combination of fever, rash, and systemic symptoms.
- Allergic Reactions: Allergic dermatitis or another allergic condition might explain the itching and rash but less likely with high fever.
- Autoimmune Disorders: Certain autoimmune diseases like lupus could cause skin rash, fever, and systemic symptoms like lethargy and fatigue.
"""

# Function to create input array
def create_input_array(sym_array):
    initial_list = [1, 1, 0]
    for element in sym_array.ravel():
        if element == 1:
            initial_list += [1, 1, 0]
        else:
            initial_list += [0, 0, 1]
    final_array = np.array(initial_list).reshape(1, -1)
    return final_array

# Function to format input for LLM
def input_format(array):
    output_list = []
    array = create_input_array(array)
    i = 0
    for col_name, value in zip(columns_to_keep, array.flatten()):
        if value != 0:
            output_list.append({"Symptom": col_name, "Value": "True"})
            i += 1
    return output_list

# Function to call LLM
def create_message(text):
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Ensure you are using the correct model name
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0,
        )
        return completion.choices[0].message["content"]  # Corrected to access message content properly
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

# Function to make predictions based on input data
def make_prediction(input_array, model):
    if input_array.ndim == 1:
        input_array = input_array.reshape(1, -1)
    
    input_array = create_input_array(input_array)
    predicted_array = model.predict_proba(input_array)
    
    emergency_value = predicted_array[0, 0]
    medicine_value = predicted_array[0, 1]
    surgery_value = predicted_array[0, 2]
    
    if emergency_value >= medicine_value and emergency_value >= surgery_value:
        return 'Emergency Room'
    elif medicine_value >= surgery_value:
        return 'Medicine Department'
    else:
        return 'Surgery Department'

@app.post("/predict")
async def predict_patient(data: Union[List[Dict], Dict]):
    """
    Predict patient department based on input data and generate a symptom summary using LLM.
    """
    if isinstance(data, dict):
        data = [data]
    
    # Validate input data
    if not data:
        raise HTTPException(status_code=400, detail="No input data provided")
    
    try:
        # Extract features from input data
        features = np.array([item.get('features', []) for item in data])

        if features.shape[1] == 0:
            raise ValueError("No features provided in input")
        
        # Make predictions
        predictions = [make_prediction(np.array(feature), logit_model) for feature in features]

        # Format symptoms for LLM
        symptom_input = [input_format(feature) for feature in features]
        
        # Generate symptom summary from LLM
        summaries = [create_message(str(symptom)) for symptom in symptom_input]

        # Clean up the LLM output (remove \n and improve format)
        formatted_summary = summaries[0].replace('\n', ' ').strip()

        # Return predictions and summaries
        return {
            "status": "Prediction Complete",
            "predictions": predictions,
            "summary": formatted_summary  # Cleaned-up summary
        }

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction error: {str(e)}")

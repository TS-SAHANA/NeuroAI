# backend/main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import uvicorn
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import base64
import tensorflow as tf

# Import your existing logic
# Ensure utils.py is in the same folder and imports are correct
from utils import (
    get_hybrid_model_inputs, 
    predict_and_process, 
    create_overlay_image, 
    generate_pdf_report,
    dice_coefficient,
    iou_score,
    MODEL_PATH
)

app = FastAPI()

# Enable CORS so the React UI can talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model Once
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={'dice_coefficient': dice_coefficient, 'iou_score': iou_score},
    compile=False
)

def array_to_base64(img_array):
    """Helper to convert numpy array to base64 string for frontend display"""
    img = Image.fromarray(img_array.astype('uint8'))
    buff = BytesIO()
    img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # Read Image
    contents = await file.read()
    
    # Save temporarily for your utils functions (or adapt utils to take bytes)
    # For now, we wrap the bytes to make it compatible with your load function
    img_stream = BytesIO(contents)
    
    # 1. Pipeline Execution (Reusing your utils logic)
    processed_img, model_inputs, _ = get_hybrid_model_inputs(img_stream)
    mask, confidence, metrics, stage, location = predict_and_process(model, model_inputs)
    
    # 2. Create Overlay
    overlay = create_overlay_image(processed_img, mask)
    
    # 3. Prepare Response
    # Convert images to Base64 for the UI
    original_b64 = array_to_base64(processed_img.squeeze() * 255)
    overlay_b64 = array_to_base64(overlay)
    
    return JSONResponse({
        "original_image": f"data:image/png;base64,{original_b64}",
        "segmented_image": f"data:image/png;base64,{overlay_b64}",
        "metrics": metrics,
        "confidence": float(confidence), # <--- ADDED float() HERE
        "stage": stage,
        "location": location
    })

@app.post("/report")
async def generate_report(
    name: str = Form(...),
    age: str = Form(...),
    contact: str = Form(...),
    original: str = Form(...), # Expecting Base64 string
    segmented: str = Form(...),
    volume: str = Form(...),
    diameter: str = Form(...),
    stage: str = Form(...),
    confidence: str = Form(...),
    location: str = Form(...)
):
    # Decode images back to bytes for the PDF generator
    def decode_base64(data_str):
        header, encoded = data_str.split(",", 1)
        return BytesIO(base64.b64decode(encoded))

    original_buff = decode_base64(original)
    segmented_buff = decode_base64(segmented)
    
    patient_data = {"name": name, "age": age, "id": contact}
    metrics = {"Volume (Approx.)": volume, "Max Diameter (Approx.)": diameter}
    
    pdf_bytes = generate_pdf_report(
        original_buff, segmented_buff, metrics, stage, 
        float(confidence), location, patient_data
    )
    
    return Response(content=pdf_bytes, media_type="application/pdf")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
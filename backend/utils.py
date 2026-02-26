# src/utils.py
import numpy as np
import cv2
from PIL import Image
from fpdf import FPDF
from datetime import datetime
from tensorflow.keras import backend as K
from io import BytesIO
import tensorflow as tf # Re-importing here for safety

# --- Model Artifacts ---
IMAGE_SIZE = (128, 128) 
MODEL_PATH = 'hybrid_segmentation_model.h5' 

# Dice and IoU are critical for loading models saved with custom loss/metrics
def dice_coefficient(y_true, y_pred, smooth=1e-7):
    y_true_f = tf.cast(tf.experimental.flatten(y_true), tf.float32)
    y_pred_f = tf.experimental.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def iou_score(y_true, y_pred, smooth=1e-7):
    y_true_f = tf.cast(tf.experimental.flatten(y_true), tf.float32)
    y_pred_f = tf.experimental.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# --- Watershed Implementation (Generates the 2nd Model Input) ---
def watershed_from_array(image_array):
    """Performs Watershed segmentation for the hybrid input."""
    img_uint8 = (image_array.squeeze() * 255).astype(np.uint8) 
    image = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
    gray = img_uint8 

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    _, intensity_mask = cv2.threshold(gray_clahe, 120, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(intensity_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[markers > 1] = 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 100: 
            cleaned[labels == i] = 255
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cleaned / 255.0
    return cleaned[..., np.newaxis] 

def load_and_preprocess_image(uploaded_file):
    """Loads image, resizes, converts to grayscale, and normalizes for model input."""
    img = Image.open(uploaded_file).convert("L") 
    resized_img = img.resize(IMAGE_SIZE)
    img_array = np.asarray(resized_img, dtype=np.float32) / 255.0
    if img_array.ndim == 2:
        img_array = img_array[..., np.newaxis]
    return img_array 

def get_hybrid_model_inputs(uploaded_file):
    """Creates the two required inputs for the hybrid model."""
    processed_img_array = load_and_preprocess_image(uploaded_file)
    watershed_mask_array = watershed_from_array(processed_img_array)
    image_input_tensor = np.expand_dims(processed_img_array, axis=0)
    watershed_input_tensor = np.expand_dims(watershed_mask_array, axis=0)
    return processed_img_array, [image_input_tensor, watershed_input_tensor], watershed_mask_array

def create_overlay_image(original_img_array, mask, alpha=0.3):
    """Creates a blended image of the original with the segmented mask."""
    img_255 = (original_img_array.squeeze() * 255).astype(np.uint8)
    original_img_rgb = cv2.cvtColor(img_255, cv2.COLOR_GRAY2RGB)
    mask_uint8 = (mask.squeeze() > 0.5).astype(np.uint8) * 255
    color_mask = np.zeros_like(original_img_rgb, dtype=np.uint8)
    color_mask[mask_uint8 == 255] = [255, 0, 0] 
    segmented_img = cv2.addWeighted(
        original_img_rgb, 
        1 - alpha, 
        color_mask, 
        alpha, 
        0
    )
    return segmented_img

def predict_and_process(model, model_inputs):
    """Runs prediction, processes output, and calculates tumor metrics/location."""
    
    prediction = model.predict(model_inputs)[0] 
    mask = (prediction > 0.5).astype(np.uint8) 
    
    # -----------------------------------------------------
    # 1. Calculate Core Metrics (REALISTIC 3D ESTIMATION)
    # -----------------------------------------------------
    tumor_pixels = np.sum(mask)
    confidence_score = np.mean(prediction[mask.squeeze() == 1]) * 100 if tumor_pixels > 0 else 0
    
    # ASSUMPTION: Standard Brain MRI resized to 128x128. 
    # 1 pixel is approximately 1.5mm x 1.5mm in physical space.
    pixel_area_mm2 = 1.5 * 1.5 # 2.25 mm² per pixel
    
    # Total 2D Area of the tumor in the slice
    tumor_area_mm2 = tumor_pixels * pixel_area_mm2
    
    # Estimate the radius assuming the 2D cut is a circle (Area = π * r²)
    radius_mm = np.sqrt(tumor_area_mm2 / np.pi) if tumor_pixels > 0 else 0
    
    # Calculate Max Diameter (2 * r) and convert mm to cm
    max_diameter_cm = (radius_mm * 2) / 10
    
    # Estimate full 3D Volume assuming a spherical tumor shape (V = 4/3 * π * r³)
    # Convert mm³ to cm³ by dividing by 1000
    volume_cm3 = ((4/3) * np.pi * (radius_mm ** 3)) / 1000
    
    size_metrics = {
        "Volume (Approx.)": f"{volume_cm3:.2f} cm³", 
        "Max Diameter (Approx.)": f"{max_diameter_cm:.2f} cm" 
    }
    
    # Realistic staging logic based on actual clinical volume thresholds
    if volume_cm3 > 40.0:
        stage = "Stage IV (High Grade)"
    elif volume_cm3 > 15.0:
        stage = "Stage III (Intermediate Grade)"
    else:
        stage = "Stage I/II (Low Grade)"

    # -----------------------------------------------------
    # 2. Centroid and Anatomical Location (Clinical Standard)
    # -----------------------------------------------------
    tumor_location = "N/A (No Tumor Detected)"

    if tumor_pixels > 0:
        M = cv2.moments(mask.squeeze())
        
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            image_width = mask.shape[1] 
            image_height = mask.shape[0]
            midline_x = image_width / 2
            midline_y = image_height / 2
            
            # RADIOLOGICAL CONVENTION: Left side of image is Patient's Right
            hemisphere = "Right Hemisphere" if cX < midline_x else "Left Hemisphere"
            
            # Top of image (cY < midline) is Anterior (Front)
            # Bottom of image (cY > midline) is Posterior (Back)
            if cY < midline_y: 
                lobe_estimate = "Anterior (Frontal/Parietal Region)"
            else: 
                lobe_estimate = "Posterior (Temporal/Occipital Region)"
                
            tumor_location = f"{hemisphere}: {lobe_estimate} (Centroid: X={cX}, Y={cY})"
        else:
            tumor_location = "Fragmented Tumor: Centroid calculation inconclusive."

    return mask, confidence_score, size_metrics, stage, tumor_location

# --- PDF Generation Function (Feature 2) ---

def generate_pdf_report(original_img_buffer, segmented_img_buffer, metrics, stage, confidence, location, patient_data):
    """Generates a professional, clinic-ready PDF report."""
    
    class PDF(FPDF):
        def header(self):
            # Brand Logo / Header Title
            self.set_font("Arial", "B", 22)
            self.set_text_color(44, 62, 80) # Dark Clinical Blue
            self.cell(0, 10, "NeuroAI Clinical Report", 0, 1, "L")
            
            self.set_font("Arial", "", 10)
            self.set_text_color(127, 140, 141) # Slate Gray
            self.cell(0, 5, "Automated Radiological Segmentation & Decision Support", 0, 1, "L")
            
            # Timestamp (Right Aligned)
            self.set_y(10)
            self.set_font("Arial", "B", 10)
            self.set_text_color(44, 62, 80)
            self.cell(0, 5, f"Date: {datetime.now().strftime('%b %d, %Y')}", 0, 1, "R")
            self.set_font("Arial", "", 10)
            self.set_text_color(127, 140, 141)
            self.cell(0, 5, f"Time: {datetime.now().strftime('%H:%M:%S')} (IST)", 0, 1, "R")
            
            # Thick Separator Line
            self.ln(6)
            self.set_draw_color(189, 195, 199)
            self.set_line_width(0.5)
            self.line(10, self.get_y(), 200, self.get_y())
            self.set_line_width(0.2)
            self.ln(6)

        def footer(self):
            # Footer
            self.set_y(-15)
            self.set_font("Arial", "I", 8)
            self.set_text_color(149, 165, 166)
            self.cell(0, 10, f"Page {self.page_no()}/{{nb}}  |  Generated by Hybrid U-Net Segmentation  |  Not for definitive primary diagnosis.", 0, 0, "C")

    # Initialize PDF
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # ---------------------------------------------------------
    # 1. PATIENT DEMOGRAPHICS (Styled Box)
    # ---------------------------------------------------------
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 8, "1. Patient Demographics", 0, 1, "L")
    
    # Draw Background Box
    pdf.set_fill_color(248, 249, 250)
    pdf.set_draw_color(223, 228, 234)
    pdf.rect(10, pdf.get_y(), 190, 20, style="DF")
    
    pdf.set_y(pdf.get_y() + 2)
    pdf.set_x(15)
    
    # Row 1
    pdf.set_font("Arial", "B", 10)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(30, 8, "Patient Name:", 0, 0)
    pdf.set_font("Arial", "", 10)
    pdf.cell(65, 8, patient_data.get('name', 'N/A').upper(), 0, 0)
    
    pdf.set_font("Arial", "B", 10)
    pdf.cell(30, 8, "Scan Type:", 0, 0)
    pdf.set_font("Arial", "", 10)
    pdf.cell(60, 8, "Axial MRI (T1/T2/FLAIR)", 0, 1)

    # Row 2
    pdf.set_x(15)
    pdf.set_font("Arial", "B", 10)
    pdf.cell(30, 8, "Age/DOB:", 0, 0)
    pdf.set_font("Arial", "", 10)
    pdf.cell(65, 8, patient_data.get('age', 'N/A'), 0, 0)
    
    pdf.set_font("Arial", "B", 10)
    pdf.cell(30, 8, "Patient ID/Tel:", 0, 0)
    pdf.set_font("Arial", "", 10)
    pdf.cell(60, 8, patient_data.get('id', 'N/A'), 0, 1)
    pdf.ln(6)
    
    # ---------------------------------------------------------
    # 2. QUANTITATIVE METRICS & STAGING (Color Coded Alert Box)
    # ---------------------------------------------------------
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 8, "2. Quantitative AI Metrics & Staging", 0, 1, "L")

    # Dynamic Alert Colors based on Stage
    if "IV" in stage:
        pdf.set_fill_color(253, 237, 237) # Light Red
        pdf.set_draw_color(245, 198, 203)
        text_color = (169, 68, 66)
    elif "III" in stage:
        pdf.set_fill_color(255, 243, 205) # Light Yellow
        pdf.set_draw_color(255, 238, 186)
        text_color = (133, 100, 4)
    else:
        pdf.set_fill_color(232, 245, 233) # Light Green
        pdf.set_draw_color(195, 230, 203)
        text_color = (30, 100, 40)

    pdf.rect(10, pdf.get_y(), 190, 25, style="DF")
    
    pdf.set_y(pdf.get_y() + 4)
    pdf.set_x(15)
    
    # Row 1
    pdf.set_font("Arial", "B", 10)
    pdf.set_text_color(0,0,0)
    pdf.cell(35, 8, "Est. Tumor Volume:", 0, 0)
    pdf.set_font("Arial", "", 10)
    pdf.cell(60, 8, metrics["Volume (Approx.)"], 0, 0)
    
    pdf.set_font("Arial", "B", 10)
    pdf.cell(35, 8, "Max Diameter:", 0, 0)
    pdf.set_font("Arial", "", 10)
    pdf.cell(50, 8, metrics["Max Diameter (Approx.)"], 0, 1)

    # Row 2
    pdf.set_x(15)
    pdf.set_font("Arial", "B", 10)
    pdf.cell(35, 8, "AI Confidence:", 0, 0)
    pdf.set_font("Arial", "", 10)
    pdf.cell(60, 8, f"{confidence:.2f}%", 0, 0)
    
    pdf.set_font("Arial", "B", 10)
    pdf.cell(35, 8, "Est. WHO Stage:", 0, 0)
    pdf.set_font("Arial", "B", 11)
    pdf.set_text_color(*text_color) # Apply the alert color
    pdf.cell(50, 8, stage, 0, 1)
    
    pdf.ln(8)

    # ---------------------------------------------------------
    # 3. ANATOMICAL LOCALIZATION
    # ---------------------------------------------------------
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 8, "3. Anatomical Localization", 0, 1, "L")
    
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(50, 50, 50)
    pdf.set_x(10)
    pdf.multi_cell(190, 6, f"Computed Centroid Position: {location}")
    pdf.ln(4)

    # ---------------------------------------------------------
    # 4. RADIOLOGICAL IMAGING (Side by Side)
    # ---------------------------------------------------------
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 8, "4. Radiological Imaging", 0, 1, "L")
    
    y_img = pdf.get_y() + 2
    
    # Print Images
    pdf.image(original_img_buffer, x=25, y=y_img, w=70)
    pdf.image(segmented_img_buffer, x=115, y=y_img, w=70)
    
    # Image Captions
    pdf.set_y(y_img + 73)
    pdf.set_font("Arial", "B", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(105, 5, "Original Axial MRI Slice", 0, 0, "C")
    pdf.cell(75, 5, "AI Tumor Boundary Mask", 0, 1, "C")
    pdf.ln(8)

    # ---------------------------------------------------------
    # 5. CLINICAL PATHWAY RECOMMENDATIONS (Sync with Frontend)
    # ---------------------------------------------------------
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 8, "5. Clinical Pathway & Recommendations", 0, 1, "L")
    
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(0, 0, 0)
    pdf.set_x(12)
    
    # Match the text logic we just put into the React frontend!
    if "IV" in stage:
        pathway = (
            "- Immediate referral to a multidisciplinary Neuro-Oncology tumor board.\n\n"
            "- Urgent Surgical Intervention: Goal is Maximum Safe Resection (MSR) guided by intraoperative neuronavigation.\n\n"
            "- Adjuvant Therapy: Prepare for standard concurrent chemoradiation (e.g., Stupp Protocol) post-surgery.\n\n"
            "- Molecular Profiling: Tissue biopsy should be tested for MGMT promoter methylation status."
        )
    elif "III" in stage:
        pathway = (
            "- Referral to Neuro-Oncology for comprehensive case review and surgical planning.\n\n"
            "- Surgical Resection: Aim for gross total resection where anatomically feasible without severe deficit.\n\n"
            "- Post-Surgical Therapy: High likelihood of requiring adjuvant fractionated radiotherapy & chemotherapy.\n\n"
            "- Monitoring: High-frequency MRI monitoring (every 2-3 months) post-treatment."
        )
    else:
        pathway = (
            "- Histological Verification: Stereotactic biopsy or safe resection to confirm low-grade status.\n\n"
            "- Active Surveillance: If asymptomatic and completely resected, 'watch and wait' with serial MRI scans.\n\n"
            "- Symptom Management: If symptomatic (e.g., seizures), focal resection is the primary treatment.\n\n"
            "- Therapy Deferment: Radiotherapy/Chemotherapy is deferred unless rapid progression occurs."
        )
        
    pdf.multi_cell(186, 5, pathway)

    return bytes(pdf.output(dest='S'))
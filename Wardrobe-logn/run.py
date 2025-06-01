
from __future__ import division, print_function

import os
import time
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gc
import uuid

import numpy as np
from flask import render_template, request, session, url_for
from flaskapp import app, login_required,redirect
from matplotlib import pyplot as plt
from werkzeug.utils import secure_filename, send_from_directory
import pymongo
import requests
from gridfs import GridFS
# import tensorflow
# from tensorflow.keras.models import load_model
# from tensorflow.keras.utils import load_img
# from keras.preprocessing import image
import calendar
import base64
from bson import ObjectId
from datetime import datetime
from flask import jsonify, request, session

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import joblib

from flask import send_file


from flask import Flask, render_template, request, jsonify, session, redirect, url_for

# material properties comented
# def extract_material_properties(img_path):
#     """
#     Extract material properties from an image including:
#     - Dominant colors
#     - Texture patterns
#     - Material type estimation based on texture analysis
#     - Pattern information
#
#     Returns a dictionary of material properties
#     """
#     # Load image
#     img = cv2.imread(img_path)
#     img = imutils.resize(img, height=300)  # Resize for consistent processing
#
#     # 1. Color analysis (using your existing KMeans approach)
#     flat_img = np.reshape(img, (-1, 3))
#     kmeans = KMeans(n_clusters=5, random_state=0)
#     kmeans.fit(flat_img)
#
#     dominant_colors = np.array(kmeans.cluster_centers_, dtype='uint')
#     percentages = (np.unique(kmeans.labels_, return_counts=True)[1]) / flat_img.shape[0]
#     p_and_c = sorted(zip(percentages, dominant_colors), reverse=True)
#
#     # 2. Texture analysis
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # GLCM (Gray Level Co-occurrence Matrix) texture features
#     # Convert to 8-bit grayscale for texture analysis
#     gray_8bit = (gray / gray.max() * 255).astype(np.uint8)
#
#     # Calculate texture features (variance as a simple measure)
#     texture_variance = np.var(gray_8bit)
#
#     # Calculate edge density (a proxy for texture complexity)
#     edges = cv2.Canny(gray_8bit, 100, 200)
#     edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
#
#     # 3. Material type estimation based on texture properties
#     material_type = "unknown"
#
#     # Simple heuristic-based material classification
#     if edge_density < 0.05 and texture_variance < 50:
#         material_type = "smooth"  # Might be leather, silk, etc.
#     elif edge_density > 0.2:
#         material_type = "textured"  # Might be denim, wool, etc.
#     elif texture_variance > 200:
#         material_type = "patterned"  # Has distinct patterns
#     else:
#         material_type = "medium"  # Medium texture, like cotton
#
#     # 4. Add basic pattern information
#     # This is a placeholder - in the full implementation you'd use the detect_pattern_type function
#     # For now we'll create a basic pattern_info structure with default values
#     pattern_info = {
#         "pattern_type": "regular" if texture_variance > 150 else "irregular",
#         "pattern_scale": "medium",
#         "pattern_strength": min(1.0, edge_density * 2),  # Simple scaling to 0-1 range
#         "has_pattern": edge_density > 0.1 or texture_variance > 100,
#         "pattern_regularity": 0.5,
#         "is_directional": False,
#         "peak_count": 0
#     }
#
#     # Return the extracted material properties
#     return {
#         "dominant_colors": [color.tolist() for _, color in p_and_c[:3]],
#         "color_percentages": [float(pct) for pct, _ in p_and_c[:3]],
#         "texture_variance": float(texture_variance),
#         "edge_density": float(edge_density),
#         "estimated_material": material_type,
#         "primary_color_rgb": p_and_c[0][1].tolist(),
#         "pattern_info": pattern_info  # Add pattern_info to the returned dict
#     }
#
#
# def determine_material_type(texture_variance, edge_density, pattern_info):
#     """Determine material type based on texture and pattern analysis"""
#
#     # Check if it's a strong pattern first
#     if pattern_info["has_pattern"] and pattern_info["pattern_strength"] > 0.4:
#         if pattern_info["pattern_type"] in ["check", "stripe"]:
#             return "woven_patterned"
#         elif pattern_info["pattern_type"] == "irregular":
#             return "printed"
#         else:
#             return "patterned"
#
#     # If no strong pattern, determine by texture
#     if edge_density < 0.05 and texture_variance < 50:
#         return "smooth"  # Might be leather, silk, etc.
#     elif edge_density > 0.2:
#         if texture_variance > 150:
#             return "rough_textured"  # Might be tweed, heavy wool
#         else:
#             return "textured"  # Might be denim, canvas
#     elif texture_variance > 200:
#         return "detailed"  # Has distinct texture details
#     else:
#         return "medium"  # Medium texture, like cotton
#
#
# def detect_pattern_type(img_path):
#     """Detect and classify pattern types in the image"""
#     img = cv2.imread(img_path)
#
#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # Resize for faster processing if needed
#     resized = cv2.resize(gray, (256, 256))
#
#     # Apply FFT to detect regular patterns
#     f = fftpack.fft2(resized)
#     fshift = fftpack.fftshift(f)
#     magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
#
#     # Threshold the magnitude spectrum to find strong frequencies
#     threshold = np.mean(magnitude_spectrum) + 3 * np.std(magnitude_spectrum)
#     peaks = magnitude_spectrum > threshold
#
#     # Count peaks in the frequency domain (excluding the DC component)
#     center_y, center_x = resized.shape[0] // 2, resized.shape[1] // 2
#     mask = np.ones_like(peaks)
#     mask[center_y - 5:center_y + 5, center_x - 5:center_x + 5] = 0  # Exclude center
#     peak_count = np.sum(peaks & mask)
#
#     # Analyze peak distribution
#     peak_locs = np.where(peaks & mask)
#     peak_distances = np.sqrt((peak_locs[0] - center_y) ** 2 + (peak_locs[1] - center_x) ** 2)
#     pattern_regularity = 0.0
#
#     if len(peak_distances) > 0:
#         # Calculate coefficient of variation (lower value = more regular)
#         if np.mean(peak_distances) > 0:
#             pattern_regularity = 1.0 - min(1.0, np.std(peak_distances) / np.mean(peak_distances))
#
#     # Gradient analysis for pattern direction
#     sobelx = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
#     sobely = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)
#
#     # Calculate gradient magnitudes and directions
#     gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
#     gradient_direction = np.arctan2(sobely, sobelx) * 180 / np.pi
#
#     # Analyze gradient directions
#     hist, _ = np.histogram(gradient_direction, bins=8, range=(-180, 180))
#     hist_normalized = hist / np.sum(hist)
#     max_dir_idx = np.argmax(hist_normalized)
#
#     # Determine if there are strong directional patterns
#     has_directional_pattern = np.max(hist_normalized) > 0.25
#
#     # Determine pattern type
#     if peak_count > 15 and pattern_regularity > 0.7:
#         # Check for grid patterns (peaks in both horizontal and vertical)
#         horizontal_peaks = np.sum(peaks[center_y, :] & mask[center_y, :])
#         vertical_peaks = np.sum(peaks[:, center_x] & mask[:, center_x])
#
#         if horizontal_peaks > 3 and vertical_peaks > 3:
#             pattern_type = "check"
#         elif has_directional_pattern:
#             if max_dir_idx in [0, 4]:  # Horizontal (0° or 180°)
#                 pattern_type = "horizontal_stripe"
#             elif max_dir_idx in [2, 6]:  # Vertical (90° or 270°)
#                 pattern_type = "vertical_stripe"
#             else:
#                 pattern_type = "diagonal_stripe"
#         else:
#             pattern_type = "regular"
#     elif peak_count > 5:
#         pattern_type = "semi_regular"
#     else:
#         # For low peak counts, further analyze texture
#         if np.max(hist_normalized) > 0.2:
#             pattern_type = "directional"
#         else:
#             pattern_type = "irregular"
#
#     # Determine pattern scale (fine, medium, large)
#     if len(peak_distances) > 0:
#         avg_distance = np.mean(peak_distances)
#         if avg_distance < 20:
#             pattern_scale = "fine"
#         elif avg_distance < 50:
#             pattern_scale = "medium"
#         else:
#             pattern_scale = "large"
#     else:
#         # Default if no peaks detected
#         pattern_scale = "medium"
#
#     # Calculate pattern strength (how dominant the pattern is)
#     pattern_strength = min(1.0, peak_count / 50)
#
#     return {
#         "pattern_type": pattern_type,
#         "pattern_scale": pattern_scale,
#         "pattern_strength": float(pattern_strength),
#         "has_pattern": peak_count > 3,
#         "pattern_regularity": float(pattern_regularity),
#         "is_directional": has_directional_pattern,
#         "peak_count": int(peak_count)
#     }


def generate_normal_map(img_path):
    """Generate a normal map from a texture for 3D relief"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to smooth while preserving edges
        smooth = cv2.bilateralFilter(gray, 9, 75, 75)

        # Generate gradients
        sobelx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=5)
        sobely = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=5)

        # Normalize gradients
        sobelx = cv2.normalize(sobelx, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        sobely = cv2.normalize(sobely, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

        # Create normal map (RGB)
        normal_map = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        normal_map[..., 0] = sobelx * 127.5 + 127.5  # Red channel (X)
        normal_map[..., 1] = sobely * 127.5 + 127.5  # Green channel (Y)
        normal_map[..., 2] = 255  # Blue channel (Z always points up)

        # Save normal map
        filename, ext = os.path.splitext(img_path)
        normal_map_path = f"{filename}_normal{ext}"
        cv2.imwrite(normal_map_path, normal_map)

        return normal_map_path

    except Exception as e:
        print(f"Error generating normal map: {str(e)}")
        return None

@app.template_filter('normalize_path')
def normalize_path_filter(file_path):
    """Normalizes file paths for template rendering"""
    return normalize_path(file_path)


client = pymongo.MongoClient('localhost', 27017)
db = client.user_login_system_test
DEFAULT_RATING = 4

import cv2
from sklearn.cluster import KMeans
import imutils

from flaskapp.ml_models import model_manager

print('🚀 Model loading optimized - models load on demand to save memory')
def model_predict(img_path, model=None):
    """Optimized model predict with lazy loading"""
    try:
        # Use the model manager instead of loading at startup
        result = model_manager.predict_clothing(img_path)

        # Convert to your expected numpy array format
        import numpy as np
        preds = np.array([result['all_predictions']])

        return preds

    except Exception as e:
        print(f"Error in model_predict: {str(e)}")
        raise


# Keep this import as is:

from flaskapp.user.texture_mapping import process_clothing_texture


@app.route('/api/process-clothing-texture', methods=['POST'])
@login_required
def api_process_clothing_texture():
    """
    API endpoint to process clothing textures for 3D visualization

    Request:
        - file: The image file
        - clothing_type: The type of clothing (e.g., "T-shirt/top", "Dress")

    Response:
        {
            "success": true,
            "model_path": "/static/models/clothing/tshirt.glb",
            "texture_url": "/static/image_users/user_id/image_texture.png",
            "normal_map_url": "/static/image_users/user_id/image_normal.png"
        }
    """
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Get clothing type from form or predict it
        clothing_type = request.form.get('clothing_type')

        # If no clothing type provided, try to predict it
        if not clothing_type:
            # You can use your existing model_predict function here
            # For now, we'll default to T-shirt/top
            clothing_type = "T-shirt/top"

        # Save the file
        user_id = session['user']['_id']
        upload_dir = os.path.join('flaskapp', 'static', 'image_users', user_id)
        os.makedirs(upload_dir, exist_ok=True)

        # Add a unique identifier to prevent filename collisions
        filename = secure_filename(file.filename)
        base, ext = os.path.splitext(filename)
        unique_filename = f"{base}_{user_id}_{int(time.time())}{ext}"
        file_path = os.path.join(upload_dir, unique_filename)

        file.save(file_path)

        # Process the texture
        result = process_clothing_texture(file_path, clothing_type)

        # Convert absolute paths to relative URLs
        base_url = f'/static/image_users/{user_id}/'
        texture_url = os.path.basename(result['texture_path'])
        normal_map_url = os.path.basename(result['normal_map_path'])

        return jsonify({
            'success': True,
            'model_path': result['model_path'],
            'texture_url': f"{base_url}{texture_url}",
            'normal_map_url': f"{base_url}{normal_map_url}"
        })

    except Exception as e:
        app.logger.error(f"Error processing texture: {str(e)}")
        return jsonify({'error': str(e)}), 500


def predict_color(img_path):
    """Wrapper for improved_predict_color for compatibility"""
    return improved_predict_color(img_path)

def improved_predict_color(img_path):
    """
    Improved color prediction that accurately identifies the dominant color of clothing
    by effectively removing the background and focusing on the central object.

    Returns: (percentage, [R, G, B])
    """
    import cv2
    import numpy as np
    from sklearn.cluster import KMeans
    import imutils

    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image from {img_path}")
        return None

    # Resize for consistent processing
    img = imutils.resize(img, height=300)

    # Step 1: Background Removal
    # Convert to RGBA to detect white/transparent backgrounds
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find contours to identify the clothing item
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the main object
    mask = np.zeros_like(gray)

    if contours:
        # Find the largest contour which is assumed to be the clothing item
        main_contour = max(contours, key=cv2.contourArea)

        # Only use the contour if it's large enough (to avoid small artifacts)
        if cv2.contourArea(main_contour) > 1000:
            # Create a mask from the contour
            cv2.drawContours(mask, [main_contour], 0, 255, -1)
        else:
            # If no large contour, use a central region of the image
            height, width = img.shape[:2]
            cv2.rectangle(mask, (width // 4, height // 4), (width * 3 // 4, height * 3 // 4), 255, -1)
    else:
        # If no contours, use a central region of the image
        height, width = img.shape[:2]
        cv2.rectangle(mask, (width // 4, height // 4), (width * 3 // 4, height * 3 // 4), 255, -1)

    # Apply the mask to the image
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # Step 2: Color Clustering (K-means)
    # Reshape the masked image to a list of pixels
    pixels = masked_img.reshape(-1, 3)

    # Filter out black background (masked areas)
    non_black_pixels = pixels[~np.all(pixels == [0, 0, 0], axis=1)]

    # If no non-black pixels, return None
    if len(non_black_pixels) == 0:
        print("Warning: No valid pixels found after masking")
        return None

    # Apply K-means clustering to find dominant colors
    clusters = 5
    kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
    kmeans.fit(non_black_pixels)

    # Get colors and percentages
    colors = kmeans.cluster_centers_.astype(np.uint8)
    labels = kmeans.labels_
    counts = np.bincount(labels)
    percentages = counts / len(labels)

    # Step 3: Color Validation and Selection
    # Sort colors by percentage
    color_percentages = list(zip(percentages, colors))
    color_percentages.sort(reverse=True)

    # Filter out very dark and very light colors
    filtered_colors = []
    for percentage, color in color_percentages:
        brightness = np.mean(color)
        # Skip very dark colors (almost black) or very light colors (almost white)
        if 15 < brightness < 240:
            filtered_colors.append((percentage, color))

    # If all colors were filtered out, use the original list
    if not filtered_colors:
        filtered_colors = color_percentages

    # Get dominant color (first in the filtered list)
    dominant_percentage, dominant_color = filtered_colors[0]

    # Convert BGR to RGB for the result
    return (float(dominant_percentage), dominant_color[::-1])  # BGR to RGB


def improved_predict_color(img_path):
    """
    Enhanced color prediction that accurately identifies clothing colors.

    Improvements:
    1. Better background removal using HSV color space
    2. Improved clothing segmentation using contour analysis
    3. Weighted color clustering to prioritize central regions
    4. Color validation to avoid dark/black misidentification

    Returns color information tuple (percentage, [R, G, B])
    """
    import cv2
    import numpy as np
    from sklearn.cluster import KMeans
    import imutils

    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image from {img_path}")
        return None

    # Resize for consistent processing
    img = imutils.resize(img, height=300)

    # Convert to HSV for better segmentation
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Step 1: Create a mask to remove white/light backgrounds
    # This range covers white/very light backgrounds
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 30, 255])
    mask_white = cv2.inRange(hsv_img, lower_white, upper_white)

    # Invert mask to keep the clothing item
    mask = cv2.bitwise_not(mask_white)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Step 2: Find contours to identify the clothing item
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If contours exist, find the largest one (likely the clothing item)
    if contours:
        main_contour = max(contours, key=cv2.contourArea)

        # Create a new mask with only the main contour
        item_mask = np.zeros_like(mask)
        cv2.drawContours(item_mask, [main_contour], 0, 255, -1)

        # Apply the mask to get just the clothing
        clothing = cv2.bitwise_and(img, img, mask=item_mask)
    else:
        # If no significant contours, use the original mask
        clothing = cv2.bitwise_and(img, img, mask=mask)

    # Step 3: Create a central weighting map (pixels in center get higher weight)
    height, width = clothing.shape[:2]
    center_y, center_x = height // 2, width // 2
    y, x = np.ogrid[:height, :width]
    # Create distance map from center
    dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    # Normalize and invert so center has high values
    max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
    weight_map = 1 - (dist_from_center / max_dist)
    # Apply threshold to make it binary (optional)
    # weight_map = (weight_map > 0.7).astype(np.uint8) * 255

    # Step 4: Apply K-means clustering with weighted samples
    # Reshape the image and remove black background
    pixels = clothing.reshape(-1, 3)
    mask_flat = mask.reshape(-1)
    valid_pixels = pixels[mask_flat > 0]

    if len(valid_pixels) == 0:
        print("Warning: No valid pixels found after masking")
        return None

    # Get weights for each pixel
    weight_flat = weight_map.reshape(-1)
    valid_weights = weight_flat[mask_flat > 0]

    # Apply K-means clustering with weights (sample_weight parameter)
    clusters = 5
    kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
    kmeans.fit(valid_pixels, sample_weight=valid_weights)

    # Get colors and calculate percentages
    colors = kmeans.cluster_centers_.astype(np.uint8)
    labels = kmeans.labels_

    # Count pixels per cluster with weighting
    weighted_counts = np.zeros(clusters)
    for i in range(clusters):
        weighted_counts[i] = np.sum(valid_weights[labels == i])

    # Calculate percentages
    percentages = weighted_counts / np.sum(weighted_counts)

    # Step 5: Validate colors (avoid very dark colors being selected as dominant)
    # Calculate brightness of each color
    brightnesses = np.sum(colors, axis=1) / 3

    # Filter out very dark colors (brightness < 30)
    valid_colors = []
    for i in range(clusters):
        if brightnesses[i] > 30 or percentages[i] > 0.5:  # Keep dark only if dominant
            valid_colors.append((percentages[i], colors[i]))

    # If no valid colors, return the original results
    if not valid_colors:
        valid_colors = [(percentages[i], colors[i]) for i in range(clusters)]

    # Sort by percentage
    valid_colors.sort(reverse=True)

    # Convert BGR to RGB for the result
    dominant_color = valid_colors[0]
    return (float(dominant_color[0]), dominant_color[1][::-1])  # Reverse BGR to RGB

def extract_material_properties(img_path):
    """
    Extract material properties from an image including:
    - Dominant colors
    - Texture patterns
    - Material type estimation based on texture analysis

    Returns a dictionary of material properties
    """
    # Load image
    img = cv2.imread(img_path)
    img = imutils.resize(img, height=300)  # Resize for consistent processing

    # 1. Color analysis (using your existing KMeans approach)
    flat_img = np.reshape(img, (-1, 3))
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(flat_img)

    dominant_colors = np.array(kmeans.cluster_centers_, dtype='uint')
    percentages = (np.unique(kmeans.labels_, return_counts=True)[1]) / flat_img.shape[0]
    p_and_c = sorted(zip(percentages, dominant_colors), reverse=True)

    # 2. Texture analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate texture features (variance as a simple measure)
    texture_variance = np.var(gray)

    # Calculate edge density (a proxy for texture complexity)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

    # 3. Material type estimation based on texture properties
    material_type = "unknown"

    # Simple heuristic-based material classification
    if edge_density < 0.05 and texture_variance < 50:
        material_type = "smooth"  # Might be leather, silk, etc.
    elif edge_density > 0.2:
        material_type = "textured"  # Might be denim, wool, etc.
    elif texture_variance > 200:
        material_type = "patterned"  # Has distinct patterns
    else:
        material_type = "medium"  # Medium texture, like cotton

    # Return the extracted material properties
    return {
        "dominant_colors": [color.tolist() for _, color in p_and_c[:3]],
        "color_percentages": [float(pct) for pct, _ in p_and_c[:3]],
        "texture_variance": float(texture_variance),
        "edge_density": float(edge_density),
        "estimated_material": material_type,
        "primary_color_rgb": p_and_c[0][1].tolist()
    }

import fastai
from fastai.vision.all import *
import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastai.vision import *
from fastai.metrics import accuracy, top_k_accuracy
from PIL import Image

PATH = r'C:\Users\Diana\Desktop\Wardrobe-login\Wardrobe-logn'
PATH1 = r"C:\Users\Diana\Desktop\Wardrobe-login\Wardrobe-logn"


@app.route('/predict', methods=['POST'])
@login_required
def upload():
    if request.method == 'POST':
        try:
            f = request.files['file']
            if not f:
                return "No file uploaded", 400

            user_id = session['user']['_id']
            upload_dir = os.path.join('flaskapp', 'static', 'image_users', user_id)
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, secure_filename(f.filename))
            f.save(file_path)

            # Make predictions with validation
            try:
                preds = model_predict(file_path)  # NEW - no model parameter
                if not isinstance(preds, np.ndarray) or preds.size == 0:
                    raise ValueError("Invalid prediction output")

                # Extract color information
                color_result = predict_color(file_path)
                if not color_result or len(color_result) < 2:
                    raise ValueError("Invalid color prediction")

                # Extract material properties - NEW
                material_properties = extract_material_properties(file_path)

                predicted_label = np.argmax(preds)
                class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

                if predicted_label >= len(class_names):
                    raise ValueError("Invalid predicted label index")

                result = class_names[predicted_label]

                # Save to database with material properties
                db.wardrobe.insert_one({
                    'label': result,
                    'color': ' '.join(map(str, color_result[1])),
                    'nota': 4,
                    'userId': user_id,
                    'file_path': f'/static/image_users/{user_id}/{secure_filename(f.filename)}',
                    'material_properties': material_properties,  # NEW - store material data
                    'texture_path': f'/static/image_users/{user_id}/{secure_filename(f.filename)}',  # Same as file_path for now
                    'created_at': datetime.now()
                })

                return result

            except Exception as e:
                print(f"Prediction error: {str(e)}")
                if os.path.exists(file_path):
                    os.remove(file_path)
                return str(e), 500

        except Exception as e:
            return str(e), 500

    return None


@app.route('/api/wardrobe/material/<item_id>', methods=['GET'])
@login_required
def get_material_properties(item_id):
    try:
        userId = session['user']['_id']
        item = db.wardrobe.find_one({'_id': ObjectId(item_id), 'userId': userId})

        if not item:
            return jsonify({'error': 'Item not found'}), 404

        # If material properties don't exist yet, extract them now
        if 'material_properties' not in item or not item['material_properties']:
            file_path = os.path.join('flaskapp', item['file_path'].lstrip('/'))
            if os.path.exists(file_path):
                material_properties = extract_material_properties(file_path)

                # Generate normal map if appropriate
                normal_map_path = None
                # Safely check for pattern_info
                has_pattern = False
                pattern_strength = 0.0

                if 'pattern_info' in material_properties:
                    pattern_info = material_properties['pattern_info']
                    has_pattern = pattern_info.get('has_pattern', False)
                    pattern_strength = pattern_info.get('pattern_strength', 0.0)

                if (material_properties.get('estimated_material') in ['textured', 'rough_textured',
                                                                      'woven_patterned'] or
                        (has_pattern and pattern_strength > 0.3)):
                    try:
                        normal_map_path = generate_normal_map(file_path)
                        if normal_map_path:
                            # Convert to database path format
                            normal_map_path = normal_map_path.replace(os.path.join('flaskapp', ''), '/')
                    except Exception as e:
                        print(f"Error generating normal map: {str(e)}")
                        # Continue even if normal map generation fails

                update_data = {'material_properties': material_properties}
                if normal_map_path:
                    update_data['normal_map_path'] = normal_map_path

                db.wardrobe.update_one(
                    {'_id': ObjectId(item_id)},
                    {'$set': update_data}
                )

                item['material_properties'] = material_properties
                item['normal_map_path'] = normal_map_path
            else:
                return jsonify({'error': 'Image file not found'}), 404

        # Ensure pattern_info exists in material_properties
        if 'material_properties' in item and 'pattern_info' not in item['material_properties']:
            # Add a default pattern_info
            item['material_properties']['pattern_info'] = {
                "pattern_type": "regular",
                "pattern_scale": "medium",
                "pattern_strength": 0.3,
                "has_pattern": False,
                "pattern_regularity": 0.5,
                "is_directional": False,
                "peak_count": 0
            }

            # Update in database too
            db.wardrobe.update_one(
                {'_id': ObjectId(item_id)},
                {'$set': {'material_properties': item['material_properties']}}
            )

        # Get normalized paths
        texture_path = normalize_path(item.get('file_path', ''))
        normal_map_path = normalize_path(item.get('normal_map_path', '')) if item.get('normal_map_path') else None

        return jsonify({
            'success': True,
            'itemId': str(item['_id']),
            'label': item['label'],
            'materialProperties': item['material_properties'],
            'texturePath': texture_path,
            'normalMapPath': normal_map_path
        })

    except Exception as e:
        print(f"Error getting material properties: {str(e)}")
        return jsonify({'error': str(e)}), 500

def load_model():
    path = r'C:\Users\Diana\Desktop\Wardrobe-login\Wardrobe-logn\atr-recognition-stage-3-resnet34.pth'
    # assert os.path.isfile(path)
    # map_location = torch.device('cpu')
    learn = load_learner(path, cpu=True)
    print(learn)
    return learn

def get_x(r):
    new_path = r["image_name"].replace('\\', '//')
    one_path = os.path.join(PATH1, new_path)
    filename = Path(one_path)
    # print(filename)
    return filename

def get_y(r): return r['labels'].split(',')

def splitter(df):
    train = df.index[df['is_valid'] == 0].tolist()
    valid = df.index[df['is_valid'] == 1].tolist()
    return train, valid

def predict_attribute(model, path, display_img=True):
    predicted = model.predict(path)
    # if display_img:
    #     size = 244,244
    #     img=Image.open(path)
    #     # img.thumbnail(size,Image.ANTIALIAS)
    #     img.show()
    return predicted[0]

def accuracy_multi(inp, targ, thresh=0.5, sigmoid=True):
    "Compute accuracy when `inp` and `targ` are the same size."
    if sigmoid: inp = inp.sigmoid()
    return ((inp > thresh) == targ.bool()).float().mean()
class LabelSmoothingBCEWithLogitsLossFlat(BCEWithLogitsLossFlat):
    def __init__(self, eps: float = 0.1, **kwargs):
        self.eps = eps
        super().__init__(thresh=0.2, **kwargs)

    def __call__(self, inp, targ, **kwargs):
        # https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/166833#929222
        targ_smooth = targ.float() * (1. - self.eps) + 0.5 * self.eps
        return super().__call__(inp, targ_smooth, **kwargs)

    def __repr__(self):
        return "FlattenedLoss of LabelSmoothingBCEWithLogits()"

from fastai.vision.all import DataBlock, ImageBlock, MultiCategoryBlock, RandomResizedCrop, aug_transforms
import pandas as pd
import torch
from fastai.vision.all import *
from functools import partial
def predict_attribute_model(img_path):
    print('alo alo')

    # Define paths to files
    TRAIN_PATH = "multilabel-train.csv"
    TEST_PATH = "multilabel-test.csv"
    CLASSES_PATH = "attribute-classes.txt"

    # Load training data
    try:
        train_df = pd.read_csv(TRAIN_PATH)
        print("Training data loaded successfully.")
    except Exception as e:
        print(f"Error loading training data: {e}")
        return

    # Parameters and functions for optimization
    wd = 5e-7  # weight decay parameter
    opt_func = partial(ranger, wd=wd)

    splitter = RandomSplitter()  # Example splitter
    get_x = lambda x: x[0]  # Example get_x function
    get_y = lambda x: x[1]  # Example get_y function

    # Define DataBlock
    print('Define datablock')
    try:
        dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                           splitter=splitter,
                           get_x=get_x,
                           get_y=get_y,
                           item_tfms=RandomResizedCrop(224, min_scale=0.8),
                           batch_tfms=aug_transforms())

        dls = dblock.dataloaders(train_df, num_workers=0)
        print('DataBlock and DataLoaders created successfully.')
    except Exception as e:
        print(f"Error creating DataBlock or DataLoaders: {e}")
        return

    # Show a batch of images
    try:
        dls.show_batch(nrows=1, ncols=6)
        print('Batch of images shown.')
    except Exception as e:
        print(f"Error showing batch: {e}")

    # Define metrics
    metrics = [FBetaMulti(2.0, 0.2, average='samples'), partial(accuracy_multi, thresh=0.2)]

    # Load test data
    try:
        test_df = pd.read_csv(TEST_PATH)
        print("Test data loaded successfully.")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return

    # Define a new DataBlock for test data
    try:
        dblock_test = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                                get_x=get_x,
                                get_y=get_y,
                                item_tfms=Resize(224))
        print('Test DataBlock created.')
    except Exception as e:
        print(f"Error creating test DataBlock: {e}")
        return

    # Initialize and load model
    try:
        learn = vision_learner(dls, resnet34, loss_func=LabelSmoothingBCEWithLogitsLossFlat(),
                               metrics=metrics, opt_func=opt_func).to_fp16()

        # Load the pre-trained model
        path = r'C:\Users\Diana\Desktop\Wardrobe-login\Wardrobe-logn\atr-recognition-stage-3-resnet34.pth'
        learn.load_state_dict(torch.load(path, map_location=torch.device('cpu'))['model'])
        print('Model loaded successfully.')
    except Exception as e:
        print(f"Error initializing or loading model: {e}")
        return

    # Predict attributes
    try:
        label_result = predict_attribute(learn, img_path)  # Ensure predict_attribute function is defined
        print(f'Prediction result: {label_result}')
        return label_result
    except Exception as e:
        print(f"Error predicting attributes: {e}")
        return

# flask app and routes
@app.route('/')
def home():
    return render_template('welcome.html')

from flaskapp.user.routes import *

@app.route('/login/')
def dologin():
    return render_template('home.html')

@app.route('/register/')
def doregister():
    return render_template('register.html')

# @app.route('/user/signOut')
# def signout():
#     # Clear the session
#     session.clear()
#     # Redirect to the login page
#     return redirect(url_for('dologin'))
#profile
# Add these imports to your existing imports
import base64
from bson.binary import Binary


# Add this route to handle profile updates
@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        try:
            user_id = session['user']['_id']

            # Handle profile picture upload
            if 'profile_picture' in request.files:
                file = request.files['profile_picture']
                if file and allowed_file(file.filename):
                    # Read the file and convert to binary for MongoDB storage
                    file_data = file.read()

                    # Create upload directory if it doesn't exist
                    upload_dir = os.path.join('flaskapp', 'static', 'profile_pictures', str(user_id))
                    os.makedirs(upload_dir, exist_ok=True)

                    # Save file to disk
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(upload_dir, filename)
                    with open(file_path, 'wb') as f:
                        f.write(file_data)

                    # Update user document with profile picture path
                    db.users.update_one(
                        {'_id': user_id},
                        {'$set': {
                            'profile_picture': f'/static/profile_pictures/{user_id}/{filename}'
                        }}
                    )

            # Update other profile information
            name = request.form.get('name')
            email = request.form.get('email')

            # Update user document
            update_data = {}
            if name:
                update_data['name'] = name
            if email:
                update_data['email'] = email

            if update_data:
                db.users.update_one(
                    {'_id': user_id},
                    {'$set': update_data}
                )

                # Update session data
                user = db.users.find_one({'_id': user_id})
                session['user'] = user

            return jsonify({'success': True, 'message': 'Profile updated successfully'})

        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500

    # GET request - display profile page
    user_data = db.users.find_one({'_id': session['user']['_id']})
    return render_template('profile.html', user=user_data)



# Helper function to get user's profile picture
def get_user_profile_picture():
    if 'user' in session and session['user']:
        user = db.users.find_one({'_id': session['user']['_id']})
        profile_picture = user.get('profile_picture') if user else None
        if profile_picture:
            return profile_picture
    return None  # Return None instead of a default image path


# Add this to your template context
@app.context_processor
def utility_processor():
    return dict(get_user_profile_picture=get_user_profile_picture)


@app.route('/dashboard/', methods=['GET', 'POST'])
@login_required
def dashboard():
    userId = session['user']['_id']
    cityByDefault = 'Bucharest'
    api_key = 'aa73cad280fbd125cc7073323a135efa'

    if request.method == 'POST':
        new_city = request.form.get('city')
        print(f"New city submitted: {new_city}")

        if new_city:
            # Check if city already exists for this user
            existing_city = db.city.find_one({'name': {'$regex': f'^{new_city}$', '$options': 'i'}, 'userId': userId})
            if existing_city:
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({
                        'success': False,
                        'error': 'duplicate',
                        'message': f'{new_city} is already in your weather list.'
                    }), 400
                else:
                    flash(f'{new_city} is already in your weather list.', 'warning')
                    return redirect(url_for('dashboard'))

            # Get geocoding data
            geocode_url = f'http://api.openweathermap.org/geo/1.0/direct?q={new_city}&limit=1&appid={api_key}'
            try:
                geocode_response = requests.get(geocode_url, timeout=5).json()
                print(f"Geocode response: {geocode_response}")

                if geocode_response:
                    lat = geocode_response[0].get('lat')
                    lon = geocode_response[0].get('lon')
                    city_name_from_api = geocode_response[0].get('name', new_city)

                    if lat and lon:
                        # Insert the new city
                        db.city.insert_one({
                            'name': city_name_from_api,
                            'lat': lat,
                            'lon': lon,
                            'userId': userId
                        })

                        # Get weather data for the new city
                        weather_url = f'https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric'
                        weather_response = requests.get(weather_url, timeout=5).json()

                        new_weather_data = []
                        if weather_response.get('weather') and weather_response.get('main'):
                            weather = {
                                'city': city_name_from_api,
                                'temperature': round(weather_response['main']['temp']),
                                'description': weather_response['weather'][0]['description'].title(),
                                'icon': weather_response['weather'][0]['icon'],
                            }
                            new_weather_data.append(weather)

                        # Return JSON response for AJAX requests
                        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                            return jsonify({
                                'success': True,
                                'weather_data': new_weather_data,
                                'message': f'{city_name_from_api} added successfully!'
                            })
                        else:
                            flash(f'{city_name_from_api} added successfully!', 'success')
                            return redirect(url_for('dashboard'))
                    else:
                        error_msg = f'Could not get coordinates for {new_city}.'
                        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                            return jsonify({'success': False, 'message': error_msg}), 400
                        else:
                            flash(error_msg, 'error')
                            return redirect(url_for('dashboard'))
                else:
                    error_msg = f'City "{new_city}" not found. Please check the spelling.'
                    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                        return jsonify({'success': False, 'message': error_msg}), 400
                    else:
                        flash(error_msg, 'error')
                        return redirect(url_for('dashboard'))

            except requests.RequestException as e:
                error_msg = 'Error connecting to weather service. Please try again.'
                print(f"API request error: {e}")
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({'success': False, 'message': error_msg}), 500
                else:
                    flash(error_msg, 'error')
                    return redirect(url_for('dashboard'))

    # GET request or fallback - render the dashboard
    filter = {'userId': userId}

    # Add default city if user has no cities
    if db.city.find_one(filter) is None:
        geocode_url = f'http://api.openweathermap.org/geo/1.0/direct?q={cityByDefault}&limit=1&appid={api_key}'
        try:
            geocode_response = requests.get(geocode_url, timeout=5).json()
            print(f"Default city geocode response: {geocode_response}")

            if geocode_response:
                lat = geocode_response[0].get('lat')
                lon = geocode_response[0].get('lon')
                if lat and lon:
                    db.city.insert_one({'name': cityByDefault, 'lat': lat, 'lon': lon, 'userId': userId})
        except requests.RequestException as e:
            print(f"Error adding default city: {e}")

    # Get all cities for the user
    cities = list(db.city.find(filter))
    weather_data = []

    for city in cities:
        try:
            # Ensure we have coordinates
            if 'lat' not in city or 'lon' not in city:
                geocode_url = f'http://api.openweathermap.org/geo/1.0/direct?q={city["name"]}&limit=1&appid={api_key}'
                geocode_response = requests.get(geocode_url, timeout=5).json()
                print(f"Geocode response for {city['name']}: {geocode_response}")

                if geocode_response:
                    lat = geocode_response[0].get('lat')
                    lon = geocode_response[0].get('lon')
                    if lat and lon:
                        db.city.update_one({'_id': city['_id']}, {'$set': {'lat': lat, 'lon': lon}})
                        city['lat'] = lat
                        city['lon'] = lon
                else:
                    continue

            # Get weather data
            weather_url = f'https://api.openweathermap.org/data/2.5/weather?lat={city["lat"]}&lon={city["lon"]}&appid={api_key}&units=metric'
            weather_response = requests.get(weather_url, timeout=5).json()
            print(f"Weather API response for {city['name']}: {weather_response}")

            if weather_response.get('weather') and weather_response.get('main'):
                weather = {
                    'city': city['name'],
                    'temperature': round(weather_response['main']['temp']),
                    'description': weather_response['weather'][0]['description'].title(),
                    'icon': weather_response['weather'][0]['icon'],
                }
                weather_data.append(weather)

        except requests.RequestException as e:
            print(f"Error getting weather for {city['name']}: {e}")
            continue

    print(f"Weather data to be rendered: {weather_data}")
    return render_template('dashboard.html', weather_data=weather_data)


@app.route('/dashboard/delete_city/<city_name>', methods=['DELETE'])
@login_required
def delete_city(city_name):
    userId = session['user']['_id']

    try:
        # Find and delete the city
        result = db.city.delete_one({
            'name': {'$regex': f'^{city_name}$', '$options': 'i'},
            'userId': userId
        })

        if result.deleted_count > 0:
            return jsonify({
                'success': True,
                'message': f'{city_name} removed from your weather list.'
            })
        else:
            return jsonify({
                'success': False,
                'message': f'City {city_name} not found in your list.'
            }), 404

    except Exception as e:
        print(f"Error deleting city {city_name}: {e}")
        return jsonify({
            'success': False,
            'message': 'Error removing city. Please try again.'
        }), 500



@app.route('/wardrobe/delete/<item_id>', methods=['DELETE'])
@login_required
def delete_wardrobe_item(item_id):
    try:
        userId = session['user']['_id']
        # Get item to delete its image file
        item = db.wardrobe.find_one({'_id': ObjectId(item_id), 'userId': userId})

        if not item:
            return jsonify({'error': 'Item not found'}), 404

        # Delete physical file if it exists
        file_path = os.path.join('flaskapp', item['file_path'].lstrip('/'))
        if os.path.exists(file_path):
            os.remove(file_path)

        # Delete from database
        result = db.wardrobe.delete_one({'_id': ObjectId(item_id), 'userId': userId})

        if result.deleted_count:
            return jsonify({'success': True})
        return jsonify({'error': 'Delete failed'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# First, define the helper function
DEFAULT_RATING = 4


def prepare_features(include_weather, event, temperature):
    """Prepare features for model prediction"""
    features = [0] * 10

    # Weather features
    if include_weather == 'yes':
        features[0] = 1
        if temperature <= 6.0:
            features[1:6] = [1, 0, 0, 0, 0]
        elif 6.0 < temperature <= 15.0:
            features[1:6] = [0, 1, 0, 0, 0]
        elif 15.0 < temperature < 26.0:
            features[1:6] = [0, 0, 1, 0, 0]
        elif temperature >= 26.0:
            features[1:6] = [0, 0, 0, 1, 0]
    else:
        features[0] = 0
        features[1:6] = [0, 0, 0, 0, 0]

    # Event features
    event_mapping = {
        'event': [1, 0, 0, 0],
        'walk': [0, 1, 0, 0],
        'work': [0, 0, 1, 0],
        'travel': [0, 0, 0, 1]
    }
    features[6:10] = event_mapping.get(event, [0, 0, 0, 0])

    return features


import re


def normalize_path(file_path):
    """
    Comprehensive path normalization that handles all cases:
    - Empty paths
    - /outfit/ prefixes
    - Duplicate /static/ paths
    - Leading/trailing slashes
    - Multiple combinations of the above
    Always returns a path starting with '/'.
    """
    if not file_path:
        return None

    # First remove /outfit/ prefix anywhere in the path
    normalized = file_path.replace('/outfit/', '/')

    # Handle multiple static occurrences
    while '/static/static/' in normalized:
        normalized = normalized.replace('/static/static/', '/static/')

    # Handle any remaining path cleanup
    normalized = normalized.strip('/')

    # Special case: ensure static paths start with static/
    if 'static' in normalized and not normalized.startswith('static'):
        normalized = 'static/' + normalized[normalized.index('static') + 6:]

    # Always return with leading slash
    if not normalized.startswith('/'):
        normalized = '/' + normalized

    return normalized



@app.route('/recommendations', methods=['GET', 'POST'])
@login_required
def get_outfit():
    print("Debug: Entering get_outfit route")

    try:
        userId = session['user']['_id']
        if not userId:
            return jsonify({"success": False, "message": "User not logged in"}), 400

        print(f"Debug: User ID: {userId}")
        user_filter = {'userId': userId, 'isFavorite': 'yes'}

        # Get user's clothes with normalized paths
        users_clothes = db.outfits.find(user_filter)
        normalized_clothes = []
        for outfit in users_clothes:
            if 'outfit' in outfit:
                for piece in outfit['outfit']:
                    piece['file_path'] = normalize_path(piece.get('file_path', ''))
                    print('olala' +  piece['file_path'])
            normalized_clothes.append(outfit)

        cityByDefault = 'Bucharest'
        DEFAULT_RATING = 4

        # Default to show generator and hide outfits
        show_generator = True
        show_outfits = False
        success_message = None
        error_message = None

        # Define available outfit combinations
        result_outfit = [
            'Dress_Sandal', 'T-shirt/top_Trouser_Sneaker', 'Shirt_Trouser',
            'Shirt_Trouser_Sneaker', 'Dress_Sandal_Coat', 'T-shirt/top_Trouser',
            'Shirt_Trouser_Coat', 'Shirt_Trouser_Coat', 'Dress_Ankle-boot_Coat',
            'Pullover_Trouser_Ankle-boot', 'Dress_Sneaker', 'Shirt_Trouser_Sandal',
            'Dress_Sandal_Bag'
        ]

        # Initialize city if not exists
        city_filter = {'userId': userId}
        if db.city.count_documents(city_filter) == 0:
            print(f"Debug: Creating new city entry for user {userId}")
            db.city.insert_one({'name': cityByDefault, 'userId': userId})

        # Get weather data
        cities = db.city.find(city_filter)
        weather_data = []
        url = 'http://api.openweathermap.org/data/2.5/weather?q={}&units=metric&appid=aa73cad280fbd125cc7073323a135efa'

        for city in cities:
            try:
                print(f"Debug: Fetching weather for {city['name']}")
                r = requests.get(url.format(city['name']), timeout=5).json()
                weather = {
                    'city': city['name'],
                    'temperature': r['main']['temp'],
                    'description': r['weather'][0]['description'],
                    'icon': r['weather'][0]['icon'],
                }
                weather_data.append(weather)
            except Exception as e:
                print(f"Error fetching weather for {city['name']}: {e}")
                weather_data.append({
                    'city': city['name'],
                    'temperature': 20,
                    'description': '',
                    'icon': ''
                })

        # Ensure we have 3 weather options
        while len(weather_data) < 3:
            weather_data.append({
                'city': cityByDefault,
                'temperature': 20,
                'description': '',
                'icon': ''
            })

        city1, city2, city3 = weather_data[:3]
        outfit1, outfit2, outfit3 = [], [], []

        if request.method == 'POST':
            print("Debug: Processing POST request")

            # Handle outfit selection
            option = request.form.get('options')
            if option:
                print(f"Debug: Selected option: {option}")
                filter_lookup = {'userId': userId, 'outfitNo': option}
                outfit_doc = db.outfits.find_one(filter_lookup, sort=[('_id', -1)])

                if outfit_doc:
                    # Update outfit pieces ratings
                    updated_pieces = []
                    for piece in outfit_doc['outfit']:
                        try:
                            current_piece = db.wardrobe.find_one({'_id': piece['_id']})
                            if current_piece:
                                piece_data = {
                                    '_id': str(current_piece['_id']),
                                    'label': current_piece.get('label', ''),
                                    'file_path': normalize_path(current_piece.get('file_path', '')),
                                    'color': current_piece.get('color', ''),
                                    'nota': current_piece.get('nota', DEFAULT_RATING)
                                }

                                # Update the rating
                                db.wardrobe.update_one(
                                    {'_id': current_piece['_id']},
                                    {'$set': {'nota': piece_data['nota'] + 1}}
                                )
                                updated_pieces.append(piece_data)
                        except Exception as e:
                            print(f"Error updating piece rating: {str(e)}")

                    try:
                        # Update outfit rating and pieces
                        current_outfit_rating = outfit_doc.get('nota', DEFAULT_RATING)
                        db.outfits.update_one(
                            {'_id': outfit_doc['_id']},
                            {
                                '$set': {
                                    'nota': current_outfit_rating + 1,
                                    'isFavorite': 'yes',
                                    'outfit': updated_pieces
                                }
                            }
                        )
                        success_message = "Outfit has been saved to your favorites!"
                        show_outfits = False
                        return render_template(
                            'outfit_of_the_day.html',
                            success_message=success_message,
                            show_generator=show_generator,
                            show_outfits=show_outfits,
                            city1=city1,
                            city2=city2,
                            city3=city3,
                            wardrobes=normalized_clothes
                        )
                    except Exception as e:
                        print(f"Error updating outfit rating: {str(e)}")
                        error_message = "Error saving outfit. Please try again."

            # Generate new outfits
            include_weather = request.form.get('weather') == 'yes'
            city = request.form.get('city')
            event = request.form.get('events')
            temperature = 20

            if include_weather and city:
                selected_weather = next(
                    (w for w in weather_data if w['city'] == city),
                    {'temperature': 20}
                )
                temperature = selected_weather['temperature']

            try:
                loaded_classifier = joblib.load("./random_forest.joblib")
                features = prepare_features(include_weather, event, temperature)
                result_forest = loaded_classifier.predict([features])
                index_of_outfit = result_forest[0]
                outfit_combination = result_outfit[index_of_outfit]
                filters_outfits = outfit_combination.split('_')

                # Generate three outfits based on classifier suggestion
                for i in range(3):
                    outfit_pieces = []
                    for filter_name in filters_outfits:
                        clothes = list(db.wardrobe.find({
                            'userId': userId,
                            'label': filter_name
                        }).sort('nota', -1))

                        if clothes:
                            # Get top rated pieces (top 5 or all if less than 5)
                            num_top_pieces = min(5, len(clothes))
                            top_pieces = clothes[:num_top_pieces]

                            # Randomly select from top pieces
                            piece = random.choice(top_pieces)
                            piece_data = {
                                '_id': str(piece['_id']),
                                'label': piece.get('label', ''),
                                'file_path': normalize_path(piece.get('file_path', '')),
                                'color': piece.get('color', ''),
                                'nota': piece.get('nota', DEFAULT_RATING)
                            }

                            if 'nota' not in piece:
                                db.wardrobe.update_one(
                                    {'_id': piece['_id']},
                                    {'$set': {'nota': DEFAULT_RATING}}
                                )
                            outfit_pieces.append(piece_data)

                    if outfit_pieces:
                        outfit_doc = {
                            'outfit': outfit_pieces,
                            'userId': userId,
                            'nota': DEFAULT_RATING,
                            'outfitNo': f'piece{i + 1}',
                            'isFavorite': 'no',
                            'created_at': datetime.now()
                        }
                        db.outfits.insert_one(outfit_doc)

                        if i == 0:
                            outfit1 = outfit_pieces
                        elif i == 1:
                            outfit2 = outfit_pieces
                        else:
                            outfit3 = outfit_pieces

                show_outfits = True

            except Exception as e:
                print(f"Error generating outfits: {e}")
                error_message = "Error generating outfits. Please try again."

        print("Debug: Rendering template")
        # Get fresh cursor with normalized paths
        users_clothes = db.outfits.find(user_filter)
        normalized_clothes = []
        for outfit in users_clothes:
            if 'outfit' in outfit:
                for piece in outfit['outfit']:
                    piece['file_path'] = normalize_path(piece.get('file_path', ''))
            normalized_clothes.append(outfit)

        return render_template(
            'outfit_of_the_day.html',
            outfit1=outfit1,
            outfit2=outfit2,
            outfit3=outfit3,
            city1=city1,
            city2=city2,
            city3=city3,
            show_generator=show_generator,
            show_outfits=show_outfits,
            success_message=success_message,
            error_message=error_message,
            wardrobes=normalized_clothes
        )

    except Exception as e:
        print(f"Error in get_outfit: {str(e)}")
        return render_template(
            'outfit_of_the_day.html',
            error_message="An error occurred. Please try again.",
            show_generator=True,
            show_outfits=False,
            city1={'city': cityByDefault, 'temperature': 20, 'description': '', 'icon': ''},
            city2={'city': cityByDefault, 'temperature': 20, 'description': '', 'icon': ''},
            city3={'city': cityByDefault, 'temperature': 20, 'description': '', 'icon': ''},
            wardrobes=[]
        )


# FIXED: Enhanced wardrobe route with proper 3D model fields
@app.route('/wardrobe', methods=['GET', 'POST'])
@login_required
def add_wardrobe():
    """Enhanced wardrobe route with proper 3D model database integration"""
    if request.method == 'POST':
        try:
            # Check if the post request has the file part
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'}), 400

            f = request.files['file']
            if f.filename == '':
                return jsonify({'error': 'No selected file'}), 400

            # Check if the file is allowed
            allowed_extensions = {'png', 'jpg', 'jpeg'}
            if not '.' in f.filename or \
                    f.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
                return jsonify({'error': 'Invalid file type'}), 400

            # Save the file
            user_id = session['user']['_id']
            upload_dir = os.path.join('flaskapp', 'static', 'image_users', user_id)
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, secure_filename(f.filename))
            f.save(file_path)
            file_path_db = f'/static/image_users/{user_id}/{secure_filename(f.filename)}'

            try:
                # Make prediction
                preds = model_predict(file_path)  # NEW - no model parameter
                class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

                # Get predicted label and confidence scores
                predicted_label = np.argmax(preds)
                clothing_type = class_names[predicted_label]

                # Get color prediction using the improved function
                color_result = improved_predict_color(file_path)
                color_percentage = float(color_result[0])
                color_rgb = color_result[1].tolist()

                # Extract material properties
                material_properties = extract_material_properties(file_path)

                # Generate normal map for textured materials
                normal_map_path = None
                has_pattern = False
                pattern_strength = 0.0

                if material_properties:
                    if 'pattern_info' in material_properties:
                        pattern_info = material_properties['pattern_info']
                        has_pattern = pattern_info.get('has_pattern', False)
                        pattern_strength = pattern_info.get('pattern_strength', 0.0)

                    if (material_properties.get('estimated_material') in ['textured', 'rough_textured',
                                                                          'woven_patterned'] or
                            (has_pattern and pattern_strength > 0.3)):
                        try:
                            normal_map_path = generate_normal_map(file_path)
                            if normal_map_path:
                                normal_map_path = normal_map_path.replace(os.path.join('flaskapp', ''), '/')
                        except Exception as e:
                            print(f"Error generating normal map: {str(e)}")

                texture_preview_path = file_path_db

                # FIXED: Save image data to database with complete 3D model fields
                wardrobe_item = {
                    'userId': user_id,
                    'label': clothing_type,
                    'confidence': float(preds[0][predicted_label]),
                    'color': {
                        'percentage': color_percentage,
                        'rgb': color_rgb
                    },
                    'material_properties': material_properties,
                    'normal_map_path': normal_map_path,
                    'filename': secure_filename(f.filename),
                    'file_path': file_path_db,
                    'texture_preview_path': texture_preview_path,
                    'created_at': datetime.now(),
                    'last_worn': None,
                    'times_worn': 0,
                    # CRITICAL: 3D MODEL FIELDS - PROPERLY INITIALIZED
                    'has_3d_model': False,
                    'model_3d_path': None,           # Will store the OBJ/GLB file path
                    'model_generated_at': None,      # When the 3D model was created
                    'model_method': None,            # 'colab', 'triposr', etc.
                    'model_file_format': None,       # 'OBJ', 'GLB', etc.
                    'model_file_size': None,         # File size in bytes
                    'model_last_updated': None,      # Last time the 3D model was updated
                    'model_generation_status': None, # 'generating', 'completed', 'failed'
                    'model_task_id': None           # Generation task ID for tracking
                }

                # INSERT AND GET THE ID - THIS IS CRUCIAL FOR AUTO-SAVE!
                print(f"💾 Inserting wardrobe item to database...")
                insert_result = db.wardrobe.insert_one(wardrobe_item)
                item_id = str(insert_result.inserted_id)

                print(f"✅ Created wardrobe item with ID: {item_id}")

                # CRITICAL: Return success response with the item_id for auto-save!
                return jsonify({
                    'success': True,
                    'item_id': item_id,  # THIS IS THE KEY FIELD FOR AUTO-SAVE!
                    'prediction': clothing_type,
                    'confidence': float(preds[0][predicted_label]),
                    'color': {
                        'percentage': color_percentage,
                        'rgb': color_rgb
                    },
                    'material_properties': material_properties,
                    'normal_map_path': normal_map_path,
                    'texture_preview_path': texture_preview_path
                })

            except Exception as e:
                print(f"Prediction error: {str(e)}")
                if os.path.exists(file_path):
                    os.remove(file_path)
                return jsonify({'error': f'Error processing image: {str(e)}'}), 500

        except Exception as e:
            print(f"Error in file upload: {str(e)}")
            return jsonify({'error': f'Error uploading file: {str(e)}'}), 500

    # GET request
    return render_template('wardrobe.html')


     # FIXED: Add this route to update existing items that don't have model_task_id
    @app.route('/api/wardrobe/fix-missing-task-ids', methods=['POST'])
    @login_required
    def fix_missing_task_ids():
        """Fix existing wardrobe items that are missing model_task_id"""
        try:
            user_id = session['user']['_id']

            # Find items without model_task_id
            items_without_task_id = list(db.wardrobe.find({
                'userId': user_id,
                '$or': [
                    {'model_task_id': None},
                    {'model_task_id': {'$exists': False}}
                ]
            }))

            updated_count = 0

            for item in items_without_task_id:
                # Generate a new model_task_id
                import uuid
                new_task_id = f"item_{user_id}_{uuid.uuid4().hex[:8]}"

                # Update the item
                result = db.wardrobe.update_one(
                    {'_id': item['_id']},
                    {'$set': {'model_task_id': new_task_id}}
                )

                if result.modified_count > 0:
                    updated_count += 1
                    print(f"✅ Updated item {item['_id']} with task_id: {new_task_id}")

            return jsonify({
                'success': True,
                'message': f'Updated {updated_count} items with model_task_id',
                'updated_count': updated_count
            })

        except Exception as e:
            print(f"❌ Error fixing task IDs: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/wardrobe/all', methods=['GET', 'POST'])
@login_required
def view_wardrobe_all():
    userId = session['user']['_id']
    filter = {'userId': userId}
    users_clothes = db.wardrobe.find(filter)

    categories = {
        'tops': ['T-shirt/top', 'Shirt', 'Pullover'],
        'bottoms': ['Trouser'],
        'dresses': ['Dress'],
        'outerwear': ['Coat'],
        'shoes': ['Sandal', 'Sneaker', 'Ankle boot'],
        'accessories': ['Bag']
    }
    grouped_items = {cat: [] for cat in categories.keys()}
    for item_doc in users_clothes:
        label = item_doc.get('label', '')
        category = item_doc.get('category')
        if not category:
            category = next((cat for cat, lbls in categories.items() if label in lbls), None)
        if category:
            # Use the same logic as calendar: always normalize file_path
            grouped_items[category].append({
                'id': str(item_doc['_id']),
                'label': item_doc['label'],
                'file_path': normalize_path(item_doc.get('file_path', '')),
                'color': item_doc.get('color', '')
            })
    return render_template('wardrobe_all2.html', wardrobe_items=grouped_items)


@app.route('/outfits/all', methods=['GET', 'POST'])
@login_required
def view_outfits_all():
    userId = session['user']['_id']
    print(userId)
    filter = {'userId': userId, 'isFavorite': 'yes'}
    users_clothes = db.outfits.find(filter)
    return render_template('outfits_all.html', wardrobes=users_clothes)







# avatar logic


@app.route('/api/avatar/<gender>', methods=['GET'])
@login_required
def get_avatar(gender):
    if gender not in ['male', 'female']:
        return jsonify({'error': 'Invalid gender specified'}), 400

    model_file = f'{gender}.gltf'
    return send_from_directory(app.config['MODELS_FOLDER'], model_file, mimetype='model/gltf+json')


@app.route('/api/clothes', methods=['GET'])
@login_required
def get_available_clothes():
    # Return list of available clothing items for the user
    clothes_path = os.path.join(app.config['MODELS_FOLDER'], 'clothing')
    available_clothes = []

    for filename in os.listdir(clothes_path):
        if allowed_file(filename):
            clothes_id = filename.rsplit('.', 1)[0]
            available_clothes.append({
                'id': clothes_id,
                'name': clothes_id.replace('_', ' ').title(),
                'path': f'/api/models/clothing/{filename}'
            })

    return jsonify(available_clothes)



def get_user_preferences(user_id):
    try:
        # You can replace this with your database query
        with open(f'user_preferences/{user_id}.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        # Return default preferences if none exist
        return {
            'gender': 'female',
            'skinColor': '#FFE0BD',
            'hairColor': '#000000',
            'hairStyle': 'default',
            'clothes': []
        }


def save_user_preferences(user_id, preferences):
    # You can replace this with your database save logic
    with open(f'user_preferences/{user_id}.json', 'w') as file:
        json.dump(preferences, file)


@app.route('/avatar')
def avatar_page():
    user_id = session.get('user_id')
    user_preferences = get_user_preferences(user_id)
    return render_template('avatar.html', user_preferences=user_preferences)


@app.route('/api/avatar/customize', methods=['POST'])
def customize_avatar():
    user_id = session.get('user_id')
    data = request.get_json()

    # Update user preferences
    user_preferences = {
        'gender': data.get('gender', 'female'),
        'skinColor': data.get('skinColor', '#FFE0BD'),
        'hairColor': data.get('hairColor', '#000000'),
        'hairStyle': data.get('hairStyle', 'default'),
        'clothes': data.get('clothes', [])
    }

    # Save the updated preferences
    save_user_preferences(user_id, user_preferences)

    return jsonify({'status': 'success', 'preferences': user_preferences})








# calendar logic

UPLOAD_FOLDER = 'static/image_users/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(os.path.join('flaskapp', UPLOAD_FOLDER), exist_ok=True)

@app.route('/api/wardrobe-items', methods=['GET'])
def get_wardrobe_items():
    """Get wardrobe items grouped by category"""
    user_id = session.get('user', {}).get('_id', '')
    if not user_id:
        return jsonify({"success": False, "message": "User not logged in"}), 400

    items_cursor = db.wardrobe.find({'userId': user_id})

    categories = {
        'tops': ['T-shirt/top', 'Shirt', 'Pullover'],
        'bottoms': ['Trouser'],
        'dresses': ['Dress'],
        'outerwear': ['Coat'],
        'shoes': ['Sandal', 'Sneaker', 'Ankle boot'],
        'accessories': ['Bag']
    }

    grouped_items = {cat: [] for cat in categories.keys()}

    for item_doc in items_cursor:
        label = item_doc.get('label', '')
        # Use stored category if present, otherwise infer from label
        category = item_doc.get('category')
        if not category:
            category = next((cat for cat, lbls in categories.items() if label in lbls), None)
        if category:
            grouped_items[category].append({
                '_id': str(item_doc['_id']),
                'label': label,
                'file_path': normalize_path(item_doc.get('file_path', '')),
                'color': item_doc.get('color', ''),
                'category': category
            })

    return jsonify(grouped_items)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

def save_uploaded_image(base64_string, user_id):
    try:
        base64_data = base64_string.split(',')[1] if ',' in base64_string else base64_string
        image_data = base64.b64decode(base64_data)
        images_dir = os.path.join('flaskapp', app.config['UPLOAD_FOLDER'], user_id)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'calendar_outfit_{timestamp}.jpg'
        full_path = os.path.join(images_dir, filename)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'wb') as f:
            f.write(image_data)
        print(f"Image saved to: {full_path}")
        return f'/static/image_users/{user_id}/{filename}'
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        return None


@app.route('/calendar', methods=['GET'])
@login_required
def calendar_view():
    year = int(request.args.get('year', datetime.now().year))
    month = int(request.args.get('month', datetime.now().month))
    user_id = session.get('user', {}).get('_id', '')
    if not user_id:
        return jsonify({"error": "User not logged in"}), 403

    calendar.setfirstweekday(calendar.MONDAY)
    weeks = calendar.monthcalendar(year, month)
    outfit_map = {}

    try:
        outfits_cursor = db.calendar.find({"user_id": user_id, "year": year, "month": month})
        for outfit_doc in outfits_cursor:
            day_number = outfit_doc['day']
            outfit_items = []
            if 'items' in outfit_doc and isinstance(outfit_doc['items'], list):
                for item_id in outfit_doc['items']:
                    item_obj = db.wardrobe.find_one({'_id': ObjectId(item_id)})
                    if item_obj:
                        outfit_items.append({
                            'id': str(item_obj['_id']),
                            'label': item_obj['label'],
                            'file_path': normalize_path(item_obj.get('file_path', '')),
                            'color': item_obj.get('color', '')
                        })
            custom_image = normalize_path(outfit_doc.get('custom_image'))
            print(f"Normalized custom image path: {custom_image}")
            if custom_image:
                full_path = os.path.join('flaskapp', custom_image.lstrip('/'))
                print(f"Full path: {full_path}")
                print(f"File exists: {os.path.exists(full_path)}")
            outfit_map[day_number] = {
                'id': str(outfit_doc['_id']),
                'outfit_items': outfit_items,
                'description': outfit_doc.get('description', ''),
                'custom_image': custom_image
            }
    except Exception as e:
        print(f"Error loading outfits: {str(e)}")
        outfit_map = {}

    return render_template(
        'calendar.html',
        year=year,
        month=month,
        weeks=weeks,
        outfits=outfit_map,
        month_names=[
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]
    )

@app.route('/calendar/add', methods=['POST'])
@login_required
def add_calendar_outfit():
    try:
        user_id = session.get('user', {}).get('_id', '')
        if not user_id:
            return jsonify({'success': False, 'message': 'User not found in session!'}), 400
        data = request.json
        if not data:
            return jsonify({'success': False, 'message': 'No data provided!'}), 400

        day = int(data.get('day', 0))
        month = int(data.get('month', 0))
        year = int(data.get('year', 0))
        description = data.get('description', '')
        selected_items = data.get('selected_items', [])
        items_object_ids = [ObjectId(i) for i in selected_items if i]

        custom_image_path = None
        uploaded_image = data.get('uploaded_image')
        if uploaded_image:
            custom_image_path = save_uploaded_image(uploaded_image, user_id)

        existing_outfit = db.calendar.find_one({
            'user_id': user_id,
            'day': day,
            'month': month,
            'year': year
        })

        if existing_outfit:
            update_data = {
                'description': description,
                'items': items_object_ids
            }
            if custom_image_path:
                update_data['custom_image'] = custom_image_path
            db.calendar.update_one(
                {'_id': existing_outfit['_id']},
                {'$set': update_data}
            )
            msg = 'Outfit updated successfully!'
        else:
            insert_data = {
                'user_id': user_id,
                'day': day,
                'month': month,
                'year': year,
                'description': description,
                'items': items_object_ids,
                'created_at': datetime.now()
            }
            if custom_image_path:
                insert_data['custom_image'] = custom_image_path
            db.calendar.insert_one(insert_data)
            msg = 'Outfit added successfully!'

        return jsonify({'success': True, 'message': msg})
    except Exception as e:
        print(f"[ERROR] add_calendar_outfit: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/calendar/delete', methods=['DELETE'])
def delete_calendar_outfit():
    """Delete an outfit from the calendar."""
    user_id = session.get('user', {}).get('_id', '')
    day, month, year = int(request.args.get('day')), int(request.args.get('month')), int(request.args.get('year'))

    result = db.calendar.delete_one({"user_id": user_id, "day": day, "month": month, "year": year})

    if result.deleted_count > 0:
        return jsonify({"success": True, "message": "Outfit deleted successfully!"})
    return jsonify({"success": False, "message": "Outfit not found!"}), 404

# rpm avatar

import requests
import base64
from io import BytesIO
from PIL import Image

# Ready Player Me API Key - Replace with your own from RPM partner dashboard
RPM_API_KEY = "your_rpm_api_key"


@app.route('/api/avatar/generate-from-photo', methods=['POST'])
@login_required
def generate_avatar_from_photo():
    try:
        # Check if photo is in request
        if 'photo' not in request.files:
            return jsonify({'success': False, 'error': 'No photo provided'}), 400

        photo = request.files['photo']

        if photo.filename == '':
            return jsonify({'success': False, 'error': 'Empty file'}), 400

        # Get user ID
        user_id = session['user']['_id']

        # Save the photo temporarily
        img = Image.open(photo)
        img_buffer = BytesIO()
        img.save(img_buffer, format='JPEG')
        img_buffer.seek(0)

        # Convert to base64 for the RPM API
        base64_image = base64.b64encode(img_buffer.read()).decode('utf-8')

        # Create directory for user avatars if it doesn't exist
        avatar_dir = os.path.join('flaskapp', 'static', 'avatars', str(user_id))
        os.makedirs(avatar_dir, exist_ok=True)

        # Contact the Ready Player Me API
        # For testing, we'll use their demo API endpoint
        # In production, use their direct API with your API key
        response = requests.post(
            'https://api.readyplayer.me/v1/avatars',
            json={
                'photo': f'data:image/jpeg;base64,{base64_image}',
                'gender': 'neutral'  # or get from form
            },
            headers={
                'Authorization': f'Bearer {RPM_API_KEY}',
                'Content-Type': 'application/json'
            }
        )

        if response.status_code != 200:
            # For testing, use their demo web API
            # In production, this approach would require your own server-side implementation
            print(f"API error: {response.status_code} - {response.text}")

            # Fallback to a demo avatar URL for testing
            avatar_url = "https://models.readyplayer.me/64c415db15199e3f53bbc65f.glb"
        else:
            # Get the avatar URL from the response
            response_data = response.json()
            avatar_url = response_data.get('avatarUrl')

        if not avatar_url:
            return jsonify({
                'success': False,
                'error': 'Failed to generate avatar'
            }), 500

        # Download the avatar model
        avatar_response = requests.get(avatar_url)
        if avatar_response.status_code == 200:
            # Save the model file
            avatar_filename = f"avatar-{uuid.uuid4()}.glb"
            avatar_filepath = os.path.join(avatar_dir, avatar_filename)

            with open(avatar_filepath, 'wb') as f:
                f.write(avatar_response.content)

            # Save the avatar info to the database
            db.avatars.update_one(
                {'userId': user_id},
                {
                    '$set': {
                        'avatarUrl': avatar_url,
                        'localPath': f'/static/avatars/{user_id}/{avatar_filename}',
                        'updated_at': datetime.now()
                    }
                },
                upsert=True
            )

            return jsonify({
                'success': True,
                'avatarUrl': avatar_url,
                'localPath': f'/static/avatars/{user_id}/{avatar_filename}'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to download avatar model: {avatar_response.status_code}'
            }), 500

    except Exception as e:
        print(f"Error generating avatar: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/avatar/get-rpm-avatar')
@login_required
def get_avatar_rpm():
    user_id = session['user']['_id']

    # Try to find an existing RPM avatar for this user
    avatar_doc = db.avatars.find_one({'userId': user_id})

    if avatar_doc and 'avatarUrl' in avatar_doc:
        return jsonify({
            'success': True,
            'avatarUrl': avatar_doc['avatarUrl'],
            'localPath': avatar_doc.get('localPath')
        })

    return jsonify({'success': False, 'error': 'No avatar found'})


@app.route('/api/wardrobe/process-clothing', methods=['POST'])
@login_required
def process_clothing():
    try:
        user_id = session['user']['_id']
        data = request.json

        # Fix this line - explicit check for each key instead of using all()
        if not data or 'itemId' not in data or 'imageUrl' not in data or 'itemType' not in data:
            return jsonify({'success': False, 'error': 'Missing required data'}), 400

        item_id = data['itemId']
        image_url = data['imageUrl']
        item_type = data['itemType']

        # Get the item from the database
        item = db.wardrobe.find_one({'_id': ObjectId(item_id), 'userId': user_id})
        if not item:
            return jsonify({'success': False, 'error': 'Item not found'}), 404

        # Fix the path normalization
        if image_url.startswith('/'):
            # Remove leading slash for joining
            image_path = os.path.join('flaskapp', image_url.lstrip('/'))
        else:
            image_path = os.path.join('flaskapp', image_url)

        print(f"Trying to find image at: {image_path}")

        # Try alternative paths if needed
        if not os.path.exists(image_path):
            # Try with static directory
            image_path = os.path.join('flaskapp', 'static', 'image_users', user_id, os.path.basename(image_url))
            print(f"Alternative path 1: {image_path}")

        if not os.path.exists(image_path):
            # One more attempt with different path construction
            base_name = os.path.basename(image_url)
            if '?' in base_name:
                base_name = base_name.split('?')[0]
            image_path = os.path.join('flaskapp', 'static', 'image_users', user_id, base_name)
            print(f"Alternative path 2: {image_path}")

        # Return basic model data using the original image URL
        model_data = {
            'itemId': str(item['_id']),
            'itemType': item_type,
            'textureUrl': image_url,  # Use the original image URL as texture
            'originalImage': image_url
        }

        return jsonify({
            'success': True,
            'modelData': model_data
        })

    except Exception as e:
        print(f"Error processing clothing: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# Add this to your Flask app (run.py or wherever your routes are)
# Add this to your Flask app (run.py)


@app.route('/api/wardrobe/item/<item_id>')
def get_wardrobe_item(item_id):
    """Get individual wardrobe item with 3D model information - FIXED"""
    try:
        from bson import ObjectId

        print(f"🔍 Looking for item {item_id}")

        # Use correct collection name 'wardrobe'
        item = db.wardrobe.find_one({"_id": ObjectId(item_id)})

        if not item:
            print(f"❌ Item {item_id} not found in 'wardrobe' collection")
            return jsonify({"success": False, "error": "Item not found"})

        print(f"✅ Found item {item_id}: {item.get('label', 'Unknown')}")

        # Ensure all required fields exist with proper defaults
        item_data = {
            "success": True,
            "_id": str(item["_id"]),
            "type": item.get("type", item.get("label", "tops")),
            "label": item.get("label", "Clothing Item"),
            "userId": item.get("userId"),
            "user_id": item.get("userId"),

            # Always provide model_task_id (auto-generate if missing)
            "model_task_id": item.get("model_task_id") or f"auto_{str(item['_id'])[:8]}",
            "modelTaskId": item.get("model_task_id") or f"auto_{str(item['_id'])[:8]}",

            "file_path": item.get("file_path"),
            "texture_preview_path": item.get("texture_preview_path", item.get("file_path")),
            "color": item.get("color"),
            "material_properties": item.get("material_properties"),
            "has_3d_model": item.get("has_3d_model", False),
            "model_3d_path": item.get("model_3d_path"),
            "model_generated_at": item.get("model_generated_at"),
            "model_generation_status": item.get("model_generation_status", "none"),

            # Add category mapping
            "category": item.get("category") or get_item_category(item.get("label", ""))
        }

        print(f"✅ Returning item data with model_task_id: {item_data['model_task_id']}")
        return jsonify(item_data)

    except Exception as e:
        print(f"❌ Error getting wardrobe item {item_id}: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


# 3. ADD this helper function if it doesn't exist:
def get_item_category(label):
    """Determine category from item label"""
    label_lower = label.lower()

    if any(word in label_lower for word in ['shirt', 'top', 'pullover']):
        return 'tops'
    elif any(word in label_lower for word in ['trouser', 'pant']):
        return 'bottoms'
    elif 'dress' in label_lower:
        return 'dresses'
    elif any(word in label_lower for word in ['coat', 'jacket']):
        return 'outerwear'
    elif any(word in label_lower for word in ['shoe', 'sandal', 'boot']):
        return 'shoes'
    elif 'bag' in label_lower:
        return 'accessories'
    else:
        return 'tops'  # Default






@app.route('/api/wardrobe/check-obj/<user_id>/<model_task_id>')
def check_obj_files(user_id, model_task_id):
    """Check which OBJ files exist for a given user and model task"""
    import os

    try:
        base_path = os.path.join('static', 'models', 'generated', user_id)

        # Check for different file variations
        possible_files = [
            f'colab_model_task_{model_task_id}_0.obj',
            f'colab_model_task_{model_task_id}_1.obj',
            f'colab_model_task_{model_task_id}_2.obj',
            f'colab_model_task_{model_task_id}_3.obj',
            f'colab_model_task_{model_task_id}_4.obj',
            f'colab_model_task_{model_task_id}.obj'
        ]

        existing_files = []
        for filename in possible_files:
            file_path = os.path.join(base_path, filename)
            if os.path.exists(file_path):
                existing_files.append({
                    'filename': filename,
                    'path': f'/static/models/generated/{user_id}/{filename}',
                    'size': os.path.getsize(file_path)
                })

        return jsonify({
            "success": True,
            "user_id": user_id,
            "model_task_id": model_task_id,
            "existing_files": existing_files,
            "total_files": len(existing_files)
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# Alternative endpoint to check if a specific OBJ file exists
@app.route('/api/wardrobe/check-model/<user_id>/<model_task_id>')
def check_model_file(user_id, model_task_id):
    """Check which model files exist for a given user and model task"""
    import os

    try:
        base_path = os.path.join('static', 'models', 'generated', user_id)

        # Check for different file variations
        possible_files = [
            f'colab_model_task_{model_task_id}_0.obj',
            f'colab_model_task_{model_task_id}_1.obj',
            f'colab_model_task_{model_task_id}_2.obj',
            f'colab_model_task_{model_task_id}_3.obj',
            f'colab_model_task_{model_task_id}_4.obj',
            f'colab_model_task_{model_task_id}.obj'
        ]

        existing_files = []
        for filename in possible_files:
            file_path = os.path.join(base_path, filename)
            if os.path.exists(file_path):
                existing_files.append(f'/static/models/generated/{user_id}/{filename}')

        return jsonify({
            "success": True,
            "existing_files": existing_files,
            "base_path": f'/static/models/generated/{user_id}',
            "checked_files": possible_files
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# Helper function to update existing items with model task IDs if missing
@app.route('/api/wardrobe/update-model-paths', methods=['POST'])
def update_model_paths():
    """Update existing wardrobe items with proper model paths"""
    try:
        # Update items that have generated models but missing model_task_id
        updated_count = 0

        # Find items that might need updating
        items = db.wardrobe_items.find({
            "has_3d_model": True,
            "$or": [
                {"model_task_id": {"$exists": False}},
                {"model_task_id": None}
            ]
        })

        for item in items:
            # Try to extract model_task_id from existing file paths or other fields
            # This is a heuristic approach - adjust based on your data structure

            if item.get("file_path"):
                # If the file path contains a pattern we can extract
                import re
                match = re.search(r'colab_model_task_([a-f0-9]+)', item.get("file_path", ""))
                if match:
                    model_task_id = match.group(1)

                    # Update the item
                    db.wardrobe_items.update_one(
                        {"_id": item["_id"]},
                        {"$set": {"model_task_id": model_task_id}}
                    )
                    updated_count += 1

        return jsonify({
            "success": True,
            "updated_count": updated_count,
            "message": f"Updated {updated_count} items with model task IDs"
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# === Ready Player Me (RPM) API Integration ===
import requests

RPM_API_BASE = "https://api.readyplayer.me/v1"
RPM_TOKEN = os.environ.get("RPM_API_TOKEN", "YOUR_RPM_API_TOKEN")  # Store securely!

# Helper: Equip an outfit (asset) on an avatar
def equip_outfit_on_avatar(avatar_id, asset_id):
    url = f"{RPM_API_BASE}/avatars/{avatar_id}/assets/{asset_id}"
    headers = {
        "Authorization": f"Bearer {RPM_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {"type": "outfit"}
    response = requests.put(url, json=data, headers=headers)
    return response.json()

# Helper: Get avatar GLB URL (update if RPM changes endpoint)
def get_avatar_url(avatar_id):
    return f"https://models.readyplayer.me/{avatar_id}.glb"

# Flask endpoint: Equip a custom outfit on the avatar
@app.route('/api/rpm/equip_outfit', methods=['POST'])
@login_required
def api_rpm_equip_outfit():
    data = request.get_json()
    avatar_id = data.get('avatar_id')
    asset_id = data.get('asset_id')
    if not avatar_id or not asset_id:
        return jsonify({'error': 'avatar_id and asset_id required'}), 400
    result = equip_outfit_on_avatar(avatar_id, asset_id)
    return jsonify(result)

# Flask endpoint: Get avatar GLB URL (with equipped outfit)
@app.route('/api/rpm/avatar_url', methods=['GET'])
@login_required
def api_rpm_avatar_url():
    avatar_id = request.args.get('avatar_id')
    if not avatar_id:
        return jsonify({'error': 'avatar_id required'}), 400
    url = get_avatar_url(avatar_id)
    return jsonify({'avatar_url': url})

@app.route('/api/rpm/current_avatar_id', methods=['GET'])
@login_required
def get_current_avatar_id():
    try:
        print('[DEBUG] /api/rpm/current_avatar_id called')
        user_id = session.get('user', {}).get('_id')
        print(f'[DEBUG] user_id: {user_id}')
        avatar_doc = db.avatars.find_one({'userId': user_id})
        print(f'[DEBUG] avatar_doc: {avatar_doc}')
        if avatar_doc and 'avatarUrl' in avatar_doc:
            avatar_url = avatar_doc['avatarUrl']
            avatar_id = avatar_url.split('/')[-1].replace('.glb', '')
            print(f'[DEBUG] avatar_id: {avatar_id}')
            return jsonify({'avatarId': avatar_id})
        print('[DEBUG] No avatar found for user')
        return jsonify({'error': 'No avatar found'}), 404
    except Exception as e:
        print(f'[ERROR] Exception in /api/rpm/current_avatar_id: {e}')
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/rpm/top_asset_id', methods=['GET'])
@login_required
def get_top_asset_id():
    headers = {
        "Authorization": f"Bearer {RPM_TOKEN}",
        "Content-Type": "application/json"
    }
    response = requests.get(f"{RPM_API_BASE}/assets", headers=headers)
    if response.status_code != 200:
        return jsonify({'error': 'Failed to fetch assets'}), 500
    assets = response.json().get('assets', [])
    # Print all asset names for debugging
    print("RPM Assets:")
    for asset in assets:
        print(asset.get('name'), asset.get('id'))
    # Try to find by name containing 'top'
    for asset in assets:
        if 'top' in asset.get('name', '').lower():
            return jsonify({'assetId': asset['id']})
    return jsonify({'error': 'No assetId found for top.glb'}), 404

@app.route('/api/rpm/save-avatar', methods=['POST'])
@login_required
def save_rpm_avatar():
    try:
        user_id = session.get('user', {}).get('_id')
        data = request.get_json()
        avatar_url = data.get('avatarUrl')
        if not avatar_url:
            return jsonify({'error': 'No avatarUrl provided'}), 400
        db.avatars.update_one(
            {'userId': user_id},
            {'$set': {'avatarUrl': avatar_url}},
            upsert=True
        )
        print(f'[DEBUG] Saved avatarUrl for user {user_id}: {avatar_url}')
        return jsonify({'success': True})
    except Exception as e:
        print(f'[ERROR] Exception in /api/rpm/save-avatar: {e}')
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/rpm/upload-top-glb', methods=['POST'])
def upload_top_glb():
    import os
    import requests
    rpm_token = os.environ.get('RPM_API_TOKEN')
    rpm_app_id = os.environ.get('RPM_APP_ID')
    if not rpm_token or not rpm_app_id:
        return jsonify({'error': 'RPM_API_TOKEN or RPM_APP_ID not set'}), 500
    glb_path = os.path.join('flaskapp', 'static', 'models', 'clothing', 'top.glb')
    if not os.path.exists(glb_path):
        return jsonify({'error': 'top.glb not found'}), 404
    # 1. Upload to temporary-media
    headers = {'Authorization': f'Bearer {rpm_token}'}
    with open(glb_path, 'rb') as f:
        files = {'file': ('top.glb', f, 'model/gltf-binary')}
        resp = requests.post('https://api.readyplayer.me/v1/temporary-media', files=files, headers=headers)
    if resp.status_code != 200:
        return jsonify({'error': 'Failed to upload to RPM', 'details': resp.text}), 500
    model_url = resp.json()['data']['url']
    # 2. Create asset
    asset_data = {
        "data": {
            "name": "top.glb",
            "type": "top",
            "modelUrl": model_url,
            "gender": "female",
            "status": "published"
        }
    }
    headers['Content-Type'] = 'application/json'
    resp = requests.post('https://api.readyplayer.me/v1/assets', json=asset_data, headers=headers)
    if resp.status_code not in (200, 201):
        return jsonify({'error': 'Failed to create asset', 'details': resp.text}), 500
    asset_id = resp.json()['data']['id']
    # 3. Add asset to your application
    add_data = {
        "data": {
            "applicationId": rpm_app_id,
            "isVisibleInEditor": True
        }
    }
    resp = requests.post(f'https://api.readyplayer.me/v1/assets/{asset_id}/application', json=add_data, headers=headers)
    if resp.status_code not in (200, 201):
        return jsonify({'error': 'Failed to add asset to app', 'details': resp.text}), 500
    return jsonify({'assetId': asset_id, 'message': 'top.glb uploaded and linked to app successfully'})


from skimage.feature import graycomatrix, graycoprops, local_binary_pattern



# material

def extract_enhanced_material_features(img_path):
    """
    Extract comprehensive material features from an image

    Parameters:
        img_path (str): Path to the image file

    Returns:
        dict: Dictionary of extracted features
    """
    import cv2
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops

    try:
        # Load and resize image for consistent processing
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image file: {img_path}")

        img = cv2.resize(img, (300, 300))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Basic texture features
        texture_variance = np.var(gray)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

        # 2. GLCM features (Gray Level Co-occurrence Matrix)
        distances = [1]  # Keep simple for efficiency
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

        # Normalize gray scale to reduce feature dimension
        gray_normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Reduce levels to improve performance
        levels = 32
        gray_reduced = (gray_normalized // (256 // levels)).astype(np.uint8)

        # Calculate GLCM
        glcm = graycomatrix(gray_reduced, distances, angles, levels, symmetric=True, normed=True)

        # Extract GLCM properties
        contrast = graycoprops(glcm, 'contrast').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()

        # 3. Color analysis
        # Convert to HSV colorspace (better for fabric analysis)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Calculate saturation variation (helps distinguish fabrics)
        saturation_variation = np.var(hsv[:, :, 1])

        # Calculate hue distribution for color consistency
        hue_histogram = cv2.calcHist([hsv], [0], None, [18], [0, 180])
        hue_histogram = hue_histogram / np.sum(hue_histogram)  # Normalize

        # Detect primary hue
        primary_hue_index = np.argmax(hue_histogram)
        primary_hue_percentage = float(hue_histogram[primary_hue_index])

        # 4. Pattern detection
        # Check if the material has visible patterns
        has_pattern = bool(edge_density > 0.1 or texture_variance > 150)

        # Estimate pattern strength
        pattern_strength = min(1.0, (edge_density * 3 + texture_variance / 400) / 2)

        # Detect pattern type
        pattern_type = "irregular"
        if has_pattern:
            # Check for regular patterns using edge direction histogram
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            # Calculate gradient directions
            gradient_direction = np.arctan2(sobely, sobelx) * 180 / np.pi

            # Analyze gradient directions
            hist, _ = np.histogram(gradient_direction, bins=8, range=(-180, 180))
            hist_normalized = hist / np.sum(hist)
            max_dir_idx = np.argmax(hist_normalized)
            max_dir_percentage = float(hist_normalized[max_dir_idx])

            # Determine pattern type
            if max_dir_percentage > 0.3:  # Strong directional pattern
                if max_dir_idx in [0, 4]:  # Horizontal (0° or 180°)
                    pattern_type = "horizontal_stripe"
                elif max_dir_idx in [2, 6]:  # Vertical (90° or 270°)
                    pattern_type = "vertical_stripe"
                else:
                    pattern_type = "diagonal_stripe"
            elif texture_variance > 200 and edge_density > 0.2:
                pattern_type = "complex"
            else:
                pattern_type = "irregular"

        # 5. Combine all features
        features = {
            # Basic features
            'texture_variance': float(texture_variance),
            'edge_density': float(edge_density),

            # GLCM features
            'glcm_contrast': float(contrast),
            'glcm_dissimilarity': float(dissimilarity),
            'glcm_homogeneity': float(homogeneity),
            'glcm_energy': float(energy),
            'glcm_correlation': float(correlation),

            # Color features
            'saturation_variation': float(saturation_variation),
            'primary_hue_percentage': float(primary_hue_percentage),

            # Pattern features
            'has_pattern': has_pattern,
            'pattern_strength': float(pattern_strength),
            'pattern_type': pattern_type
        }

        return features

    except Exception as e:
        print(f"Error extracting material features: {str(e)}")
        # Return basic features if anything fails
        return {
            'texture_variance': 100.0,
            'edge_density': 0.1,
            'glcm_energy': 0.5,
            'glcm_homogeneity': 0.5,
            'has_pattern': False
        }


def get_material_prediction_with_confidence(features):
    """
    Predict material type with confidence score

    Parameters:
        features (dict): Dictionary of extracted features

    Returns:
        dict: Prediction results with confidence
    """
    import json
    import os

    def load_material_database():
        """Load material database from JSON file"""
        try:
            db_path = os.path.join('flaskapp', 'static', 'data', 'material_database.json')
            with open(db_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading material database: {str(e)}")
            # Return a minimal fallback database
            return {
                "cotton": {
                    "texture_variance_range": [70, 160],
                    "edge_density_range": [0.05, 0.15],
                    "glcm_contrast_range": [0.2, 0.5],
                    "glcm_energy_range": [0.1, 0.3],
                    "weight": 0.8
                },
                "denim": {
                    "texture_variance_range": [150, 300],
                    "edge_density_range": [0.15, 0.25],
                    "glcm_contrast_range": [0.4, 0.7],
                    "glcm_energy_range": [0.05, 0.2],
                    "weight": 0.9
                }
            }

    def determine_material_type(features):
        """Simple rule-based material determination"""
        if features.get('edge_density', 0) < 0.05 and features.get('texture_variance', 0) < 50:
            return "silk"
        elif features.get('edge_density', 0) > 0.2:
            if features.get('texture_variance', 0) > 200:
                return "wool"
            else:
                return "denim"
        elif features.get('edge_density', 0) > 0.1:
            return "cotton"
        elif features.get('glcm_energy', 0) > 0.5:
            return "polyester"
        else:
            return "unknown"

    def calculate_material_similarity(features, properties):
        """Calculate similarity score between features and reference"""
        score = 0
        total_weight = 0

        # Helper function to calculate range score
        def range_score(value, range_min, range_max):
            if value < range_min:
                return max(0, 1 - min(1, (range_min - value) / range_min))
            elif value > range_max:
                return max(0, 1 - min(1, (value - range_max) / range_max))
            else:
                # Calculate how close to center of range
                range_center = (range_min + range_max) / 2
                range_width = max(1, range_max - range_min)
                distance = abs(value - range_center)
                return max(0, 1 - (distance / (range_width / 2)))

        # Check texture variance
        if 'texture_variance' in features and 'texture_variance_range' in properties:
            var_score = range_score(
                features['texture_variance'],
                properties['texture_variance_range'][0],
                properties['texture_variance_range'][1]
            )
            score += var_score * 0.3
            total_weight += 0.3

        # Check edge density
        if 'edge_density' in features and 'edge_density_range' in properties:
            edge_score = range_score(
                features['edge_density'],
                properties['edge_density_range'][0],
                properties['edge_density_range'][1]
            )
            score += edge_score * 0.3
            total_weight += 0.3

        # Check GLCM contrast if available
        if 'glcm_contrast' in features and 'glcm_contrast_range' in properties:
            contrast_score = range_score(
                features['glcm_contrast'],
                properties['glcm_contrast_range'][0],
                properties['glcm_contrast_range'][1]
            )
            score += contrast_score * 0.2
            total_weight += 0.2

        # Check GLCM energy if available
        if 'glcm_energy' in features and 'glcm_energy_range' in properties:
            energy_score = range_score(
                features['glcm_energy'],
                properties['glcm_energy_range'][0],
                properties['glcm_energy_range'][1]
            )
            score += energy_score * 0.2
            total_weight += 0.2

        # Normalize final score
        if total_weight > 0:
            final_score = score / total_weight
        else:
            final_score = 0

        # Adjust by material weight if available
        weight = properties.get('weight', 1.0)
        return final_score * weight

    def compare_with_reference_materials(features):
        """Compare features with reference database"""
        material_db = load_material_database()
        scores = {}

        for material, properties in material_db.items():
            similarity = calculate_material_similarity(features, properties)
            scores[material] = similarity

        # Find best match
        if not scores:
            return "unknown", 0

        best_match = max(scores.items(), key=lambda x: x[1])
        return best_match[0], best_match[1]

    def calculate_feature_distinctiveness(features):
        """Calculate how distinctive the features are"""
        distinctiveness = 0

        # Very low or high variance is distinctive
        if 'texture_variance' in features:
            variance = features['texture_variance']
            if variance < 50 or variance > 200:
                distinctiveness += 0.3
            else:
                distinctiveness += 0.1

        # Very smooth or rough textures are distinctive
        if 'edge_density' in features:
            edge_density = features['edge_density']
            if edge_density < 0.05 or edge_density > 0.2:
                distinctiveness += 0.3
            else:
                distinctiveness += 0.1

        # GLCM features add distinctiveness
        if 'glcm_energy' in features:
            energy = features['glcm_energy']
            if energy > 0.6 or energy < 0.1:
                distinctiveness += 0.2
            else:
                distinctiveness += 0.1

        # Pattern detection increases distinctiveness
        if features.get('has_pattern', False):
            distinctiveness += 0.2

        return min(1.0, distinctiveness)

    def get_alternative_materials(features):
        """Find alternative material matches"""
        material_db = load_material_database()
        scores = {}

        for material, properties in material_db.items():
            similarity = calculate_material_similarity(features, properties)
            scores[material] = similarity

        # Sort by score
        sorted_materials = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Return top alternatives (excluding the best match)
        alternatives = []
        for material, score in sorted_materials[1:4]:  # Get 2nd to 4th best matches
            if score > 0.3:  # Only include reasonable matches
                alternatives.append({
                    "material": material,
                    "confidence": round(min(score * 100, 100), 1)
                })

        return alternatives

    try:
        # 1. Direct classification
        rule_based_material = determine_material_type(features)

        # 2. Reference comparison
        reference_material, similarity = compare_with_reference_materials(features)

        # 3. Calculate confidence
        feature_clarity = calculate_feature_distinctiveness(features)
        confidence = (similarity * 0.7) + (feature_clarity * 0.3)

        # 4. Choose final material
        # Use reference material if similarity is high enough
        if similarity > 0.6:
            final_material = reference_material
        else:
            final_material = rule_based_material

        # 5. Get alternative materials
        alternatives = get_alternative_materials(features)

        # 6. Return comprehensive result
        return {
            "material_type": final_material,
            "confidence": round(min(confidence * 100, 100), 1),
            "feature_distinctiveness": round(feature_clarity * 100, 1),
            "reference_match": reference_material,
            "reference_similarity": round(similarity * 100, 1),
            "rule_based_material": rule_based_material,
            "alternative_materials": alternatives
        }

    except Exception as e:
        print(f"Error in material prediction: {str(e)}")
        return {
            "material_type": "unknown",
            "confidence": 0,
            "alternative_materials": []
        }

# Add a new route to handle material analysis
@app.route('/api/wardrobe/analyze-material', methods=['POST'])
@login_required
def analyze_material():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save uploaded file
        user_id = session['user']['_id']
        upload_dir = os.path.join('flaskapp', 'static', 'temp_analysis', user_id)
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, secure_filename(file.filename))
        file.save(file_path)

        # Extract enhanced features
        features = extract_enhanced_material_features(file_path)

        # Get material prediction
        material_prediction = get_material_prediction_with_confidence(features)

        # Clean up temp file
        os.remove(file_path)

        return jsonify({
            'success': True,
            'materialAnalysis': material_prediction
        })

    except Exception as e:
        print(f"Error analyzing material: {str(e)}")
        return jsonify({'error': str(e)}), 500

# 3d clothes

from gradio_client import Client, handle_file
import tempfile
import threading
import queue
import time
import requests
import os
import shutil
from werkzeug.utils import secure_filename
from datetime import datetime
import json
import uuid


# colab api api TRIPOSR
# Add this to your existing run.py file

import requests
import time
import uuid
from datetime import datetime
import threading
import os

# Clean Colab-Only 3D Generation System
# Remove all Hugging Face Spaces / TripoSG integration

import requests
import time
import uuid
from datetime import datetime
import threading
import os
import tempfile
import shutil
from werkzeug.utils import secure_filename
import random

# Global variables for Colab-only integration
COLAB_API_URL = 'https://f606-34-72-94-244.ngrok-free.app/'
colab_api_available = True
generation_tasks = {}
generation_lock = threading.Lock()
active_generations = 0
MAX_CONCURRENT_GENERATIONS = 2  # Allow more since we're only using Colab


class ColabTripoSRClient:
    """Simplified Colab-only client"""

    def __init__(self):
        self.colab_url = None
        self.session = requests.Session()
        self.session.timeout = 60  # Longer timeout for 3D generation
        self.available = False

    def set_url(self, url):
        """Set the Colab API URL"""
        self.colab_url = url.rstrip('/')
        self.check_availability()

    def check_availability(self):
        """Check if Colab API is available"""
        if not self.colab_url:
            self.available = False
            return False

        try:
            print(f"🔍 Checking Colab API at: {self.colab_url}")
            response = self.session.get(f"{self.colab_url}/health", timeout=15)
            self.available = response.status_code == 200
            print(f"✅ Colab API {'available' if self.available else 'unavailable'}")
            return self.available
        except requests.exceptions.RequestException as e:
            print(f"❌ Colab API check failed: {str(e)}")
            self.available = False
            return False

    def generate_3d_model(self, image_path, task_id, user_id):
        """Generate 3D model using Colab API only"""
        if not self.available:
            raise Exception("Colab API not available")

        try:
            print(f"🚀 Starting Colab 3D generation for task {task_id}")

            # Update task status
            generation_tasks[task_id].update({
                'status': 'processing',
                'message': 'Uploading image to Colab...',
                'progress': 0.1,
                'updated_at': time.time()
            })

            # Prepare file for upload
            with open(image_path, 'rb') as f:
                files = {'image': ('image.jpg', f, 'image/jpeg')}

                # Update progress
                generation_tasks[task_id].update({
                    'message': 'Processing with Colab GPU...',
                    'progress': 0.3,
                    'updated_at': time.time()
                })

                print(f"📤 Uploading to Colab API: {self.colab_url}/generate")

                # Call Colab API with longer timeout
                response = self.session.post(
                    f"{self.colab_url}/generate",
                    files=files,
                    timeout=300  # 5 minutes timeout for 3D generation
                )

            print(f"📥 Colab response: HTTP {response.status_code}")

            if response.status_code == 200:
                # Update progress
                generation_tasks[task_id].update({
                    'message': 'Saving generated model...',
                    'progress': 0.9,
                    'updated_at': time.time()
                })

                # Check if response contains actual file data
                content_length = len(response.content)
                print(f"📦 Response content length: {content_length:,} bytes")

                if content_length < 1000:  # Less than 1KB suggests an error response
                    # Try to parse as JSON error message
                    try:
                        error_data = response.json()
                        error_msg = error_data.get('message', error_data.get('error', 'Unknown error'))
                        print(f"❌ Colab API returned error: {error_msg}")
                        raise Exception(f"Colab processing failed: {error_msg}")
                    except:
                        print(f"❌ Colab API returned insufficient data: {response.text[:200]}...")
                        raise Exception("Colab API returned insufficient data")

                # Save the returned model file
                user_model_dir = os.path.join('flaskapp', 'static', 'models', 'generated', user_id)
                os.makedirs(user_model_dir, exist_ok=True)

                # Determine file extension based on content type or default to OBJ
                content_type = response.headers.get('content-type', '').lower()
                if 'glb' in content_type or 'gltf' in content_type:
                    file_extension = 'glb'
                elif 'obj' in content_type or 'octet-stream' in content_type:
                    file_extension = 'obj'
                else:
                    # Default to OBJ since that's what we expect
                    file_extension = 'obj'

                model_filename = f"colab_model_{task_id}_{int(time.time())}.{file_extension}"
                model_path = os.path.join(user_model_dir, model_filename)

                # Save the model file
                with open(model_path, 'wb') as f:
                    f.write(response.content)

                file_size = len(response.content)
                print(f"💾 Model saved: {model_path} ({file_size:,} bytes)")

                # Verify the saved file exists and has content
                if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
                    print("❌ Saved model file is empty or doesn't exist")
                    raise Exception("Failed to save model file properly")

                # Update task as completed
                generation_tasks[task_id].update({
                    'status': 'completed',
                    'message': f'3D model generated successfully via Colab! ({file_extension.upper()} format)',
                    'progress': 1.0,
                    'updated_at': time.time(),
                    'model_path': f'/static/models/generated/{user_id}/{model_filename}',
                    'local_path': model_path,
                    'file_size': file_size,
                    'file_format': file_extension.upper(),
                    'completed_at': datetime.now().isoformat(),
                    'method': 'colab'
                })

                print(f"✅ Task {task_id} completed successfully!")
                return True

            else:
                error_msg = f"Colab API error: HTTP {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', error_data.get('message', error_msg))
                    print(f"❌ Colab API error details: {error_data}")

                    # Handle specific error cases
                    if 'No OBJ file found' in str(error_data):
                        error_msg = "3D model generation failed - TripoSR couldn't create the model file. This might be due to image quality or processing issues."
                    elif 'timeout' in str(error_data).lower():
                        error_msg = "3D model generation timed out - try with a simpler image"
                    elif 'memory' in str(error_data).lower():
                        error_msg = "Insufficient GPU memory for processing - try with a smaller image"

                except:
                    print(f"❌ Colab API error (raw response): {response.text[:500]}...")
                    if response.status_code == 500:
                        error_msg = "Colab server error - the 3D generation process failed. Try again or use a different image."
                    elif response.status_code == 404:
                        error_msg = "Colab API endpoint not found - make sure the Colab notebook is running properly"
                    elif response.status_code == 413:
                        error_msg = "Image file too large - please use a smaller image"

                raise Exception(error_msg)

        except requests.exceptions.Timeout:
            error_msg = "Colab API timeout - 3D generation took too long (>5 minutes). Try with a simpler image or check if Colab is still running."
            print(f"❌ {error_msg}")
            raise Exception(error_msg)

        except requests.exceptions.ConnectionError:
            error_msg = "Cannot connect to Colab API. Make sure your Colab notebook is still running and the ngrok tunnel is active."
            print(f"❌ {error_msg}")
            raise Exception(error_msg)

        except Exception as e:
            error_str = str(e)
            print(f"❌ Colab generation error: {error_str}")

            # Provide more helpful error messages
            if "No OBJ file found" in error_str:
                raise Exception(
                    "3D model generation failed - the AI couldn't process your image properly. Try a different image with clearer object details.")
            elif "Processing failed" in error_str:
                raise Exception(
                    "3D model generation failed - there was an issue processing your image. Please try again or use a different image.")
            else:
                raise Exception(f"Colab API error: {error_str}")

# Create global Colab client (ONLY client we need)
colab_client = ColabTripoSRClient()

# Auto-configure the Colab client
if COLAB_API_URL:
    colab_client.set_url(COLAB_API_URL)
    print(f"🔗 Colab API configured: {COLAB_API_URL}")
    print(f"✅ Colab available: {colab_client.available}")


def generate_3d_model_colab_thread(image_path, task_id, user_id):
    """Thread function for Colab-only 3D model generation"""
    global active_generations

    try:
        with generation_lock:
            active_generations += 1

        print(f"🔄 Starting Colab 3D generation for task {task_id} (active: {active_generations})")

        success = colab_client.generate_3d_model(image_path, task_id, user_id)

        if success:
            print(f"✅ Task {task_id} completed successfully")
        else:
            print(f"❌ Task {task_id} failed")

    except Exception as e:
        error_message = str(e)
        print(f"❌ Task {task_id} thread failed: {error_message}")

        current_time = time.time()
        generation_tasks[task_id] = {
            **generation_tasks.get(task_id, {}),
            'status': 'failed',
            'message': error_message,
            'error': error_message,
            'user_id': user_id,
            'updated_at': current_time,
            'failed_at': datetime.now().isoformat()
        }
    finally:
        with generation_lock:
            active_generations -= 1

        # Clean up temporary image file
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"🗑️ Cleaned up temp file: {image_path}")
        except Exception as e:
            print(f"⚠️ Failed to clean up temp file {image_path}: {str(e)}")


# FLASK ROUTES - Simplified for Colab-only

@app.route('/api/generate-3d-model', methods=['POST'])
@login_required
def api_generate_3d_model_colab():
    """Generate 3D model using Colab API only"""
    global active_generations

    try:
        # Check if Colab is available
        if not colab_client.available:
            return jsonify({
                'success': False,
                'error': 'Colab API not available. Please check the Colab connection.',
                'colab_url': COLAB_API_URL,
                'setup_required': True
            }), 503

        # Check concurrent generation limit
        if active_generations >= MAX_CONCURRENT_GENERATIONS:
            return jsonify({
                'success': False,
                'error': f'Server is busy processing {active_generations} models. Please wait and try again.',
                'active_count': active_generations,
                'max_count': MAX_CONCURRENT_GENERATIONS
            }), 429

        # Validate request
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'success': False, 'error': 'Empty file'}), 400

        # Validate file size (15MB limit for Colab)
        image_file.seek(0, 2)
        file_size = image_file.tell()
        image_file.seek(0)

        if file_size > 15 * 1024 * 1024:  # 15MB
            return jsonify({'success': False, 'error': 'File too large. Maximum size is 15MB.'}), 400

        # Validate file type
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png']
        if image_file.content_type not in allowed_types:
            return jsonify({'success': False, 'error': 'Invalid file type. Please use JPEG or PNG.'}), 400

        # Get user ID
        user_id = session['user']['_id']

        # Save uploaded image temporarily
        temp_dir = os.path.join('flaskapp', 'static', 'temp_3d_generation')
        os.makedirs(temp_dir, exist_ok=True)

        # Create unique filename
        file_extension = os.path.splitext(image_file.filename)[1].lower()
        if not file_extension:
            file_extension = '.jpg'

        temp_filename = f"colab_{user_id}_{uuid.uuid4().hex[:8]}{file_extension}"
        temp_image_path = os.path.join(temp_dir, temp_filename)
        image_file.save(temp_image_path)

        # Verify file was saved
        if not os.path.exists(temp_image_path):
            return jsonify({'success': False, 'error': 'Failed to save uploaded image'}), 500

        # Generate unique task ID
        task_id = f"task_{user_id}_{uuid.uuid4().hex[:8]}_{int(time.time())}"

        # Initialize task status
        current_time = time.time()
        generation_tasks[task_id] = {
            'status': 'started',
            'message': 'Preparing 3D model generation with Colab GPU...',
            'progress': 0.0,
            'created_at': current_time,
            'updated_at': current_time,
            'user_id': user_id,
            'temp_image_path': temp_image_path,
            'method': 'colab',
            'started_at': datetime.now().isoformat()
        }

        # Start generation in a separate thread
        thread = threading.Thread(
            target=generate_3d_model_colab_thread,
            args=(temp_image_path, task_id, user_id),
            name=f"ColabGen-{task_id}"
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            'success': True,
            'task_id': task_id,
            'method': 'colab',
            'message': '3D model generation started with Colab GPU',
            'estimated_time': '2-4 minutes',
            'colab_url': COLAB_API_URL
        })

    except Exception as e:
        app.logger.error(f"Error starting Colab 3D generation: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/check-generation-status', methods=['GET'])
@login_required
def api_check_colab_generation_status():
    """Check the status of Colab 3D model generation"""
    try:
        task_id = request.args.get('task_id')
        if not task_id:
            return jsonify({'success': False, 'error': 'No task ID provided'}), 400

        # Check if task exists
        if task_id not in generation_tasks:
            return jsonify({'success': False, 'error': 'Task not found'}), 404

        task = generation_tasks[task_id]

        # Verify user owns this task
        user_id = session['user']['_id']
        if task.get('user_id') != user_id:
            return jsonify({'success': False, 'error': 'Unauthorized'}), 403

        # Check if task is too old
        current_time = time.time()
        task_age = current_time - task.get('created_at', current_time)

        if task_age > 1800:  # 30 minutes max
            print(f"Task {task_id} expired after {task_age} seconds")
            del generation_tasks[task_id]
            return jsonify({'success': False, 'error': 'Task expired'}), 410

        # Update the task's last access time
        task['updated_at'] = current_time

        # Prepare response
        response_data = {
            'success': True,
            'status': task['status'],
            'message': task['message'],
            'progress': task['progress'],
            'task_id': task_id,
            'method': 'colab',
            'active_generations': active_generations,
            'task_age': int(task_age),
            'colab_url': COLAB_API_URL
        }

        # Add additional data based on status
        if task['status'] == 'completed':
            response_data.update({
                'model_path': task.get('model_path'),
                'file_size': task.get('file_size'),
                'completed_at': task.get('completed_at')
            })
        elif task['status'] == 'failed':
            response_data.update({
                'error': task.get('error'),
                'failed_at': task.get('failed_at')
            })

        return jsonify(response_data)

    except Exception as e:
        app.logger.error(f"Error checking Colab generation status: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/generation-methods', methods=['GET'])
@login_required
def get_colab_only_methods():
    """Get Colab-only generation method status"""
    return jsonify({
        'success': True,
        'methods': {
            'colab': {
                'name': 'Google Colab API',
                'available': colab_client.available,
                'description': 'Generate 3D models using Google Colab with free GPU',
                'url': COLAB_API_URL,
                'pros': ['Free GPU access', 'No local hardware requirements', 'Always up-to-date'],
                'cons': ['Requires internet', 'May have usage limits', 'Depends on external service']
            }
        },
        'current_method': 'colab',
        'active_generations': active_generations,
        'max_concurrent': MAX_CONCURRENT_GENERATIONS
    })


@app.route('/api/set-colab-url', methods=['POST'])
@login_required
def set_colab_url():
    """Update the Google Colab API URL"""
    try:
        data = request.json
        new_colab_url = data.get('colab_url', '').strip()

        if not new_colab_url:
            return jsonify({'success': False, 'error': 'No URL provided'}), 400

        # Validate URL format
        if not new_colab_url.startswith(('http://', 'https://')):
            return jsonify({'success': False, 'error': 'URL must start with http:// or https://'}), 400

        # Update global URL
        global COLAB_API_URL, colab_api_available
        COLAB_API_URL = 'https://df44-34-72-94-244.ngrok-free.app/'

        # Test the new URL
        colab_client.set_url(new_colab_url)

        if colab_client.available:
            colab_api_available = True
            return jsonify({
                'success': True,
                'message': 'Colab API connected successfully!',
                'url': new_colab_url
            })
        else:
            colab_api_available = False
            return jsonify({
                'success': False,
                'error': 'Could not connect to new Colab API URL. Make sure it\'s running and accessible.',
                'url': new_colab_url
            }), 400

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/test-colab-connection', methods=['GET'])
@login_required
def test_colab_connection():
    """Test current Colab connection"""
    try:
        if not COLAB_API_URL:
            return jsonify({
                'success': False,
                'error': 'No Colab URL configured',
                'connected': False
            })

        # Test the connection
        is_available = colab_client.check_availability()

        return jsonify({
            'success': is_available,
            'connected': is_available,
            'url': COLAB_API_URL,
            'message': 'Colab API is responding normally' if is_available else 'Colab API is not responding'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'connected': False,
            'error': str(e)
        })


@app.route('/api/generation-stats', methods=['GET'])
@login_required
def get_colab_generation_stats():
    """Get Colab generation statistics"""
    try:
        return jsonify({
            'success': True,
            'active_generations': active_generations,
            'max_concurrent': MAX_CONCURRENT_GENERATIONS,
            'can_start_new': active_generations < MAX_CONCURRENT_GENERATIONS,
            'method': 'colab',
            'colab_available': colab_client.available,
            'colab_url': COLAB_API_URL,
            'total_tasks': len(generation_tasks)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# Cleanup function for old tasks
def cleanup_old_colab_tasks():
    """Clean up old Colab generation tasks"""
    try:
        current_time = time.time()
        tasks_to_remove = []

        for task_id, task in list(generation_tasks.items()):
            task_age = current_time - task.get('created_at', 0)

            # Remove tasks older than 1 hour
            if task_age > 3600:
                tasks_to_remove.append(task_id)

                # Clean up temporary files
                temp_path = task.get('temp_image_path')
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception as e:
                        print(f"Failed to remove temp file {temp_path}: {e}")

        # Remove old tasks
        for task_id in tasks_to_remove:
            del generation_tasks[task_id]

        if len(tasks_to_remove) > 0:
            print(f"🧹 Cleaned up {len(tasks_to_remove)} old Colab tasks")

    except Exception as e:
        print(f"Error in Colab cleanup: {str(e)}")


# Periodic cleanup
def periodic_colab_cleanup():
    """Run Colab cleanup periodically"""
    while True:
        time.sleep(1800)  # 30 minutes
        cleanup_old_colab_tasks()


# Start cleanup thread
cleanup_thread = threading.Thread(target=periodic_colab_cleanup, name="Colab-Cleanup")
cleanup_thread.daemon = True
cleanup_thread.start()

print("🚀 Colab-only 3D Generation System initialized!")
print(f"🔗 Colab URL: {COLAB_API_URL}")
print(f"✅ Colab Available: {colab_client.available}")


# FIXED: Enhanced save 3D model route with better error handling
@app.route('/api/save-3d-model', methods=['POST'])
@login_required
def save_3d_model_to_wardrobe():
    """Save generated 3D model to wardrobe item with enhanced validation"""
    try:
        data = request.json
        item_id = data.get('item_id')
        model_path = data.get('model_path')
        method = data.get('method', 'colab')
        file_format = data.get('file_format', 'OBJ')
        file_size = data.get('file_size', 0)

        print(f"💾 Saving 3D model to wardrobe:")
        print(f"   Item ID: {item_id}")
        print(f"   Model Path: {model_path}")
        print(f"   Method: {method}")
        print(f"   Format: {file_format}")
        print(f"   Size: {file_size}")

        if not item_id or not model_path:
            return jsonify({
                'success': False,
                'error': 'Missing item_id or model_path'
            }), 400

        user_id = session['user']['_id']

        # Verify the model file exists
        if model_path.startswith('/'):
            full_model_path = os.path.join('flaskapp', model_path.lstrip('/'))
        else:
            full_model_path = os.path.join('flaskapp', model_path)

        if not os.path.exists(full_model_path):
            print(f"❌ Model file not found at: {full_model_path}")
            return jsonify({
                'success': False,
                'error': f'Model file not found at path: {model_path}'
            }), 404

        # Get actual file size if not provided
        if file_size == 0:
            try:
                file_size = os.path.getsize(full_model_path)
            except:
                file_size = 0

        print(f"✅ Model file verified: {full_model_path} ({file_size} bytes)")

        # FIXED: Update the wardrobe item with comprehensive 3D model info
        update_data = {
            'model_3d_path': model_path,
            'has_3d_model': True,
            'model_generated_at': datetime.now(),
            'model_method': method,
            'model_file_format': file_format,
            'model_file_size': file_size,
            'model_last_updated': datetime.now(),
            'model_generation_status': 'completed'
        }

        result = db.wardrobe.update_one(
            {'_id': ObjectId(item_id), 'userId': user_id},
            {'$set': update_data}
        )

        if result.modified_count > 0:
            print(f"✅ Successfully updated wardrobe item {item_id} with 3D model")

            # VERIFY the update worked by fetching the item
            updated_item = db.wardrobe.find_one({'_id': ObjectId(item_id), 'userId': user_id})
            if updated_item and updated_item.get('has_3d_model'):
                print(f"✅ Database verification successful - 3D model saved!")
                return jsonify({
                    'success': True,
                    'message': '3D model saved to wardrobe successfully',
                    'model_info': {
                        'path': model_path,
                        'format': file_format,
                        'size': file_size,
                        'method': method,
                        'item_label': updated_item.get('label', 'Unknown')
                    }
                })
            else:
                print(f"❌ Database verification failed - 3D model not properly saved")
                raise Exception("Database update verification failed")

        elif result.matched_count > 0:
            print(f"⚠️ Wardrobe item {item_id} found but not modified (already up to date?)")
            return jsonify({
                'success': True,
                'message': '3D model info already up to date',
                'model_info': {
                    'path': model_path,
                    'format': file_format,
                    'size': file_size,
                    'method': method
                }
            })
        else:
            print(f"❌ Wardrobe item {item_id} not found for user {user_id}")
            return jsonify({
                'success': False,
                'error': 'Wardrobe item not found'
            }), 404

    except Exception as e:
        print(f"❌ Error saving 3D model: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/wardrobe/3d-model/<item_id>', methods=['GET'])
@login_required
def get_wardrobe_3d_model(item_id):
    """Get 3D model info for a wardrobe item"""
    try:
        user_id = session['user']['_id']
        item = db.wardrobe.find_one({'_id': ObjectId(item_id), 'userId': user_id})

        if not item:
            return jsonify({'success': False, 'error': 'Item not found'}), 404

        if not item.get('has_3d_model') or not item.get('model_3d_path'):
            return jsonify({'success': False, 'error': 'No 3D model available for this item'}), 404

        return jsonify({
            'success': True,
            'model_path': item['model_3d_path'],
            'generated_at': item.get('model_generated_at'),
            'method': item.get('model_method', 'unknown'),
            'item_label': item.get('label', 'Unknown'),
            'has_3d_model': True
        })

    except Exception as e:
        print(f"Error getting 3D model: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


def handle_colab_generation_success(data, task_id, user_id):
    """Enhanced success handler with automatic database saving"""
    try:
        print(f"🎉 Colab generation completed for task {task_id}")

        # Update task status
        generation_tasks[task_id].update({
            'status': 'completed',
            'message': f'3D model generated successfully via Colab! ({data.get("file_format", "OBJ")} format)',
            'progress': 1.0,
            'updated_at': time.time(),
            'model_path': data['model_path'],
            'local_path': data['local_path'],
            'file_size': data['file_size'],
            'file_format': data.get('file_format', 'OBJ'),
            'completed_at': datetime.now().isoformat(),
            'method': 'colab'
        })

        print(f"✅ Task {task_id} marked as completed")
        return True

    except Exception as e:
        print(f"❌ Error in success handler: {str(e)}")
        return False


# Enhanced Colab generation method in your existing ColabTripoSRClient class
def enhanced_colab_generate_3d_model(self, image_path, task_id, user_id):
    """Enhanced Colab generation with better error handling and validation"""
    if not self.available:
        raise Exception("Colab API not available")

    try:
        print(f"🚀 Starting enhanced Colab 3D generation for task {task_id}")

        # Update task status
        generation_tasks[task_id].update({
            'status': 'processing',
            'message': 'Uploading image to Colab GPU...',
            'progress': 0.1,
            'updated_at': time.time()
        })

        # Prepare file for upload
        with open(image_path, 'rb') as f:
            files = {'image': ('image.jpg', f, 'image/jpeg')}

            # Update progress
            generation_tasks[task_id].update({
                'message': 'Processing with Colab TripoSR...',
                'progress': 0.3,
                'updated_at': time.time()
            })

            print(f"📤 Uploading to enhanced Colab API: {self.colab_url}/generate")

            # Call Colab API with extended timeout
            response = self.session.post(
                f"{self.colab_url}/generate",
                files=files,
                timeout=400  # Extended to 6+ minutes for complex models
            )

        print(f"📥 Colab response: HTTP {response.status_code}")
        print(f"📦 Response headers: {dict(response.headers)}")

        if response.status_code == 200:
            # Update progress
            generation_tasks[task_id].update({
                'message': 'Saving and validating 3D model...',
                'progress': 0.9,
                'updated_at': time.time()
            })

            # Validate response content
            content_length = len(response.content)
            print(f"📦 Response content length: {content_length:,} bytes")

            # More lenient validation - allow smaller OBJ files
            if content_length < 500:  # Changed from 1000 to 500
                try:
                    error_data = response.json()
                    error_msg = error_data.get('message', error_data.get('error', 'Unknown error'))
                    print(f"❌ Colab API returned error: {error_msg}")
                    raise Exception(f"Colab processing failed: {error_msg}")
                except:
                    print(f"❌ Colab API returned insufficient data: {response.text[:200]}...")
                    raise Exception("Colab API returned insufficient data - model may be too simple")

            # Create user model directory
            user_model_dir = os.path.join('flaskapp', 'static', 'models', 'generated', user_id)
            os.makedirs(user_model_dir, exist_ok=True)

            # Determine file format
            content_type = response.headers.get('content-type', '').lower()
            if 'glb' in content_type or 'gltf' in content_type:
                file_extension = 'glb'
            elif 'obj' in content_type or 'octet-stream' in content_type:
                file_extension = 'obj'
            else:
                # Check content for OBJ markers
                content_sample = response.content[:1000].decode('utf-8', errors='ignore')
                if 'v ' in content_sample and 'f ' in content_sample:
                    file_extension = 'obj'
                else:
                    file_extension = 'obj'  # Default assumption

            # Generate unique filename
            timestamp = int(time.time())
            model_filename = f"colab_model_{task_id}_{timestamp}.{file_extension}"
            model_path = os.path.join(user_model_dir, model_filename)

            # Save the model file
            with open(model_path, 'wb') as f:
                f.write(response.content)

            file_size = len(response.content)
            print(f"💾 Model saved: {model_path} ({file_size:,} bytes)")

            # Verify saved file
            if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
                print("❌ Saved model file is empty or doesn't exist")
                raise Exception("Failed to save model file properly")

            # Additional validation for OBJ files
            if file_extension == 'obj':
                try:
                    with open(model_path, 'r') as f:
                        content_check = f.read(1000)
                        if not ('v ' in content_check or 'f ' in content_check):
                            print("⚠️ OBJ file may not contain valid geometry data")
                            print(f"Content preview: {content_check[:200]}...")
                except Exception as e:
                    print(f"⚠️ Could not validate OBJ content: {e}")

            # Success data
            success_data = {
                'model_path': f'/static/models/generated/{user_id}/{model_filename}',
                'local_path': model_path,
                'file_size': file_size,
                'file_format': file_extension.upper(),
                'method': 'colab'
            }

            # Update task as completed
            generation_tasks[task_id].update({
                'status': 'completed',
                'message': f'3D model generated successfully! ({file_extension.upper()}, {(file_size / 1024 / 1024):.1f}MB)',
                'progress': 1.0,
                'updated_at': time.time(),
                **success_data,
                'completed_at': datetime.now().isoformat()
            })

            print(f"✅ Enhanced task {task_id} completed successfully!")
            return True

        else:
            # Enhanced error handling
            error_msg = f"Colab API error: HTTP {response.status_code}"
            try:
                error_data = response.json()
                error_msg = error_data.get('error', error_data.get('message', error_msg))
                print(f"❌ Colab API error details: {error_data}")
            except:
                print(f"❌ Colab API error (raw): {response.text[:500]}...")

            # Provide specific error messages
            if response.status_code == 500:
                error_msg = "Colab server error during 3D generation. The image may be too complex or the server is overloaded."
            elif response.status_code == 404:
                error_msg = "Colab API endpoint not found. Please check if the Colab notebook is running properly."
            elif response.status_code == 413:
                error_msg = "Image file too large for Colab processing. Please use a smaller image."
            elif response.status_code == 408:
                error_msg = "Colab processing timeout. Try with a simpler image or check server load."

            raise Exception(error_msg)

    except requests.exceptions.Timeout:
        error_msg = "Colab processing timeout (>6 minutes). The 3D generation is taking too long - try with a simpler image."
        print(f"❌ {error_msg}")
        raise Exception(error_msg)

    except requests.exceptions.ConnectionError:
        error_msg = "Cannot connect to Colab API. Please verify the Colab notebook is running and ngrok tunnel is active."
        print(f"❌ {error_msg}")
        raise Exception(error_msg)

    except Exception as e:
        error_str = str(e)
        print(f"❌ Enhanced Colab generation error: {error_str}")
        raise Exception(f"3D generation failed: {error_str}")


print("🚀 Enhanced Auto-Save 3D Model System loaded!")


# FIXED: Route to get all wardrobe items WITH 3D model info
@app.route('/api/wardrobe/all-with-3d', methods=['GET'])
@login_required
def get_all_wardrobe_with_3d():
    """Get all wardrobe items including 3D model information"""
    try:
        user_id = session['user']['_id']
        items_cursor = db.wardrobe.find({'userId': user_id})

        items_with_3d = []
        for item in items_cursor:
            item_data = {
                'id': str(item['_id']),
                'label': item.get('label', ''),
                'file_path': normalize_path(item.get('file_path', '')),
                'color': item.get('color', ''),
                'created_at': item.get('created_at'),
                # 3D MODEL INFO
                'has_3d_model': item.get('has_3d_model', False),
                'model_3d_path': normalize_path(item.get('model_3d_path', '')) if item.get('model_3d_path') else None,
                'model_method': item.get('model_method'),
                'model_file_format': item.get('model_file_format'),
                'model_file_size': item.get('model_file_size'),
                'model_generated_at': item.get('model_generated_at'),
                'model_generation_status': item.get('model_generation_status', 'none')
            }
            items_with_3d.append(item_data)

        return jsonify({
            'success': True,
            'items': items_with_3d,
            'total_items': len(items_with_3d),
            'items_with_3d': len([item for item in items_with_3d if item['has_3d_model']])
        })

    except Exception as e:
        print(f"Error getting wardrobe with 3D: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

        @app.route('/api/debug/wardrobe-items', methods=['GET'])
        @login_required
        def debug_wardrobe_items():
            """Debug endpoint to see what wardrobe items exist"""
            try:
                user_id = session['user']['_id']

                # Get all items for the user
                items = list(db.wardrobe.find({'userId': user_id}))

                debug_info = {
                    'user_id': user_id,
                    'total_items': len(items),
                    'collection_name': 'wardrobe',
                    'sample_items': []
                }

                # Add sample items for debugging
                for item in items[:5]:  # First 5 items
                    debug_info['sample_items'].append({
                        'id': str(item['_id']),
                        'label': item.get('label', 'No label'),
                        'type': item.get('type', 'No type'),
                        'file_path': item.get('file_path', 'No file_path'),
                        'has_model_task_id': 'model_task_id' in item,
                        'model_task_id': item.get('model_task_id', 'None'),
                        'has_3d_model': item.get('has_3d_model', False)
                    })

                return jsonify(debug_info)

            except Exception as e:
                return jsonify({'error': str(e)})


# ADD this route to your run.py to help find OBJ files

# ADD this route to your run.py to help match specific OBJ files to items

@app.route('/api/find-obj-for-item/<item_id>')
@login_required
def find_obj_for_item(item_id):
    """Find the specific OBJ file for a wardrobe item"""
    try:
        user_id = session['user']['_id']

        # Get the item from database
        item = db.wardrobe.find_one({"_id": ObjectId(item_id)})
        if not item:
            return jsonify({"success": False, "error": "Item not found"})

        model_task_id = item.get('model_task_id')
        if not model_task_id:
            return jsonify({"success": False, "error": "No model_task_id found"})

        import os
        import glob

        base_path = os.path.join('flaskapp', 'static', 'models', 'generated', user_id)

        if not os.path.exists(base_path):
            return jsonify({"success": False, "error": "No models directory"})

        # Look for files matching this specific model_task_id
        patterns = [
            f"colab_model_task_{model_task_id}_*.obj",
            f"colab_model_task_{model_task_id}.obj",
            f"*{model_task_id}*.obj",
            f"*{item_id[-8:]}*.obj"  # Last 8 chars of item ID
        ]

        found_files = []
        for pattern in patterns:
            pattern_path = os.path.join(base_path, pattern)
            matches = glob.glob(pattern_path)
            for match in matches:
                filename = os.path.basename(match)
                if filename not in [f['filename'] for f in found_files]:
                    found_files.append({
                        'filename': filename,
                        'path': f'/static/models/generated/{user_id}/{filename}',
                        'size': os.path.getsize(match),
                        'pattern_matched': pattern
                    })

        return jsonify({
            "success": True,
            "item_id": item_id,
            "model_task_id": model_task_id,
            "found_files": found_files,
            "total_matches": len(found_files)
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


def calculate_match_score(filename, item_id, model_task_id):
    """Calculate how well a filename matches an item"""
    score = 0

    # Exact model_task_id match gets highest score
    if model_task_id and model_task_id in filename:
        score += 100

    # Item ID matches
    if item_id in filename:
        score += 80

    # Partial item ID matches
    item_id_short = item_id[-8:]
    if item_id_short in filename:
        score += 60

    # File recency (newer files get slightly higher score)
    # This is a simple heuristic based on filename patterns
    if '_174826' in filename:  # Recent timestamp pattern
        score += 5

    return score


# ALSO ADD: Route to check all wardrobe items and their OBJ matches
@app.route('/api/debug/check-obj-matches')
@login_required
def debug_check_obj_matches():
    """Debug route to check OBJ matches for all wardrobe items"""
    try:
        user_id = session['user']['_id']

        # Get all wardrobe items
        items = list(db.wardrobe.find({'userId': user_id}))

        results = []
        for item in items:
            item_id = str(item['_id'])

            # Find matches for this item
            match_response = find_obj_for_item(item_id)
            match_data = match_response.get_json()

            results.append({
                'item_id': item_id,
                'label': item.get('label', 'Unknown'),
                'model_task_id': item.get('model_task_id'),
                'has_matches': match_data.get('success', False),
                'match_count': match_data.get('total_matches', 0),
                'best_match': match_data.get('best_match')
            })

        # Summary statistics
        total_items = len(results)
        items_with_matches = len([r for r in results if r['has_matches'] and r['match_count'] > 0])

        return jsonify({
            'success': True,
            'total_items': total_items,
            'items_with_matches': items_with_matches,
            'match_percentage': round((items_with_matches / total_items * 100), 1) if total_items > 0 else 0,
            'results': results
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# avatar make human
from bson import ObjectId
import json


# Avatar Save/Load Routes
@app.route('/api/avatar/save', methods=['POST'])
@login_required
def save_avatar():
    """Save avatar configuration to database"""
    try:
        user_id = session['user']['_id']
        data = request.json

        if not data or 'configuration' not in data:
            return jsonify({'success': False, 'error': 'No configuration provided'}), 400

        configuration = data['configuration']
        avatar_type = data.get('avatarType', 'glb')
        format_type = data.get('format', 'glb')
        hair_system = data.get('hairSystem', 'glb')

        # Create avatar document
        avatar_doc = {
            'userId': user_id,
            'configuration': configuration,
            'avatarType': avatar_type,
            'format': format_type,
            'hairSystem': hair_system,
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'version': '1.0'
        }

        # Check if user already has a saved avatar - update it or create new
        existing_avatar = db.saved_avatars.find_one({'userId': user_id})

        if existing_avatar:
            # Update existing avatar
            avatar_doc['updated_at'] = datetime.now()
            result = db.saved_avatars.update_one(
                {'userId': user_id},
                {'$set': avatar_doc}
            )
            message = 'Avatar configuration updated successfully'
        else:
            # Create new avatar
            result = db.saved_avatars.insert_one(avatar_doc)
            message = 'Avatar configuration saved successfully'

        if result:
            print(f"✅ Avatar saved for user {user_id}")
            return jsonify({
                'success': True,
                'message': message,
                'avatarId': str(existing_avatar['_id']) if existing_avatar else str(result.inserted_id)
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to save avatar'}), 500

    except Exception as e:
        print(f"❌ Error saving avatar: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/avatar/get-saved', methods=['GET'])
@login_required
def get_saved_avatar():
    """Get the most recent saved avatar for the current user"""
    try:
        user_id = session['user']['_id']

        # Get the most recent saved avatar
        avatar = db.saved_avatars.find_one(
            {'userId': user_id},
            sort=[('updated_at', -1)]
        )

        if avatar:
            # Convert ObjectId to string for JSON serialization
            avatar['_id'] = str(avatar['_id'])

            return jsonify({
                'success': True,
                'avatar': avatar
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No saved avatar found'
            }), 404

    except Exception as e:
        print(f"❌ Error getting saved avatar: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/avatar/get-all-saved', methods=['GET'])
@login_required
def get_all_saved_avatars():
    """Get all saved avatars for the current user"""
    try:
        user_id = session['user']['_id']

        # Get all saved avatars for this user, sorted by most recent
        avatars = list(db.saved_avatars.find(
            {'userId': user_id}
        ).sort('updated_at', -1).limit(10))  # Limit to last 10 avatars

        # Convert ObjectIds to strings for JSON serialization
        for avatar in avatars:
            avatar['_id'] = str(avatar['_id'])

        return jsonify({
            'success': True,
            'avatars': avatars,
            'count': len(avatars)
        })

    except Exception as e:
        print(f"❌ Error getting all saved avatars: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/avatar/get-saved/<avatar_id>', methods=['GET'])
@login_required
def get_saved_avatar_by_id(avatar_id):
    """Get a specific saved avatar by ID"""
    try:
        user_id = session['user']['_id']

        # Get the specific avatar
        avatar = db.saved_avatars.find_one({
            '_id': ObjectId(avatar_id),
            'userId': user_id
        })

        if avatar:
            # Convert ObjectId to string for JSON serialization
            avatar['_id'] = str(avatar['_id'])

            return jsonify({
                'success': True,
                'avatar': avatar
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Avatar not found'
            }), 404

    except Exception as e:
        print(f"❌ Error getting saved avatar by ID: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/avatar/delete/<avatar_id>', methods=['DELETE'])
@login_required
def delete_saved_avatar(avatar_id):
    """Delete a saved avatar"""
    try:
        user_id = session['user']['_id']

        # Delete the avatar (only if it belongs to the current user)
        result = db.saved_avatars.delete_one({
            '_id': ObjectId(avatar_id),
            'userId': user_id
        })

        if result.deleted_count > 0:
            return jsonify({
                'success': True,
                'message': 'Avatar deleted successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Avatar not found or access denied'
            }), 404

    except Exception as e:
        print(f"❌ Error deleting saved avatar: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/avatar/load-on-startup', methods=['GET'])
@login_required
def load_avatar_on_startup():
    """Load avatar configuration on application startup"""
    try:
        user_id = session['user']['_id']

        # Get the most recent saved avatar
        avatar = db.saved_avatars.find_one(
            {'userId': user_id},
            sort=[('updated_at', -1)]
        )

        if avatar:
            # Convert ObjectId to string for JSON serialization
            avatar['_id'] = str(avatar['_id'])

            print(f"✅ Loading saved avatar for user {user_id} on startup")

            return jsonify({
                'success': True,
                'hasAvatar': True,
                'avatar': avatar,
                'message': 'Saved avatar configuration loaded'
            })
        else:
            print(f"ℹ️ No saved avatar found for user {user_id}, loading defaults")

            # Return default configuration
            default_config = {
                'gender': 'female',
                'bodySize': 'm',
                'height': 'medium',
                'skinColor': 'light',
                'hairType': 'elvis_hazel',
                'hairColor': 'brown',
                'eyeColor': 'brown'
            }

            return jsonify({
                'success': True,
                'hasAvatar': False,
                'defaultConfig': default_config,
                'message': 'No saved avatar found, using defaults'
            })

    except Exception as e:
        print(f"❌ Error loading avatar on startup: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# Optional: Avatar statistics route
@app.route('/api/avatar/stats', methods=['GET'])
@login_required
def get_avatar_stats():
    """Get avatar statistics for the current user"""
    try:
        user_id = session['user']['_id']

        # Count saved avatars
        avatar_count = db.saved_avatars.count_documents({'userId': user_id})

        # Get creation date of first avatar
        first_avatar = db.saved_avatars.find_one(
            {'userId': user_id},
            sort=[('created_at', 1)]
        )

        # Get last updated avatar
        last_avatar = db.saved_avatars.find_one(
            {'userId': user_id},
            sort=[('updated_at', -1)]
        )

        stats = {
            'totalAvatars': avatar_count,
            'firstCreated': first_avatar['created_at'].isoformat() if first_avatar else None,
            'lastUpdated': last_avatar['updated_at'].isoformat() if last_avatar else None,
            'hasAvatars': avatar_count > 0
        }

        return jsonify({
            'success': True,
            'stats': stats
        })

    except Exception as e:
        print(f"❌ Error getting avatar stats: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# Database initialization - add this to your app startup
def initialize_avatar_collections():
    """Initialize avatar-related database collections and indexes"""
    try:
        # Create indexes for better performance
        db.saved_avatars.create_index([("userId", 1), ("updated_at", -1)])
        db.saved_avatars.create_index([("userId", 1), ("created_at", -1)])

        print("✅ Avatar database collections and indexes initialized")

    except Exception as e:
        print(f"❌ Error initializing avatar collections: {str(e)}")


#         saved outfits 3d
# Add these routes to your run.py file

@app.route('/api/outfits/save', methods=['POST'])
@login_required
def save_outfit():
    try:
        user_id = session['user']['_id']
        data = request.json

        outfit_doc = {
            'userId': user_id,
            'name': data['name'],
            'items': data['items'],
            'itemsDetails': data.get('itemsDetails', []),
            'avatarConfig': data.get('avatarConfig', {}),
            'outfitType': data.get('outfitType', '3d'),
            'renderingType': data.get('renderingType', '3d'),
            'itemCount': data.get('itemCount', 0),
            'clothingPositions': data.get('clothingPositions', {}),
            'clothingScales': data.get('clothingScales', {}),
            'created_at': datetime.now(),
            'isFavorite': False
        }

        result = db.saved_outfits.insert_one(outfit_doc)

        return jsonify({
            'success': True,
            'outfit_id': str(result.inserted_id),
            'message': 'Outfit saved successfully'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/outfits/saved', methods=['GET'])
@login_required
def get_saved_outfits():
    try:
        user_id = session['user']['_id']
        outfits = list(db.saved_outfits.find(
            {'userId': user_id}
        ).sort('created_at', -1))

        # Convert ObjectIds to strings
        for outfit in outfits:
            outfit['_id'] = str(outfit['_id'])

        return jsonify({
            'success': True,
            'outfits': outfits,
            'count': len(outfits)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/outfits/delete/<outfit_id>', methods=['DELETE'])
@login_required
def delete_saved_outfit(outfit_id):
    try:
        user_id = session['user']['_id']
        result = db.saved_outfits.delete_one({
            '_id': ObjectId(outfit_id),
            'userId': user_id
        })

        if result.deleted_count > 0:
            return jsonify({'success': True, 'message': 'Outfit deleted successfully'})
        else:
            return jsonify({'success': False, 'error': 'Outfit not found'}), 404

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

        @app.route('/api/outfits/update-snapshot/<outfit_id>', methods=['PUT'])
        @login_required
        def update_outfit_snapshot(outfit_id):
            try:
                user_id = session['user']['_id']
                data = request.json

                result = db.saved_outfits.update_one(
                    {'_id': ObjectId(outfit_id), 'userId': user_id},
                    {'$set': {'snapshot': data['snapshot'], 'updated_at': datetime.now()}}
                )

                if result.modified_count > 0:
                    return jsonify({'success': True, 'message': 'Snapshot updated successfully'})
                else:
                    return jsonify({'success': False, 'error': 'Outfit not found'}), 404

            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    initialize_avatar_collections()
    app.run(debug=True, use_reloader=False)

from __future__ import division, print_function
import os
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
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img
from keras.preprocessing import image
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


def extract_material_properties(img_path):
    """
    Extract material properties from an image including:
    - Dominant colors
    - Texture patterns
    - Material type estimation based on texture analysis
    - Pattern information

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

    # GLCM (Gray Level Co-occurrence Matrix) texture features
    # Convert to 8-bit grayscale for texture analysis
    gray_8bit = (gray / gray.max() * 255).astype(np.uint8)

    # Calculate texture features (variance as a simple measure)
    texture_variance = np.var(gray_8bit)

    # Calculate edge density (a proxy for texture complexity)
    edges = cv2.Canny(gray_8bit, 100, 200)
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

    # 4. Add basic pattern information
    # This is a placeholder - in the full implementation you'd use the detect_pattern_type function
    # For now we'll create a basic pattern_info structure with default values
    pattern_info = {
        "pattern_type": "regular" if texture_variance > 150 else "irregular",
        "pattern_scale": "medium",
        "pattern_strength": min(1.0, edge_density * 2),  # Simple scaling to 0-1 range
        "has_pattern": edge_density > 0.1 or texture_variance > 100,
        "pattern_regularity": 0.5,
        "is_directional": False,
        "peak_count": 0
    }

    # Return the extracted material properties
    return {
        "dominant_colors": [color.tolist() for _, color in p_and_c[:3]],
        "color_percentages": [float(pct) for pct, _ in p_and_c[:3]],
        "texture_variance": float(texture_variance),
        "edge_density": float(edge_density),
        "estimated_material": material_type,
        "primary_color_rgb": p_and_c[0][1].tolist(),
        "pattern_info": pattern_info  # Add pattern_info to the returned dict
    }


def determine_material_type(texture_variance, edge_density, pattern_info):
    """Determine material type based on texture and pattern analysis"""

    # Check if it's a strong pattern first
    if pattern_info["has_pattern"] and pattern_info["pattern_strength"] > 0.4:
        if pattern_info["pattern_type"] in ["check", "stripe"]:
            return "woven_patterned"
        elif pattern_info["pattern_type"] == "irregular":
            return "printed"
        else:
            return "patterned"

    # If no strong pattern, determine by texture
    if edge_density < 0.05 and texture_variance < 50:
        return "smooth"  # Might be leather, silk, etc.
    elif edge_density > 0.2:
        if texture_variance > 150:
            return "rough_textured"  # Might be tweed, heavy wool
        else:
            return "textured"  # Might be denim, canvas
    elif texture_variance > 200:
        return "detailed"  # Has distinct texture details
    else:
        return "medium"  # Medium texture, like cotton


def detect_pattern_type(img_path):
    """Detect and classify pattern types in the image"""
    img = cv2.imread(img_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize for faster processing if needed
    resized = cv2.resize(gray, (256, 256))

    # Apply FFT to detect regular patterns
    f = fftpack.fft2(resized)
    fshift = fftpack.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    # Threshold the magnitude spectrum to find strong frequencies
    threshold = np.mean(magnitude_spectrum) + 3 * np.std(magnitude_spectrum)
    peaks = magnitude_spectrum > threshold

    # Count peaks in the frequency domain (excluding the DC component)
    center_y, center_x = resized.shape[0] // 2, resized.shape[1] // 2
    mask = np.ones_like(peaks)
    mask[center_y - 5:center_y + 5, center_x - 5:center_x + 5] = 0  # Exclude center
    peak_count = np.sum(peaks & mask)

    # Analyze peak distribution
    peak_locs = np.where(peaks & mask)
    peak_distances = np.sqrt((peak_locs[0] - center_y) ** 2 + (peak_locs[1] - center_x) ** 2)
    pattern_regularity = 0.0

    if len(peak_distances) > 0:
        # Calculate coefficient of variation (lower value = more regular)
        if np.mean(peak_distances) > 0:
            pattern_regularity = 1.0 - min(1.0, np.std(peak_distances) / np.mean(peak_distances))

    # Gradient analysis for pattern direction
    sobelx = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate gradient magnitudes and directions
    gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    gradient_direction = np.arctan2(sobely, sobelx) * 180 / np.pi

    # Analyze gradient directions
    hist, _ = np.histogram(gradient_direction, bins=8, range=(-180, 180))
    hist_normalized = hist / np.sum(hist)
    max_dir_idx = np.argmax(hist_normalized)

    # Determine if there are strong directional patterns
    has_directional_pattern = np.max(hist_normalized) > 0.25

    # Determine pattern type
    if peak_count > 15 and pattern_regularity > 0.7:
        # Check for grid patterns (peaks in both horizontal and vertical)
        horizontal_peaks = np.sum(peaks[center_y, :] & mask[center_y, :])
        vertical_peaks = np.sum(peaks[:, center_x] & mask[:, center_x])

        if horizontal_peaks > 3 and vertical_peaks > 3:
            pattern_type = "check"
        elif has_directional_pattern:
            if max_dir_idx in [0, 4]:  # Horizontal (0° or 180°)
                pattern_type = "horizontal_stripe"
            elif max_dir_idx in [2, 6]:  # Vertical (90° or 270°)
                pattern_type = "vertical_stripe"
            else:
                pattern_type = "diagonal_stripe"
        else:
            pattern_type = "regular"
    elif peak_count > 5:
        pattern_type = "semi_regular"
    else:
        # For low peak counts, further analyze texture
        if np.max(hist_normalized) > 0.2:
            pattern_type = "directional"
        else:
            pattern_type = "irregular"

    # Determine pattern scale (fine, medium, large)
    if len(peak_distances) > 0:
        avg_distance = np.mean(peak_distances)
        if avg_distance < 20:
            pattern_scale = "fine"
        elif avg_distance < 50:
            pattern_scale = "medium"
        else:
            pattern_scale = "large"
    else:
        # Default if no peaks detected
        pattern_scale = "medium"

    # Calculate pattern strength (how dominant the pattern is)
    pattern_strength = min(1.0, peak_count / 50)

    return {
        "pattern_type": pattern_type,
        "pattern_scale": pattern_scale,
        "pattern_strength": float(pattern_strength),
        "has_pattern": peak_count > 3,
        "pattern_regularity": float(pattern_regularity),
        "is_directional": has_directional_pattern,
        "peak_count": int(peak_count)
    }


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

MODEL_PATH = 'my_second_model.h5'
# MODEL_PATH = 'my_model_june.h5'
# Load your trained model
model = load_model(MODEL_PATH)
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(28, 28))
        img_array = np.asarray(img)
        x = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        result = int(img_array[0][0][0])

        img = cv2.bitwise_not(x) if result > 128 else x
        img = img / 255
        img = np.expand_dims(img, 0)

        preds = model.predict(img)
        if preds is None or len(preds) == 0:
            raise ValueError("Model prediction returned None or empty")

        return preds

    except Exception as e:
        print(f"Error in model_predict: {str(e)}")
        raise

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
    """
    Improved color prediction that focuses on the clothing item in the image.

    Steps:
    1. Load the image
    2. Detect and crop the clothing item
    3. Apply color segmentation on the cropped region
    4. Return the dominant color and its percentage

    Returns a tuple of (percentage, [R, G, B])
    """
    import cv2
    import numpy as np
    from sklearn.cluster import KMeans
    import imutils
    import matplotlib.pyplot as plt

    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image from {img_path}")
        return None

    # Make a copy for visualization
    org_img = img.copy()

    # Step 1: Apply preprocessing to enhance the clothing item
    # Convert to grayscale for processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edge map
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour which is likely to be the clothing item
    if contours:
        main_contour = max(contours, key=cv2.contourArea)

        # Create a mask for the largest contour
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [main_contour], 0, 255, -1)

        # Apply the mask to get the clothing item
        clothing_item = cv2.bitwise_and(img, img, mask=mask)

        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(main_contour)

        # Crop the clothing item
        cropped_item = clothing_item[y:y + h, x:x + w]

        # Ensure the cropped item is not empty
        if cropped_item.size == 0:
            print("Warning: Cropped image is empty, using original image")
            cropped_item = img
    else:
        print("Warning: No contours found, using original image")
        cropped_item = img

    # Step 2: Resize the cropped item for consistent processing
    cropped_item = imutils.resize(cropped_item, height=300)

    # Step 3: Apply color segmentation using K-means clustering
    # Reshape the image to be a list of pixels
    pixels = cropped_item.reshape(-1, 3)

    # Remove black background (mask pixels)
    non_black_pixels = pixels[~np.all(pixels == [0, 0, 0], axis=1)]

    # If there are no non-black pixels, revert to original image
    if len(non_black_pixels) == 0:
        print("Warning: No non-black pixels found, using original image")
        non_black_pixels = img.reshape(-1, 3)

    # Apply K-means clustering to find dominant colors
    clusters = 5
    kmeans = KMeans(n_clusters=clusters, random_state=0, n_init=10)
    kmeans.fit(non_black_pixels)

    # Get the colors and their percentages
    colors = np.array(kmeans.cluster_centers_, dtype='uint8')
    percentages = np.bincount(kmeans.labels_) / len(kmeans.labels_)

    # Sort colors by percentage
    p_and_c = sorted(zip(percentages, colors), reverse=True)

    # Debug: Visualize the color segmentation
    if False:  # Set to True to debug color segmentation
        plt.figure(figsize=(12, 6))

        # Plot the original image
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        # Plot the cropped clothing item
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(cropped_item, cv2.COLOR_BGR2RGB))
        plt.title('Cropped Clothing Item')
        plt.axis('off')

        # Plot the color palette
        plt.subplot(1, 3, 3)
        # Create a color palette
        palette = np.zeros((100, 500, 3), dtype='uint8')
        start = 0
        for i, (percentage, color) in enumerate(p_and_c):
            end = start + int(percentage * 500)
            palette[:, start:end] = color[::-1]  # BGR to RGB
            start = end

        plt.imshow(palette)
        plt.title('Color Palette')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    # Return the dominant color (first in the sorted list)
    # Format: (percentage, [R, G, B])
    return (float(p_and_c[0][0]), p_and_c[0][1])


def predict_color(img_path):
    clusters = 5
    img = cv2.imread(img_path)
    org_img = img.copy()
    img = imutils.resize(img, height=200)
    flat_img = np.reshape(img, (-1, 3))
    print('After Flattening shape --> ', flat_img.shape)

    kmeans = KMeans(n_clusters=clusters, random_state=0)
    kmeans.fit(flat_img)

    dominant_colors = np.array(kmeans.cluster_centers_, dtype='uint')

    percentages = (np.unique(kmeans.labels_, return_counts=True)[1]) / flat_img.shape[0]
    p_and_c = zip(percentages, dominant_colors)
    p_and_c = sorted(p_and_c, reverse=True)
    print("the colour in the db")
    print(p_and_c[1])
    block = np.ones((50, 50, 3), dtype='uint')
    plt.figure(figsize=(12, 8))
    for i in range(clusters):
        plt.subplot(1, clusters, i + 1)
        block[:] = p_and_c[i][1][::-1]  # we have done this to convert bgr(opencv) to rgb(matplotlib)
        plt.imshow(block)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(str(round(p_and_c[i][0] * 100, 2)) + '%')

    bar = np.ones((50, 500, 3), dtype='uint')
    plt.figure(figsize=(12, 8))
    plt.title('Proportions of colors in the image')
    start = 0
    i = 1
    for p, c in p_and_c:
        end = start + int(p * bar.shape[1])
        if i == clusters:
            bar[:, start:] = c[::-1]
        else:
            bar[:, start:end] = c[::-1]
        start = end
        i += 1

    plt.imshow(bar)
    plt.xticks([])
    plt.yticks([])

    rows = 1000
    cols = int((org_img.shape[0] / org_img.shape[1]) * rows)
    img = cv2.resize(org_img, dsize=(rows, cols), interpolation=cv2.INTER_LINEAR)

    copy = img.copy()
    cv2.rectangle(copy, (rows // 2 - 250, cols // 2 - 90), (rows // 2 + 250, cols // 2 + 110), (255, 255, 255), -1)

    final = cv2.addWeighted(img, 0.1, copy, 0.9, 0)
    cv2.putText(final, 'Most Dominant Colors in the Image',
                (rows // 2 - 230, cols // 2 - 40),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8, (0, 0, 0), 1, cv2.LINE_AA)

    start = rows // 2 - 220
    for i in range(5):
        end = start + 70
        final[cols // 2:cols // 2 + 70, start:end] = p_and_c[i][1]
        cv2.putText(final, str(i + 1), (start + 25, cols // 2 + 45), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1,
                    cv2.LINE_AA)
        start = end + 20

    plt.show()
    return p_and_c[1]


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
                preds = model_predict(file_path, model)
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
# @app.route('/predict', methods=['POST'])
# @login_required
# def upload():
#     if request.method == 'POST':
#         try:
#             f = request.files['file']
#             if not f:
#                 return "No file uploaded", 400
#
#             user_id = session['user']['_id']
#             upload_dir = os.path.join('flaskapp', 'static', 'image_users', user_id)
#             os.makedirs(upload_dir, exist_ok=True)
#             file_path = os.path.join(upload_dir, secure_filename(f.filename))
#             f.save(file_path)
#
#             # Make predictions with validation
#             try:
#                 preds = model_predict(file_path, model)
#                 if not isinstance(preds, np.ndarray) or preds.size == 0:
#                     raise ValueError("Invalid prediction output")
#
#                 color_result = predict_color(file_path)
#                 if not color_result or len(color_result) < 2:
#                     raise ValueError("Invalid color prediction")
#
#                 predicted_label = np.argmax(preds)
#                 class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#
#                 if predicted_label >= len(class_names):
#                     raise ValueError("Invalid predicted label index")
#
#                 result = class_names[predicted_label]
#
#                 # Save to database
#                 db.wardrobe.insert_one({
#                     'label': result,
#                     'color': ' '.join(map(str, color_result[1])),
#                     'nota': 4,
#                     'userId': user_id,
#                     'file_path': f'/static/image_users/{user_id}/{secure_filename(f.filename)}'
#                 })
#
#                 return result
#
#             except Exception as e:
#                 print(f"Prediction error: {str(e)}")
#                 if os.path.exists(file_path):
#                     os.remove(file_path)
#                 return str(e), 500
#
#         except Exception as e:
#             return str(e), 500
#
#     return None
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
            geocode_url = f'http://api.openweathermap.org/geo/1.0/direct?q={new_city}&limit=1&appid={api_key}'
            geocode_response = requests.get(geocode_url).json()
            print(f"Geocode response: {geocode_response}")

            if geocode_response:
                lat = geocode_response[0].get('lat')
                lon = geocode_response[0].get('lon')
                if lat and lon:
                    db.city.insert_one({'name': new_city, 'lat': lat, 'lon': lon, 'userId': userId})

    filter = {'userId': userId}
    if db.city.find_one(filter) is None:
        geocode_url = f'http://api.openweathermap.org/geo/1.0/direct?q={cityByDefault}&limit=1&appid={api_key}'
        geocode_response = requests.get(geocode_url).json()
        print(f"Default city geocode response: {geocode_response}")

        if geocode_response:
            lat = geocode_response[0].get('lat')
            lon = geocode_response[0].get('lon')
            if lat and lon:
                db.city.insert_one({'name': cityByDefault, 'lat': lat, 'lon': lon, 'userId': userId})

    cities = db.city.find(filter)
    url = 'https://api.openweathermap.org/data/2.5/weather?lat={}&lon={}&appid={}&units=metric'

    weather_data = []

    for city in cities:
        if 'lat' not in city or 'lon' not in city:
            geocode_url = f'http://api.openweathermap.org/geo/1.0/direct?q={city["name"]}&limit=1&appid={api_key}'
            geocode_response = requests.get(geocode_url).json()
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

        r = requests.get(url.format(city['lat'], city['lon'], api_key)).json()
        print(f"Weather API response for {city['name']}: {r}")

        if r.get('weather') and r.get('main'):
            weather = {
                'city': city['name'],
                'temperature': r['main']['temp'],
                'description': r['weather'][0]['description'],
                'icon': r['weather'][0]['icon'],
            }
            weather_data.append(weather)

    print(f"Weather data to be rendered: {weather_data}")
    return render_template('dashboard.html', weather_data=weather_data)


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

# @app.route('/outfit/day', methods=['GET', 'POST'])
# @login_required
# def get_outfit():
#     print("Debug: Entering get_outfit route")
#
#     try:
#         userId = session['user']['_id']
#         print(f"Debug: User ID: {userId}")
#         filter = {'userId': userId, 'isFavorite': 'yes'}
#         users_clothes = db.outfits.find(filter)
#
#         cityByDefault = 'Bucharest'
#         DEFAULT_RATING = 4
#
#         # Default to show generator and hide outfits
#         show_generator = True
#         show_outfits = False
#         success_message = None
#         error_message = None
#
#         # Define available outfit combinations
#         result_outfit = [
#             'Dress_Sandal', 'T-shirt/top_Trouser_Sneaker', 'Shirt_Trouser',
#             'Shirt_Trouser_Sneaker', 'Dress_Sandal_Coat', 'T-shirt/top_Trouser',
#             'Shirt_Trouser_Coat', 'Shirt_Trouser_Coat', 'Dress_Ankle-boot_Coat',
#             'Pullover_Trouser_Ankle-boot', 'Dress_Sneaker', 'Shirt_Trouser_Sandal',
#             'Dress_Sandal_Bag'
#         ]
#
#         # Initialize city if not exists
#         filter = {'userId': userId}
#         if db.city.count_documents(filter) == 0:
#             print(f"Debug: Creating new city entry for user {userId}")
#             db.city.insert_one({'name': cityByDefault, 'userId': userId})
#
#         # Get weather data
#         cities = db.city.find(filter)
#         weather_data = []
#         url = 'http://api.openweathermap.org/data/2.5/weather?q={}&units=metric&appid=aa73cad280fbd125cc7073323a135efa'
#
#         for city in cities:
#             try:
#                 print(f"Debug: Fetching weather for {city['name']}")
#                 r = requests.get(url.format(city['name']), timeout=5).json()
#                 weather = {
#                     'city': city['name'],
#                     'temperature': r['main']['temp'],
#                     'description': r['weather'][0]['description'],
#                     'icon': r['weather'][0]['icon'],
#                 }
#                 weather_data.append(weather)
#             except Exception as e:
#                 print(f"Error fetching weather for {city['name']}: {e}")
#                 weather_data.append({
#                     'city': city['name'],
#                     'temperature': 20,
#                     'description': '',
#                     'icon': ''
#                 })
#
#         # Ensure we have 3 weather options
#         while len(weather_data) < 3:
#             weather_data.append({
#                 'city': cityByDefault,
#                 'temperature': 20,
#                 'description': '',
#                 'icon': ''
#             })
#
#         city1, city2, city3 = weather_data[:3]
#         outfit1, outfit2, outfit3 = [], [], []
#
#         if request.method == 'POST':
#             print("Debug: Processing POST request")
#
#             # Handle outfit selection
#             option = request.form.get('options')
#             if option:
#                 print(f"Debug: Selected option: {option}")
#                 filter_lookup = {'userId': userId, 'outfitNo': option}
#                 outfit_doc = db.outfits.find_one(filter_lookup, sort=[('_id', -1)])
#
#                 if outfit_doc:
#                     # Update outfit pieces ratings
#                     for piece in outfit_doc['outfit']:
#                         try:
#                             current_piece = db.wardrobe.find_one({'_id': piece['_id']})
#                             current_rating = current_piece.get('nota',
#                                                                DEFAULT_RATING) if current_piece else DEFAULT_RATING
#
#                             db.wardrobe.update_one(
#                                 {'_id': piece['_id']},
#                                 {'$set': {'nota': current_rating + 1}},
#                                 upsert=True
#                             )
#                         except Exception as e:
#                             print(f"Error updating piece rating: {str(e)}")
#
#                     try:
#                         # Update outfit rating
#                         current_outfit_rating = outfit_doc.get('nota', DEFAULT_RATING)
#                         db.outfits.update_one(
#                             {'_id': outfit_doc['_id']},
#                             {
#                                 '$set': {
#                                     'nota': current_outfit_rating + 1,
#                                     'isFavorite': 'yes'
#                                 }
#                             }
#                         )
#                         # Show success message and hide outfits
#                         success_message = "Outfit has been saved to your favorites!"
#                         show_outfits = False
#                         return render_template(
#                             'outfit_of_the_day.html',
#                             success_message=success_message,
#                             show_generator=show_generator,
#                             show_outfits=show_outfits,
#                             city1=city1,
#                             city2=city2,
#                             city3=city3
#                         )
#                     except Exception as e:
#                         print(f"Error updating outfit rating: {str(e)}")
#                         error_message = "Error saving outfit. Please try again."
#
#             # Generate new outfits
#             include_weather = request.form.get('weather') == 'yes'
#             city = request.form.get('city')
#             event = request.form.get('events')
#             temperature = 20  # Default temperature
#
#             print(f"Debug: Form data - weather: {include_weather}, city: {city}, event: {event}")
#
#             if include_weather and city:
#                 selected_weather = next(
#                     (w for w in weather_data if w['city'] == city),
#                     {'temperature': 20}
#                 )
#                 temperature = selected_weather['temperature']
#
#             try:
#                 loaded_classifier = joblib.load("./random_forest.joblib")
#                 features = prepare_features(include_weather, event, temperature)
#                 result_forest = loaded_classifier.predict([features])
#                 index_of_outfit = result_forest[0]
#                 outfit_combination = result_outfit[index_of_outfit]
#                 filters_outfits = outfit_combination.split('_')
#
#                 print(f"Debug: Generated outfit combination: {outfit_combination}")
#
#                 # Generate three outfits
#                 for i, outfit_list in enumerate([outfit1, outfit2, outfit3]):
#                     outfit_pieces = []
#                     for filter_name in filters_outfits:
#                         clothes = list(db.wardrobe.find({
#                             'userId': userId,
#                             'label': filter_name
#                         }).sort('nota', -1))
#
#                         if clothes:
#                             index = min(i, len(clothes) - 1)
#                             piece = clothes[index]
#                             if not piece.get('file_path'):
#                                 piece['file_path'] = None
#                             if 'nota' not in piece:
#                                 piece['nota'] = DEFAULT_RATING
#                                 db.wardrobe.update_one(
#                                     {'_id': piece['_id']},
#                                     {'$set': {'nota': DEFAULT_RATING}}
#                                 )
#                             outfit_pieces.append(piece)
#
#                     if outfit_pieces:
#                         outfit_doc = {
#                             'outfit': outfit_pieces,
#                             'userId': userId,
#                             'nota': DEFAULT_RATING,
#                             'outfitNo': f'piece{i + 1}',
#                             'isFavorite': 'no',
#                             'created_at': datetime.now()
#                         }
#                         db.outfits.insert_one(outfit_doc)
#
#                         if i == 0:
#                             outfit1 = outfit_pieces
#                         elif i == 1:
#                             outfit2 = outfit_pieces
#                         else:
#                             outfit3 = outfit_pieces
#
#                 show_outfits = True
#
#             except Exception as e:
#                 print(f"Error generating outfits: {e}")
#                 error_message = "Error generating outfits. Please try again."
#
#         print("Debug: Rendering template")
#         return render_template(
#             'outfit_of_the_day.html',
#             outfit1=outfit1,
#             outfit2=outfit2,
#             outfit3=outfit3,
#             city1=city1,
#             city2=city2,
#             city3=city3,
#             show_generator=show_generator,
#             show_outfits=show_outfits,
#             success_message=success_message,
#             error_message=error_message
#         )
#
#     except Exception as e:
#         print(f"Error in get_outfit: {str(e)}")
#         return render_template(
#             'outfit_of_the_day.html',
#             error_message="An error occurred. Please try again.",
#             show_generator=True,
#             show_outfits=False,
#             city1={'city': cityByDefault, 'temperature': 20, 'description': '', 'icon': ''},
#             city2={'city': cityByDefault, 'temperature': 20, 'description': '', 'icon': ''},
#             city3={'city': cityByDefault, 'temperature': 20, 'description': '', 'icon': ''}
#         )

@app.route('/wardrobe', methods=['GET', 'POST'])
@login_required
def add_wardrobe():
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
                preds = model_predict(file_path, model)
                class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

                # Get predicted label and confidence scores
                predicted_label = np.argmax(preds)
                clothing_type = class_names[predicted_label]

                # Get color prediction
                color_result = predict_color(file_path)
                # color_result is a tuple of (percentage, RGB array)
                color_percentage = float(color_result[0])
                color_rgb = color_result[1].tolist()  # Convert numpy array to list

                # Extract material properties with pattern detection
                material_properties = extract_material_properties(file_path)

                # Generate normal map for textured materials - with safety checks
                normal_map_path = None
                # Check if material_properties contains pattern_info before accessing it
                has_pattern = False
                pattern_strength = 0.0

                if material_properties:
                    # Safely check for pattern_info
                    if 'pattern_info' in material_properties:
                        pattern_info = material_properties['pattern_info']
                        has_pattern = pattern_info.get('has_pattern', False)
                        pattern_strength = pattern_info.get('pattern_strength', 0.0)

                    # Determine if we should generate a normal map based on material type and pattern
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

                # Save image data
                userId = session['user']['_id']
                db.wardrobe.insert_one({
                    'userId': userId,
                    'label': clothing_type,
                    'confidence': float(preds[0][predicted_label]),
                    'color': {
                        'percentage': color_percentage,
                        'rgb': color_rgb
                    },
                    # Add material properties
                    'material_properties': material_properties,
                    'normal_map_path': normal_map_path,
                    'filename': secure_filename(f.filename),
                    'file_path': file_path_db,
                    'created_at': datetime.now(),
                    'last_worn': None,
                    'times_worn': 0
                })

                # Return success response
                return jsonify({
                    'success': True,
                    'prediction': clothing_type,
                    'confidence': float(preds[0][predicted_label]),
                    'color': {
                        'percentage': color_percentage,
                        'rgb': color_rgb
                    },
                    'material_properties': material_properties,
                    'normal_map_path': normal_map_path
                })

            except Exception as e:
                print(f"Error in prediction: {str(e)}")
                if os.path.exists(file_path):
                    os.remove(file_path)
                return jsonify({'error': f'Error processing image: {str(e)}'}), 500

        except Exception as e:
            print(f"Error in file upload: {str(e)}")
            return jsonify({'error': f'Error uploading file: {str(e)}'}), 500

    # GET request
    return render_template('wardrobe.html')


# @app.route('/api/wardrobe/material/<item_id>', methods=['GET'])
# @login_required
# def get_material_properties(item_id):
#     try:
#         userId = session['user']['_id']
#         item = db.wardrobe.find_one({'_id': ObjectId(item_id), 'userId': userId})
#
#         if not item:
#             return jsonify({'error': 'Item not found'}), 404
#
#         # If material properties don't exist yet, extract them now
#         if 'material_properties' not in item:
#             file_path = os.path.join('flaskapp', item['file_path'].lstrip('/'))
#             if os.path.exists(file_path):
#                 material_properties = extract_material_properties(file_path)
#
#                 # Generate normal map if appropriate
#                 normal_map_path = None
#                 if material_properties and (
#                         material_properties['estimated_material'] in ['textured', 'rough_textured',
#                                                                       'woven_patterned'] or
#                         (material_properties['pattern_info']['has_pattern'] and
#                          material_properties['pattern_info']['pattern_strength'] > 0.3)
#                 ):
#                     normal_map_path = generate_normal_map(file_path)
#                     if normal_map_path:
#                         # Convert to database path format
#                         normal_map_path = normal_map_path.replace(os.path.join('flaskapp', ''), '/')
#
#                 update_data = {'material_properties': material_properties}
#                 if normal_map_path:
#                     update_data['normal_map_path'] = normal_map_path
#
#                 db.wardrobe.update_one(
#                     {'_id': ObjectId(item_id)},
#                     {'$set': update_data}
#                 )
#
#                 item['material_properties'] = material_properties
#                 item['normal_map_path'] = normal_map_path
#             else:
#                 return jsonify({'error': 'Image file not found'}), 404
#
#         # Get normalized paths
#         texture_path = normalize_path(item.get('file_path', ''))
#         normal_map_path = normalize_path(item.get('normal_map_path', '')) if item.get('normal_map_path') else None
#
#         return jsonify({
#             'success': True,
#             'itemId': str(item['_id']),
#             'label': item['label'],
#             'materialProperties': item['material_properties'],
#             'texturePath': texture_path,
#             'normalMapPath': normal_map_path
#         })
#
#     except Exception as e:
#         print(f"Error getting material properties: {str(e)}")
#         return jsonify({'error': str(e)}), 500
# @app.route('/wardrobe', methods=['GET', 'POST'])
# @login_required
# def add_wardrobe():
#     if request.method == 'POST':
#         try:
#             # Check if the post request has the file part
#             if 'file' not in request.files:
#                 return jsonify({'error': 'No file part'}), 400
#
#             f = request.files['file']
#             if f.filename == '':
#                 return jsonify({'error': 'No selected file'}), 400
#
#             # Check if the file is allowed
#             allowed_extensions = {'png', 'jpg', 'jpeg'}
#             if not '.' in f.filename or \
#                     f.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
#                 return jsonify({'error': 'Invalid file type'}), 400
#
#             # Save the file
#             user_id = session['user']['_id']
#             upload_dir = os.path.join('flaskapp', 'static', 'image_users', user_id)
#             print('dir' + upload_dir);
#             os.makedirs(upload_dir, exist_ok=True)
#             file_path = os.path.join(upload_dir, secure_filename(f.filename))
#             f.save(file_path)
#             print(file_path + 'fsss')
#             file_path_db = f'/static/image_users/{user_id}/{secure_filename(f.filename)}'
#             print('fileeeDB'+ file_path_db);
#             # Rest of the code remains the same...
#
#             try:
#                 # Make prediction
#                 preds = model_predict(file_path, model)
#                 class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#
#                 # Get predicted label and confidence scores
#                 predicted_label = np.argmax(preds)
#                 clothing_type = class_names[predicted_label]
#
#                 # Get color prediction
#                 color_result = predict_color(file_path)
#                 # color_result is a tuple of (percentage, RGB array)
#                 color_percentage = float(color_result[0])
#                 color_rgb = color_result[1].tolist()  # Convert numpy array to list
#
#                 # Save image data
#                 userId = session['user']['_id']
#                 db.wardrobe.insert_one({
#                     'userId': userId,
#                     'label': clothing_type,
#                     'confidence': float(preds[0][predicted_label]),
#                     'color': {
#                         'percentage': color_percentage,
#                         'rgb': color_rgb
#                     },
#                     'filename': secure_filename(f.filename),
#                     'file_path': file_path_db,
#                     'created_at': datetime.now(),
#                     'last_worn': None,
#                     'times_worn': 0
#                 })
#
#                 # # Clean up the uploaded file
#                 # os.remove(file_path)
#
#                 # Return success response
#                 return jsonify({
#                     'success': True,
#                     'prediction': clothing_type,
#                     'confidence': float(preds[0][predicted_label]),
#                     'color': {
#                         'percentage': color_percentage,
#                         'rgb': color_rgb
#                     }
#                 })
#
#             except Exception as e:
#                 print(f"Error in prediction: {str(e)}")
#                 if os.path.exists(file_path):
#                     os.remove(file_path)
#                 return jsonify({'error': f'Error processing image: {str(e)}'}), 500
#
#         except Exception as e:
#             print(f"Error in file upload: {str(e)}")
#             return jsonify({'error': f'Error uploading file: {str(e)}'}), 500
#
#     # GET request
#     return render_template('wardrobe.html')

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

# # Enhanced Avatar generation imports
# import mediapipe as mp
# import cv2
# import numpy as np
# from PIL import Image
# import json
# import base64
# from io import BytesIO
# import colorsys
# from sklearn.cluster import KMeans
#
# # Initialize MediaPipe Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(
#     static_image_mode=True,
#     max_num_faces=1,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )
#
# def extract_facial_features(image_path):
#     """Extract facial features using MediaPipe Face Mesh."""
#     image = cv2.imread(image_path)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(image_rgb)
#
#     if not results.multi_face_landmarks:
#         raise ValueError("No face detected in the image")
#
#     landmarks = results.multi_face_landmarks[0]
#
#     # Extract key facial features
#     features = {
#         'face_width': landmarks.landmark[234].x - landmarks.landmark[454].x,
#         'face_height': landmarks.landmark[152].y - landmarks.landmark[10].y,
#         'eye_distance': landmarks.landmark[33].x - landmarks.landmark[263].x,
#         'nose_length': landmarks.landmark[6].y - landmarks.landmark[94].y,
#         'mouth_width': landmarks.landmark[61].x - landmarks.landmark[291].x
#     }
#
#     return features
#
# def analyze_skin_tone(image_path):
#     """Analyze skin tone from the uploaded image."""
#     image = cv2.imread(image_path)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(image_rgb)
#
#     if not results.multi_face_landmarks:
#         raise ValueError("No face detected in the image")
#
#     # Get face region
#     landmarks = results.multi_face_landmarks[0]
#     face_points = np.array([[int(l.x * image.shape[1]), int(l.y * image.shape[0])]
#                           for l in landmarks.landmark])
#
#     # Create mask for face region
#     mask = np.zeros(image.shape[:2], dtype=np.uint8)
#     cv2.fillConvexPoly(mask, face_points, 255)
#
#     # Get average skin color
#     face_region = cv2.bitwise_and(image, image, mask=mask)
#     skin_color = cv2.mean(face_region, mask=mask)[:3]
#
#     return [int(c) for c in skin_color]
#
# @app.route('/api/avatar/generate', methods=['POST'])
# @login_required
# def generate_avatar():
#     try:
#         if 'photo' not in request.files:
#             return jsonify({'error': 'No photo uploaded'}), 400
#
#         photo = request.files['photo']
#         gender = request.form.get('gender', 'female')
#
#         if photo.filename == '':
#             return jsonify({'error': 'No selected file'}), 400
#
#         # Save uploaded photo
#         filename = secure_filename(photo.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         photo.save(filepath)
#
#         # Extract facial features
#         features = extract_facial_features(filepath)
#
#         # Analyze skin tone
#         skin_color = analyze_skin_tone(filepath)
#
#         # Prepare avatar data
#         avatar_data = {
#             'model_path': MODEL_PATHS[gender]['model'],
#             'textures': MODEL_PATHS[gender]['textures'],
#             'features': features,
#             'skin_color': skin_color,
#             'gender': gender
#         }
#
#         # Save avatar data to user's profile
#         user_id = session.get('user_id')
#         if user_id:
#             db.users.update_one(
#                 {'_id': ObjectId(user_id)},
#                 {'$set': {'avatar_data': avatar_data}}
#             )
#
#         return jsonify(avatar_data)
#
#     except Exception as e:
#         app.logger.error(f"Error in generate_avatar: {str(e)}")
#         return jsonify({'error': str(e)}), 500
#
# @app.route('/api/avatar/get', methods=['GET'])
# @login_required
# def get_user_avatar_data():
#     """Get user's current avatar data"""
#     try:
#         user_id = session['user']['_id']
#         avatar_doc = db.avatars.find_one({'userId': user_id})
#
#         if not avatar_doc:
#             return jsonify({'error': 'No avatar found'}), 404
#
#         # Get the avatar data
#         avatar_data = avatar_doc.get('avatarData', {})
#
#         # Add the model path if not present
#         if 'model_path' not in avatar_data and 'gender' in avatar_data:
#             avatar_data['model_path'] = MODEL_PATHS[avatar_data['gender'].lower()]
#
#         return jsonify({
#             'success': True,
#             'avatarData': avatar_data
#         })
#
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
#
# # New endpoint for trying on clothes
# @app.route('/api/avatar/try-on', methods=['POST'])
# @login_required
# def try_on_clothes():
#     """Try on clothes from wardrobe on the avatar"""
#     try:
#         user_id = session['user']['_id']
#         item_id = request.json.get('itemId')
#
#         if not item_id:
#             return jsonify({'error': 'No item ID provided'}), 400
#
#         # Get the clothing item from the wardrobe
#         item = db.wardrobe.find_one({'_id': ObjectId(item_id), 'userId': user_id})
#
#         if not item:
#             return jsonify({'error': 'Item not found'}), 404
#
#         # Get the avatar data
#         avatar_data = db.avatars.find_one({'userId': user_id})
#
#         if not avatar_data:
#             return jsonify({'error': 'No avatar found. Please create an avatar first.'}), 404
#
#         # Return the item data for the avatar to wear
#         return jsonify({
#             'success': True,
#             'item': {
#                 'id': str(item['_id']),
#                 'type': item.get('label', '').lower(),
#                 'color': item.get('color', ''),
#                 'image_url': normalize_path(item.get('file_path', ''))
#             }
#         })
#
#     except Exception as e:
#         print(f"Error in try_on_clothes: {str(e)}")
#         return jsonify({'error': str(e)}), 500
#
# @app.route('/model-inspector')
# @login_required
# def model_inspector():
#     return render_template('model_inspector.html')
#
#
# # Update avatar data
# @app.route('/api/avatar/update', methods=['POST'])
# @login_required
# def update_avatar():
#     try:
#         user_id = session['user']['_id']
#         avatar_data = request.json
#
#         if not avatar_data:
#             return jsonify({'error': 'No avatar data provided'}), 400
#
#         # Update avatar document
#         result = db.avatars.update_one(
#             {'userId': user_id},
#             {
#                 '$set': {
#                     'avatarData': avatar_data,
#                     'updatedAt': datetime.now()
#                 }
#             },
#             upsert=True
#         )
#
#         return jsonify({
#             'success': True,
#             'message': 'Avatar updated successfully'
#         })
#
#     except Exception as e:
#         print(f"Error updating avatar: {str(e)}")
#         return jsonify({'error': str(e)}), 500
#
#


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

    @app.route('/api/wardrobe/item/<item_id>', methods=['GET'])
    @login_required
    def get_wardrobe_item(item_id):
        try:
            user_id = session['user']['_id']

            # Find the item in the database
            item = db.wardrobe.find_one({'_id': ObjectId(item_id), 'userId': user_id})

            if not item:
                return jsonify({'success': False, 'error': 'Item not found'}), 404

            # Convert ObjectId to string for JSON serialization
            item['_id'] = str(item['_id'])

            # Normalize file path
            if 'file_path' in item:
                item['file_path'] = normalize_path(item['file_path'])

            return jsonify(item)

        except Exception as e:
            print(f"Error getting wardrobe item: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 500

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

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

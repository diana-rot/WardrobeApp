"""
Texture Mapping Utilities for WardrobeApp
Add this to a new file called 'texture_mapping.py' in your flaskapp directory
"""

import os
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import uuid

try:
    from rembg import remove as remove_bg

    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("Warning: rembg not available. Install with 'pip install rembg' for better background removal.")


# Fallback background removal if rembg is not available
def simple_background_removal(img_path):
    """Simple background removal using OpenCV"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")

    # Convert to RGBA
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # Create a grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a mask
    _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    # Dilate the mask to get rid of small holes
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Apply the mask to the alpha channel
    rgba[:, :, 3] = mask

    output_path = img_path.replace('.jpg', '_nobg.png').replace('.jpeg', '_nobg.png').replace('.png', '_nobg.png')
    cv2.imwrite(output_path, rgba)

    return output_path


def remove_background(img_path):
    """Remove background from clothing image"""
    if REMBG_AVAILABLE:
        try:
            # Load image with PIL
            input_img = Image.open(img_path)

            # Remove background
            output_img = remove_bg(input_img)

            # Save as transparent PNG
            output_path = img_path.replace('.jpg', '_nobg.png').replace('.jpeg', '_nobg.png').replace('.png',
                                                                                                      '_nobg.png')
            output_img.save(output_path)

            return output_path
        except Exception as e:
            print(f"Error in rembg: {str(e)}. Falling back to simple method.")
            return simple_background_removal(img_path)
    else:
        return simple_background_removal(img_path)


def get_model_and_uv_map(clothing_type):
    """Return the 3D model path and UV map for a given clothing type"""
    # Map clothing types to model paths
    model_map = {
        'T-shirt/top': 'tshirt',
        'Shirt': 'shirt',
        'Trouser': 'pants',
        'Pullover': 'pullover',
        'Dress': 'dress',
        'Coat': 'coat',
        'Sandal': 'sandal',
        'Sneaker': 'sneaker',
        'Bag': 'bag',
        'Ankle boot': 'ankle_boot'
    }

    # Default to tshirt if clothing type not found
    model_name = model_map.get(clothing_type, 'tshirt')

    # Model path should point to the GLB file
    model_path = f'/static/models/clothing/{model_name}.glb'

    # UV map is typically a 2D template that matches the 3D model's unwrapped coordinates
    # In practice, you might have these as actual files
    uv_map = {
        'type': model_name,
        'front_area': {
            'x': 0.1,
            'y': 0.1,
            'width': 0.8,
            'height': 0.6
        }
    }

    return model_path, uv_map


def warp_image_to_uv(img, uv_map):
    """
    Warp the clothing image to fit the UV map of the 3D model
    This is a simplified version - production would need more sophisticated mapping
    """
    if isinstance(img, str):
        # If img is a path, load it
        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)

    # Get the dimensions of the original image
    h, w = img.shape[:2]

    # Create a standard UV texture (typically 1024x1024 for clothing)
    texture_size = 1024
    texture = np.zeros((texture_size, texture_size, 4), dtype=np.uint8)

    # Get front area from UV map (simplified for this example)
    front_area = uv_map.get('front_area', {
        'x': 0.25, 'y': 0.25, 'width': 0.5, 'height': 0.5
    })

    # Calculate target area in texture
    tx = int(front_area['x'] * texture_size)
    ty = int(front_area['y'] * texture_size)
    tw = int(front_area['width'] * texture_size)
    th = int(front_area['height'] * texture_size)

    # Resize the image to fit the target area
    resized_img = cv2.resize(img, (tw, th))

    # Place the resized image in the texture
    if resized_img.shape[2] == 4:  # With alpha channel
        # Place the image with alpha blending
        for c in range(3):  # RGB channels
            texture[ty:ty + th, tx:tx + tw, c] = (
                    resized_img[:, :, c] * (resized_img[:, :, 3] / 255.0)
            ).astype(np.uint8)
        # Copy alpha channel
        texture[ty:ty + th, tx:tx + tw, 3] = resized_img[:, :, 3]
    else:  # Without alpha channel
        texture[ty:ty + th, tx:tx + tw, :3] = resized_img
        texture[ty:ty + th, tx:tx + tw, 3] = 255  # Fully opaque

    return texture


def generate_normal_map(texture):
    """Generate a normal map from a texture to add surface detail"""
    if isinstance(texture, str):
        # If texture is a path, load it
        texture = cv2.imread(texture, cv2.IMREAD_UNCHANGED)

    # Ensure we have at least 3 channels
    if texture.shape[2] < 3:
        # Convert grayscale to RGB
        texture = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)

    # Convert to grayscale for height information
    if texture.shape[2] >= 3:
        gray = cv2.cvtColor(texture[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        gray = texture

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

    return normal_map


def process_clothing_texture(img_path, clothing_type):
    """
    Process a clothing image for 3D visualization
    Returns paths to model, texture, and normal map
    """
    # 1. Remove background
    print(f"Removing background from {img_path}")
    no_bg_path = remove_background(img_path)

    # 2. Get model and UV map for clothing type
    print(f"Getting model for {clothing_type}")
    model_path, uv_map = get_model_and_uv_map(clothing_type)

    # 3. Warp image to UV map
    print("Warping image to UV map")
    warped_texture = warp_image_to_uv(no_bg_path, uv_map)

    # 4. Generate file paths
    base_dir = os.path.dirname(img_path)
    filename = os.path.basename(img_path).split(".")[0]

    # 5. Save texture
    texture_path = os.path.join(base_dir, f"{filename}_texture.png")
    cv2.imwrite(texture_path, warped_texture)

    # 6. Generate and save normal map
    print("Generating normal map")
    normal_map = generate_normal_map(warped_texture)
    normal_map_path = os.path.join(base_dir, f"{filename}_normal.png")
    cv2.imwrite(normal_map_path, normal_map)

    # 7. Return paths
    return {
        'model_path': model_path,
        'texture_path': texture_path,
        'normal_map_path': normal_map_path
    }


def process_texture(image_path, clothing_type):
    """
    Simplified texture processing that:
    1. Removes background
    2. Creates a simple texture map
    3. Returns paths to save in database
    """
    try:
        # 1. Load and process image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # 2. Create a simple texture map (1024x1024)
        texture_size = 1024
        texture = np.zeros((texture_size, texture_size, 4), dtype=np.uint8)
        
        # 3. Resize and center the image
        h, w = img.shape[:2]
        scale = min(texture_size/w, texture_size/h) * 0.8  # 80% of the texture
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        
        # 4. Center the image in the texture
        x = (texture_size - new_w) // 2
        y = (texture_size - new_h) // 2
        texture[y:y+new_h, x:x+new_w, :3] = resized
        texture[y:y+new_h, x:x+new_w, 3] = 255  # Alpha channel

        # 5. Save the texture
        base_dir = os.path.dirname(image_path)
        filename = os.path.basename(image_path).split(".")[0]
        texture_path = os.path.join(base_dir, f"{filename}_texture.png")
        cv2.imwrite(texture_path, texture)

        # 6. Create a simple normal map
        normal_map = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
        normal_map[..., 2] = 255  # Default normal pointing up
        normal_map_path = os.path.join(base_dir, f"{filename}_normal.png")
        cv2.imwrite(normal_map_path, normal_map)

        return {
            'texture_path': texture_path,
            'normal_map_path': normal_map_path,
            'material_properties': {
                'roughness': 0.7,
                'metalness': 0.1
            }
        }

    except Exception as e:
        print(f"Error processing texture: {str(e)}")
        return None


def apply_texture_to_model(model, texture_data):
    """
    Apply texture to a 3D model
    """
    try:
        # Load textures
        texture = cv2.imread(texture_data['texture_path'], cv2.IMREAD_UNCHANGED)
        normal_map = cv2.imread(texture_data['normal_map_path'])
        
        # Convert to base64 for storage
        _, texture_buffer = cv2.imencode('.png', texture)
        _, normal_buffer = cv2.imencode('.png', normal_map)
        
        texture_base64 = texture_buffer.tobytes()
        normal_base64 = normal_buffer.tobytes()
        
        return {
            'texture_data': texture_base64,
            'normal_map_data': normal_base64,
            'material_properties': texture_data['material_properties']
        }
        
    except Exception as e:
        print(f"Error applying texture: {str(e)}")
        return None
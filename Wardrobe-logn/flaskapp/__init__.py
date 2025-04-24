from flask import Flask, session, redirect, render_template, request, jsonify
from functools import wraps
import pymongo
import mediapipe as mp
import numpy as np
import cv2
from PIL import Image
import io
import base64
import logging
import os
import json
import gc
from bson.objectid import ObjectId

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = b'\xcc^\x91\xea\x17-\xd0W\x03\xa7\xf8J0\xac8\xc5'

# Database

# from flask_mongoengine import MongoEngine

# app = Flask(__name__)
# app.config.from_pyfile('the-config.cfg')
# db = MongoEngine(app)
client = pymongo.MongoClient('localhost', 27017)
db = client.user_login_system_test

# Decorators
def login_required(f):
  @wraps(f)
  def wrap(*args, **kwargs):
    if 'logged_in' in session:
      return f(*args, **kwargs)
    else:
      return redirect('/set')

  return wrap

# Initialize MediaPipe solutions globally
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def cleanup_resources():
    """Helper function to clean up resources"""
    gc.collect()
    cv2.destroyAllWindows()

@app.route('/api/extract-features', methods=['POST'])
def extract_features():
    features = {}
    cv_image = None
    rgb_image = None
    
    try:
        logger.info("Starting feature extraction process...")
        
        # Get the image data from the request
        image_data = request.json.get('image')
        if not image_data:
            logger.error("No image data provided in request")
            return jsonify({'success': False, 'error': 'No image data provided'})

        # Convert base64 to image
        try:
            # Clean up any previous resources
            cleanup_resources()
            
            image_data = image_data.split(',')[1] if ',' in image_data else image_data
            image_bytes = base64.b64decode(image_data)
            
            # Use context manager for PIL Image
            with Image.open(io.BytesIO(image_bytes)) as image:
                logger.info(f"Image loaded successfully - Format: {image.format}, Size: {image.size}, Mode: {image.mode}")
                
                if image.mode != 'RGB':
                    logger.info(f"Converting image from {image.mode} to RGB")
                    image = image.convert('RGB')
                
                # Convert to CV2 format with explicit memory management
                cv_image = np.ascontiguousarray(cv2.cvtColor(np.array(image, dtype=np.uint8), cv2.COLOR_RGB2BGR))
                
            # Clear the original image data
            del image_bytes
            gc.collect()
            
            height, width = cv_image.shape[:2]
            logger.info(f"Image dimensions - Width: {width}px, Height: {height}px")
            
            image_size = {'width': width, 'height': height}
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}", exc_info=True)
            cleanup_resources()
            return jsonify({'success': False, 'error': 'Error processing image format'})

        # Convert to RGB once for both face and body detection
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Extract face features with resource management
        try:
            logger.info("Starting face feature extraction...")
            face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            try:
                face_results = face_mesh.process(rgb_image)
                
                if face_results.multi_face_landmarks:
                    logger.info("Face detected in image")
                    face_landmarks = face_results.multi_face_landmarks[0]
                    features['face'] = extract_face_features(face_landmarks, image_size)
                    logger.info("Face features extracted:")
                    logger.info(json.dumps(features['face'], indent=2))
                else:
                    logger.warning("No face detected in the image")
            finally:
                face_mesh.close()
                del face_mesh
                gc.collect()
                
        except Exception as e:
            logger.error(f"Error in face feature extraction: {str(e)}", exc_info=True)
            features['face'] = None

        # Extract body features with resource management
        try:
            logger.info("Starting body feature extraction...")
            pose = mp_pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5
            )
            
            try:
                pose_results = pose.process(rgb_image)
                
                if pose_results.pose_landmarks:
                    features['body'] = extract_body_features(pose_results.pose_landmarks, image_size)
                    logger.info("Body features extracted:")
                    logger.info(json.dumps(features['body'], indent=2))
                else:
                    logger.warning("No body pose detected in the image")
            finally:
                pose.close()
                del pose
                gc.collect()
                
        except Exception as e:
            logger.error(f"Error in body feature extraction: {str(e)}", exc_info=True)
            features['body'] = None

        # Clean up image resources
        if cv_image is not None:
            del cv_image
        if rgb_image is not None:
            del rgb_image
        cleanup_resources()

        # Return whatever features we were able to extract
        if not features.get('face') and not features.get('body'):
            logger.error("No features could be extracted from the image")
            return jsonify({'success': False, 'error': 'No features could be extracted from the image'})

        logger.info("Feature extraction completed")
        logger.info("Final feature set:")
        logger.info(json.dumps(features, indent=2))

        return jsonify({
            'success': True,
            'features': features,
            'image_size': image_size
        })

    except Exception as e:
        logger.error(f"Error in feature extraction: {str(e)}", exc_info=True)
        cleanup_resources()
        return jsonify({'success': False, 'error': str(e)})
    finally:
        # Ensure resources are cleaned up
        cleanup_resources()

def extract_face_features(landmarks, image_size):
    """Extract normalized face measurements"""
    logger.info("Starting face feature measurements...")
    
    width, height = image_size['width'], image_size['height']
    features = {}
    
    try:
        def get_distance(p1, p2):
            try:
                return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
            except Exception as e:
                logger.error(f"Error calculating distance: {str(e)}")
                return 0.0

        # Dictionary of landmark pairs for measurements
        landmark_pairs = {
            'face_width': (234, 454),
            'face_height': (10, 152),
            'eye_distance': (133, 362),
            'nose_length': (6, 4),
            'mouth_width': (57, 287),
            'jaw_width': (132, 361)
        }

        # Take measurements with error handling
        for name, (idx1, idx2) in landmark_pairs.items():
            try:
                value = get_distance(landmarks.landmark[idx1], landmarks.landmark[idx2])
                features[name] = float(value)  # Ensure value is a regular float
                logger.info(f"{name} measurement: {value:.4f}")
            except Exception as e:
                logger.error(f"Error measuring {name}: {str(e)}")
                features[name] = 0.0

        # Calculate confidence
        try:
            visible_landmarks = [lm for lm in landmarks.landmark if hasattr(lm, 'visibility') and lm.visibility > 0.5]
            features['confidence'] = float(len(visible_landmarks)) / len(landmarks.landmark)
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            features['confidence'] = 0.0

        # Normalize measurements
        base_scale = features.get('face_width', 0)
        if base_scale > 0:
            features = {k: float(v/base_scale) for k, v in features.items()}

        return features

    except Exception as e:
        logger.error(f"Error in face feature extraction: {str(e)}", exc_info=True)
        return {'error': str(e)}

def extract_body_features(landmarks, image_size):
    """Extract normalized body measurements"""
    logger.info("Starting body feature measurements...")
    
    width, height = image_size['width'], image_size['height']
    features = {}
    
    try:
        def get_distance(p1, p2):
            try:
                if not (hasattr(p1, 'visibility') and hasattr(p2, 'visibility')):
                    return 0.0
                if p1.visibility < 0.5 or p2.visibility < 0.5:
                    return 0.0
                return float(np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2))
            except Exception as e:
                logger.error(f"Error calculating body distance: {str(e)}")
                return 0.0

        # Dictionary of measurements and their landmark pairs
        measurements = {
            'shoulder_width': (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
            'hip_width': (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
            'torso_length': (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_HIP)  # Using nose to hip as torso length
        }

        # Take measurements
        for name, (lm1, lm2) in measurements.items():
            try:
                value = get_distance(landmarks.landmark[lm1.value], landmarks.landmark[lm2.value])
                features[name] = float(value)
                logger.info(f"{name} measurement: {value:.4f}")
            except Exception as e:
                logger.error(f"Error measuring {name}: {str(e)}")
                features[name] = 0.0

        # Calculate arm length
        try:
            left_upper_arm = get_distance(
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            )
            left_forearm = get_distance(
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value]
            )
            features['arm_length'] = float(left_upper_arm + left_forearm)
            logger.info(f"Arm length measurement: {features['arm_length']:.4f}")
        except Exception as e:
            logger.error(f"Error measuring arm length: {str(e)}")
            features['arm_length'] = 0.0

        # Calculate leg length
        try:
            left_thigh = get_distance(
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value],
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value]
            )
            left_calf = get_distance(
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value],
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            )
            features['leg_length'] = float(left_thigh + left_calf)
            logger.info(f"Leg length measurement: {features['leg_length']:.4f}")
        except Exception as e:
            logger.error(f"Error measuring leg length: {str(e)}")
            features['leg_length'] = 0.0

        # Add additional measurements
        try:
            # Shoulder to hip distance (alternative torso measurement)
            features['shoulder_to_hip'] = float(get_distance(
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value]
            ))
            
            # Chest width (measured between armpits)
            chest_width = get_distance(
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            )
            features['chest_width'] = float(chest_width)
            
            logger.info(f"Additional measurements - Shoulder to hip: {features['shoulder_to_hip']:.4f}, Chest width: {features['chest_width']:.4f}")
        except Exception as e:
            logger.error(f"Error measuring additional features: {str(e)}")

        # Calculate confidence and normalize
        try:
            visible_landmarks = [lm for lm in landmarks.landmark if hasattr(lm, 'visibility') and lm.visibility > 0.5]
            features['confidence'] = float(len(visible_landmarks)) / len(landmarks.landmark)
            logger.info(f"Confidence score: {features['confidence']:.4f}")
            
            # Use shoulder width as the base scale
            base_scale = features.get('shoulder_width', 0)
            if base_scale > 0:
                normalized_features = {k: float(v/base_scale) for k, v in features.items()}
                logger.info("Normalized measurements:")
                for key, value in normalized_features.items():
                    logger.info(f"  {key}: {value:.4f}")
                return normalized_features
            else:
                logger.warning("Invalid base scale (shoulder_width is 0), returning unnormalized features")
                return features

        except Exception as e:
            logger.error(f"Error in normalization: {str(e)}")
            features['confidence'] = 0.0
            return features

    except Exception as e:
        logger.error(f"Error in body feature extraction: {str(e)}", exc_info=True)
        return {'error': str(e)}

@app.route('/rpm-avatar', methods=['GET'])
@login_required
def rpm_avatar():
    """Route for Ready Player Me avatar creation and management"""
    return render_template('rpm_avatar.html')

@app.route('/api/rpm/save-avatar', methods=['POST'])
@login_required
def save_rpm_avatar():
    """Save the Ready Player Me avatar URL to the user's profile"""
    try:
        data = request.json
        avatar_url = data.get('avatarUrl')
        
        if not avatar_url:
            return jsonify({'success': False, 'error': 'No avatar URL provided'})
        
        # Get user ID from session
        user_id = session.get('user_id')
        if not user_id:
            # Fallback to user object if user_id not directly in session
            user = session.get('user')
            if user and '_id' in user:
                user_id = user['_id']
            else:
                return jsonify({'success': False, 'error': 'User not logged in'})
        
        # Update user's avatar URL in database
        result = db.users.update_one(
            {'_id': user_id},
            {'$set': {'rpm_avatar_url': avatar_url}}
        )
        
        if result.modified_count > 0:
            return jsonify({'success': True, 'avatarUrl': avatar_url})
        else:
            return jsonify({'success': False, 'error': 'Failed to update avatar'})
            
    except Exception as e:
        logger.error(f"Error saving RPM avatar: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/rpm/get-avatar', methods=['GET'])
@login_required
def get_rpm_avatar():
    """Get the user's saved Ready Player Me avatar URL"""
    try:
        # Get user ID from session
        user_id = session.get('user_id')
        if not user_id:
            # Fallback to user object if user_id not directly in session
            user = session.get('user')
            if user and '_id' in user:
                user_id = user['_id']
            else:
                return jsonify({'success': False, 'error': 'User not logged in'})
        
        # Find user in database
        user = db.users.find_one({'_id': user_id})
        if not user:
            return jsonify({'success': False, 'error': 'User not found'})
        
        # Get avatar URL
        avatar_url = user.get('rpm_avatar_url')
        if avatar_url:
            return jsonify({'success': True, 'avatarUrl': avatar_url})
        else:
            return jsonify({'success': False, 'error': 'No avatar found'})
            
    except Exception as e:
        logger.error(f"Error getting RPM avatar: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})
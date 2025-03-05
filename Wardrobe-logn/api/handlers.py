from flask import jsonify, request, current_app
from bson import ObjectId
from datetime import datetime
import os
import numpy as np
import random
import requests
from .utils import normalize_path, save_base64_image
from .auth import db


# Don't import directly
# Instead, we'll get these from the main application context

def handle_login():
    """Handle API login request"""
    data = request.get_json()

    if not data or 'email' not in data or 'password' not in data:
        return jsonify({'success': False, 'message': 'Email and password required'}), 400

    # Find user in database
    from .auth import generate_token
    user = db.users.find_one({'email': data['email']})

    # Use your existing password verification logic
    if user and user['password'] == data['password']:  # You should use proper hashing
        # Generate JWT token
        token = generate_token(user['_id'])

        return jsonify({
            'success': True,
            'token': token,
            'user': {
                'id': str(user['_id']),
                'name': user.get('name', ''),
                'email': user['email']
            }
        })

    return jsonify({'success': False, 'message': 'Invalid credentials'}), 401


def handle_get_wardrobe(current_user):
    """Handle getting all wardrobe items"""
    try:
        user_id = current_user['_id']
        items = list(db.wardrobe.find({'userId': user_id}))

        # Convert ObjectIds to strings and normalize file paths
        for item in items:
            item['_id'] = str(item['_id'])
            if 'file_path' in item:
                item['file_path'] = normalize_path(item['file_path'])

        return jsonify({
            'success': True,
            'items': items
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


def handle_add_wardrobe_item(current_user):
    """Handle adding a new wardrobe item with image"""
    try:
        if not request.json or 'image' not in request.json:
            return jsonify({'success': False, 'message': 'No image data provided'}), 400

        user_id = current_user['_id']
        image_data = request.json['image']

        # Save the image
        file_path, db_path = save_base64_image(image_data, user_id)

        # Get the global functions from the main application
        # (These will be passed in from run.py when the blueprint is registered)
        import sys
        model = sys.modules['__main__'].model
        model_predict = sys.modules['__main__'].model_predict
        predict_color = sys.modules['__main__'].predict_color

        # Make predictions using your model
        preds = model_predict(file_path, model)
        if not isinstance(preds, np.ndarray) or preds.size == 0:
            raise ValueError("Invalid prediction output")

        # Get color prediction
        color_result = predict_color(file_path)
        if not color_result or len(color_result) < 2:
            raise ValueError("Invalid color prediction")

        # Determine clothing type
        predicted_label = np.argmax(preds)
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        if predicted_label >= len(class_names):
            raise ValueError("Invalid predicted label index")

        result = class_names[predicted_label]

        # Save to database
        item_id = db.wardrobe.insert_one({
            'label': result,
            'color': ' '.join(map(str, color_result[1])),
            'nota': 4,  # Default rating
            'userId': user_id,
            'file_path': db_path,
            'created_at': datetime.utcnow(),
            'last_worn': None,
            'times_worn': 0
        }).inserted_id

        return jsonify({
            'success': True,
            'item': {
                'id': str(item_id),
                'label': result,
                'file_path': db_path,
                'color': ' '.join(map(str, color_result[1]))
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


def handle_delete_wardrobe_item(current_user, item_id):
    """Handle deleting a wardrobe item"""
    try:
        user_id = current_user['_id']

        # Find the item
        item = db.wardrobe.find_one({'_id': ObjectId(item_id), 'userId': user_id})
        if not item:
            return jsonify({'success': False, 'message': 'Item not found'}), 404

        # Delete physical file if exists
        if 'file_path' in item:
            file_path = os.path.join('flaskapp', item['file_path'].lstrip('/'))
            if os.path.exists(file_path):
                os.remove(file_path)

        # Delete from database
        result = db.wardrobe.delete_one({'_id': ObjectId(item_id), 'userId': user_id})

        if result.deleted_count:
            return jsonify({'success': True})
        return jsonify({'success': False, 'message': 'Delete failed'}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


def handle_register():
    """Handle API registration request"""
    data = request.get_json()

    if not data or 'email' not in data or 'password' not in data:
        return jsonify({'success': False, 'message': 'Email and password required'}), 400

    # Check if user already exists
    if db.users.find_one({'email': data['email']}):
        return jsonify({'success': False, 'message': 'Email already registered'}), 409

    # Create new user
    try:
        from .auth import generate_token
        user_id = db.users.insert_one({
            'email': data['email'],
            'password': data['password'],  # You should hash this in production
            'name': data.get('name', ''),
            'profile_picture': '/static/image/default-profile.png',
            'created_at': datetime.utcnow()
        }).inserted_id

        token = generate_token(user_id)

        return jsonify({
            'success': True,
            'token': token,
            'user': {
                'id': str(user_id),
                'name': data.get('name', ''),
                'email': data['email']
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# api/auth.py
from functools import wraps
from flask import request, jsonify, current_app
from datetime import datetime, timedelta
import jwt
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client.user_login_system_test


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization')

        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]

        if not token:
            return jsonify({'success': False, 'message': 'Token is missing'}), 401

        try:
            # Decode the token
            data = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = db.users.find_one({'_id': data['user_id']})
            if not current_user:
                return jsonify({'success': False, 'message': 'User not found'}), 401
        except Exception as e:
            return jsonify({'success': False, 'message': 'Invalid token', 'error': str(e)}), 401

        # Pass the user to the route
        return f(current_user, *args, **kwargs)

    return decorated


def generate_token(user_id):
    """Generate a JWT token for authentication"""
    payload = {
        'user_id': str(user_id),
        'exp': datetime.utcnow() + timedelta(days=7)
    }
    return jwt.encode(payload, current_app.config['SECRET_KEY'])

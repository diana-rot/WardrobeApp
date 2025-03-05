import os
import base64
from datetime import datetime
from bson import ObjectId


# Don't import directly - we'll pass these as parameters instead
# from flaskapp import model, model_predict, predict_color

def normalize_path(file_path):
    """Normalize file paths for consistent formatting"""
    if not file_path:
        return None
    return file_path.replace('/outfit/', '/').replace('/static/static/', '/static/').lstrip('/')


def save_base64_image(base64_string, user_id, prefix="wardrobe"):
    """Save a base64 encoded image to disk and return the file path"""
    if base64_string.startswith('data:image'):
        base64_string = base64_string.split(',')[1]

    image_data = base64.b64decode(base64_string)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{prefix}_{timestamp}.jpg'

    # Create directory if it doesn't exist
    upload_dir = os.path.join('flaskapp', 'static', 'image_users', str(user_id))
    os.makedirs(upload_dir, exist_ok=True)

    # Save file
    file_path = os.path.join(upload_dir, filename)
    with open(file_path, 'wb') as f:
        f.write(image_data)

    # Return the path to be stored in DB
    db_path = f'/static/image_users/{user_id}/{filename}'
    return file_path, db_path
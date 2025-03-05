from flask import Flask
from flask_cors import CORS
from api import api_bp
import os


def create_app():
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes

    # Set configuration
    app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'

    # Register the API blueprint
    app.register_blueprint(api_bp)

    return app


if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 5001))  # Use a different port than your main app
    app.run(debug=True, host='0.0.0.0', port=port)
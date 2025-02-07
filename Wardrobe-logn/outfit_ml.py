# outfit_ml.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import joblib


class OutfitRecommender:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,  # More trees for better stability
            max_depth=15,  # Control overfitting
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',  # Handle imbalanced classes
            random_state=42
        )
        self.scaler = StandardScaler()

    def prepare_features(self, weather_data, event_type, user_preferences=None):
        """
        Prepare features for prediction with enhanced feature engineering
        """
        features = []

        # Weather features
        temp = weather_data.get('temperature', 20)
        features.extend([
            temp,  # Raw temperature
            1 if temp <= 10 else 0,  # Cold
            1 if 10 < temp <= 20 else 0,  # Mild
            1 if 20 < temp <= 25 else 0,  # Warm
            1 if temp > 25 else 0,  # Hot
            weather_data.get('humidity', 50) / 100,  # Normalized humidity
            1 if 'rain' in weather_data.get('description', '').lower() else 0,
            1 if 'snow' in weather_data.get('description', '').lower() else 0
        ])

        # Time features
        current_time = datetime.now()
        features.extend([
            current_time.hour / 24,  # Normalized hour
            1 if current_time.weekday() >= 5 else 0,  # Weekend flag
            np.sin(2 * np.pi * current_time.month / 12),  # Seasonal cycle
            np.cos(2 * np.pi * current_time.month / 12)
        ])

        # Event features
        event_features = {
            'casual': [1, 0, 0, 0, 0],
            'work': [0, 1, 0, 0, 0],
            'formal': [0, 0, 1, 0, 0],
            'sport': [0, 0, 0, 1, 0],
            'travel': [0, 0, 0, 0, 1]
        }
        features.extend(event_features.get(event_type, [0, 0, 0, 0, 0]))

        # User preference features (if available)
        if user_preferences:
            features.extend([
                user_preferences.get('style_preference', 0),
                user_preferences.get('color_preference', 0),
                user_preferences.get('formality_preference', 0)
            ])
        else:
            features.extend([0, 0, 0])

        return np.array(features).reshape(1, -1)

    def train(self, training_data, user_ratings):
        """
        Train the model with user feedback
        """
        X = []
        y = []

        for outfit in training_data:
            features = self.prepare_features(
                outfit['weather_data'],
                outfit['event_type'],
                outfit['user_preferences']
            )
            X.append(features.flatten())
            y.append(outfit['outfit_type'])

        X = np.array(X)
        y = np.array(y)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model.fit(X_scaled, y)

    def get_recommendations(self, weather_data, event_type, user_preferences=None, n_recommendations=3):
        """
        Get outfit recommendations with confidence scores
        """
        features = self.prepare_features(weather_data, event_type, user_preferences)
        features_scaled = self.scaler.transform(features)

        # Get predictions and probabilities
        outfit_probas = self.model.predict_proba(features_scaled)[0]
        top_indices = np.argsort(outfit_probas)[-n_recommendations:][::-1]

        recommendations = []
        for idx in top_indices:
            outfit_type = self.model.classes_[idx]
            confidence = outfit_probas[idx]

            recommendation = {
                'outfit_type': outfit_type,
                'confidence': confidence,
                'weather_appropriate': self._check_weather_appropriateness(
                    outfit_type, weather_data
                ),
                'event_appropriate': self._check_event_appropriateness(
                    outfit_type, event_type
                )
            }
            recommendations.append(recommendation)

        return recommendations

    def _check_weather_appropriateness(self, outfit_type, weather_data):
        """
        Check if outfit is appropriate for weather conditions
        """
        temp = weather_data.get('temperature', 20)
        is_raining = 'rain' in weather_data.get('description', '').lower()
        is_snowing = 'snow' in weather_data.get('description', '').lower()

        checks = {
            'temperature': True,
            'precipitation': True
        }

        # Temperature checks
        if temp < 10 and 'Coat' not in outfit_type:
            checks['temperature'] = False
        elif temp > 25 and ('Coat' in outfit_type or 'Pullover' in outfit_type):
            checks['temperature'] = False

        # Precipitation checks
        if (is_raining or is_snowing) and 'Sandal' in outfit_type:
            checks['precipitation'] = False
        elif is_snowing and 'Sneaker' in outfit_type:
            checks['precipitation'] = False

        return all(checks.values())

    def _check_event_appropriateness(self, outfit_type, event_type):
        """
        Check if outfit is appropriate for event type
        """
        formal_pieces = ['Shirt', 'Trouser', 'Dress']
        casual_pieces = ['T-shirt/top', 'Sneaker', 'Sandal']
        sport_pieces = ['T-shirt/top', 'Sneaker']

        outfit_pieces = outfit_type.split('_')

        if event_type == 'formal':
            return any(piece in formal_pieces for piece in outfit_pieces)
        elif event_type == 'casual':
            return any(piece in casual_pieces for piece in outfit_pieces)
        elif event_type == 'sport':
            return any(piece in sport_pieces for piece in outfit_pieces)

        return True  # Default to True for other event types

    def save_model(self, path='outfit_model.joblib'):
        """Save model and scaler"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, path)

    def load_model(self, path='outfit_model.joblib'):
        """Load model and scaler"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
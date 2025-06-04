# flaskapp/ml_models/__init__.py
"""
Machine Learning Models package initialization with enhanced hybrid recognition
"""

import os

try:
    # Try to import the enhanced model manager first
    from .enhanced_model_manager import EnhancedModelManager, enhanced_model_manager

    # Set as the primary model manager
    ModelManager = EnhancedModelManager
    model_manager = enhanced_model_manager

    print("✅ Enhanced Model Manager imported successfully")
    print("🎯 Hybrid prediction system active (Filename + Model + Fallback)")

except ImportError as e:
    print(f"⚠️ Warning: Could not import enhanced model manager: {e}")

    try:
        # Fallback to original model manager
        from .model_manager import ModelManager, model_manager

        print("✅ Original Model Manager imported as fallback")

    except ImportError as e2:
        print(f"⚠️ Warning: Could not import original model manager: {e2}")


        # Create minimal emergency fallback
        class ModelManager:
            def __init__(self):
                self.class_names = [
                    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
                ]

            def predict_clothing(self, img_path):
                """Emergency fallback prediction"""
                print(f"🆘 Using emergency fallback for: {img_path}")

                # Simple filename-based prediction
                filename = os.path.basename(img_path).lower()

                # Basic keyword detection
                if any(word in filename for word in ['shirt', 'top', 'blouse']):
                    return {'all_predictions': [0.6, 0.1, 0.1, 0.05, 0.05, 0.02, 0.05, 0.01, 0.01, 0.01]}
                elif any(word in filename for word in ['trouser', 'pant', 'jean']):
                    return {'all_predictions': [0.1, 0.6, 0.1, 0.05, 0.05, 0.02, 0.05, 0.01, 0.01, 0.01]}
                elif any(word in filename for word in ['dress', 'gown']):
                    return {'all_predictions': [0.05, 0.05, 0.1, 0.6, 0.1, 0.02, 0.05, 0.01, 0.01, 0.01]}
                elif any(word in filename for word in ['shoe', 'sneaker', 'boot']):
                    return {'all_predictions': [0.05, 0.05, 0.05, 0.05, 0.05, 0.15, 0.05, 0.3, 0.05, 0.2]}
                else:
                    # Balanced fallback
                    return {'all_predictions': [0.15, 0.12, 0.13, 0.14, 0.11, 0.08, 0.09, 0.07, 0.06, 0.05]}

            def get_class_name(self, class_idx):
                if 0 <= class_idx < len(self.class_names):
                    return self.class_names[class_idx]
                return f"Unknown({class_idx})"

            def get_memory_info(self):
                return {'keras_loaded': False, 'text_classifier_loaded': False, 'last_used': {}}


        model_manager = ModelManager()
        print("🆘 Emergency fallback model manager created")

__all__ = ['ModelManager', 'model_manager']

print("✅ ML Models package initialized with hybrid recognition system")
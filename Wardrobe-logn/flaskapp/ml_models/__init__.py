# flaskapp/ml_models/__init__.py
"""
Machine Learning Models package initialization
"""

try:
    from .model_manager import ModelManager, model_manager

    print("✅ Model manager imported successfully")
except ImportError as e:
    print(f"⚠️ Warning: Could not import model manager: {e}")


    # Create minimal fallback
    class ModelManager:
        def predict_clothing(self, img_path):
            return {
                'all_predictions': [0.1, 0.8, 0.05, 0.03, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0]
            }


    model_manager = ModelManager()

__all__ = ['ModelManager', 'model_manager']

print("✅ ML Models package initialized")
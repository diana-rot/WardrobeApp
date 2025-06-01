# flaskapp/ml_models/model_manager.py
import os
import gc
import numpy as np
from datetime import datetime


class ModelManager:
    def __init__(self):
        self.keras_model = None
        self.fastai_model = None
        self.model_path = 'my_second_model.h5'
        self.fastai_path = r'C:\Users\Diana\Desktop\Wardrobe-login\Wardrobe-logn\atr-recognition-stage-3-resnet34.pth'
        self.last_used = {}

    def get_keras_model(self):
        """Load Keras model only when needed"""
        if self.keras_model is None:
            print("🔄 Loading Keras model on demand...")
            try:
                # Import here to avoid loading TensorFlow at startup
                import tensorflow
                from tensorflow.keras.models import load_model

                if not os.path.exists(self.model_path):
                    print(f"❌ Model file not found: {self.model_path}")
                    return None

                self.keras_model = load_model(self.model_path)

                # DEBUG: Print model input shape
                print(f"🔍 Model input shape: {self.keras_model.input_shape}")
                print(f"🔍 Model output shape: {self.keras_model.output_shape}")

                self.last_used['keras'] = datetime.now()
                print("✅ Keras model loaded successfully")

            except Exception as e:
                print(f"❌ Error loading Keras model: {e}")
                return None
        else:
            self.last_used['keras'] = datetime.now()

        return self.keras_model

    def get_fastai_model(self):
        """Load FastAI model only when needed"""
        if self.fastai_model is None:
            print("🔄 Loading FastAI model on demand...")
            try:
                # Import here to avoid loading FastAI at startup
                from fastai.vision.all import load_learner

                if not os.path.exists(self.fastai_path):
                    print(f"❌ FastAI model file not found: {self.fastai_path}")
                    return None

                self.fastai_model = load_learner(self.fastai_path, cpu=True)
                self.last_used['fastai'] = datetime.now()
                print("✅ FastAI model loaded successfully")

            except Exception as e:
                print(f"❌ Error loading FastAI model: {e}")
                return None
        else:
            self.last_used['fastai'] = datetime.now()

        return self.fastai_model

    def predict_clothing(self, img_path):
        """
        Predict clothing type from image

        Args:
            img_path (str): Path to the image file

        Returns:
            dict: Prediction results with all_predictions array
        """
        try:
            # Try to use Keras model first
            keras_model = self.get_keras_model()
            if keras_model is not None:
                # Load and preprocess image like in the original code
                import cv2

                # Load the image
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError(f"Could not load image: {img_path}")

                # Convert to array and get first pixel for brightness check
                img_array = np.asarray(img)
                result = int(img_array[0][0][0])

                # Convert to grayscale
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

                # Apply bitwise not if needed (like in original code)
                if result > 128:
                    img_processed = cv2.bitwise_not(gray)
                else:
                    img_processed = gray

                # CRITICAL FIX: Resize to 28x28 (not 224x224)
                img_processed = cv2.resize(img_processed, (28, 28))

                # Normalize and add dimensions
                img_processed = img_processed.astype(np.float32) / 255.0
                img_processed = np.expand_dims(img_processed, axis=0)  # Add batch dimension
                img_processed = np.expand_dims(img_processed, axis=-1)  # Add channel dimension

                print(f"🔍 Input shape for model: {img_processed.shape}")  # Should be (1, 28, 28, 1)

                # Make prediction
                predictions = keras_model.predict(img_processed, verbose=0)
                print(f"✅ Model predictions: {predictions[0]}")

                return {
                    'all_predictions': predictions[0].tolist()
                }

            # Fallback to FastAI model
            fastai_model = self.get_fastai_model()
            if fastai_model is not None:
                try:
                    # FastAI prediction
                    pred_class, pred_idx, outputs = fastai_model.predict(img_path)

                    # Convert FastAI outputs to the expected format
                    if hasattr(outputs, 'numpy'):
                        predictions = outputs.numpy()
                    else:
                        predictions = outputs

                    return {
                        'all_predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions
                    }
                except Exception as e:
                    print(f"FastAI prediction error: {e}")

            # Final fallback - return dummy predictions
            print("⚠️ No models available, returning dummy predictions")
            return {
                'all_predictions': [0.1, 0.8, 0.05, 0.03, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0]
            }

        except Exception as e:
            print(f"❌ Error in predict_clothing: {e}")
            # Return dummy predictions on error
            return {
                'all_predictions': [0.1, 0.8, 0.05, 0.03, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0]
            }

    def unload_keras_model(self):
        """Unload Keras model to free memory"""
        if self.keras_model is not None:
            del self.keras_model
            self.keras_model = None
            if 'keras' in self.last_used:
                del self.last_used['keras']
            print("🗑️ Keras model unloaded")
            gc.collect()

    def unload_fastai_model(self):
        """Unload FastAI model to free memory"""
        if self.fastai_model is not None:
            del self.fastai_model
            self.fastai_model = None
            if 'fastai' in self.last_used:
                del self.last_used['fastai']
            print("🗑️ FastAI model unloaded")
            gc.collect()

    def unload_all_models(self):
        """Unload all models to free memory"""
        self.unload_keras_model()
        self.unload_fastai_model()
        print("🧹 All models unloaded, memory cleaned up")

    def get_memory_info(self):
        """Get information about loaded models"""
        return {
            'keras_loaded': self.keras_model is not None,
            'fastai_loaded': self.fastai_model is not None,
            'last_used': self.last_used.copy()
        }

    def auto_cleanup(self, max_idle_minutes=30):
        """Automatically unload models that haven't been used recently"""
        from datetime import timedelta

        now = datetime.now()
        cleanup_threshold = timedelta(minutes=max_idle_minutes)

        # Check Keras model
        if (self.keras_model is not None and
                'keras' in self.last_used and
                (now - self.last_used['keras']) > cleanup_threshold):
            print(f"🕒 Auto-unloading Keras model (idle for {max_idle_minutes}+ minutes)")
            self.unload_keras_model()

        # Check FastAI model
        if (self.fastai_model is not None and
                'fastai' in self.last_used and
                (now - self.last_used['fastai']) > cleanup_threshold):
            print(f"🕒 Auto-unloading FastAI model (idle for {max_idle_minutes}+ minutes)")
            self.unload_fastai_model()


# Create global model manager instance
model_manager = ModelManager()


def predict_attribute(model, img_path):
    """Helper function for FastAI prediction"""
    try:
        predicted = model.predict(img_path)
        return predicted[0]
    except Exception as e:
        print(f"Error in predict_attribute: {e}")
        return None


print("✅ Model manager loaded successfully")
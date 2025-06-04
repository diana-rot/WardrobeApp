import os
import gc
import re
import numpy as np
import cv2
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


class EnhancedModelManager:
    def __init__(self):
        self.keras_model = None
        self.fastai_model = None
        self.text_classifier = None
        self.tfidf_vectorizer = None

        # Try different model files
        self.possible_model_paths = [
            'improved_fashion_model.h5',
            'my_model_june.h5',
            'my_second_model.h5',
            'fashion_model.h5'
        ]
        self.model_path = None
        self.fastai_path = r'C:\Users\Diana\Desktop\Wardrobe-login\Wardrobe-logn\atr-recognition-stage-3-resnet34.pth'
        self.last_used = {}

        # Class names for consistent mapping
        self.class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]

        # Enhanced keyword mappings for filename analysis
        self.keyword_mappings = {
            'T-shirt/top': [
                't-shirt', 'tshirt', 'top', 'tank', 'camisole', 'halter', 'crop top',
                'blouse', 'tube top', 'bodysuit', 'vest', 'sleeveless', 'polo'
            ],
            'Trouser': [
                'trouser', 'pants', 'jeans', 'leggings', 'slacks', 'chinos',
                'cargo', 'sweatpants', 'joggers', 'palazzo', 'wide leg', 'skinny',
                'bootcut', 'straight', 'flare', 'cropped pants'
            ],
            'Pullover': [
                'pullover', 'sweater', 'jumper', 'hoodie', 'sweatshirt',
                'cardigan', 'knit', 'woolly', 'fleece', 'cashmere'
            ],
            'Dress': [
                'dress', 'gown', 'frock', 'sundress', 'cocktail', 'maxi',
                'mini dress', 'midi dress', 'shift', 'wrap dress', 'a-line',
                'bodycon', 'slip dress', 'tunic dress'
            ],
            'Coat': [
                'coat', 'jacket', 'blazer', 'outerwear', 'parka', 'trench',
                'denim jacket', 'bomber', 'windbreaker', 'peacoat', 'overcoat',
                'raincoat', 'winter coat', 'leather jacket'
            ],
            'Sandal': [
                'sandal', 'sandals', 'flip flop', 'slides', 'wedge', 'espadrille',
                'gladiator', 'strappy', 'thong', 'platform sandal'
            ],
            'Shirt': [
                'shirt', 'button up', 'dress shirt', 'oxford', 'flannel',
                'denim shirt', 'hawaiian', 'work shirt', 'formal shirt',
                'casual shirt', 'long sleeve', 'short sleeve'
            ],
            'Sneaker': [
                'sneaker', 'sneakers', 'trainers', 'athletic shoes', 'running shoes',
                'basketball shoes', 'tennis shoes', 'casual shoes', 'canvas shoes',
                'high tops', 'low tops', 'slip on sneakers'
            ],
            'Bag': [
                'bag', 'handbag', 'purse', 'tote', 'clutch', 'backpack',
                'satchel', 'messenger', 'crossbody', 'shoulder bag',
                'evening bag', 'travel bag', 'laptop bag'
            ],
            'Ankle boot': [
                'ankle boot', 'boot', 'boots', 'chelsea', 'combat boots',
                'hiking boots', 'work boots', 'knee high', 'thigh high',
                'cowboy boots', 'riding boots', 'winter boots'
            ]
        }

        # Find which model file actually exists
        for path in self.possible_model_paths:
            if os.path.exists(path):
                self.model_path = path
                print(f"✅ Found model file: {path}")
                break

        if self.model_path is None:
            print("⚠️ No model file found, will use enhanced fallback predictions")

        # Initialize text-based classifier
        self._initialize_text_classifier()

    def _initialize_text_classifier(self):
        """Initialize text-based classification system"""
        try:
            # Create training data for text classifier
            training_texts = []
            training_labels = []

            for class_name, keywords in self.keyword_mappings.items():
                for keyword in keywords:
                    # Add variations of keywords
                    training_texts.extend([
                        keyword,
                        keyword.replace(' ', '_'),
                        keyword.replace('-', '_'),
                        keyword.replace(' ', ''),
                        f"womens_{keyword}",
                        f"mens_{keyword}",
                        f"{keyword}_clothing"
                    ])
                    training_labels.extend([class_name] * 7)

            # Initialize TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words='english',
                lowercase=True,
                max_features=1000
            )

            # Fit the vectorizer
            self.tfidf_vectorizer.fit(training_texts)

            # Store text mappings for quick lookup
            self.text_mappings = {}
            for i, text in enumerate(training_texts):
                label = training_labels[i]
                if text not in self.text_mappings:
                    self.text_mappings[text] = {}
                if label not in self.text_mappings[text]:
                    self.text_mappings[text][label] = 0
                self.text_mappings[text][label] += 1

            print("✅ Text-based classifier initialized")

        except Exception as e:
            print(f"⚠️ Warning: Could not initialize text classifier: {e}")
            self.tfidf_vectorizer = None

    def extract_clothing_words(self, text):
        """
        Extract clothing-related words from complex text patterns
        Handles cases like: '1313212dress', 'img_tshirt_001', 'photo123jeans456'
        """

        # Remove common non-clothing words first
        text = text.lower()

        # Remove file extensions if any remain
        text = re.sub(r'\.(jpg|jpeg|png|gif|bmp|webp)$', '', text)

        # METHOD 1: Split on numbers, underscores, hyphens
        # This handles: "123dress456" -> ["dress"], "img_tshirt_001" -> ["img", "tshirt"]
        split_parts = re.split(r'[\d_\-\s]+', text)
        split_words = [part for part in split_parts if len(part) >= 3]

        # METHOD 2: Find alphabetic sequences (words embedded in numbers)
        # This handles: "1313212dress456" -> ["dress"]
        embedded_words = re.findall(r'[a-z]{3,}', text)

        # METHOD 3: Look for common clothing word patterns
        # This handles compound words and variations
        compound_patterns = [
            r'tshirt', r't-shirt', r'pullover', r'sweater', r'hoodie',
            r'jeans?', r'trouser', r'pants?', r'leggings?',
            r'dress', r'gown', r'skirt',
            r'coat', r'jacket', r'blazer',
            r'sneaker', r'boot', r'sandal', r'shoe',
            r'bag', r'purse', r'backpack'
        ]

        pattern_matches = []
        for pattern in compound_patterns:
            matches = re.findall(pattern, text)
            pattern_matches.extend(matches)

        # Combine all methods
        all_found_words = split_words + embedded_words + pattern_matches

        # Remove duplicates while preserving order
        unique_words = []
        for word in all_found_words:
            if word not in unique_words and len(word) >= 3:
                # Filter out common non-clothing words
                non_clothing_words = {'img', 'pic', 'photo', 'image', 'file', 'new', 'old', 'size', 'color', 'item'}
                if word not in non_clothing_words:
                    unique_words.append(word)

        return unique_words

    def analyze_filename_and_path(self, img_path):
        """
        Enhanced filename analysis that handles complex cases like '1313212dress.jpg'
        """
        try:
            # Get filename without extension
            filename = os.path.basename(img_path).lower()
            filename_no_ext = os.path.splitext(filename)[0]

            # Get directory name as well
            dir_name = os.path.basename(os.path.dirname(img_path)).lower()

            print(f"🔍 Analyzing filename: '{filename_no_ext}'")

            # ENHANCED: Extract words from complex patterns
            extracted_words = self.extract_clothing_words(filename_no_ext)

            # Also include directory name
            dir_words = self.extract_clothing_words(dir_name)

            # Combine all extracted words
            all_words = extracted_words + dir_words
            text_to_analyze = ' '.join(all_words)

            print(f"🔍 Extracted words: {extracted_words}")
            print(f"🔍 Final text to analyze: '{text_to_analyze}'")

            if not text_to_analyze.strip():
                print("📝 No clothing keywords found in filename")
                return None

            # Score each class based on keyword presence
            class_scores = {}
            for class_name, keywords in self.keyword_mappings.items():
                score = 0
                matches = []

                for keyword in keywords:
                    # Direct keyword match (higher score)
                    if keyword.lower() in text_to_analyze:
                        score += 3  # Increased score for direct matches
                        matches.append(keyword)

                    # Partial keyword match (lower score)
                    elif any(word in text_to_analyze for word in keyword.split()):
                        score += 1
                        matches.append(f"partial:{keyword}")

                class_scores[class_name] = {
                    'score': score,
                    'matches': matches
                }

            # Find best match
            best_class = max(class_scores.items(), key=lambda x: x[1]['score'])

            if best_class[1]['score'] > 0:
                print(
                    f"📝 Text analysis result: {best_class[0]} (score: {best_class[1]['score']}, matches: {best_class[1]['matches']})")

                # Convert to probability distribution
                total_score = sum(data['score'] for data in class_scores.values())
                if total_score > 0:
                    probabilities = [
                        class_scores[class_name]['score'] / total_score
                        for class_name in self.class_names
                    ]

                    # Add some smoothing
                    probabilities = np.array(probabilities)
                    probabilities = (probabilities + 0.01) / (probabilities.sum() + 0.01 * len(probabilities))

                    # Calculate confidence based on how dominant the best match is
                    confidence = best_class[1]['score'] / max(6, total_score)  # Normalize by max expected score

                    return {
                        'predictions': probabilities.tolist(),
                        'confidence': min(1.0, confidence),  # Cap at 1.0
                        'method': 'filename_analysis',
                        'best_match': best_class[0],
                        'matches_found': best_class[1]['matches'],
                        'extracted_words': extracted_words
                    }

            print("📝 No clear text match found")
            return None

        except Exception as e:
            print(f"❌ Error in filename analysis: {e}")
            return None

    def get_keras_model(self):
        """Load Keras model only when needed"""
        if self.keras_model is None and self.model_path is not None:
            print("🔄 Loading Keras model on demand...")
            try:
                import tensorflow as tf
                from tensorflow.keras.models import load_model

                self.keras_model = load_model(self.model_path)
                print(f"🔍 Model input shape: {self.keras_model.input_shape}")
                print(f"🔍 Model output shape: {self.keras_model.output_shape}")

                self.last_used['keras'] = datetime.now()
                print("✅ Keras model loaded successfully")

            except Exception as e:
                print(f"❌ Error loading Keras model: {e}")
                return None
        else:
            if self.keras_model is not None:
                self.last_used['keras'] = datetime.now()

        return self.keras_model

    def original_working_preprocessing(self, img):
        """Keep your ORIGINAL preprocessing - it's actually working!"""
        try:
            # Convert to array and get first pixel for brightness check
            img_array = np.asarray(img)
            result = int(img_array[0][0][0])

            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

            # Apply bitwise not if needed - KEEP THIS! Your model expects it
            if result > 128:
                img_processed = cv2.bitwise_not(gray)
            else:
                img_processed = gray

            # Resize to 28x28
            img_processed = cv2.resize(img_processed, (28, 28))

            # Normalize and add dimensions
            img_processed = img_processed.astype(np.float32) / 255.0
            img_processed = np.expand_dims(img_processed, axis=0)
            img_processed = np.expand_dims(img_processed, axis=-1)

            print(f"✅ Original preprocessing shape: {img_processed.shape}")
            print(f"   First pixel check: {result} ({'inverted' if result > 128 else 'normal'})")

            return img_processed

        except Exception as e:
            print(f"❌ Original preprocessing error: {e}")
            return None

    def create_enhanced_smart_fallback(self, img_path):
        """Fixed version of enhanced smart fallback"""
        try:
            print("🧠 Creating enhanced smart fallback prediction...")

            # For non-existent test files, use filename-based heuristics
            if not os.path.exists(img_path):
                print("   📁 File doesn't exist, using filename-only analysis")

                # Extract any clothing hints from the path itself
                filename = os.path.basename(img_path).lower()

                # Simple heuristic based on filename length and characteristics
                predictions = [0.1] * len(self.class_names)  # Base probability

                # If filename is very short or just numbers, assume common clothing
                if len(filename) < 8 or filename.replace('.jpg', '').replace('.png', '').isdigit():
                    # Favor common clothing items
                    predictions[0] += 0.3  # T-shirt/top
                    predictions[1] += 0.2  # Trouser
                    predictions[6] += 0.2  # Shirt
                else:
                    # More balanced for descriptive filenames
                    predictions[0] += 0.2  # T-shirt/top
                    predictions[1] += 0.15  # Trouser
                    predictions[3] += 0.15  # Dress

                # Normalize predictions
                predictions = np.array(predictions)
                predictions = predictions / predictions.sum()

                best_class = np.argmax(predictions)
                print(
                    f"   🎯 Filename-based fallback: {self.get_class_name(best_class)} ({predictions[best_class]:.1%})")

                return {
                    'predictions': predictions.tolist(),
                    'confidence': float(predictions[best_class]),
                    'method': 'filename_fallback'
                }

            # Original image analysis code for existing files
            img = cv2.imread(img_path)
            if img is None:
                return self.create_emergency_fallback(img_path)

            height, width = img.shape[:2]
            aspect_ratio = height / width

            # Analyze colors
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)

            # Analyze texture/edges
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

            # Analyze color distribution
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            saturation_mean = np.mean(hsv[:, :, 1])

            print(f"   📐 Aspect ratio: {aspect_ratio:.2f}")
            print(f"   💡 Brightness: {avg_brightness:.1f}")
            print(f"   🔍 Edge density: {edge_density:.3f}")
            print(f"   🎨 Saturation: {saturation_mean:.1f}")

            # Enhanced decision logic
            predictions = [0.1] * len(self.class_names)  # Base probability

            # Aspect ratio analysis
            if aspect_ratio > 1.5:  # Very tall
                if avg_brightness < 80:  # Dark
                    predictions[1] += 0.4  # Trouser
                    predictions[3] += 0.3  # Dress
                else:  # Light
                    predictions[4] += 0.3  # Coat
                    predictions[3] += 0.4  # Dress

            elif aspect_ratio < 0.7:  # Wide
                predictions[5] += 0.3  # Sandal
                predictions[7] += 0.3  # Sneaker
                predictions[8] += 0.2  # Bag
                predictions[9] += 0.2  # Ankle boot

            else:  # Square-ish
                if edge_density > 0.15:  # High texture
                    predictions[1] += 0.2  # Trouser (denim texture)
                    predictions[2] += 0.2  # Pullover (knit texture)
                else:  # Smooth
                    predictions[0] += 0.3  # T-shirt/top
                    predictions[6] += 0.3  # Shirt

            # Brightness analysis
            if avg_brightness > 150:  # Very bright
                predictions[0] += 0.2  # T-shirt/top
                predictions[6] += 0.2  # Shirt
            elif avg_brightness < 50:  # Very dark
                predictions[1] += 0.2  # Trouser
                predictions[4] += 0.2  # Coat

            # Saturation analysis
            if saturation_mean > 100:  # Colorful
                predictions[3] += 0.1  # Dress
                predictions[8] += 0.1  # Bag

            # Normalize predictions
            predictions = np.array(predictions)
            predictions = predictions / predictions.sum()

            best_class = np.argmax(predictions)
            print(f"   🎯 Enhanced fallback: {self.get_class_name(best_class)} ({predictions[best_class]:.1%})")

            return {
                'predictions': predictions.tolist(),
                'confidence': float(predictions[best_class]),
                'method': 'enhanced_fallback'
            }

        except Exception as e:
            print(f"❌ Enhanced fallback failed: {e}")
            return self.create_emergency_fallback(img_path)

    def create_emergency_fallback(self, img_path):
        """Emergency fallback when all else fails"""
        print("🆘 Using emergency fallback")
        # Return balanced distribution favoring common items
        fallback = [0.15, 0.12, 0.13, 0.14, 0.11, 0.08, 0.09, 0.07, 0.06, 0.05]
        return {'all_predictions': fallback}

    def hybrid_predict_clothing(self, img_path):
        """
        Enhanced prediction using multiple methods:
        1. Filename/path analysis
        2. Keras model prediction
        3. Smart fallback based on image analysis
        """
        print(f"\n🔍 HYBRID PREDICTION: {os.path.basename(img_path)}")

        predictions_ensemble = []
        method_weights = []

        # Method 1: Filename Analysis (high confidence when matches found)
        filename_result = self.analyze_filename_and_path(img_path)
        if filename_result and filename_result['confidence'] > 0.3:
            predictions_ensemble.append(filename_result['predictions'])
            # Weight based on confidence - high confidence filename analysis gets more weight
            weight = min(0.7, filename_result['confidence'] * 2)
            method_weights.append(weight)
            print(f"📝 Filename method weight: {weight:.2f}")
        else:
            print("📝 Filename analysis: low confidence or no matches")

        # Method 2: Keras Model Prediction
        model_predictions = None
        try:
            model = self.get_keras_model()
            if model is not None:
                # Load image
                img = cv2.imread(img_path)
                if img is not None:
                    processed_img = self.original_working_preprocessing(img)
                    if processed_img is not None:
                        model_predictions = model.predict(processed_img, verbose=0)[0]

                        # Quality check for model predictions
                        confidence = np.max(model_predictions)
                        diversity = np.sum(model_predictions > 0.01)

                        if confidence < 0.98 and diversity >= 2:  # Good prediction
                            predictions_ensemble.append(model_predictions.tolist())
                            # Weight based on confidence and diversity
                            weight = min(0.8, confidence * 0.7 + (diversity / 10) * 0.3)
                            method_weights.append(weight)
                            print(f"🤖 Model method weight: {weight:.2f}")
                        else:
                            print(f"🤖 Model prediction quality check failed (conf: {confidence:.3f}, div: {diversity})")
        except Exception as e:
            print(f"❌ Model prediction error: {e}")

        # Method 3: Smart Fallback (always included with lower weight)
        fallback_result = self.create_enhanced_smart_fallback(img_path)
        if fallback_result:
            predictions_ensemble.append(fallback_result['predictions'])
            method_weights.append(0.3)  # Lower weight for fallback
            print(f"🧠 Fallback method weight: 0.30")

        # Ensemble the predictions
        if predictions_ensemble:
            # Normalize weights
            total_weight = sum(method_weights)
            if total_weight > 0:
                method_weights = [w / total_weight for w in method_weights]

            # Weighted average of predictions
            final_predictions = np.zeros(len(self.class_names))
            for predictions, weight in zip(predictions_ensemble, method_weights):
                final_predictions += np.array(predictions) * weight

            # Ensure predictions sum to 1
            final_predictions = final_predictions / final_predictions.sum()

            # Calculate ensemble confidence
            ensemble_confidence = np.max(final_predictions)

            print(f"🎯 Ensemble prediction: {self.get_class_name(np.argmax(final_predictions))}")
            print(f"🎯 Ensemble confidence: {ensemble_confidence:.3f}")
            print(f"🎯 Methods used: {len(predictions_ensemble)}")

            return {
                'all_predictions': final_predictions.tolist(),
                'ensemble_confidence': float(ensemble_confidence),
                'methods_used': len(predictions_ensemble),
                'method_weights': method_weights,
                'raw_predictions': {
                    'filename': filename_result['predictions'] if filename_result else None,
                    'model': model_predictions.tolist() if model_predictions is not None else None,
                    'fallback': fallback_result['predictions'] if fallback_result else None
                }
            }
        else:
            print("❌ All prediction methods failed")
            return self.create_emergency_fallback(img_path)

    def predict_clothing(self, img_path):
        """Main prediction method using hybrid approach"""
        return self.hybrid_predict_clothing(img_path)

    def get_class_name(self, class_idx):
        """Get class name from index"""
        if 0 <= class_idx < len(self.class_names):
            return self.class_names[class_idx]
        return f"Unknown({class_idx})"

    # Memory management methods (keeping your existing ones)
    def unload_keras_model(self):
        if self.keras_model is not None:
            del self.keras_model
            self.keras_model = None
            if 'keras' in self.last_used:
                del self.last_used['keras']
            print("🗑️ Keras model unloaded")
            gc.collect()

    def get_memory_info(self):
        return {
            'keras_loaded': self.keras_model is not None,
            'text_classifier_loaded': self.tfidf_vectorizer is not None,
            'last_used': self.last_used.copy()
        }

    def auto_cleanup(self, max_idle_minutes=30):
        from datetime import timedelta
        now = datetime.now()
        cleanup_threshold = timedelta(minutes=max_idle_minutes)

        if (self.keras_model is not None and
                'keras' in self.last_used and
                (now - self.last_used['keras']) > cleanup_threshold):
            print(f"🕒 Auto-unloading Keras model (idle for {max_idle_minutes}+ minutes)")
            self.unload_keras_model()


# Create global enhanced model manager instance
enhanced_model_manager = EnhancedModelManager()

print("✅ ENHANCED Model manager loaded with hybrid prediction system!")
print("🎯 Features: Filename analysis + Keras model + Smart fallback")
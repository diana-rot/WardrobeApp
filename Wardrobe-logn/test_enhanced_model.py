# test_enhanced_model.py
"""
Test script for the enhanced clothing recognition model
"""

import os
import sys
import numpy as np
from datetime import datetime

# Add your project directory to path
sys.path.append('.')


def test_enhanced_model():
    """Test the enhanced model with various scenarios"""

    try:
        # Import the enhanced model
        from flaskapp.ml_models.enhanced_model_manager import EnhancedModelManager

        # Create model instance
        model = EnhancedModelManager()

        print("🧪 Testing Enhanced Clothing Recognition Model")
        print("=" * 60)

        # Test cases with different filename patterns
        test_cases = [
            # Filename that should be easily recognized
            "womens_blue_tshirt_summer_2024.jpg",
            "black_skinny_jeans_size_m.png",
            "red_cocktail_dress_evening.jpg",
            "nike_running_sneakers_white.jpg",
            "leather_ankle_boots_brown.jpg",
            "wool_pullover_sweater_gray.jpg",
            "designer_handbag_luxury.jpg",
            "winter_coat_warm_jacket.jpg",
            "casual_shirt_button_up.jpg",
            "beach_sandals_summer.jpg",
            # Ambiguous filenames
            "IMG_12345.jpg",
            "photo_2024_01_15.png",
            "unknown_item.jpg"
        ]

        for i, filename in enumerate(test_cases, 1):
            print(f"\n🔍 Test {i}: {filename}")
            print("-" * 40)

            # Create a dummy image path (for testing filename analysis)
            fake_path = f"/path/to/images/{filename}"

            try:
                # Test filename analysis only
                filename_result = model.analyze_filename_and_path(fake_path)

                if filename_result:
                    best_pred_idx = np.argmax(filename_result['predictions'])
                    best_class = model.get_class_name(best_pred_idx)
                    confidence = filename_result['confidence']

                    print(f"📝 Filename Analysis:")
                    print(f"   Predicted: {best_class}")
                    print(f"   Confidence: {confidence:.3f}")
                    print(f"   Matches: {filename_result.get('matches_found', [])}")

                    # Show top 3 predictions
                    predictions = np.array(filename_result['predictions'])
                    top_3_indices = np.argsort(predictions)[-3:][::-1]

                    print("   Top 3 predictions:")
                    for idx in top_3_indices:
                        class_name = model.get_class_name(idx)
                        prob = predictions[idx]
                        print(f"      {class_name}: {prob:.3f}")
                else:
                    print("📝 Filename Analysis: No clear matches found")

            except Exception as e:
                print(f"❌ Error testing {filename}: {e}")

        print("\n" + "=" * 60)
        print("✅ Enhanced model testing completed!")

        # Test with actual image if available
        print("\n🖼️ Testing with actual image (if available)...")

        # Look for any image in common directories
        test_image_paths = [
            "static/test_image.jpg",
            "flaskapp/static/test_image.jpg",
            "test_images/sample.jpg"
        ]

        actual_image_found = False
        for test_path in test_image_paths:
            if os.path.exists(test_path):
                print(f"Found test image: {test_path}")
                try:
                    result = model.predict_clothing(test_path)

                    predictions = np.array(result['all_predictions'])
                    best_idx = np.argmax(predictions)
                    best_class = model.get_class_name(best_idx)

                    print(f"🎯 Hybrid Prediction Result:")
                    print(f"   Best prediction: {best_class}")
                    print(f"   Confidence: {predictions[best_idx]:.3f}")
                    print(f"   Methods used: {result.get('methods_used', 1)}")
                    print(f"   Ensemble confidence: {result.get('ensemble_confidence', 0):.3f}")

                    actual_image_found = True
                    break

                except Exception as e:
                    print(f"❌ Error testing actual image: {e}")

        if not actual_image_found:
            print("ℹ️ No test images found. Create a test image to see full functionality.")

        return True

    except ImportError as e:
        print(f"❌ Could not import enhanced model: {e}")
        print("📝 Make sure the enhanced_model_manager.py file is in flaskapp/ml_models/")
        return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def benchmark_comparison():
    """Compare different prediction methods"""
    print("\n🏃 Running benchmark comparison...")

    try:
        from flaskapp.ml_models.enhanced_model_manager import EnhancedModelManager
        model = EnhancedModelManager()

        test_filenames = [
            "blue_denim_jeans_womens.jpg",
            "white_cotton_tshirt.jpg",
            "black_evening_dress.jpg",
            "IMG_random_12345.jpg"  # Ambiguous case
        ]

        for filename in test_filenames:
            print(f"\n📊 Comparing methods for: {filename}")
            fake_path = f"/test/{filename}"

            # Test filename analysis
            filename_result = model.analyze_filename_and_path(fake_path)
            if filename_result:
                fn_pred = np.argmax(filename_result['predictions'])
                fn_conf = filename_result['confidence']
                print(f"   Filename method: {model.get_class_name(fn_pred)} (conf: {fn_conf:.3f})")
            else:
                print("   Filename method: No matches")

            # Test fallback
            fallback_result = model.create_enhanced_smart_fallback(fake_path)
            if fallback_result:
                fb_pred = np.argmax(fallback_result['predictions'])
                fb_conf = fallback_result['confidence']
                print(f"   Fallback method: {model.get_class_name(fb_pred)} (conf: {fb_conf:.3f})")

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")


if __name__ == "__main__":
    print("🚀 Starting Enhanced Clothing Recognition Model Tests")

    success = test_enhanced_model()

    if success:
        benchmark_comparison()

        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("\n📋 Next steps:")
        print("1. Save the enhanced_model_manager.py in flaskapp/ml_models/")
        print("2. Update your __init__.py file")
        print("3. Test with real images from your dataset")
        print("4. Fine-tune the keyword mappings based on your specific use case")
    else:
        print("\n❌ Tests failed. Please check the error messages above.")
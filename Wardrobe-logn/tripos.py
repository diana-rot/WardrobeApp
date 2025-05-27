#!/usr/bin/env python3
"""
Working test script for TripoSG integration with local image
"""

from gradio_client import Client, handle_file
import time
import os
from PIL import Image, ImageDraw


def create_test_image():
    """Create a simple test image of a T-shirt"""
    print("Creating test T-shirt image...")

    # Create a simple T-shirt shape
    img = Image.new('RGB', (400, 400), 'white')
    draw = ImageDraw.Draw(img)

    # Draw a simple T-shirt shape in blue
    # Body
    draw.rectangle([150, 150, 250, 350], fill='blue', outline='darkblue', width=2)
    # Left sleeve
    draw.rectangle([100, 150, 150, 200], fill='blue', outline='darkblue', width=2)
    # Right sleeve
    draw.rectangle([250, 150, 300, 200], fill='blue', outline='darkblue', width=2)
    # Neckline
    draw.ellipse([180, 140, 220, 160], fill='white', outline='darkblue', width=2)

    img.save('test_tshirt.png')
    print("✅ Test image created: test_tshirt.png")

    return 'test_tshirt.png'


def test_triposg_full_process():
    """Test the complete TripoSG process"""
    print("🚀 Testing Complete TripoSG Process")
    print("=" * 50)

    try:
        # Step 1: Connect to TripoSG
        print("1. Connecting to TripoSG...")
        client = Client("https://vast-ai-triposg.hf.space")
        print("✅ Connected successfully!")

        # Step 2: Create test image
        image_path = create_test_image()

        # Step 3: Run segmentation
        print("\n2. Running image segmentation...")
        start_time = time.time()

        segmentation_result = client.predict(
            image=handle_file(image_path),
            api_name="/run_segmentation"
        )

        seg_time = time.time() - start_time
        print(f"✅ Segmentation completed in {seg_time:.1f} seconds")
        print(f"   Segmentation result type: {type(segmentation_result)}")

        # Step 4: Get random seed
        print("\n3. Generating random seed...")
        seed = client.predict(
            randomize_seed=True,
            seed=0,
            api_name="/get_random_seed"
        )
        print(f"✅ Generated seed: {seed}")

        # Step 5: Generate 3D model
        print(f"\n4. Generating 3D model (this may take 30-60 seconds)...")
        model_start_time = time.time()

        model_result = client.predict(
            image=segmentation_result,
            seed=seed,
            num_inference_steps=30,  # Reduced for faster testing
            guidance_scale=7.0,
            simplify=True,
            target_face_num=50000,  # Reduced for faster testing
            api_name="/image_to_3d"
        )

        model_time = time.time() - model_start_time
        print(f"✅ 3D model generated in {model_time:.1f} seconds!")
        print(f"   Model result type: {type(model_result)}")
        print(f"   Model result: {model_result}")

        # Step 6: Apply texture (optional)
        print(f"\n5. Applying texture...")
        texture_start_time = time.time()

        try:
            textured_result = client.predict(
                image=handle_file(image_path),
                mesh_path=model_result,
                seed=seed,
                api_name="/run_texture"
            )

            texture_time = time.time() - texture_start_time
            print(f"✅ Texture applied in {texture_time:.1f} seconds!")
            print(f"   Final result: {textured_result}")
            final_result = textured_result

        except Exception as e:
            print(f"⚠️  Texture application failed: {e}")
            print("   Using model without texture")
            final_result = model_result

        # Summary
        total_time = time.time() - start_time
        print(f"\n" + "=" * 50)
        print(f"🎉 FULL PROCESS COMPLETED!")
        print(f"   Total time: {total_time:.1f} seconds")
        print(f"   Segmentation: {seg_time:.1f}s")
        print(f"   3D Generation: {model_time:.1f}s")
        print(f"   Final result: {final_result}")

        # Check if result is a file
        if isinstance(final_result, str) and os.path.exists(final_result):
            file_size = os.path.getsize(final_result)
            print(f"   Generated file size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")

        return True

    except Exception as e:
        print(f"\n❌ Process failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean up
        if os.path.exists('test_tshirt.png'):
            os.remove('test_tshirt.png')
            print("\n🧹 Cleaned up test image")


def quick_connection_test():
    """Quick test to verify connection works"""
    print("🔄 Quick Connection Test")
    print("=" * 30)

    try:
        client = Client("https://vast-ai-triposg.hf.space")
        print("✅ Connection successful!")

        # Test segmentation with a simple image
        image_path = create_test_image()

        result = client.predict(
            image=handle_file(image_path),
            api_name="/run_segmentation"
        )

        print(f"✅ Segmentation test successful!")
        print(f"   Result type: {type(result)}")

        # Clean up
        if os.path.exists('test_tshirt.png'):
            os.remove('test_tshirt.png')

        return True

    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False


def main():
    print("🧪 TripoSG Integration Test Suite")
    print("=" * 50)

    # Ask user what they want to test
    print("\nChoose test type:")
    print("1. Quick connection test (30 seconds)")
    print("2. Full 3D generation test (2-3 minutes)")
    print("3. Skip tests")

    try:
        choice = input("\nEnter choice (1, 2, or 3): ").strip()

        if choice == '1':
            success = quick_connection_test()
        elif choice == '2':
            success = test_triposg_full_process()
        elif choice == '3':
            print("Skipping tests.")
            success = True
        else:
            print("Invalid choice. Running quick test...")
            success = quick_connection_test()

        if success:
            print(f"\n✅ Test completed successfully!")
            print(f"\nYour TripoSG integration should work. Next steps:")
            print("1. Update your Flask app with the corrected code")
            print("2. Restart your Flask application")
            print("3. Try uploading a clothing image and generating a 3D model")
        else:
            print(f"\n❌ Test failed. Check the error messages above.")

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
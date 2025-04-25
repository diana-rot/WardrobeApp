import os
import sys
from avatar_clothing_generator import AvatarClothingGenerator

def test_generator():
    """Test the AvatarClothingGenerator with a sample image"""
    print("Testing AvatarClothingGenerator...")
    
    # Create output directories
    os.makedirs('flaskapp/static/avatar_assets/models', exist_ok=True)
    os.makedirs('flaskapp/static/avatar_assets/textures', exist_ok=True)
    
    # Initialize the generator
    generator = AvatarClothingGenerator(
        wardrobe_path='flaskapp/static/image_users',
        output_path='avatar_assets'
    )
    
    # Find a sample image in the wardrobe
    sample_image = None
    for root, dirs, files in os.walk('flaskapp/static/image_users'):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                sample_image = os.path.join(root, file)
                break
        if sample_image:
            break
    
    if not sample_image:
        print("No sample images found in the wardrobe.")
        return False
    
    print(f"Using sample image: {sample_image}")
    
    # Process the image
    texture = generator.process_wardrobe_image(sample_image)
    if texture is None:
        print("Failed to process image.")
        return False
    
    # Save the processed texture
    texture_path = 'flaskapp/static/avatar_assets/textures/test_texture.png'
    import cv2
    cv2.imwrite(texture_path, cv2.cvtColor(texture, cv2.COLOR_RGB2BGR))
    print(f"Saved texture to: {texture_path}")
    
    # Create the basic top mesh
    mesh = generator.create_basic_top_mesh()
    print("Created basic top mesh.")
    
    # Create the GLTF file
    output_path = 'flaskapp/static/avatar_assets/models/test_top.glb'
    generator.create_gltf(mesh, texture_path, output_path)
    print(f"Saved GLB file to: {output_path}")
    
    print("Test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_generator()
    sys.exit(0 if success else 1) 
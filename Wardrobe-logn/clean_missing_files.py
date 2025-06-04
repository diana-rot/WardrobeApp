#!/usr/bin/env python3
"""
Quick cleanup script to remove broken file references from database
"""

import pymongo
from bson import ObjectId
import os


def cleanup_broken_references():
    """Remove file_path references for files that don't exist"""

    try:
        # Connect to MongoDB
        client = pymongo.MongoClient('localhost', 27017)
        db = client.user_login_system_test

        # The specific missing files from your diagnostic
        missing_files = [
            "/static/image_users/a26981595e554e2baaddfb8ee0113127/1a031093-9b54-432a-973a-6df4346f1816.jpeg",
            "/static/image_users/a26981595e554e2baaddfb8ee0113127/download_-_2025-02-07T231846.701.jpeg",
            "/static/image_users/a26981595e554e2baaddfb8ee0113127/Luxe_Longline_Tee.jpeg",
            "/static/image_users/a26981595e554e2baaddfb8ee0113127/download_-_2025-02-07T231603.227.jpeg",
            "/static/image_users/a26981595e554e2baaddfb8ee0113127/21fcd3f8-536f-4e8b-86e7-e65ee60550c9.jpeg"
        ]

        print(f"🧹 Cleaning up {len(missing_files)} broken file references...")

        fixed_count = 0
        for file_path in missing_files:
            # Find the item with this file path
            item = db.wardrobe.find_one({'file_path': file_path})

            if item:
                label = item.get('label', 'Unknown')
                print(f"   Removing broken reference from: {label}")

                # Remove the file_path field so it shows "No image" instead of broken link
                result = db.wardrobe.update_one(
                    {'_id': item['_id']},
                    {'$unset': {'file_path': 1}}
                )

                if result.modified_count > 0:
                    fixed_count += 1
                    print(f"   ✅ Fixed: {label}")
                else:
                    print(f"   ❌ Failed to fix: {label}")
            else:
                print(f"   ⚠️ Item not found for path: {file_path}")

        print(f"\n🎉 Successfully cleaned up {fixed_count} broken references!")
        print(f"💡 These items will now show 'No image' instead of broken links")

        return fixed_count

    except Exception as e:
        print(f"❌ Error during cleanup: {str(e)}")
        return 0


def create_placeholder_image():
    """Create the missing no-image.png placeholder"""

    try:
        from PIL import Image, ImageDraw, ImageFont

        # Create directories if they don't exist
        os.makedirs('flaskapp/static/image', exist_ok=True)

        # Create a simple placeholder image
        img = Image.new('RGB', (300, 300), color='#f7f3f0')  # Match your background color
        draw = ImageDraw.Draw(img)

        # Try to use a font
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None

        # Draw placeholder text
        text = "No Image\nAvailable"
        if font:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (300 - text_width) // 2
            y = (300 - text_height) // 2
            draw.text((x, y), text, fill='#bbb', font=font, align='center')
        else:
            draw.text((120, 140), text, fill='#bbb', align='center')

        # Add a subtle border
        draw.rectangle([20, 20, 280, 280], outline='#ddd', width=2)

        # Save the placeholder
        placeholder_path = 'flaskapp/static/image/no-image.png'
        img.save(placeholder_path)
        print(f"✅ Created placeholder image: {placeholder_path}")

        return True

    except ImportError:
        print("⚠️ PIL not available, creating a simple placeholder...")
        # Create a simple HTML file as fallback
        placeholder_path = 'flaskapp/static/image/no-image.png'
        # Copy an existing image as placeholder
        try:
            import shutil
            # Use one of your existing images as a placeholder
            source = 'flaskapp/static/image_users/a26981595e554e2baaddfb8ee0113127/3.jpeg'
            if os.path.exists(source):
                shutil.copy(source, placeholder_path)
                print(f"✅ Created placeholder by copying existing image")
                return True
        except:
            pass

        print("❌ Could not create placeholder image")
        return False

    except Exception as e:
        print(f"❌ Error creating placeholder: {str(e)}")
        return False


def main():
    """Main cleanup function"""

    print("🚀 Starting quick cleanup...\n")

    # Step 1: Create placeholder image
    print("Step 1: Creating placeholder image")
    create_placeholder_image()

    # Step 2: Clean up broken references
    print("\nStep 2: Cleaning up broken file references")
    fixed_count = cleanup_broken_references()

    print(f"\n🎉 Cleanup completed!")
    print(f"   - Created placeholder image for 404 errors")
    print(f"   - Cleaned up {fixed_count} broken file references")
    print(f"\n💡 Now restart your Flask app and the 404 errors should be gone!")
    print(f"💡 Items with missing files will show 'No image' instead of broken links")


if __name__ == "__main__":
    main()
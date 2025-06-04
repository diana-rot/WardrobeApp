#!/usr/bin/env python3
"""
Diagnostic script to check path issues with wardrobe images
Run this from your Flask app directory
"""

import os
import pymongo
from bson import ObjectId


def normalize_path(file_path):
    """Your current normalize_path function"""
    if not file_path:
        return None

    # First remove /outfit/ prefix anywhere in the path
    normalized = file_path.replace('/outfit/', '/')

    # Handle multiple static occurrences
    while '/static/static/' in normalized:
        normalized = normalized.replace('/static/static/', '/static/')

    # Handle any remaining path cleanup
    normalized = normalized.strip('/')

    # Special case: ensure static paths start with static/
    if 'static' in normalized and not normalized.startswith('static'):
        normalized = 'static/' + normalized[normalized.index('static') + 6:]

    # Always return with leading slash
    if not normalized.startswith('/'):
        normalized = '/' + normalized

    return normalized


def check_file_paths():
    """Check actual file paths vs database paths"""

    try:
        # Connect to MongoDB
        client = pymongo.MongoClient('localhost', 27017)
        db = client.user_login_system_test

        print("🔍 Checking file paths for user...")

        # Get items for the specific user from your logs
        user_id = "a26981595e554e2baaddfb8ee0113127"
        items = list(db.wardrobe.find({'userId': user_id}))

        print(f"📊 Found {len(items)} items for user {user_id}")

        # Check the actual directory
        user_dir = f"flaskapp/static/image_users/{user_id}"
        if os.path.exists(user_dir):
            actual_files = os.listdir(user_dir)
            print(f"📁 Actual files in directory ({len(actual_files)}):")
            for f in sorted(actual_files)[:10]:  # Show first 10
                print(f"   ✅ {f}")
            if len(actual_files) > 10:
                print(f"   ... and {len(actual_files) - 10} more")
        else:
            print(f"❌ Directory doesn't exist: {user_dir}")
            return

        print(f"\n🔍 Checking database paths vs actual files:")

        missing_in_db = []
        missing_files = []
        path_mismatches = []

        # Check each database item
        for item in items:
            file_path = item.get('file_path', '')
            item_label = item.get('label', 'Unknown')

            if not file_path:
                missing_in_db.append(f"{item_label} (ID: {str(item['_id'])[:8]})")
                continue

            # Original path
            print(f"\n📝 Item: {item_label}")
            print(f"   DB path: {file_path}")

            # Normalized path
            normalized = normalize_path(file_path)
            print(f"   Normalized: {normalized}")

            # Convert to actual file system path
            if normalized and normalized.startswith('/static/'):
                actual_path = os.path.join('flaskapp', normalized.lstrip('/'))
            elif normalized and normalized.startswith('static/'):
                actual_path = os.path.join('flaskapp', normalized)
            else:
                actual_path = os.path.join('flaskapp', 'static', normalized.lstrip('/') if normalized else '')

            print(f"   Actual path: {actual_path}")
            print(f"   Exists: {os.path.exists(actual_path)}")

            if not os.path.exists(actual_path):
                # Try to find the file with a different approach
                filename = os.path.basename(file_path) if file_path else ''
                if filename and filename in actual_files:
                    correct_path = f"/static/image_users/{user_id}/{filename}"
                    print(f"   🔧 FOUND WITH CORRECT PATH: {correct_path}")
                    path_mismatches.append({
                        'item_id': item['_id'],
                        'label': item_label,
                        'wrong_path': file_path,
                        'correct_path': correct_path
                    })
                else:
                    missing_files.append(f"{item_label}: {file_path}")

        # Check for files not in database
        db_filenames = []
        for item in items:
            if item.get('file_path'):
                filename = os.path.basename(item['file_path'])
                if filename:
                    db_filenames.append(filename)

        orphaned_files = [f for f in actual_files if f not in db_filenames]

        print(f"\n📈 SUMMARY:")
        print(f"   Items missing file_path in DB: {len(missing_in_db)}")
        print(f"   Items with wrong paths: {len(path_mismatches)}")
        print(f"   Items with missing files: {len(missing_files)}")
        print(f"   Orphaned files (not in DB): {len(orphaned_files)}")

        # Show specific issues
        if missing_in_db:
            print(f"\n❌ Items missing file_path:")
            for item in missing_in_db[:5]:
                print(f"   - {item}")

        if path_mismatches:
            print(f"\n🔧 Items with fixable path issues:")
            for mismatch in path_mismatches[:5]:
                print(f"   - {mismatch['label']}")
                print(f"     Wrong: {mismatch['wrong_path']}")
                print(f"     Correct: {mismatch['correct_path']}")

        if missing_files:
            print(f"\n❌ Items with truly missing files:")
            for missing in missing_files[:5]:
                print(f"   - {missing}")

        if orphaned_files:
            print(f"\n📄 Orphaned files (exist but not in DB):")
            for orphan in orphaned_files[:5]:
                print(f"   - {orphan}")

        return path_mismatches

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return []


def fix_path_mismatches(mismatches):
    """Fix the path mismatches found"""

    if not mismatches:
        print("No path mismatches to fix!")
        return

    try:
        client = pymongo.MongoClient('localhost', 27017)
        db = client.user_login_system_test

        print(f"\n🔧 Fixing {len(mismatches)} path mismatches...")

        fixed_count = 0
        for mismatch in mismatches:
            result = db.wardrobe.update_one(
                {'_id': mismatch['item_id']},
                {'$set': {'file_path': mismatch['correct_path']}}
            )

            if result.modified_count > 0:
                print(f"   ✅ Fixed: {mismatch['label']}")
                fixed_count += 1
            else:
                print(f"   ❌ Failed to fix: {mismatch['label']}")

        print(f"\n🎉 Fixed {fixed_count} out of {len(mismatches)} items!")

    except Exception as e:
        print(f"❌ Error fixing paths: {str(e)}")


def main():
    """Main diagnostic function"""

    print("🔍 Starting path diagnostic...\n")

    # Run the check
    mismatches = check_file_paths()

    # Ask if user wants to fix the mismatches
    if mismatches:
        print(f"\n❓ Found {len(mismatches)} fixable path issues.")
        response = input("Do you want to fix them? (y/n): ")

        if response.lower() in ['y', 'yes']:
            fix_path_mismatches(mismatches)
            print(f"\n💡 Please restart your Flask app to see the changes!")
        else:
            print(f"Skipping fixes. You can run this script again later.")
    else:
        print(f"\n✅ No fixable path issues found!")


if __name__ == "__main__":
    main()
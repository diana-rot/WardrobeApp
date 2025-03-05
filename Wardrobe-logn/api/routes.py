from flask import jsonify
from . import api_bp
from .auth import token_required
from .handlers import (
    handle_login, handle_get_wardrobe,
    handle_add_wardrobe_item, handle_delete_wardrobe_item
    # Remove handle_register from this import
)

# Auth routes
@api_bp.route('/login', methods=['POST'])
def login():
    return handle_login()

# Wardrobe routes
@api_bp.route('/wardrobe/all', methods=['GET'])
@token_required
def get_wardrobe(current_user):
    return handle_get_wardrobe(current_user)

@api_bp.route('/wardrobe/add', methods=['POST'])
@token_required
def add_wardrobe_item(current_user):
    return handle_add_wardrobe_item(current_user)

@api_bp.route('/wardrobe/delete/<item_id>', methods=['DELETE'])
@token_required
def delete_wardrobe_item(current_user, item_id):
    return handle_delete_wardrobe_item(current_user, item_id)

# Remove the register route for now or add the function
from .handlers import (
    handle_login, handle_register, handle_get_wardrobe,
    handle_add_wardrobe_item, handle_delete_wardrobe_item
)

# And add the route
@api_bp.route('/register', methods=['POST'])
def register():
    return handle_register()
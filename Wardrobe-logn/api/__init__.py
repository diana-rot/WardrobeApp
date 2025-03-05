# api/__init__.py
from flask import Blueprint

api_bp = Blueprint('api', __name__, url_prefix='/api')

from . import routes




# Add similar routes for outfits, calendar, etc.
# For example:

# @api_bp.route('/outfit/all', methods=['GET'])
# @token_required
# def get_outfits(current_user):
#     # Implementation here
#     pass


from flask import Blueprint, request, jsonify, session
from ..utils import community_inference as ci

community_bp = Blueprint('community', __name__)

@community_bp.route('/api/infer_community', methods=['POST'])
def infer_community():
    """Proxy endpoint to use community inference services.
    Request JSON fields:
      - provider: 'space' or 'hf'
      - id: space_id (e.g. 'username/space') or model_id
      - inputs: payload (string or dict)
      - api_key: optional (user-provided, not stored)

    Returns JSON response from provider or error.
    """
    data = request.get_json() or {}
    provider = data.get('provider')
    identifier = data.get('id')
    inputs = data.get('inputs')
    api_key = data.get('api_key')

    if not provider or not identifier or inputs is None:
        return jsonify({'error': 'provider, id and inputs are required'}), 400

    try:
        result = ci.infer(provider=provider, identifier=identifier, inputs=inputs, api_key=api_key)
        return jsonify({'ok': True, 'result': result})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

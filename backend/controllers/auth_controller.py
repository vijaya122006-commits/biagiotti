# controllers/auth_controller.py
from utils.response_builder import build_success_response, build_error_response

def authenticate_user(data):
    email = data.get('email')
    password = data.get('password')
    
    # Simple explicit dummy auth
    if email == 'dealer@example.com' and password == '123456':
        return build_success_response(data={"token": "dummy_jwt_token_123"}, message="Login successful")
    
    return build_error_response("Invalid credentials", status_code=401)

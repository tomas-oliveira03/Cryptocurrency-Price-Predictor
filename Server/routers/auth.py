from flask import json, jsonify, request
from db.mongoConnection import getMongoConnectionForUser

def authentication(app, prefix):
    
    userDB = getMongoConnectionForUser()
    
    @app.route(f"{prefix}/login", methods=["POST"])
    def login():
        try:
            # Parse request data
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400
                
            # Validate required fields
            if "email" not in data or "password" not in data:
                return jsonify({"error": "Email and password are required"}), 400
                
            email = data["email"]
            password = data["password"]
            
            # Find user in database
            user = userDB.find_one({"email": email})
            
            # Check if user exists and password matches
            if not user or user["password"] != password:
                return jsonify({"error": "Invalid email or password"}), 401
              
            return jsonify({
                "message": "Login successful",
                "user": {
                    "id": str(user["_id"]),
                    "name": user["name"]
                }
            }), 200
            
        except Exception as e:
            return jsonify({"error": f"Login failed: {str(e)}"}), 500


    @app.route(f"{prefix}/register", methods=["POST"])
    def register():
        try:
            # Parse request data
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400
                
            # Validate required fields
            if "email" not in data or "password" not in data or "name" not in data:
                return jsonify({"error": "Email, password and name are required"}), 400
                
            email = data["email"]
            password = data["password"]
            name = data["name"]
            
            # Check if user already exists
            existing_user = userDB.find_one({"email": email})
            if existing_user:
                return jsonify({"error": "User with this email already exists"}), 409
            
            # Create new user
            user_data = {
                "email": email,
                "password": password,
                "name": name
            }
            
            # Insert into database
            result = userDB.insert_one(user_data)
            
            return jsonify({
                "message": "Registration successful",
                "user": {
                    "id": str(result.inserted_id),
                    "name": name
                }
            }), 200
            
        except Exception as e:
            return jsonify({"error": f"Registration failed: {str(e)}"}), 500
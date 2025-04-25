from flask import json, jsonify, request
from bson import ObjectId

def notification(app, prefix, userDB, notificatioDB):
    
    @app.route(f"{prefix}/<cryptoCurrency>")
    def getAllUserNotificationsOfACryptoCurrency(cryptoCurrency):
        try:
            userId = request.args.get("userId")
            if not userId:
                return jsonify({"error": "Missing userId in query parameters"}), 400

            # Validate and convert userId to ObjectId
            try:
                userObjectId = ObjectId(userId)
            except Exception:
                return jsonify({"error": "Invalid userId format"}), 400

            # Find user in database
            user = userDB.find_one({"_id": userObjectId})
            if not user:
                return jsonify({"error": "User not found"}), 404


            notifications = notificatioDB.find({"userId": userObjectId, "cryptoCurrency": cryptoCurrency})
            
            notificationsParsed = [
                {
                    "id": str(notification["_id"]),
                    "price": notification["price"],
                    "isActive": notification["isActive"],
                    "alertCondition": notification["alertCondition"],
                    "monitoredPriceType": notification["monitoredPriceType"],
                }
                for notification in notifications
            ]
            
            return jsonify(notificationsParsed), 200

        except Exception as e:
            return jsonify({"error": f"Failed to retrieve notifications: {str(e)}"}), 500


    @app.route(f"{prefix}")
    def getNotification():
        try:
            notificationId = request.args.get("notificationId")
            if not notificationId:
                return jsonify({"error": "Missing notificationId in query parameters"}), 400

            # Validate and convert notificationId to ObjectId
            try:
                notificationObjectId = ObjectId(notificationId)
            except Exception:
                return jsonify({"error": "Invalid notificationId format"}), 400

            # Find notification in database
            notification = notificatioDB.find_one({"_id": notificationObjectId})
            if not notification:
                return jsonify({"error": "Notification not found"}), 404

            return jsonify({
                "_id": str(notification["_id"]),
                "price": notification["price"],
                "isActive": notification["isActive"],
                "alertCondition": notification["alertCondition"],
                "monitoredPriceType": notification["monitoredPriceType"],
            }), 200

        except Exception as e:
            return jsonify({"error": f"Failed to retrieve notification: {str(e)}"}), 500
    
    
    @app.route(f"{prefix}", methods=["DELETE"])
    def deleteNotification():
        try:
            notificationId = request.args.get("notificationId")
            if not notificationId:
                return jsonify({"error": "Missing notificationId in query parameters"}), 400

            # Validate and convert userId to ObjectId
            try:
                notificationObjectId = ObjectId(notificationId)
            except Exception:
                return jsonify({"error": "Invalid notificationId format"}), 400

            # Find notification in database
            notification = notificatioDB.find_one({"_id": notificationObjectId})
            if not notification:
                return jsonify({"error": "Notification not found"}), 404

            # Delete the notification
            notificatioDB.delete_one({"_id": notificationObjectId})
            
            return jsonify({
                "message": "Notification deleted successfully",
                }), 200

        except Exception as e:
            return jsonify({"error": f"Failed to delete notification: {str(e)}"}), 500
        
        
    @app.route(f"{prefix}/edit", methods=["POST"])
    def editNotification():
        try:
            data = request.get_json()
            notificationId = request.args.get("notificationId")
            if not notificationId:
                return jsonify({"error": "Missing notificationId in query parameters"}), 400

            # Validate and convert ID
            try:
                notificationObjectId = ObjectId(notificationId)
            except Exception:
                return jsonify({"error": "Invalid notificationId format"}), 400

            # Build update fields
            update_fields = {}
            if "price" in data:
                update_fields["price"] = data["price"]
            if "isActive" in data:
                update_fields["isActive"] = data["isActive"]
            if "alertCondition" in data:
                update_fields["alertCondition"] = data["alertCondition"]
            if "monitoredPriceType" in data:
                update_fields["monitoredPriceType"] = data["monitoredPriceType"]

            if not update_fields:
                return jsonify({"error": "No fields provided to update"}), 400

            # Update document
            result = notificatioDB.update_one(
                {"_id": notificationObjectId},
                {"$set": update_fields}
            )

            if result.matched_count == 0:
                return jsonify({"error": "Notification not found"}), 404

            return jsonify({
                "message": "Notification updated successfully",
            }), 200

        except Exception as e:
            return jsonify({"error": f"Failed to update notification: {str(e)}"}), 500


    @app.route(f"{prefix}/add", methods=["POST"])
    def addNotification():
        try:
            userId = request.args.get("userId")
            if not userId:
                return jsonify({"error": "Missing userId in query parameters"}), 400

            # Convert and validate ObjectId
            try:
                userObjectId = ObjectId(userId)
            except Exception:
                return jsonify({"error": "Invalid userId format"}), 400

            # Check if user exists
            user = userDB.find_one({"_id": userObjectId})
            if not user:
                return jsonify({"error": "User not found"}), 404

            # Parse payload
            data = request.get_json()
            coin = data.get("coin")
            price = data.get("price")
            isActive = data.get("isActive")
            alertCondition = data.get("alertCondition")
            monitoredPriceType = data.get("monitoredPriceType")

            if coin is None or price is None or isActive is None or alertCondition is None or monitoredPriceType is None:
                return jsonify({"error": "Missing one or more fields: coin, price, isActive, alertCondition, monitoredPriceType"}), 400

            # Create and insert notification
            new_notification = {
                "userId": userObjectId,
                "cryptoCurrency": coin,
                "price": price,
                "isActive": isActive,
                "alertCondition": alertCondition,
                "monitoredPriceType": monitoredPriceType
            }

            result = notificatioDB.insert_one(new_notification)

            return jsonify({"notificationId": str(result.inserted_id)}), 200

        except Exception as e:
            return jsonify({"error": f"Failed to add notification: {str(e)}"}), 500

from flask_socketio import SocketIO

def createWebSocket(app):
    socketio = SocketIO(cors_allowed_origins=["http://localhost:5173", "http://127.0.0.1:5173"])
    socketio.init_app(app)
    
    # Register Socket.IO event handlers
    @socketio.on('connect')
    def handle_connect():
        print('Client connected to WebSocket')
        
    @socketio.on('disconnect')
    def handle_disconnect():
        print('Client disconnected from WebSocket')
        
    @socketio.on('message')
    def handle_message(data):
        try:
            coin = data.get('coin')
            price = data.get('price')

            if coin is None or price is None:
                raise ValueError("Missing 'coin' or 'price' in the message")

            print(f"Received coin: {coin}, price: {price}")

        except Exception as e:
            print(f"Error handling message: {e}")
            socketio.emit('response', {
                'status': 'error',
                'message': str(e)
            })
            
    return socketio
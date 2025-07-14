import { io, Socket } from 'socket.io-client';
import { CoinPriceUpdate } from '../types';

type PriceUpdateCallback = (coin: string, price: number, previousPrice: number | null) => void;

class WebSocketService {
  private socket: Socket | null = null;
  private callbacks: PriceUpdateCallback[] = [];
  private lastPrices: Record<string, number> = {}; // Track last price by coin

  // Initialize connection to WebSocket server
  connect(): void {
    if (this.socket) return; // Already connected
    
    console.log('Connecting to WebSocket server...');
    
    this.socket = io('http://localhost:3001', {
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });
    
    this.socket.on('connect', () => {
      console.log('Connected to WebSocket server');
    });
    
    this.socket.on('disconnect', () => {
      console.log('Disconnected from WebSocket server');
    });
    
    this.socket.on('message', (data: CoinPriceUpdate[] | CoinPriceUpdate) => {
      // Handle both array and single object formats for backward compatibility
      const updates = Array.isArray(data) ? data : [data];
      
      // Process each update in the array
      updates.forEach(update => {
        const { coin, price } = update;
        
        // Only notify callbacks if the price has actually changed
        const previousPrice = this.lastPrices[coin] || null;
        console.log(`Received new price update: ${coin} - $${price} (was: ${previousPrice ? '$' + previousPrice : 'unknown'})`);
        
        this.lastPrices[coin] = price; // Update stored price
        this.notifyCallbacks(coin, price, previousPrice);
      });
    });
    
    this.socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
    });
  }
  
  // Register a callback for price updates
  onPriceUpdate(callback: PriceUpdateCallback): void {
    this.callbacks.push(callback);
  }
  
  // Remove a callback
  removeCallback(callback: PriceUpdateCallback): void {
    this.callbacks = this.callbacks.filter(cb => cb !== callback);
  }
  
  // Notify all registered callbacks
  private notifyCallbacks(coin: string, price: number, previousPrice: number | null): void {
    this.callbacks.forEach(callback => callback(coin, price, previousPrice));
  }
  
  // Disconnect from server
  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
      this.lastPrices = {}; // Reset price tracking when disconnecting
    }
  }
}

// Create a singleton instance
export const websocketService = new WebSocketService();
export default websocketService;

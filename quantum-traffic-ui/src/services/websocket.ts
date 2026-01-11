// ================================
// WebSocket Service for Real-time Updates
// ================================

import { API_CONFIG, type WSReoptimizeMessage } from '../types';

type MessageHandler = (message: WSReoptimizeMessage) => void;
type StatusHandler = (connected: boolean) => void;

class WebSocketService {
    private ws: WebSocket | null = null;
    private messageHandlers: Set<MessageHandler> = new Set();
    private statusHandlers: Set<StatusHandler> = new Set();
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 5;
    private reconnectDelay = 3000;

    connect(): void {
        if (this.ws?.readyState === WebSocket.OPEN) {
            return;
        }

        const token = localStorage.getItem('access_token');
        const wsUrl = `${API_CONFIG.wsUrl}/reoptimize${token ? `?token=${token}` : ''}`;

        try {
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.reconnectAttempts = 0;
                this.notifyStatus(true);
            };

            this.ws.onmessage = (event) => {
                try {
                    const message: WSReoptimizeMessage = JSON.parse(event.data);
                    this.messageHandlers.forEach(handler => handler(message));
                } catch (error) {
                    console.error('Failed to parse WebSocket message:', error);
                }
            };

            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.notifyStatus(false);
                this.attemptReconnect();
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        } catch (error) {
            console.error('Failed to create WebSocket:', error);
        }
    }

    disconnect(): void {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    private attemptReconnect(): void {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
            setTimeout(() => this.connect(), this.reconnectDelay);
        }
    }

    private notifyStatus(connected: boolean): void {
        this.statusHandlers.forEach(handler => handler(connected));
    }

    // Subscribe to route updates
    subscribeToRoute(routeId: string): void {
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'subscribe',
                route_id: routeId,
            }));
        }
    }

    // Request reoptimization with new traffic conditions
    requestReoptimize(routeId: string, trafficMultiplier: number): void {
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'reoptimize',
                route_id: routeId,
                traffic_multiplier: trafficMultiplier,
            }));
        }
    }

    // Add message handler
    onMessage(handler: MessageHandler): () => void {
        this.messageHandlers.add(handler);
        return () => this.messageHandlers.delete(handler);
    }

    // Add connection status handler
    onStatusChange(handler: StatusHandler): () => void {
        this.statusHandlers.add(handler);
        return () => this.statusHandlers.delete(handler);
    }

    // Check if connected
    isConnected(): boolean {
        return this.ws?.readyState === WebSocket.OPEN;
    }
}

// Export singleton instance
export const wsService = new WebSocketService();
export default wsService;

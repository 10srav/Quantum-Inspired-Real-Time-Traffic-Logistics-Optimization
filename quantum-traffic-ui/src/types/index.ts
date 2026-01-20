// ================================
// TypeScript Interfaces for Quantum Traffic Optimizer
// ================================

// Delivery Point
export interface DeliveryPoint {
    lat: number;
    lng: number;
    priority: number;
    name?: string;
}

// Sequence Stop (from optimization result)
export interface SequenceStop {
    position: number;
    delivery: DeliveryPoint;
    distance_from_prev: number;
    eta_from_prev: number;
}

// Optimization Result
export interface OptimizeResult {
    route_id: string;
    sequence: SequenceStop[];
    total_distance: number;
    total_eta: number;
    improvement_over_greedy: number;
    optimization_time: number;
    traffic_level: string;
    route_geometry?: number[][][]; // Array of path segments, each segment is array of [lat, lng] coords
}

// Optimization Request
export interface OptimizeRequest {
    current_loc: [number, number];
    deliveries: DeliveryPoint[];
    traffic_level: 'low' | 'medium' | 'high';
    include_map?: boolean;
}

// Auth Types
export interface User {
    id: string;
    username: string;
    email?: string;
    scopes: string[];
}

export interface AuthTokens {
    access_token: string;
    token_type: string;
    expires_in: number;
}

export interface LoginCredentials {
    username: string;
    password: string;
}

// API Types
export interface APIInfo {
    name: string;
    version: string;
    environment: string;
    features: {
        websocket: boolean;
        map_generation: boolean;
        qaoa: boolean;
    };
    limits: {
        max_deliveries: number;
        rate_limit_per_minute: number;
    };
}

export interface HealthResponse {
    status: 'healthy' | 'unhealthy';
    version: string;
    environment: string;
    checks: Record<string, boolean>;
}

// Route History Entry
export interface RouteHistoryEntry {
    route_id: string;
    n_stops: number;
    total_eta: number;
    timestamp: string;
}

// WebSocket Message Types
export interface WSReoptimizeMessage {
    type: 'traffic_update' | 'route_update' | 'error';
    data: {
        route_id?: string;
        traffic_multiplier?: number;
        new_sequence?: SequenceStop[];
        error?: string;
    };
}

// Vijayawada Configuration
export const VIJAYAWADA_CONFIG = {
    center: [16.525, 80.65] as [number, number],
    bbox: {
        lat_min: 16.50,
        lat_max: 16.55,
        lng_min: 80.62,
        lng_max: 80.68,
    },
    zoom: 13,
} as const;

// API Configuration
export const API_CONFIG = {
    baseUrl: import.meta.env.VITE_API_URL || 'http://localhost:8000',
    wsUrl: import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws',
} as const;

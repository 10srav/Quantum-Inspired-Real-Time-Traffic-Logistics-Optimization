// ================================
// API Service for FastAPI Integration
// ================================

import axios, { type AxiosInstance, type AxiosError } from 'axios';
import {
    API_CONFIG,
    type OptimizeRequest,
    type OptimizeResult,
    type LoginCredentials,
    type AuthTokens,
    type APIInfo,
    type HealthResponse
} from '../types';

// Create axios instance with base configuration
const api: AxiosInstance = axios.create({
    baseURL: API_CONFIG.baseUrl,
    timeout: 30000,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Request interceptor - add auth token
api.interceptors.request.use(
    (config) => {
        const token = localStorage.getItem('access_token');
        if (token) {
            config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
    },
    (error) => Promise.reject(error)
);

// Response interceptor - handle errors
api.interceptors.response.use(
    (response) => response,
    (error: AxiosError) => {
        if (error.response?.status === 401) {
            // Token expired or invalid
            localStorage.removeItem('access_token');
            window.location.href = '/login';
        }
        return Promise.reject(error);
    }
);

// ================================
// Auth API
// ================================

export const authAPI = {
    login: async (credentials: LoginCredentials): Promise<AuthTokens> => {
        // FastAPI OAuth2 expects form data
        const formData = new URLSearchParams();
        formData.append('username', credentials.username);
        formData.append('password', credentials.password);

        const response = await api.post<AuthTokens>('/token', formData, {
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        });

        // Store token
        localStorage.setItem('access_token', response.data.access_token);
        return response.data;
    },

    logout: () => {
        localStorage.removeItem('access_token');
    },

    isAuthenticated: (): boolean => {
        return !!localStorage.getItem('access_token');
    },
};

// ================================
// Optimization API
// ================================

export const optimizationAPI = {
    optimize: async (request: OptimizeRequest): Promise<OptimizeResult> => {
        const response = await api.post<OptimizeResult>('/optimize', request);
        return response.data;
    },

    getRouteMap: async (routeId: string): Promise<string> => {
        const response = await api.get<string>(`/routes/${routeId}/map`);
        return response.data;
    },

    listRoutes: async (params?: {
        skip?: number;
        limit?: number;
        traffic_level?: string;
    }) => {
        const response = await api.get('/api/v1/routes', { params });
        return response.data;
    },

    deleteRoute: async (routeId: string) => {
        const response = await api.delete(`/routes/${routeId}`);
        return response.data;
    },
};

// ================================
// Health & Info API
// ================================

export const systemAPI = {
    getHealth: async (): Promise<HealthResponse> => {
        const response = await api.get<HealthResponse>('/health');
        return response.data;
    },

    getInfo: async (): Promise<APIInfo> => {
        const response = await api.get<APIInfo>('/api/v1/info');
        return response.data;
    },

    checkConnection: async (): Promise<boolean> => {
        try {
            await api.get('/health/live');
            return true;
        } catch {
            return false;
        }
    },
};

export default api;

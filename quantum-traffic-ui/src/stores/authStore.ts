// ================================
// Auth Store using Zustand
// ================================

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { jwtDecode } from 'jwt-decode';
import { type User, type LoginCredentials, type AuthTokens } from '../types';
import { authAPI } from '../services/api';

interface AuthState {
    user: User | null;
    token: string | null;
    isAuthenticated: boolean;
    isLoading: boolean;
    error: string | null;

    // Actions
    login: (credentials: LoginCredentials) => Promise<boolean>;
    logout: () => void;
    checkAuth: () => void;
    clearError: () => void;
}

// Decode JWT and extract user info
const decodeToken = (token: string): User | null => {
    try {
        const decoded = jwtDecode<{ sub: string; scopes?: string[] }>(token);
        return {
            id: decoded.sub,
            username: decoded.sub,
            scopes: decoded.scopes || [],
        };
    } catch {
        return null;
    }
};

export const useAuthStore = create<AuthState>()(
    persist(
        (set, _get) => ({
            user: null,
            token: null,
            isAuthenticated: false,
            isLoading: false,
            error: null,

            login: async (credentials: LoginCredentials) => {
                set({ isLoading: true, error: null });

                try {
                    const tokens: AuthTokens = await authAPI.login(credentials);
                    const user = decodeToken(tokens.access_token);

                    set({
                        token: tokens.access_token,
                        user,
                        isAuthenticated: true,
                        isLoading: false,
                    });

                    return true;
                } catch (error: unknown) {
                    const message = error instanceof Error ? error.message : 'Login failed';
                    set({
                        error: message,
                        isLoading: false,
                        isAuthenticated: false,
                    });
                    return false;
                }
            },

            logout: () => {
                authAPI.logout();
                set({
                    user: null,
                    token: null,
                    isAuthenticated: false,
                    error: null,
                });
            },

            checkAuth: () => {
                const token = localStorage.getItem('access_token');
                if (token) {
                    const user = decodeToken(token);
                    if (user) {
                        set({ token, user, isAuthenticated: true });
                    } else {
                        // Token is invalid
                        localStorage.removeItem('access_token');
                        set({ token: null, user: null, isAuthenticated: false });
                    }
                }
            },

            clearError: () => set({ error: null }),
        }),
        {
            name: 'auth-storage',
            partialize: (state) => ({ token: state.token }),
        }
    )
);

export default useAuthStore;

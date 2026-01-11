// ================================
// Route Store using Zustand
// ================================

import { create } from 'zustand';
import {
    type DeliveryPoint,
    type OptimizeResult,
    type RouteHistoryEntry,
    VIJAYAWADA_CONFIG
} from '../types';
import { optimizationAPI } from '../services/api';

interface RouteState {
    // Current state
    currentLocation: [number, number];
    deliveries: DeliveryPoint[];
    trafficLevel: 'low' | 'medium' | 'high';

    // Result
    lastResult: OptimizeResult | null;
    isOptimizing: boolean;
    error: string | null;

    // History
    routeHistory: RouteHistoryEntry[];

    // Actions
    setCurrentLocation: (loc: [number, number]) => void;
    addDelivery: (delivery: DeliveryPoint) => void;
    removeDelivery: (index: number) => void;
    updateDelivery: (index: number, delivery: Partial<DeliveryPoint>) => void;
    reorderDeliveries: (fromIndex: number, toIndex: number) => void;
    clearDeliveries: () => void;
    setTrafficLevel: (level: 'low' | 'medium' | 'high') => void;

    // Optimization
    optimizeRoute: () => Promise<OptimizeResult | null>;
    clearResult: () => void;

    // Sample data
    addSampleDeliveries: () => void;

    // Reset
    reset: () => void;
}

const SAMPLE_DELIVERIES: DeliveryPoint[] = [
    { lat: 16.51, lng: 80.63, priority: 2.0, name: 'Delivery A' },
    { lat: 16.52, lng: 80.64, priority: 1.0, name: 'Delivery B' },
    { lat: 16.53, lng: 80.65, priority: 3.0, name: 'Delivery C' },
    { lat: 16.54, lng: 80.66, priority: 1.5, name: 'Delivery D' },
];

export const useRouteStore = create<RouteState>((set, get) => ({
    // Initial state
    currentLocation: VIJAYAWADA_CONFIG.center,
    deliveries: [],
    trafficLevel: 'medium',
    lastResult: null,
    isOptimizing: false,
    error: null,
    routeHistory: [],

    // Location actions
    setCurrentLocation: (loc) => set({ currentLocation: loc }),

    // Delivery actions
    addDelivery: (delivery) => {
        const name = delivery.name || `Delivery ${get().deliveries.length + 1}`;
        set((state) => ({
            deliveries: [...state.deliveries, { ...delivery, name }],
        }));
    },

    removeDelivery: (index) => {
        set((state) => ({
            deliveries: state.deliveries.filter((_, i) => i !== index),
        }));
    },

    updateDelivery: (index, updates) => {
        set((state) => ({
            deliveries: state.deliveries.map((d, i) =>
                i === index ? { ...d, ...updates } : d
            ),
        }));
    },

    reorderDeliveries: (fromIndex, toIndex) => {
        set((state) => {
            const newDeliveries = [...state.deliveries];
            const [removed] = newDeliveries.splice(fromIndex, 1);
            newDeliveries.splice(toIndex, 0, removed);
            return { deliveries: newDeliveries };
        });
    },

    clearDeliveries: () => set({ deliveries: [], lastResult: null }),

    setTrafficLevel: (level) => set({ trafficLevel: level }),

    // Optimization
    optimizeRoute: async () => {
        const { currentLocation, deliveries, trafficLevel } = get();

        if (deliveries.length === 0) {
            set({ error: 'Add at least one delivery first!' });
            return null;
        }

        set({ isOptimizing: true, error: null });

        try {
            const result = await optimizationAPI.optimize({
                current_loc: currentLocation,
                deliveries,
                traffic_level: trafficLevel,
                include_map: false,
            });

            // Add to history
            const historyEntry: RouteHistoryEntry = {
                route_id: result.route_id,
                n_stops: result.sequence.length,
                total_eta: result.total_eta,
                timestamp: new Date().toLocaleTimeString(),
            };

            set((state) => ({
                lastResult: result,
                isOptimizing: false,
                routeHistory: [historyEntry, ...state.routeHistory.slice(0, 9)], // Keep last 10
            }));

            return result;
        } catch (error: unknown) {
            const message = error instanceof Error ? error.message : 'Optimization failed';
            set({ error: message, isOptimizing: false });
            return null;
        }
    },

    clearResult: () => set({ lastResult: null, error: null }),

    // Sample data
    addSampleDeliveries: () => {
        set((state) => {
            // Only add samples that don't already exist
            const existing = new Set(
                state.deliveries.map(d => `${d.lat}-${d.lng}`)
            );
            const newDeliveries = SAMPLE_DELIVERIES.filter(
                d => !existing.has(`${d.lat}-${d.lng}`)
            );
            return { deliveries: [...state.deliveries, ...newDeliveries] };
        });
    },

    // Reset all
    reset: () => set({
        currentLocation: VIJAYAWADA_CONFIG.center,
        deliveries: [],
        trafficLevel: 'medium',
        lastResult: null,
        isOptimizing: false,
        error: null,
    }),
}));

export default useRouteStore;

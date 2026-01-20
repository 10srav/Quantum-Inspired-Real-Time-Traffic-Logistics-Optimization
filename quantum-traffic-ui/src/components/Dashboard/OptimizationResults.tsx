// ================================
// Optimization Results Component
// ================================

import useRouteStore from '../../stores/routeStore';
import MetricsCard from './MetricsCard';

const OptimizationResults = () => {
    const { lastResult } = useRouteStore();

    if (!lastResult) {
        return (
            <div className="glass-card p-6 text-center">
                <div className="text-4xl mb-3">📈</div>
                <p className="text-gray-400">No optimization results yet</p>
                <p className="text-sm text-gray-500 mt-1">
                    Add locations and click "Optimize Route" to see results
                </p>
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {/* Metrics Grid */}
            <div className="grid grid-cols-3 gap-4">
                <MetricsCard
                    icon="📏"
                    label="Total Distance"
                    value={(lastResult.total_distance / 1000).toFixed(2)}
                    unit="km"
                />
                <MetricsCard
                    icon="⏱️"
                    label="Total ETA"
                    value={lastResult.total_eta.toFixed(1)}
                    unit="min"
                />
                <MetricsCard
                    icon="📊"
                    label="vs Greedy"
                    value={lastResult.improvement_over_greedy.toFixed(1)}
                    unit="%"
                    trend={lastResult.improvement_over_greedy}
                />
            </div>

            {/* Optimization Info */}
            <div className="glass-card p-4">
                <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-400">
                        ⚡ Optimized in {lastResult.optimization_time.toFixed(3)}s
                    </span>
                    <span className="text-gray-400">
                        🚦 Traffic: <span className="capitalize">{lastResult.traffic_level}</span>
                    </span>
                </div>
            </div>

            {/* Optimized Sequence */}
            <div>
                <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                    🛣️ Optimized Sequence
                </h3>
                <div className="space-y-2">
                    {lastResult.sequence.map((stop) => (
                        <div key={stop.position} className="stop-card">
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                    <span className="w-8 h-8 flex items-center justify-center bg-primary-500 text-white text-sm font-bold rounded-full">
                                        {stop.position + 1}
                                    </span>
                                    <div>
                                        <span className="font-medium text-white">
                                            {stop.delivery.name || `Location ${stop.position + 1}`}
                                        </span>
                                        <div className="text-sm text-gray-400">
                                            📍 {stop.delivery.lat.toFixed(4)}, {stop.delivery.lng.toFixed(4)}
                                        </div>
                                    </div>
                                </div>
                                <div className="text-right text-sm">
                                    <div className="text-gray-300">
                                        +{(stop.distance_from_prev / 1000).toFixed(2)} km
                                    </div>
                                    <div className="text-gray-500">
                                        +{stop.eta_from_prev.toFixed(1)} min
                                    </div>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default OptimizationResults;

// ================================
// Optimization Panel Component
// ================================

import useRouteStore from '../../stores/routeStore';
import { VIJAYAWADA_CONFIG } from '../../types';

const OptimizationPanel = () => {
    const {
        currentLocation,
        setCurrentLocation,
        trafficLevel,
        setTrafficLevel,
        optimizeRoute,
        isOptimizing,
        error,
        reset,
    } = useRouteStore();

    const { bbox } = VIJAYAWADA_CONFIG;

    const handleOptimize = async () => {
        await optimizeRoute();
    };

    return (
        <div className="space-y-6">
            {/* Current Location */}
            <div>
                <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                    📍 Current Location
                </h3>
                <div className="grid grid-cols-2 gap-3">
                    <div>
                        <label className="block text-xs text-gray-400 mb-1">Latitude</label>
                        <input
                            type="number"
                            value={currentLocation[0]}
                            onChange={(e) => setCurrentLocation([
                                Math.max(bbox.lat_min, Math.min(bbox.lat_max, parseFloat(e.target.value) || bbox.lat_min)),
                                currentLocation[1]
                            ])}
                            step="0.001"
                            className="input-modern text-sm"
                        />
                    </div>
                    <div>
                        <label className="block text-xs text-gray-400 mb-1">Longitude</label>
                        <input
                            type="number"
                            value={currentLocation[1]}
                            onChange={(e) => setCurrentLocation([
                                currentLocation[0],
                                Math.max(bbox.lng_min, Math.min(bbox.lng_max, parseFloat(e.target.value) || bbox.lng_min))
                            ])}
                            step="0.001"
                            className="input-modern text-sm"
                        />
                    </div>
                </div>
            </div>

            {/* Traffic Level */}
            <div>
                <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                    🚦 Traffic Level
                </h3>
                <div className="flex gap-2">
                    {(['low', 'medium', 'high'] as const).map((level) => (
                        <button
                            key={level}
                            onClick={() => setTrafficLevel(level)}
                            className={`flex-1 py-2 px-3 rounded-lg text-sm font-medium transition-all duration-200
                ${trafficLevel === level
                                    ? 'bg-primary-500 text-white shadow-glow'
                                    : 'bg-white/5 text-gray-400 hover:bg-white/10'
                                }`}
                        >
                            {level === 'low' && '🟢'}
                            {level === 'medium' && '🟡'}
                            {level === 'high' && '🔴'}
                            <span className="ml-1 capitalize">{level}</span>
                        </button>
                    ))}
                </div>
            </div>

            {/* Error display */}
            {error && (
                <div className="text-red-400 text-sm bg-red-500/10 p-3 rounded-lg border border-red-500/20">
                    ❌ {error}
                </div>
            )}

            {/* Action Buttons */}
            <div className="space-y-3 pt-2">
                <button
                    onClick={handleOptimize}
                    disabled={isOptimizing}
                    className="btn-gradient w-full disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    {isOptimizing ? (
                        <span className="flex items-center justify-center gap-2">
                            <div className="loading-spinner w-5 h-5" />
                            Optimizing...
                        </span>
                    ) : (
                        <span className="flex items-center justify-center gap-2">
                            🚀 Optimize Route
                        </span>
                    )}
                </button>

                <button
                    onClick={reset}
                    className="w-full py-2 px-4 rounded-lg bg-white/5 text-gray-400 
                     hover:bg-white/10 transition-colors text-sm"
                >
                    🔄 Reset All
                </button>
            </div>
        </div>
    );
};

export default OptimizationPanel;

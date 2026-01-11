// ================================
// Delivery List Component
// ================================

import { type DeliveryPoint } from '../../types';
import useRouteStore from '../../stores/routeStore';

const DeliveryList = () => {
    const {
        deliveries,
        removeDelivery,
        lastResult,
        addSampleDeliveries,
        clearDeliveries
    } = useRouteStore();

    // Get optimized sequence position for a delivery
    const getSequencePosition = (delivery: DeliveryPoint): number | null => {
        if (!lastResult?.sequence) return null;

        for (const stop of lastResult.sequence) {
            if (
                Math.abs(stop.delivery.lat - delivery.lat) < 0.0001 &&
                Math.abs(stop.delivery.lng - delivery.lng) < 0.0001
            ) {
                return stop.position + 1;
            }
        }
        return null;
    };

    return (
        <div className="space-y-4">
            {/* Header */}
            <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                    📦 Deliveries
                    <span className="bg-primary-500/20 text-primary-400 text-sm px-2 py-0.5 rounded-full">
                        {deliveries.length}
                    </span>
                </h3>

                <div className="flex gap-2">
                    <button
                        onClick={addSampleDeliveries}
                        className="text-xs text-primary-400 hover:text-primary-300 transition-colors"
                    >
                        + Add Samples
                    </button>
                    {deliveries.length > 0 && (
                        <button
                            onClick={clearDeliveries}
                            className="text-xs text-red-400 hover:text-red-300 transition-colors"
                        >
                            Clear All
                        </button>
                    )}
                </div>
            </div>

            {/* List */}
            <div className="space-y-2 max-h-80 overflow-y-auto pr-2">
                {deliveries.length === 0 ? (
                    <div className="text-center py-8 text-gray-500">
                        <div className="text-4xl mb-2">🗺️</div>
                        <p className="text-sm">No deliveries yet</p>
                        <p className="text-xs mt-1">Click on the map to add locations</p>
                    </div>
                ) : (
                    deliveries.map((delivery, index) => {
                        const seqPos = getSequencePosition(delivery);

                        return (
                            <div
                                key={index}
                                className="stop-card group"
                            >
                                <div className="flex items-start justify-between">
                                    <div className="flex-1">
                                        <div className="flex items-center gap-2">
                                            {seqPos && (
                                                <span className="w-6 h-6 flex items-center justify-center bg-primary-500 text-white text-xs font-bold rounded-full">
                                                    {seqPos}
                                                </span>
                                            )}
                                            <span className="font-medium text-white">
                                                {delivery.name || `Delivery ${index + 1}`}
                                            </span>
                                        </div>
                                        <div className="mt-1 text-sm text-gray-400 flex items-center gap-3">
                                            <span>📍 {delivery.lat.toFixed(4)}, {delivery.lng.toFixed(4)}</span>
                                            <span className="flex items-center gap-1">
                                                ⭐ {delivery.priority}
                                            </span>
                                        </div>
                                    </div>

                                    <button
                                        onClick={() => removeDelivery(index)}
                                        className="opacity-0 group-hover:opacity-100 text-red-400 hover:text-red-300 
                               transition-all duration-200 text-lg"
                                        title="Remove delivery"
                                    >
                                        ×
                                    </button>
                                </div>
                            </div>
                        );
                    })
                )}
            </div>
        </div>
    );
};

export default DeliveryList;

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

    const getSequencePosition = (delivery: DeliveryPoint): number | null => {
        if (!lastResult?.sequence) return null;
        for (const stop of lastResult.sequence) {
            if (Math.abs(stop.delivery.lat - delivery.lat) < 0.0001 && Math.abs(stop.delivery.lng - delivery.lng) < 0.0001) {
                return stop.position + 1;
            }
        }
        return null;
    };

    return (
        <div>
            {/* Header */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px' }}>
                <h3 style={{ fontSize: '16px', fontWeight: 600, color: 'white', display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <span style={{ width: '36px', height: '36px', borderRadius: '8px', background: 'linear-gradient(135deg, #3b82f6, #06b6d4)', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 4px 12px rgba(59,130,246,0.3)' }}>
                        <svg style={{ width: '20px', height: '20px', color: 'white' }} fill="currentColor" viewBox="0 0 20 20">
                            <path d="M4 3a2 2 0 100 4h12a2 2 0 100-4H4z" />
                            <path fillRule="evenodd" d="M3 8h14v7a2 2 0 01-2 2H5a2 2 0 01-2-2V8zm5 3a1 1 0 011-1h2a1 1 0 110 2H9a1 1 0 01-1-1z" clipRule="evenodd" />
                        </svg>
                    </span>
                    Deliveries
                    {deliveries.length > 0 && (
                        <span style={{ padding: '4px 10px', borderRadius: '8px', background: 'rgba(59,130,246,0.2)', color: '#60a5fa', fontSize: '14px', fontWeight: 600, border: '1px solid rgba(59,130,246,0.3)' }}>
                            {deliveries.length}
                        </span>
                    )}
                </h3>

                <div style={{ display: 'flex', gap: '10px' }}>
                    <button
                        onClick={addSampleDeliveries}
                        style={{ fontSize: '12px', padding: '8px 16px', borderRadius: '8px', background: 'rgba(75,85,99,0.5)', color: '#d1d5db', border: '1px solid rgba(75,85,99,0.5)', cursor: 'pointer', fontWeight: 500, transition: 'all 0.2s' }}
                    >
                        + Samples
                    </button>
                    {deliveries.length > 0 && (
                        <button
                            onClick={clearDeliveries}
                            style={{ fontSize: '12px', padding: '8px 16px', borderRadius: '8px', background: 'rgba(75,85,99,0.5)', color: '#d1d5db', border: '1px solid rgba(75,85,99,0.5)', cursor: 'pointer', fontWeight: 500, transition: 'all 0.2s' }}
                        >
                            Clear
                        </button>
                    )}
                </div>
            </div>

            {/* List */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', maxHeight: '320px', overflowY: 'auto', paddingRight: '4px' }}>
                {deliveries.length === 0 ? (
                    <div style={{ textAlign: 'center', padding: '48px 16px' }}>
                        <div style={{ width: '64px', height: '64px', margin: '0 auto 16px', borderRadius: '16px', background: 'rgba(31,41,55,0.8)', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px solid rgba(75,85,99,0.5)' }}>
                            <svg style={{ width: '32px', height: '32px', color: '#4b5563' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                            </svg>
                        </div>
                        <p style={{ color: '#d1d5db', fontWeight: 500 }}>No deliveries yet</p>
                        <p style={{ color: '#6b7280', fontSize: '14px', marginTop: '8px' }}>Search for a place or click on the map</p>
                    </div>
                ) : (
                    deliveries.map((delivery, index) => {
                        const seqPos = getSequencePosition(delivery);

                        return (
                            <div
                                key={index}
                                style={{ padding: '16px', borderRadius: '12px', background: 'rgba(31,41,55,0.5)', border: '1px solid rgba(75,85,99,0.5)', transition: 'all 0.2s' }}
                            >
                                <div style={{ display: 'flex', alignItems: 'flex-start', gap: '16px' }}>
                                    {/* Badge */}
                                    <div style={{
                                        width: '44px',
                                        height: '44px',
                                        borderRadius: '12px',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        fontSize: '14px',
                                        fontWeight: 700,
                                        flexShrink: 0,
                                        background: seqPos
                                            ? 'linear-gradient(135deg, #10b981, #059669)'
                                            : 'rgba(55,65,81,0.8)',
                                        color: seqPos ? 'white' : '#9ca3af',
                                        border: seqPos ? 'none' : '1px solid rgba(75,85,99,0.5)',
                                        boxShadow: seqPos ? '0 4px 12px rgba(16,185,129,0.3)' : 'none',
                                    }}>
                                        {seqPos || index + 1}
                                    </div>

                                    {/* Info */}
                                    <div style={{ flex: 1, minWidth: 0 }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                                            <span style={{ fontWeight: 600, color: 'white', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                                {delivery.name || `Delivery ${index + 1}`}
                                            </span>
                                            {seqPos && (
                                                <span style={{ padding: '2px 8px', borderRadius: '6px', fontSize: '12px', background: 'rgba(16,185,129,0.2)', color: '#34d399', border: '1px solid rgba(16,185,129,0.3)', fontWeight: 500 }}>
                                                    Stop #{seqPos}
                                                </span>
                                            )}
                                        </div>
                                        <div style={{ marginTop: '10px', display: 'flex', alignItems: 'center', gap: '16px', fontSize: '12px', color: '#9ca3af' }}>
                                            <span style={{ display: 'flex', alignItems: 'center', gap: '6px', fontFamily: 'monospace', background: 'rgba(17,24,39,0.5)', padding: '4px 8px', borderRadius: '4px' }}>
                                                <svg style={{ width: '14px', height: '14px', color: '#6b7280' }} fill="currentColor" viewBox="0 0 20 20">
                                                    <path fillRule="evenodd" d="M5.05 4.05a7 7 0 119.9 9.9L10 18.9l-4.95-4.95a7 7 0 010-9.9zM10 11a2 2 0 100-4 2 2 0 000 4z" clipRule="evenodd" />
                                                </svg>
                                                {delivery.lat.toFixed(4)}, {delivery.lng.toFixed(4)}
                                            </span>
                                            <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                                <svg style={{ width: '14px', height: '14px', color: '#f59e0b' }} fill="currentColor" viewBox="0 0 20 20">
                                                    <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                                                </svg>
                                                Priority: {delivery.priority}
                                            </span>
                                        </div>
                                    </div>

                                    {/* Remove */}
                                    <button
                                        onClick={() => removeDelivery(index)}
                                        style={{ width: '36px', height: '36px', borderRadius: '8px', background: 'rgba(239,68,68,0.1)', color: '#f87171', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px solid transparent', cursor: 'pointer', transition: 'all 0.2s' }}
                                    >
                                        <svg style={{ width: '16px', height: '16px' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                        </svg>
                                    </button>
                                </div>
                            </div>
                        );
                    })
                )}
            </div>

            {/* Optimized Status */}
            {deliveries.length > 0 && lastResult && (
                <div style={{ marginTop: '20px', paddingTop: '20px', borderTop: '1px solid rgba(75,85,99,0.5)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <span style={{ color: '#9ca3af', fontSize: '14px' }}>Route Status</span>
                        <span style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#34d399', fontSize: '14px', fontWeight: 500 }}>
                            <svg style={{ width: '20px', height: '20px' }} fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                            </svg>
                            Optimized
                        </span>
                    </div>
                </div>
            )}
        </div>
    );
};

export default DeliveryList;

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
        deliveries,
    } = useRouteStore();

    const { bbox } = VIJAYAWADA_CONFIG;

    const handleOptimize = async () => {
        await optimizeRoute();
    };

    const trafficColors = {
        low: { bg: '#22c55e', activeBg: 'rgba(34,197,94,0.2)', border: 'rgba(34,197,94,0.5)', text: '#4ade80' },
        medium: { bg: '#eab308', activeBg: 'rgba(234,179,8,0.2)', border: 'rgba(234,179,8,0.5)', text: '#facc15' },
        high: { bg: '#ef4444', activeBg: 'rgba(239,68,68,0.2)', border: 'rgba(239,68,68,0.5)', text: '#f87171' },
    };

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
            {/* Start Location */}
            <div>
                <h3 style={{ fontSize: '16px', fontWeight: 600, color: 'white', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <span style={{ width: '36px', height: '36px', borderRadius: '8px', background: 'linear-gradient(135deg, #ef4444, #f97316)', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 4px 12px rgba(239,68,68,0.3)' }}>
                        <svg style={{ width: '20px', height: '20px', color: 'white' }} fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M5.05 4.05a7 7 0 119.9 9.9L10 18.9l-4.95-4.95a7 7 0 010-9.9zM10 11a2 2 0 100-4 2 2 0 000 4z" clipRule="evenodd" />
                        </svg>
                    </span>
                    Start Location
                </h3>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                    <div>
                        <label style={{ display: 'block', fontSize: '12px', color: '#9ca3af', marginBottom: '8px', fontWeight: 500 }}>Latitude</label>
                        <input
                            type="number"
                            value={currentLocation[0]}
                            onChange={(e) => setCurrentLocation([
                                Math.max(bbox.lat_min, Math.min(bbox.lat_max, parseFloat(e.target.value) || bbox.lat_min)),
                                currentLocation[1]
                            ])}
                            step="0.001"
                            style={{ width: '100%', padding: '12px 16px', borderRadius: '10px', background: 'rgba(31,41,55,0.8)', border: '1px solid rgba(75,85,99,0.5)', color: 'white', fontSize: '14px', fontFamily: 'monospace', outline: 'none', boxSizing: 'border-box' }}
                        />
                    </div>
                    <div>
                        <label style={{ display: 'block', fontSize: '12px', color: '#9ca3af', marginBottom: '8px', fontWeight: 500 }}>Longitude</label>
                        <input
                            type="number"
                            value={currentLocation[1]}
                            onChange={(e) => setCurrentLocation([
                                currentLocation[0],
                                Math.max(bbox.lng_min, Math.min(bbox.lng_max, parseFloat(e.target.value) || bbox.lng_min))
                            ])}
                            step="0.001"
                            style={{ width: '100%', padding: '12px 16px', borderRadius: '10px', background: 'rgba(31,41,55,0.8)', border: '1px solid rgba(75,85,99,0.5)', color: 'white', fontSize: '14px', fontFamily: 'monospace', outline: 'none', boxSizing: 'border-box' }}
                        />
                    </div>
                </div>
            </div>

            {/* Traffic Level */}
            <div>
                <h3 style={{ fontSize: '16px', fontWeight: 600, color: 'white', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <span style={{ width: '36px', height: '36px', borderRadius: '8px', background: 'linear-gradient(135deg, #f59e0b, #eab308)', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 4px 12px rgba(245,158,11,0.3)' }}>
                        <svg style={{ width: '20px', height: '20px', color: 'white' }} fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M11.3 1.046A1 1 0 0112 2v5h4a1 1 0 01.82 1.573l-7 10A1 1 0 018 18v-5H4a1 1 0 01-.82-1.573l7-10a1 1 0 011.12-.38z" clipRule="evenodd" />
                        </svg>
                    </span>
                    Traffic Level
                </h3>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px' }}>
                    {(['low', 'medium', 'high'] as const).map((level) => {
                        const isActive = trafficLevel === level;
                        const c = trafficColors[level];

                        return (
                            <button
                                key={level}
                                onClick={() => setTrafficLevel(level)}
                                style={{
                                    padding: '14px 16px',
                                    borderRadius: '12px',
                                    fontSize: '14px',
                                    fontWeight: 500,
                                    display: 'flex',
                                    flexDirection: 'column',
                                    alignItems: 'center',
                                    gap: '10px',
                                    cursor: 'pointer',
                                    transition: 'all 0.2s',
                                    background: isActive ? c.activeBg : 'rgba(31,41,55,0.5)',
                                    color: isActive ? c.text : '#9ca3af',
                                    border: isActive ? `2px solid ${c.border}` : '2px solid rgba(75,85,99,0.5)',
                                    boxShadow: isActive ? `0 0 0 3px ${c.activeBg}` : 'none',
                                }}
                            >
                                <span style={{ width: '14px', height: '14px', borderRadius: '50%', background: c.bg, boxShadow: isActive ? `0 0 12px ${c.bg}` : 'none' }} />
                                <span style={{ textTransform: 'capitalize' }}>{level}</span>
                            </button>
                        );
                    })}
                </div>
            </div>

            {/* Error Display */}
            {error && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '16px', borderRadius: '12px', background: 'rgba(239,68,68,0.15)', border: '1px solid rgba(239,68,68,0.3)' }}>
                    <svg style={{ width: '20px', height: '20px', color: '#f87171', flexShrink: 0 }} fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                    </svg>
                    <span style={{ color: '#f87171', fontSize: '14px' }}>{error}</span>
                </div>
            )}

            {/* Action Buttons */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', paddingTop: '8px' }}>
                <button
                    onClick={handleOptimize}
                    disabled={isOptimizing || deliveries.length === 0}
                    style={{
                        width: '100%',
                        padding: '16px 24px',
                        borderRadius: '12px',
                        fontWeight: 700,
                        color: 'white',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '12px',
                        cursor: deliveries.length === 0 ? 'not-allowed' : 'pointer',
                        border: 'none',
                        fontSize: '15px',
                        background: deliveries.length === 0
                            ? '#374151'
                            : isOptimizing
                                ? 'linear-gradient(135deg, #7c3aed, #9333ea)'
                                : 'linear-gradient(135deg, #7c3aed, #9333ea, #c026d3)',
                        boxShadow: deliveries.length === 0 ? 'none' : '0 8px 24px rgba(147,51,234,0.4)',
                        opacity: deliveries.length === 0 ? 0.6 : 1,
                        transition: 'all 0.3s',
                    }}
                >
                    {isOptimizing ? (
                        <>
                            <div style={{ width: '20px', height: '20px', border: '3px solid rgba(255,255,255,0.3)', borderTopColor: 'white', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
                            <span>Optimizing Route...</span>
                        </>
                    ) : (
                        <>
                            <svg style={{ width: '24px', height: '24px' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
                            </svg>
                            <span>Optimize Route</span>
                            {deliveries.length > 0 && (
                                <span style={{ padding: '4px 10px', borderRadius: '8px', background: 'rgba(255,255,255,0.2)', fontSize: '12px', fontWeight: 600 }}>
                                    {deliveries.length} {deliveries.length === 1 ? 'stop' : 'stops'}
                                </span>
                            )}
                        </>
                    )}
                </button>

                <button
                    onClick={reset}
                    style={{
                        width: '100%',
                        padding: '14px 16px',
                        borderRadius: '12px',
                        background: 'rgba(31,41,55,0.5)',
                        color: '#9ca3af',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '8px',
                        fontSize: '14px',
                        fontWeight: 500,
                        border: '1px solid rgba(75,85,99,0.5)',
                        cursor: 'pointer',
                        transition: 'all 0.2s',
                    }}
                >
                    <svg style={{ width: '16px', height: '16px' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                    Reset All
                </button>
            </div>
        </div>
    );
};

export default OptimizationPanel;

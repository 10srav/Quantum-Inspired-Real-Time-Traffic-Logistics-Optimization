import { useState } from 'react';
import useRouteStore from '../../stores/routeStore';
import { VIJAYAWADA_CONFIG } from '../../types';

const OptimizationPanel = () => {
    const {
        currentLocation,
        setCurrentLocation,
        startingLocationSet,
        selectionMode,
        setSelectionMode,
        trafficLevel,
        setTrafficLevel,
        optimizeRoute,
        isOptimizing,
        error,
        reset,
        deliveries,
    } = useRouteStore();

    const { bbox } = VIJAYAWADA_CONFIG;

    // Geolocation state
    const [isLocating, setIsLocating] = useState(false);
    const [geoError, setGeoError] = useState<string | null>(null);

    const handleUseMyLocation = () => {
        if (!navigator.geolocation) {
            setGeoError('Geolocation is not supported by your browser');
            return;
        }

        setIsLocating(true);
        setGeoError(null);

        navigator.geolocation.getCurrentPosition(
            (position) => {
                const { latitude, longitude } = position.coords;

                // Validate within Vijayawada bbox
                if (latitude >= bbox.lat_min && latitude <= bbox.lat_max &&
                    longitude >= bbox.lng_min && longitude <= bbox.lng_max) {
                    setCurrentLocation([
                        parseFloat(latitude.toFixed(4)),
                        parseFloat(longitude.toFixed(4))
                    ]);
                    setGeoError(null);
                } else {
                    setGeoError(`Location (${latitude.toFixed(4)}, ${longitude.toFixed(4)}) is outside Vijayawada service area`);
                }
                setIsLocating(false);
            },
            (err) => {
                let message = 'Failed to get location';
                switch (err.code) {
                    case err.PERMISSION_DENIED:
                        message = 'Location permission denied';
                        break;
                    case err.POSITION_UNAVAILABLE:
                        message = 'Location unavailable';
                        break;
                    case err.TIMEOUT:
                        message = 'Location request timed out';
                        break;
                }
                setGeoError(message);
                setIsLocating(false);
            },
            { enableHighAccuracy: true, timeout: 10000, maximumAge: 300000 }
        );
    };

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

                {/* Select on Map Button */}
                <button
                    onClick={() => setSelectionMode(selectionMode === 'startingLocation' ? 'delivery' : 'startingLocation')}
                    style={{
                        width: '100%',
                        marginTop: '12px',
                        padding: '12px 16px',
                        borderRadius: '10px',
                        background: selectionMode === 'startingLocation'
                            ? 'rgba(239,68,68,0.2)'
                            : 'rgba(34,197,94,0.15)',
                        border: selectionMode === 'startingLocation'
                            ? '2px solid rgba(239,68,68,0.5)'
                            : '1px solid rgba(34,197,94,0.4)',
                        color: selectionMode === 'startingLocation' ? '#f87171' : '#4ade80',
                        fontSize: '14px',
                        fontWeight: 500,
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '8px',
                        transition: 'all 0.2s',
                    }}
                >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/>
                    </svg>
                    {selectionMode === 'startingLocation' ? 'Click Map to Set Start Point' : 'Select on Map'}
                </button>

                {/* Selection Mode Indicator */}
                {selectionMode === 'startingLocation' && (
                    <div style={{
                        marginTop: '8px',
                        padding: '10px 12px',
                        borderRadius: '8px',
                        background: 'rgba(239,68,68,0.1)',
                        border: '1px solid rgba(239,68,68,0.3)',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px'
                    }}>
                        <div style={{
                            width: '8px',
                            height: '8px',
                            borderRadius: '50%',
                            background: '#ef4444',
                            animation: 'pulse 1.5s infinite'
                        }} />
                        <span style={{ color: '#f87171', fontSize: '12px' }}>
                            Click anywhere on the map to set your starting location
                        </span>
                    </div>
                )}

                {/* Starting Location Status */}
                {startingLocationSet && selectionMode !== 'startingLocation' && (
                    <div style={{
                        marginTop: '8px',
                        padding: '8px 12px',
                        borderRadius: '8px',
                        background: 'rgba(34,197,94,0.1)',
                        border: '1px solid rgba(34,197,94,0.3)',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px'
                    }}>
                        <svg width="14" height="14" viewBox="0 0 20 20" fill="#4ade80">
                            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                        </svg>
                        <span style={{ color: '#4ade80', fontSize: '12px' }}>Starting location set</span>
                    </div>
                )}

                {/* Use My Location Button - Keep as optional */}
                <button
                    onClick={handleUseMyLocation}
                    disabled={isLocating}
                    style={{
                        width: '100%',
                        marginTop: '8px',
                        padding: '10px 16px',
                        borderRadius: '10px',
                        background: 'rgba(59,130,246,0.1)',
                        border: '1px solid rgba(59,130,246,0.3)',
                        color: '#60a5fa',
                        fontSize: '13px',
                        fontWeight: 500,
                        cursor: isLocating ? 'wait' : 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '8px',
                        transition: 'all 0.2s',
                        opacity: 0.8,
                    }}
                >
                    {isLocating ? (
                        <>
                            <div style={{
                                width: '14px',
                                height: '14px',
                                border: '2px solid rgba(96,165,250,0.3)',
                                borderTopColor: '#60a5fa',
                                borderRadius: '50%',
                                animation: 'spin 1s linear infinite'
                            }} />
                            Locating...
                        </>
                    ) : (
                        <>
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                                <path d="M12 8c-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4-1.79-4-4-4zm8.94 3A8.994 8.994 0 0013 3.06V1h-2v2.06A8.994 8.994 0 003.06 11H1v2h2.06A8.994 8.994 0 0011 20.94V23h2v-2.06A8.994 8.994 0 0020.94 13H23v-2h-2.06zM12 19c-3.87 0-7-3.13-7-7s3.13-7 7-7 7 3.13 7 7-3.13 7-7 7z"/>
                            </svg>
                            Use GPS (if in Vijayawada)
                        </>
                    )}
                </button>

                {/* Geolocation Error */}
                {geoError && (
                    <div style={{
                        marginTop: '8px',
                        padding: '10px 12px',
                        borderRadius: '8px',
                        background: 'rgba(239,68,68,0.1)',
                        border: '1px solid rgba(239,68,68,0.3)',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px'
                    }}>
                        <svg width="14" height="14" viewBox="0 0 20 20" fill="#f87171">
                            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                        </svg>
                        <span style={{ color: '#f87171', fontSize: '12px' }}>{geoError}</span>
                    </div>
                )}
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

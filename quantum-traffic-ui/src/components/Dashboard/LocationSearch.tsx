import { useState, useCallback, useRef, useEffect } from 'react';
import { VIJAYAWADA_CONFIG } from '../../types';
import useRouteStore from '../../stores/routeStore';

interface SearchResult {
    place_id: number;
    display_name: string;
    lat: string;
    lon: string;
}

const LocationSearch = () => {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState<SearchResult[]>([]);
    const [isSearching, setIsSearching] = useState(false);
    const [showResults, setShowResults] = useState(false);
    const [showManual, setShowManual] = useState(false);
    const [lat, setLat] = useState('');
    const [lng, setLng] = useState('');
    const [name, setName] = useState('');
    const [isLocating, setIsLocating] = useState(false);
    const [geoError, setGeoError] = useState<string | null>(null);
    const searchRef = useRef<HTMLDivElement>(null);
    const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    const { addDelivery } = useRouteStore();
    const { bbox } = VIJAYAWADA_CONFIG;

    const handleAddMyLocation = () => {
        if (!navigator.geolocation) {
            setGeoError('Geolocation not supported');
            return;
        }

        setIsLocating(true);
        setGeoError(null);

        navigator.geolocation.getCurrentPosition(
            (position) => {
                const { latitude, longitude } = position.coords;

                if (latitude >= bbox.lat_min && latitude <= bbox.lat_max &&
                    longitude >= bbox.lng_min && longitude <= bbox.lng_max) {
                    addDelivery({
                        lat: parseFloat(latitude.toFixed(4)),
                        lng: parseFloat(longitude.toFixed(4)),
                        priority: 5,
                        name: 'My Location'
                    });
                    setGeoError(null);
                } else {
                    setGeoError('Your location is outside Vijayawada service area');
                }
                setIsLocating(false);
            },
            (err) => {
                let message = 'Failed to get location';
                if (err.code === err.PERMISSION_DENIED) message = 'Permission denied';
                else if (err.code === err.POSITION_UNAVAILABLE) message = 'Location unavailable';
                else if (err.code === err.TIMEOUT) message = 'Request timed out';
                setGeoError(message);
                setIsLocating(false);
            },
            { enableHighAccuracy: true, timeout: 10000, maximumAge: 300000 }
        );
    };

    useEffect(() => {
        const handleClickOutside = (e: MouseEvent) => {
            if (searchRef.current && !searchRef.current.contains(e.target as Node)) {
                setShowResults(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const searchLocations = useCallback(async (q: string) => {
        if (q.length < 2) { setResults([]); return; }
        setIsSearching(true);
        try {
            const viewbox = `${bbox.lng_min},${bbox.lat_min},${bbox.lng_max},${bbox.lat_max}`;
            const res = await fetch(
                `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(q + ', Vijayawada')}&viewbox=${viewbox}&bounded=1&limit=5`,
                { headers: { 'User-Agent': 'QuantumTrafficOptimizer/1.0' } }
            );
            const data: SearchResult[] = await res.json();
            setResults(data.filter(r => {
                const lat = parseFloat(r.lat), lng = parseFloat(r.lon);
                return lat >= bbox.lat_min && lat <= bbox.lat_max && lng >= bbox.lng_min && lng <= bbox.lng_max;
            }));
            setShowResults(true);
        } catch { setResults([]); }
        setIsSearching(false);
    }, [bbox]);

    const handleSearch = (e: React.ChangeEvent<HTMLInputElement>) => {
        setQuery(e.target.value);
        if (debounceRef.current) clearTimeout(debounceRef.current);
        debounceRef.current = setTimeout(() => searchLocations(e.target.value), 300);
    };

    const selectLocation = (r: SearchResult) => {
        addDelivery({ lat: parseFloat(parseFloat(r.lat).toFixed(4)), lng: parseFloat(parseFloat(r.lon).toFixed(4)), priority: 5, name: r.display_name.split(',')[0] });
        setQuery(''); setResults([]); setShowResults(false);
    };

    const addManual = () => {
        const latNum = parseFloat(lat), lngNum = parseFloat(lng);
        if (!isNaN(latNum) && !isNaN(lngNum) && latNum >= bbox.lat_min && latNum <= bbox.lat_max && lngNum >= bbox.lng_min && lngNum <= bbox.lng_max) {
            addDelivery({ lat: parseFloat(latNum.toFixed(4)), lng: parseFloat(lngNum.toFixed(4)), priority: 5, name: name || `Point ${latNum.toFixed(3)}, ${lngNum.toFixed(3)}` });
            setLat(''); setLng(''); setName(''); setShowManual(false);
        }
    };

    return (
        <div ref={searchRef}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <h3 style={{ fontSize: '16px', fontWeight: 600, color: 'white', display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <span style={{ width: '36px', height: '36px', borderRadius: '8px', background: 'linear-gradient(135deg, #10b981, #059669)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', fontSize: '18px' }}>+</span>
                    Add Location
                </h3>
                <button
                    onClick={() => setShowManual(!showManual)}
                    style={{ fontSize: '12px', padding: '8px 16px', borderRadius: '8px', background: showManual ? 'rgba(16,185,129,0.2)' : 'rgba(75,85,99,0.5)', color: showManual ? '#10b981' : '#d1d5db', border: showManual ? '1px solid rgba(16,185,129,0.4)' : '1px solid rgba(75,85,99,0.5)', cursor: 'pointer' }}
                >
                    {showManual ? 'Search Mode' : 'Manual Entry'}
                </button>
            </div>

            {!showManual ? (
                <div style={{ position: 'relative' }}>
                    <input
                        type="text"
                        value={query}
                        onChange={handleSearch}
                        onFocus={() => results.length > 0 && setShowResults(true)}
                        placeholder="Type location name..."
                        style={{ width: '100%', padding: '14px 16px', borderRadius: '12px', background: 'rgba(31,41,55,0.8)', border: '1px solid rgba(75,85,99,0.5)', color: 'white', fontSize: '14px', outline: 'none', boxSizing: 'border-box' }}
                    />
                    {isSearching && <div style={{ position: 'absolute', right: '16px', top: '50%', transform: 'translateY(-50%)', width: '20px', height: '20px', border: '2px solid rgba(16,185,129,0.3)', borderTopColor: '#10b981', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />}

                    {showResults && results.length > 0 && (
                        <div style={{ position: 'absolute', top: '100%', left: 0, right: 0, marginTop: '8px', background: '#1f2937', border: '1px solid rgba(75,85,99,0.5)', borderRadius: '12px', overflow: 'hidden', zIndex: 50 }}>
                            {results.map((r, i) => (
                                <button key={r.place_id} onClick={() => selectLocation(r)} style={{ width: '100%', padding: '12px 16px', textAlign: 'left', background: 'transparent', border: 'none', borderBottom: i < results.length - 1 ? '1px solid rgba(55,65,81,0.5)' : 'none', color: 'white', cursor: 'pointer' }}>
                                    <div style={{ fontSize: '14px', fontWeight: 500 }}>{r.display_name.split(',')[0]}</div>
                                    <div style={{ fontSize: '12px', color: '#9ca3af', marginTop: '4px' }}>{r.display_name.split(',').slice(1, 3).join(', ')}</div>
                                </button>
                            ))}
                        </div>
                    )}
                </div>
            ) : (
                <div style={{ padding: '16px', borderRadius: '12px', background: 'rgba(31,41,55,0.5)', border: '1px solid rgba(75,85,99,0.3)' }}>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '16px' }}>
                        <div>
                            <label style={{ display: 'block', fontSize: '12px', color: '#9ca3af', marginBottom: '8px' }}>Latitude</label>
                            <input type="number" value={lat} onChange={e => setLat(e.target.value)} placeholder="16.5200" step="0.0001" style={{ width: '100%', padding: '12px', borderRadius: '8px', background: 'rgba(55,65,81,0.5)', border: '1px solid rgba(75,85,99,0.5)', color: 'white', fontSize: '14px', outline: 'none', boxSizing: 'border-box' }} />
                        </div>
                        <div>
                            <label style={{ display: 'block', fontSize: '12px', color: '#9ca3af', marginBottom: '8px' }}>Longitude</label>
                            <input type="number" value={lng} onChange={e => setLng(e.target.value)} placeholder="80.6500" step="0.0001" style={{ width: '100%', padding: '12px', borderRadius: '8px', background: 'rgba(55,65,81,0.5)', border: '1px solid rgba(75,85,99,0.5)', color: 'white', fontSize: '14px', outline: 'none', boxSizing: 'border-box' }} />
                        </div>
                    </div>
                    <div style={{ marginBottom: '16px' }}>
                        <label style={{ display: 'block', fontSize: '12px', color: '#9ca3af', marginBottom: '8px' }}>Name (optional)</label>
                        <input type="text" value={name} onChange={e => setName(e.target.value)} placeholder="e.g., Customer Office" style={{ width: '100%', padding: '12px', borderRadius: '8px', background: 'rgba(55,65,81,0.5)', border: '1px solid rgba(75,85,99,0.5)', color: 'white', fontSize: '14px', outline: 'none', boxSizing: 'border-box' }} />
                    </div>
                    <button onClick={addManual} style={{ width: '100%', padding: '14px', borderRadius: '8px', background: 'linear-gradient(135deg, #10b981, #059669)', color: 'white', fontSize: '14px', fontWeight: 600, border: 'none', cursor: 'pointer' }}>
                        Add Location
                    </button>
                </div>
            )}

            {/* Quick Add My Location Button */}
            <button
                onClick={handleAddMyLocation}
                disabled={isLocating}
                style={{
                    width: '100%',
                    marginTop: '12px',
                    padding: '12px 16px',
                    borderRadius: '10px',
                    background: 'rgba(59,130,246,0.15)',
                    border: '1px solid rgba(59,130,246,0.4)',
                    color: '#60a5fa',
                    fontSize: '13px',
                    fontWeight: 500,
                    cursor: isLocating ? 'wait' : 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '8px',
                    transition: 'all 0.2s',
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
                        Add My Current Location
                    </>
                )}
            </button>

            {/* Geolocation Error */}
            {geoError && (
                <div style={{
                    marginTop: '8px',
                    padding: '8px 12px',
                    borderRadius: '8px',
                    background: 'rgba(239,68,68,0.1)',
                    border: '1px solid rgba(239,68,68,0.3)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px'
                }}>
                    <svg width="12" height="12" viewBox="0 0 20 20" fill="#f87171">
                        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                    </svg>
                    <span style={{ color: '#f87171', fontSize: '11px' }}>{geoError}</span>
                </div>
            )}

            <p style={{ marginTop: '12px', fontSize: '12px', color: '#6b7280' }}>
                Search for places, use GPS, or click directly on the map
            </p>
        </div>
    );
};

export default LocationSearch;

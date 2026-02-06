// ================================
// Route Map Component with React-Leaflet
// ================================

import { useEffect, useMemo } from 'react';
import {
    MapContainer,
    TileLayer,
    Marker,
    Popup,
    Polyline,
    Rectangle,
    useMapEvents,
    useMap
} from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

import { VIJAYAWADA_CONFIG, type DeliveryPoint } from '../../types';
import useRouteStore from '../../stores/routeStore';

// Fix for default marker icons in React-Leaflet
delete (L.Icon.Default.prototype as unknown as { _getIconUrl?: () => void })._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
    iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
    shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

// Custom icons
const createIcon = (color: string, emoji: string) => {
    return L.divIcon({
        className: 'custom-marker',
        html: `
      <div style="
        background: ${color};
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        border: 2px solid white;
      ">${emoji}</div>
    `,
        iconSize: [32, 32],
        iconAnchor: [16, 32],
        popupAnchor: [0, -32],
    });
};

const depotIcon = createIcon('linear-gradient(135deg, #ef4444 0%, #dc2626 100%)', '🏠');

const deliveryColors = [
    '#3b82f6', // blue
    '#22c55e', // green
    '#a855f7', // purple
    '#f97316', // orange
    '#ec4899', // pink
    '#06b6d4', // cyan
    '#eab308', // yellow
];

// Route segment colors - distinct colors for each leg of the journey
const routeColors = [
    '#ef4444', // red (from depot)
    '#f97316', // orange
    '#eab308', // yellow
    '#22c55e', // green
    '#14b8a6', // teal
    '#06b6d4', // cyan
    '#3b82f6', // blue
    '#8b5cf6', // violet
    '#a855f7', // purple
    '#ec4899', // pink
    '#f43f5e', // rose
];

// Map click handler component
const MapClickHandler = () => {
    const { addDelivery, setCurrentLocation, selectionMode } = useRouteStore();
    const { bbox } = VIJAYAWADA_CONFIG;

    useMapEvents({
        click: (e) => {
            const { lat, lng } = e.latlng;

            // Check if within bounds
            if (
                lat >= bbox.lat_min && lat <= bbox.lat_max &&
                lng >= bbox.lng_min && lng <= bbox.lng_max
            ) {
                const roundedLat = parseFloat(lat.toFixed(4));
                const roundedLng = parseFloat(lng.toFixed(4));

                if (selectionMode === 'startingLocation') {
                    setCurrentLocation([roundedLat, roundedLng]);
                } else {
                    addDelivery({
                        lat: roundedLat,
                        lng: roundedLng,
                        priority: 5.0,
                    });
                }
            }
        },
    });

    return null;
};

// Map center updater
const MapCenterUpdater = ({ center }: { center: [number, number] }) => {
    const map = useMap();

    useEffect(() => {
        map.setView(center, map.getZoom());
    }, [center, map]);

    return null;
};

// Route Legend Component
const RouteLegend = ({ sequence }: { sequence: { position: number; delivery: DeliveryPoint }[] | undefined }) => {
    if (!sequence || sequence.length === 0) return null;

    return (
        <div style={{
            position: 'absolute',
            bottom: '20px',
            left: '20px',
            zIndex: 1000,
            background: 'rgba(17, 24, 39, 0.95)',
            borderRadius: '12px',
            padding: '12px 16px',
            border: '1px solid rgba(75, 85, 99, 0.5)',
            maxHeight: '200px',
            overflowY: 'auto',
            minWidth: '180px',
        }}>
            <div style={{ fontSize: '12px', fontWeight: 600, color: '#9ca3af', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                Route Legs
            </div>
            {sequence.map((stop, index) => {
                const color = routeColors[index % routeColors.length];
                const fromLabel = index === 0 ? '🏠 Depot' : `${index}`;
                const toLabel = `${index + 1}`;
                const stopName = stop.delivery.name || `Stop ${index + 1}`;

                return (
                    <div key={index} style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                        <div style={{
                            width: '24px',
                            height: '4px',
                            background: color,
                            borderRadius: '2px',
                            flexShrink: 0,
                        }} />
                        <span style={{ fontSize: '11px', color: '#d1d5db' }}>
                            {fromLabel} → {toLabel}
                        </span>
                        <span style={{ fontSize: '10px', color: '#6b7280', marginLeft: 'auto', maxWidth: '80px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {stopName}
                        </span>
                    </div>
                );
            })}
        </div>
    );
};

const RouteMap = () => {
    const {
        currentLocation,
        deliveries,
        lastResult
    } = useRouteStore();

    const { center, bbox, zoom } = VIJAYAWADA_CONFIG;

    // Create route coordinates for polyline - use actual road geometry if available
    const routeSegments = useMemo(() => {
        if (!lastResult?.sequence || lastResult.sequence.length === 0) return null;

        // Build the ordered list of locations: depot first, then deliveries in sequence order
        const orderedLocations: [number, number][] = [currentLocation];
        for (const stop of lastResult.sequence) {
            orderedLocations.push([stop.delivery.lat, stop.delivery.lng]);
        }

        // If we have road geometry from backend, enhance it to connect to exact marker positions
        if (lastResult.route_geometry && lastResult.route_geometry.length > 0) {
            return lastResult.route_geometry.map((segment, index) => {
                const segmentCoords = segment.map(coord => [coord[0], coord[1]] as [number, number]);

                // Get exact start and end positions for this segment
                const startPos = orderedLocations[index];
                const endPos = orderedLocations[index + 1];

                if (!startPos || !endPos) return segmentCoords;

                // Ensure segment starts at exact marker position
                if (segmentCoords.length > 0) {
                    const firstPoint = segmentCoords[0];
                    // If first point is far from start marker, prepend start position
                    const startDist = Math.abs(firstPoint[0] - startPos[0]) + Math.abs(firstPoint[1] - startPos[1]);
                    if (startDist > 0.0005) { // ~50m threshold
                        segmentCoords.unshift(startPos);
                    }

                    // If last point is far from end marker, append end position
                    const lastPoint = segmentCoords[segmentCoords.length - 1];
                    const endDist = Math.abs(lastPoint[0] - endPos[0]) + Math.abs(lastPoint[1] - endPos[1]);
                    if (endDist > 0.0005) {
                        segmentCoords.push(endPos);
                    }
                } else {
                    // Empty segment - just connect the two points directly
                    return [startPos, endPos];
                }

                return segmentCoords;
            });
        }

        // Fallback: create individual segments with straight lines between consecutive stops
        const segments: [number, number][][] = [];
        for (let i = 0; i < orderedLocations.length - 1; i++) {
            segments.push([orderedLocations[i], orderedLocations[i + 1]]);
        }
        return segments;
    }, [lastResult, currentLocation]);

    // Get sequence position for a delivery
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
        <div style={{ position: 'relative', width: '100%', height: '100%' }}>
        <MapContainer
            center={center}
            zoom={zoom}
            className="w-full h-full min-h-[500px] rounded-xl"
            style={{ background: '#1a1a3e' }}
        >
            <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />

            <MapClickHandler />
            <MapCenterUpdater center={currentLocation} />

            {/* Service area boundary */}
            <Rectangle
                bounds={[
                    [bbox.lat_min, bbox.lng_min],
                    [bbox.lat_max, bbox.lng_max],
                ]}
                pathOptions={{
                    color: '#6b7280',
                    weight: 2,
                    dashArray: '10, 5',
                    fill: false,
                }}
            >
                <Popup>Vijayawada Service Area</Popup>
            </Rectangle>

            {/* Depot marker */}
            <Marker position={currentLocation} icon={depotIcon}>
                <Popup>
                    <div className="text-gray-900">
                        <strong>📍 Current Location (Depot)</strong>
                        <br />
                        {currentLocation[0].toFixed(4)}, {currentLocation[1].toFixed(4)}
                    </div>
                </Popup>
            </Marker>

            {/* Delivery markers */}
            {deliveries.map((delivery, index) => {
                const seqPos = getSequencePosition(delivery);
                const color = deliveryColors[index % deliveryColors.length];
                const icon = createIcon(
                    color,
                    seqPos ? seqPos.toString() : '📦'
                );

                return (
                    <Marker
                        key={index}
                        position={[delivery.lat, delivery.lng]}
                        icon={icon}
                    >
                        <Popup>
                            <div className="text-gray-900">
                                <strong>{delivery.name || `Location ${index + 1}`}</strong>
                                {seqPos && <span className="ml-2 text-primary-600">Stop #{seqPos}</span>}
                                <br />
                                <span className="text-sm">
                                    📍 {delivery.lat.toFixed(4)}, {delivery.lng.toFixed(4)}
                                    <br />
                                    ⭐ Priority: {delivery.priority}
                                </span>
                            </div>
                        </Popup>
                    </Marker>
                );
            })}

            {/* Optimized route polylines - following actual roads with distinct colors */}
            {routeSegments && routeSegments.map((segment, index) => {
                const segmentColor = routeColors[index % routeColors.length];
                const fromLabel = index === 0 ? 'Depot' : `Stop ${index}`;
                const toLabel = `Stop ${index + 1}`;
                const segmentDistance = lastResult?.sequence?.[index]?.distance_from_prev;
                const segmentEta = lastResult?.sequence?.[index]?.eta_from_prev;

                return (
                    <Polyline
                        key={index}
                        positions={segment}
                        pathOptions={{
                            color: segmentColor,
                            weight: 6,
                            opacity: 0.9,
                        }}
                    >
                        <Popup>
                            <div className="text-gray-900">
                                <strong style={{ color: segmentColor }}>
                                    Leg {index + 1}: {fromLabel} → {toLabel}
                                </strong>
                                <br />
                                {segmentDistance !== undefined && (
                                    <span className="text-sm">
                                        Distance: {(segmentDistance / 1000).toFixed(2)} km
                                        <br />
                                        ETA: {segmentEta?.toFixed(1)} min
                                    </span>
                                )}
                                {index === 0 && lastResult && (
                                    <>
                                        <hr style={{ margin: '8px 0', borderColor: '#e5e7eb' }} />
                                        <span className="text-sm">
                                            <strong>Total Route:</strong>
                                            <br />
                                            {(lastResult.total_distance / 1000).toFixed(2)} km | {lastResult.total_eta.toFixed(1)} min
                                        </span>
                                    </>
                                )}
                            </div>
                        </Popup>
                    </Polyline>
                );
            })}
        </MapContainer>

        {/* Route Legend - shows color coding for each leg */}
        <RouteLegend sequence={lastResult?.sequence} />
        </div>
    );
};

export default RouteMap;

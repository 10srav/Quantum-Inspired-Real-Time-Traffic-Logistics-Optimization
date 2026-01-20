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

// Map click handler component
const MapClickHandler = () => {
    const { addDelivery } = useRouteStore();
    const { bbox } = VIJAYAWADA_CONFIG;

    useMapEvents({
        click: (e) => {
            const { lat, lng } = e.latlng;

            // Check if within bounds
            if (
                lat >= bbox.lat_min && lat <= bbox.lat_max &&
                lng >= bbox.lng_min && lng <= bbox.lng_max
            ) {
                addDelivery({
                    lat: parseFloat(lat.toFixed(4)),
                    lng: parseFloat(lng.toFixed(4)),
                    priority: 5.0,
                });
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

const RouteMap = () => {
    const {
        currentLocation,
        deliveries,
        lastResult
    } = useRouteStore();

    const { center, bbox, zoom } = VIJAYAWADA_CONFIG;

    // Create route coordinates for polyline - use actual road geometry if available
    const routeSegments = useMemo(() => {
        if (!lastResult?.sequence) return null;

        // If we have road geometry from backend, use it
        if (lastResult.route_geometry && lastResult.route_geometry.length > 0) {
            return lastResult.route_geometry.map(segment =>
                segment.map(coord => [coord[0], coord[1]] as [number, number])
            );
        }

        // Fallback to straight lines if no geometry available
        const coords: [number, number][] = [currentLocation];
        for (const stop of lastResult.sequence) {
            coords.push([stop.delivery.lat, stop.delivery.lng]);
        }
        return [coords]; // Wrap in array to match segments format
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
                                <strong>{delivery.name || `Delivery ${index + 1}`}</strong>
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

            {/* Optimized route polylines - following actual roads */}
            {routeSegments && routeSegments.map((segment, index) => (
                <Polyline
                    key={index}
                    positions={segment}
                    pathOptions={{
                        color: '#667eea',
                        weight: 5,
                        opacity: 0.85,
                    }}
                >
                    {index === 0 && (
                        <Popup>
                            <div className="text-gray-900">
                                <strong>🚀 Optimized Route</strong>
                                <br />
                                {lastResult && (
                                    <span className="text-sm">
                                        Distance: {(lastResult.total_distance / 1000).toFixed(2)} km
                                        <br />
                                        ETA: {lastResult.total_eta.toFixed(1)} min
                                    </span>
                                )}
                            </div>
                        </Popup>
                    )}
                </Polyline>
            ))}
        </MapContainer>
    );
};

export default RouteMap;

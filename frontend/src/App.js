import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Polyline, Popup, Rectangle } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix Leaflet default marker icons
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
});

// Configuration
const API_URL = 'http://localhost:8000';
const VIJAYAWADA_CENTER = [16.5063, 80.6480];
const VIJAYAWADA_BBOX = [[16.5, 80.6], [16.7, 80.7]];

// Custom marker icons
const createIcon = (color, label) => L.divIcon({
  className: 'custom-marker',
  html: `<div style="
    background: ${color};
    color: white;
    padding: 5px 10px;
    border-radius: 20px;
    font-weight: bold;
    font-size: 12px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.3);
    white-space: nowrap;
  ">${label}</div>`,
  iconSize: [40, 20],
  iconAnchor: [20, 10],
});

const depotIcon = createIcon('#dc3545', '📍 Depot');

function App() {
  // State
  const [currentLoc, setCurrentLoc] = useState(VIJAYAWADA_CENTER);
  const [deliveries, setDeliveries] = useState([]);
  const [trafficLevel, setTrafficLevel] = useState('medium');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // New delivery form
  const [newDelivery, setNewDelivery] = useState({
    name: '',
    lat: 16.52,
    lng: 80.65,
    priority: 5
  });

  // Add sample deliveries
  const addSampleDeliveries = () => {
    const samples = [
      { name: 'Delivery A', lat: 16.5175, lng: 80.6198, priority: 2.0 },
      { name: 'Delivery B', lat: 16.5412, lng: 80.6352, priority: 1.0 },
      { name: 'Delivery C', lat: 16.5628, lng: 80.6521, priority: 3.0 },
      { name: 'Delivery D', lat: 16.5890, lng: 80.6705, priority: 1.5 },
    ];
    setDeliveries(prev => {
      const existing = new Set(prev.map(d => `${d.lat},${d.lng}`));
      const newOnes = samples.filter(s => !existing.has(`${s.lat},${s.lng}`));
      return [...prev, ...newOnes];
    });
  };

  // Add delivery
  const addDelivery = () => {
    const delivery = {
      ...newDelivery,
      name: newDelivery.name || `Delivery ${deliveries.length + 1}`
    };
    setDeliveries([...deliveries, delivery]);
    setNewDelivery({ name: '', lat: 16.52, lng: 80.65, priority: 5 });
  };

  // Remove delivery
  const removeDelivery = (index) => {
    setDeliveries(deliveries.filter((_, i) => i !== index));
  };

  // Optimize route
  const optimizeRoute = async () => {
    if (deliveries.length === 0) {
      setError('Add at least one delivery first!');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/optimize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          current_loc: currentLoc,
          deliveries: deliveries,
          traffic_level: trafficLevel,
          include_map: false
        })
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(`Error: ${err.message}. Is the API server running?`);
    } finally {
      setLoading(false);
    }
  };

  // Clear all
  const clearAll = () => {
    setDeliveries([]);
    setResult(null);
    setError(null);
  };

  // Build route coordinates
  const routeCoords = result ? [
    currentLoc,
    ...result.sequence.map(s => [s.delivery.lat, s.delivery.lng])
  ] : [];

  // Styles
  const styles = {
    container: {
      display: 'flex',
      height: '100vh',
      fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
    },
    sidebar: {
      width: '350px',
      background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)',
      color: '#fff',
      padding: '20px',
      overflowY: 'auto',
    },
    header: {
      textAlign: 'center',
      marginBottom: '20px',
    },
    title: {
      fontSize: '1.5rem',
      background: 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
      WebkitBackgroundClip: 'text',
      WebkitTextFillColor: 'transparent',
      marginBottom: '5px',
    },
    section: {
      marginBottom: '20px',
    },
    sectionTitle: {
      fontSize: '1rem',
      marginBottom: '10px',
      paddingBottom: '5px',
      borderBottom: '1px solid rgba(255,255,255,0.1)',
    },
    input: {
      width: '100%',
      padding: '10px',
      marginBottom: '10px',
      border: '1px solid rgba(255,255,255,0.1)',
      borderRadius: '5px',
      background: 'rgba(255,255,255,0.05)',
      color: '#fff',
      fontSize: '14px',
    },
    select: {
      width: '100%',
      padding: '10px',
      border: '1px solid rgba(255,255,255,0.1)',
      borderRadius: '5px',
      background: 'rgba(255,255,255,0.1)',
      color: '#fff',
      fontSize: '14px',
    },
    button: {
      width: '100%',
      padding: '12px',
      border: 'none',
      borderRadius: '5px',
      fontSize: '14px',
      cursor: 'pointer',
      marginBottom: '10px',
      transition: 'all 0.3s',
    },
    primaryButton: {
      background: 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
      color: '#fff',
    },
    secondaryButton: {
      background: 'rgba(255,255,255,0.1)',
      color: '#fff',
    },
    deliveryItem: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '10px',
      background: 'rgba(255,255,255,0.05)',
      borderRadius: '5px',
      marginBottom: '5px',
    },
    removeBtn: {
      cursor: 'pointer',
      color: '#ff6b6b',
      fontWeight: 'bold',
      border: 'none',
      background: 'none',
    },
    metric: {
      display: 'flex',
      justifyContent: 'space-between',
      padding: '8px 0',
    },
    metricValue: {
      fontWeight: 'bold',
      color: '#667eea',
    },
    error: {
      background: 'rgba(220, 53, 69, 0.2)',
      color: '#dc3545',
      padding: '10px',
      borderRadius: '5px',
      marginBottom: '10px',
    },
    success: {
      background: 'rgba(40, 167, 69, 0.2)',
      color: '#28a745',
      padding: '10px',
      borderRadius: '5px',
      marginBottom: '10px',
    },
    mapContainer: {
      flex: 1,
    },
  };

  return (
    <div style={styles.container}>
      {/* Sidebar */}
      <div style={styles.sidebar}>
        <div style={styles.header}>
          <h1 style={styles.title}>🚀 Quantum Traffic Optimizer</h1>
          <p style={{ opacity: 0.8, fontSize: '12px' }}>QUBO/QAOA Route Optimization</p>
        </div>

        {/* Current Location */}
        <div style={styles.section}>
          <h3 style={styles.sectionTitle}>📍 Current Location</h3>
          <input
            type="number"
            style={styles.input}
            placeholder="Latitude"
            value={currentLoc[0]}
            onChange={e => setCurrentLoc([parseFloat(e.target.value), currentLoc[1]])}
            step="0.001"
            min="16.5"
            max="16.7"
          />
          <input
            type="number"
            style={styles.input}
            placeholder="Longitude"
            value={currentLoc[1]}
            onChange={e => setCurrentLoc([currentLoc[0], parseFloat(e.target.value)])}
            step="0.001"
            min="80.6"
            max="80.7"
          />
        </div>

        {/* Add Delivery */}
        <div style={styles.section}>
          <h3 style={styles.sectionTitle}>➕ Add Delivery</h3>
          <input
            type="text"
            style={styles.input}
            placeholder="Name (optional)"
            value={newDelivery.name}
            onChange={e => setNewDelivery({...newDelivery, name: e.target.value})}
          />
          <input
            type="number"
            style={styles.input}
            placeholder="Latitude"
            value={newDelivery.lat}
            onChange={e => setNewDelivery({...newDelivery, lat: parseFloat(e.target.value)})}
            step="0.005"
            min="16.5"
            max="16.7"
          />
          <input
            type="number"
            style={styles.input}
            placeholder="Longitude"
            value={newDelivery.lng}
            onChange={e => setNewDelivery({...newDelivery, lng: parseFloat(e.target.value)})}
            step="0.005"
            min="80.6"
            max="80.7"
          />
          <input
            type="number"
            style={styles.input}
            placeholder="Priority (1-10)"
            value={newDelivery.priority}
            onChange={e => setNewDelivery({...newDelivery, priority: parseFloat(e.target.value)})}
            step="0.5"
            min="1"
            max="10"
          />
          <button
            style={{...styles.button, ...styles.secondaryButton}}
            onClick={addDelivery}
          >
            Add Delivery
          </button>
          <button
            style={{...styles.button, ...styles.secondaryButton}}
            onClick={addSampleDeliveries}
          >
            📦 Add Sample Deliveries
          </button>
        </div>

        {/* Deliveries List */}
        <div style={styles.section}>
          <h3 style={styles.sectionTitle}>📦 Deliveries ({deliveries.length})</h3>
          <div style={{ maxHeight: '150px', overflowY: 'auto' }}>
            {deliveries.map((d, i) => (
              <div key={i} style={styles.deliveryItem}>
                <span>{i + 1}. {d.name} (P: {d.priority})</span>
                <button style={styles.removeBtn} onClick={() => removeDelivery(i)}>✕</button>
              </div>
            ))}
          </div>
        </div>

        {/* Traffic Level */}
        <div style={styles.section}>
          <h3 style={styles.sectionTitle}>🚦 Traffic Level</h3>
          <select
            style={styles.select}
            value={trafficLevel}
            onChange={e => setTrafficLevel(e.target.value)}
          >
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
          </select>
        </div>

        {/* Actions */}
        <button
          style={{...styles.button, ...styles.primaryButton}}
          onClick={optimizeRoute}
          disabled={loading}
        >
          {loading ? '⏳ Optimizing...' : '🚀 Optimize Route'}
        </button>
        <button
          style={{...styles.button, ...styles.secondaryButton}}
          onClick={clearAll}
        >
          🗑️ Clear All
        </button>

        {/* Error */}
        {error && <div style={styles.error}>{error}</div>}

        {/* Results */}
        {result && (
          <div style={styles.section}>
            <div style={styles.success}>✅ Route optimized!</div>
            <h3 style={styles.sectionTitle}>📊 Results</h3>
            <div style={styles.metric}>
              <span>Total Distance</span>
              <span style={styles.metricValue}>{(result.total_distance / 1000).toFixed(2)} km</span>
            </div>
            <div style={styles.metric}>
              <span>Total ETA</span>
              <span style={styles.metricValue}>{result.total_eta.toFixed(1)} min</span>
            </div>
            <div style={styles.metric}>
              <span>vs Greedy</span>
              <span style={styles.metricValue}>{result.improvement_over_greedy?.toFixed(1) || 0}%</span>
            </div>
            <div style={styles.metric}>
              <span>Compute Time</span>
              <span style={styles.metricValue}>{result.optimization_time.toFixed(3)}s</span>
            </div>
          </div>
        )}
      </div>

      {/* Map */}
      <div style={styles.mapContainer}>
        <MapContainer
          center={VIJAYAWADA_CENTER}
          zoom={13}
          style={{ height: '100%', width: '100%' }}
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          
          {/* Bounding box */}
          <Rectangle
            bounds={VIJAYAWADA_BBOX}
            pathOptions={{ color: '#667eea', weight: 2, fill: false, dashArray: '5, 5' }}
          />
          
          {/* Depot marker */}
          <Marker position={currentLoc} icon={depotIcon}>
            <Popup>
              <b>Depot (Start)</b><br />
              {currentLoc[0].toFixed(4)}, {currentLoc[1].toFixed(4)}
            </Popup>
          </Marker>
          
          {/* Delivery markers */}
          {deliveries.map((d, i) => {
            const seqNum = result?.sequence.findIndex(s => 
              Math.abs(s.delivery.lat - d.lat) < 0.0001 && 
              Math.abs(s.delivery.lng - d.lng) < 0.0001
            );
            const label = seqNum >= 0 ? `${seqNum + 1}` : `${i + 1}`;
            const colors = ['#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#e74c3c'];
            
            return (
              <Marker
                key={i}
                position={[d.lat, d.lng]}
                icon={createIcon(colors[i % colors.length], label)}
              >
                <Popup>
                  <b>{d.name}</b><br />
                  Priority: {d.priority}<br />
                  {d.lat.toFixed(4)}, {d.lng.toFixed(4)}
                </Popup>
              </Marker>
            );
          })}
          
          {/* Route polyline */}
          {routeCoords.length > 1 && (
            <Polyline
              positions={routeCoords}
              pathOptions={{ color: '#667eea', weight: 4, opacity: 0.8 }}
            />
          )}
        </MapContainer>
      </div>
    </div>
  );
}

export default App;

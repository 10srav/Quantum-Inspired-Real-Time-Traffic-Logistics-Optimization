"""
Streamlit Frontend for Quantum Traffic Optimization.

This provides an interactive UI for route optimization with map visualization.
Run with: streamlit run frontend/streamlit_app.py
"""

import json
import time
from typing import List, Tuple

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

# Configuration
API_BASE_URL = "http://localhost:8000"
VIJAYAWADA_CENTER = [16.525, 80.65]
VIJAYAWADA_BBOX = {
    "lat_min": 16.50,
    "lat_max": 16.55,
    "lng_min": 80.62,
    "lng_max": 80.68
}

# Page config
st.set_page_config(
    page_title="Quantum Traffic Optimizer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stop-card {
        background: #f8f9fa;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        color: #1a1a1a !important;
    }
    .stop-card b {
        color: #333 !important;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'deliveries' not in st.session_state:
        st.session_state.deliveries = []
    if 'current_loc' not in st.session_state:
        st.session_state.current_loc = VIJAYAWADA_CENTER.copy()
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
    if 'route_history' not in st.session_state:
        st.session_state.route_history = []


def add_delivery(lat: float, lng: float, priority: float, name: str = ""):
    """Add a delivery to the list."""
    delivery = {
        "lat": lat,
        "lng": lng,
        "priority": priority,
        "name": name or f"Delivery {len(st.session_state.deliveries) + 1}"
    }
    st.session_state.deliveries.append(delivery)


def remove_delivery(index: int):
    """Remove a delivery by index."""
    if 0 <= index < len(st.session_state.deliveries):
        st.session_state.deliveries.pop(index)


def call_optimize_api(
    current_loc: Tuple[float, float],
    deliveries: List[dict],
    traffic_level: str
) -> dict:
    """Call the optimization API."""
    payload = {
        "current_loc": list(current_loc),
        "deliveries": deliveries,
        "traffic_level": traffic_level,
        "include_map": False  # We'll render our own map
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/optimize",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure the server is running.")
        st.code("uvicorn src.main:app --reload", language="bash")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API Error: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None


def create_map(current_loc, deliveries, result=None):
    """Create Folium map with current state."""
    m = folium.Map(
        location=current_loc,
        zoom_start=13,
        tiles="OpenStreetMap"
    )
    
    # Add depot marker
    folium.Marker(
        location=current_loc,
        popup="<b>Current Location (Depot)</b>",
        tooltip="Depot",
        icon=folium.Icon(color='red', icon='home', prefix='fa')
    ).add_to(m)
    
    # Add delivery markers
    colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue', 'pink']
    
    for i, delivery in enumerate(deliveries):
        color = colors[i % len(colors)]
        
        # Check if we have optimization result
        seq_num = None
        if result and 'sequence' in result:
            for stop in result['sequence']:
                if (abs(stop['delivery']['lat'] - delivery['lat']) < 0.0001 and
                    abs(stop['delivery']['lng'] - delivery['lng']) < 0.0001):
                    seq_num = stop['position'] + 1
                    break
        
        label = f"Stop {seq_num}" if seq_num else f"Delivery {i + 1}"
        
        folium.Marker(
            location=[delivery['lat'], delivery['lng']],
            popup=f"<b>{delivery.get('name', label)}</b><br>"
                  f"Priority: {delivery['priority']}<br>"
                  f"Coords: ({delivery['lat']:.4f}, {delivery['lng']:.4f})",
            tooltip=label,
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(m)
    
    # Add route polyline if result exists
    if result and 'sequence' in result:
        route_coords = [current_loc]
        for stop in result['sequence']:
            route_coords.append([stop['delivery']['lat'], stop['delivery']['lng']])
        
        folium.PolyLine(
            locations=route_coords,
            weight=4,
            color='blue',
            opacity=0.7,
            popup="Optimized Route"
        ).add_to(m)
    
    # Add bounding box rectangle
    folium.Rectangle(
        bounds=[
            [VIJAYAWADA_BBOX['lat_min'], VIJAYAWADA_BBOX['lng_min']],
            [VIJAYAWADA_BBOX['lat_max'], VIJAYAWADA_BBOX['lng_max']]
        ],
        color='gray',
        weight=1,
        fill=False,
        dash_array='5',
        popup="Vijayawada Service Area"
    ).add_to(m)
    
    return m


def main():
    """Main Streamlit app."""
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🚀 Quantum Traffic Optimizer</h1>', unsafe_allow_html=True)
    st.markdown("*QUBO/QAOA-based delivery route optimization for Vijayawada*")
    
    # Sidebar - Controls
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Current Location
        st.subheader("📍 Current Location")
        col1, col2 = st.columns(2)
        with col1:
            depot_lat = st.number_input(
                "Latitude",
                min_value=VIJAYAWADA_BBOX['lat_min'],
                max_value=VIJAYAWADA_BBOX['lat_max'],
                value=st.session_state.current_loc[0],
                step=0.001,
                format="%.4f",
                key="depot_lat"
            )
        with col2:
            depot_lng = st.number_input(
                "Longitude",
                min_value=VIJAYAWADA_BBOX['lng_min'],
                max_value=VIJAYAWADA_BBOX['lng_max'],
                value=st.session_state.current_loc[1],
                step=0.001,
                format="%.4f",
                key="depot_lng"
            )
        st.session_state.current_loc = [depot_lat, depot_lng]
        
        st.divider()
        
        # Add Delivery
        st.subheader("➕ Add Delivery")
        with st.form("add_delivery_form"):
            new_name = st.text_input("Name (optional)")
            col1, col2 = st.columns(2)
            with col1:
                new_lat = st.number_input(
                    "Lat",
                    min_value=VIJAYAWADA_BBOX['lat_min'],
                    max_value=VIJAYAWADA_BBOX['lat_max'],
                    value=16.52,
                    step=0.005,
                    format="%.4f"
                )
            with col2:
                new_lng = st.number_input(
                    "Lng",
                    min_value=VIJAYAWADA_BBOX['lng_min'],
                    max_value=VIJAYAWADA_BBOX['lng_max'],
                    value=80.65,
                    step=0.005,
                    format="%.4f"
                )
            new_priority = st.slider("Priority", 1.0, 10.0, 5.0, 0.5)
            
            if st.form_submit_button("Add Delivery", use_container_width=True):
                add_delivery(new_lat, new_lng, new_priority, new_name)
                st.rerun()
        
        # Quick add sample deliveries
        if st.button("📦 Add Sample Deliveries", use_container_width=True):
            samples = [
                {"lat": 16.51, "lng": 80.63, "priority": 2.0, "name": "Delivery A"},
                {"lat": 16.52, "lng": 80.64, "priority": 1.0, "name": "Delivery B"},
                {"lat": 16.53, "lng": 80.65, "priority": 3.0, "name": "Delivery C"},
                {"lat": 16.54, "lng": 80.66, "priority": 1.5, "name": "Delivery D"},
            ]
            for s in samples:
                if s not in st.session_state.deliveries:
                    st.session_state.deliveries.append(s)
            st.rerun()
        
        st.divider()
        
        # Traffic Level
        st.subheader("🚦 Traffic Level")
        traffic_level = st.select_slider(
            "Select traffic condition",
            options=["low", "medium", "high"],
            value="medium"
        )
        
        st.divider()
        
        # Optimize Button
        if st.button("🚀 Optimize Route", use_container_width=True, type="primary"):
            if len(st.session_state.deliveries) == 0:
                st.warning("Add at least one delivery first!")
            else:
                with st.spinner("Optimizing route..."):
                    result = call_optimize_api(
                        tuple(st.session_state.current_loc),
                        st.session_state.deliveries,
                        traffic_level
                    )
                    if result:
                        st.session_state.last_result = result
                        st.session_state.route_history.append({
                            "route_id": result.get('route_id'),
                            "n_stops": len(result.get('sequence', [])),
                            "total_eta": result.get('total_eta'),
                            "timestamp": time.strftime("%H:%M:%S")
                        })
                        st.success("✅ Route optimized!")
                        st.rerun()
        
        # Clear buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.deliveries = []
                st.session_state.last_result = None
                st.rerun()
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.current_loc = VIJAYAWADA_CENTER.copy()
                st.rerun()
    
    # Main content
    col_map, col_info = st.columns([2, 1])
    
    with col_map:
        st.subheader("🗺️ Route Map")
        
        # Create and display map
        m = create_map(
            st.session_state.current_loc,
            st.session_state.deliveries,
            st.session_state.last_result
        )
        st_folium(m, width=700, height=500)
    
    with col_info:
        # Deliveries list
        st.subheader(f"📦 Deliveries ({len(st.session_state.deliveries)})")
        
        if st.session_state.deliveries:
            for i, delivery in enumerate(st.session_state.deliveries):
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"""
                        <div class="stop-card">
                            <b>{delivery.get('name', f'Delivery {i+1}')}</b><br>
                            📍 ({delivery['lat']:.4f}, {delivery['lng']:.4f})<br>
                            ⭐ Priority: {delivery['priority']}
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        if st.button("❌", key=f"del_{i}"):
                            remove_delivery(i)
                            st.rerun()
        else:
            st.info("No deliveries added yet. Use the sidebar to add locations.")
        
        # Results
        if st.session_state.last_result:
            st.divider()
            st.subheader("📊 Optimization Results")
            
            result = st.session_state.last_result
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Distance", f"{result.get('total_distance', 0)/1000:.2f} km")
            with col2:
                st.metric("Total ETA", f"{result.get('total_eta', 0):.1f} min")
            with col3:
                improvement = result.get('improvement_over_greedy', 0)
                st.metric("vs Greedy", f"{improvement:+.1f}%")
            
            # Optimization time
            st.caption(f"⏱️ Computed in {result.get('optimization_time', 0):.3f}s")
            
            # Sequence
            st.subheader("🛣️ Optimized Sequence")
            for stop in result.get('sequence', []):
                delivery = stop['delivery']
                st.markdown(f"""
                <div class="stop-card">
                    <b>Stop {stop['position'] + 1}:</b> {delivery.get('name', 'N/A')}<br>
                    📏 +{stop['distance_from_prev']/1000:.2f} km | 
                    ⏱️ +{stop['eta_from_prev']:.1f} min
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.divider()
    st.markdown(
        """
        <div style="text-align: center; color: gray; font-size: 0.8rem;">
            Quantum Traffic Optimizer v1.0.0 | 
            QUBO/QAOA-based Route Optimization | 
            Vijayawada, India
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

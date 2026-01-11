# Quantum-Inspired Real-Time Traffic & Logistics Optimization

A standalone Python/FastAPI application that optimizes multi-delivery sequences/routes for logistics partners under simulated dynamic traffic conditions. Uses QUBO/QAOA via Qiskit classical simulator for NP-hard TSP/VRP solving.

## Features

- **Quantum-Inspired Optimization**: QUBO encoding with QAOA solver (Qiskit)
- **Real-Time Traffic Simulation**: Dynamic congestion modeling
- **Interactive Map Visualization**: Folium-based route display
- **REST API**: FastAPI backend with Swagger documentation
- **React/Streamlit Frontend**: User-friendly interface

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ (for React frontend, optional)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd quantum-traffic-opt

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Start the FastAPI server
uvicorn src.main:app --reload

# Access the API
# Swagger UI: http://localhost:8000/docs
# API: http://localhost:8000
```

### Using the Streamlit Frontend (Alternative)

```bash
streamlit run frontend/streamlit_app.py
```

## API Endpoints

### POST /optimize

Optimize delivery sequence using QAOA.

**Request Body:**
```json
{
  "current_loc": [16.52, 80.63],
  "deliveries": [
    {"lat": 16.54, "lng": 80.65, "priority": 2},
    {"lat": 16.56, "lng": 80.62, "priority": 1},
    {"lat": 16.51, "lng": 80.68, "priority": 3}
  ],
  "traffic_level": "medium"
}
```

**Response:**
```json
{
  "sequence": [...],
  "total_distance": 12.5,
  "total_eta": 25.0,
  "optimization_time": 2.3,
  "route_id": "abc123"
}
```

### GET /map/{route_id}

Get interactive Folium map for a route.

### GET /health

Health check endpoint.

## Example Usage

```bash
# Optimize a route
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "current_loc": [16.5063, 80.6480],
    "deliveries": [
      {"lat": 16.5175, "lng": 80.6198, "priority": 2},
      {"lat": 16.5412, "lng": 80.6352, "priority": 1},
      {"lat": 16.5628, "lng": 80.6521, "priority": 3}
    ],
    "traffic_level": "low"
  }'
```

## Project Structure

```
quantum-traffic-opt/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── Dockerfile               # Container deployment
├── src/
│   ├── main.py              # FastAPI application
│   ├── graph_builder.py     # OSMnx graph operations
│   ├── traffic_sim.py       # Traffic simulation
│   ├── qubo_optimizer.py    # QUBO/QAOA optimization
│   ├── models.py            # Pydantic schemas
│   └── utils.py             # Utility functions
├── frontend/
│   ├── src/App.js           # React frontend
│   └── streamlit_app.py     # Streamlit alternative
├── data/                    # Graph cache
├── tests/                   # Unit & integration tests
└── experiments/             # Benchmarking notebooks
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Optimization time (n=5) | <5 seconds |
| API response time | <6 seconds |
| QAOA quality vs greedy | ≥0% improvement |

## Configuration

Key constants (configurable in respective modules):

- **Vijayawada Bounding Box**: (16.5, 16.7, 80.6, 80.7)
- **Traffic Multipliers**: low=1.0, medium=1.5, high=2.5
- **QAOA Layers**: p=3
- **Adaptive λ**: 2.0 (high traffic), 0.5 (otherwise)

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run performance tests
pytest tests/test_full_system.py -v
```

## Docker Deployment

```bash
# Build the image
docker build -t quantum-traffic-opt .

# Run the container
docker run -p 8000:8000 quantum-traffic-opt

# Access at http://localhost:8000
```

## License

MIT License

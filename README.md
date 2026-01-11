# Quantum-Inspired Real-Time Traffic & Logistics Optimization

A production-grade Python/FastAPI + React/TypeScript application that optimizes multi-delivery sequences/routes for logistics partners under simulated dynamic traffic conditions. Uses QUBO/QAOA via Qiskit classical simulator for NP-hard TSP/VRP solving.

## 🚀 Features

- **Quantum-Inspired Optimization**: QUBO encoding with QAOA solver (Qiskit)
- **Real-Time Traffic Simulation**: Dynamic congestion modeling
- **Interactive Map Visualization**: React-Leaflet with click-to-add markers
- **Modern React Dashboard**: TypeScript, Vite, Tailwind CSS, Zustand
- **REST API**: FastAPI backend with JWT authentication
- **WebSocket Support**: Real-time route updates
- **Production Ready**: Docker, Kubernetes, Prometheus, Grafana

## 📊 Architecture

```
┌────────────────────┐     ┌─────────────────────┐
│  React Dashboard   │────▶│   FastAPI Backend   │
│  (TypeScript)      │◀────│   (Python)          │
└────────────────────┘     └─────────────────────┘
        │                          │
        │                    ┌─────┴─────┐
        │                    ▼           ▼
   ┌────▼────┐         ┌─────────┐  ┌─────────┐
   │ Leaflet │         │PostgreSQL│ │  Redis  │
   │  Maps   │         └─────────┘  └─────────┘
   └─────────┘
```

## 🛠️ Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+ (for React frontend)

### Installation

```bash
# Clone the repository
git clone https://github.com/10srav/Quantum-Inspired-Real-Time-Traffic-Logistics-Optimization.git
cd Quantum-Inspired-Real-Time-Traffic-Logistics-Optimization

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install Python dependencies
pip install -r requirements.txt

# Install React dashboard
cd quantum-traffic-ui
npm install
```

### Running the Application

**Start Backend:**
```bash
uvicorn src.main:app --reload
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

**Start Frontend:**
```bash
cd quantum-traffic-ui
npm run dev
# Dashboard: http://localhost:5173
```

## 📱 React Dashboard

The modern React dashboard provides:

| Feature | Description |
|---------|-------------|
| 🗺️ Interactive Map | Click to add delivery points |
| ⚡ Real-time Updates | WebSocket-based route optimization |
| 🔐 JWT Authentication | Secure API access |
| 📊 Metrics Display | Distance, ETA, improvement stats |
| 🌙 Dark Mode | Glassmorphism UI design |
| 📱 Responsive | Mobile-friendly layout |

### Dashboard Tech Stack

- **React 18** with TypeScript
- **Vite** for blazing fast builds
- **Tailwind CSS** with glassmorphism design
- **Zustand** for state management
- **React-Leaflet** for maps
- **Axios** with JWT interceptors

## 🔌 API Endpoints

### POST /optimize

Optimize delivery sequence using QAOA.

**Request:**
```json
{
  "current_loc": [16.52, 80.63],
  "deliveries": [
    {"lat": 16.54, "lng": 80.65, "priority": 2},
    {"lat": 16.56, "lng": 80.62, "priority": 1}
  ],
  "traffic_level": "medium"
}
```

**Response:**
```json
{
  "route_id": "abc123",
  "sequence": [...],
  "total_distance": 12.5,
  "total_eta": 25.0,
  "improvement_over_greedy": 15.2
}
```

### Other Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/routes` | GET | List routes |
| `/map/{route_id}` | GET | Route map HTML |
| `/reoptimize` | WebSocket | Real-time updates |

## 📁 Project Structure

```
├── src/                      # FastAPI Backend
│   ├── main.py              # Application entry
│   ├── graph_builder.py     # OSMnx graph ops
│   ├── traffic_sim.py       # Traffic simulation
│   ├── qubo_optimizer.py    # QUBO/QAOA core
│   ├── security.py          # JWT authentication
│   └── models.py            # Pydantic schemas
│
├── quantum-traffic-ui/       # React Dashboard
│   ├── src/
│   │   ├── components/      # UI components
│   │   ├── pages/           # Route pages
│   │   ├── services/        # API integration
│   │   ├── stores/          # Zustand stores
│   │   └── types/           # TypeScript types
│   ├── Dockerfile           # Production build
│   └── nginx.conf           # Web server config
│
├── k8s/                      # Kubernetes manifests
├── terraform/                # Infrastructure as Code
├── monitoring/               # Prometheus + Grafana
├── tests/                    # 74+ test cases
└── docker-compose.yml        # Full stack deployment
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Performance tests
pytest tests/test_full_system.py -v
```

## 🐳 Docker Deployment

**Development:**
```bash
# Start full stack
docker-compose up -d

# With React frontend
docker-compose --profile frontend up -d
```

**Production:**
```bash
# Build images
docker-compose build

# Deploy
docker-compose -f docker-compose.yml up -d

# Access:
# - Frontend: http://localhost:3001
# - API: http://localhost:8000
# - Grafana: http://localhost:3000
```

## ☸️ Kubernetes

```bash
# Apply manifests
kubectl apply -f k8s/

# Check pods
kubectl get pods -n quantum-traffic
```

## 📈 Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| Optimization (n=5) | <5s | ✅ ~2.3s |
| API Response | <6s | ✅ ~3.5s |
| QAOA vs Greedy | ≥0% | ✅ ~15% |
| Frontend Build | <10s | ✅ 4.6s |

## 🔧 Configuration

Environment variables (see `.env.example`):

```env
# Backend
DATABASE_URL=postgresql+asyncpg://...
REDIS_URL=redis://...
JWT_SECRET_KEY=your-secret
CORS_ORIGINS=http://localhost:5173

# Frontend
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws
```

## 📜 License

MIT License

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

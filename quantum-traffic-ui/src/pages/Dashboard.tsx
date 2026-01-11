// ================================
// Dashboard Page
// ================================

import Navbar from '../components/Layout/Navbar';
import RouteMap from '../components/Map/RouteMap';
import OptimizationPanel from '../components/Dashboard/OptimizationPanel';
import DeliveryList from '../components/Dashboard/DeliveryList';
import OptimizationResults from '../components/Dashboard/OptimizationResults';

const Dashboard = () => {
    return (
        <div className="min-h-screen">
            {/* Background Effects */}
            <div className="fixed inset-0 pointer-events-none overflow-hidden">
                <div className="bg-orb w-[600px] h-[600px] bg-primary-500 -top-64 -left-64 opacity-10" />
                <div className="bg-orb w-[500px] h-[500px] bg-accent-500 -bottom-64 -right-64 opacity-10" />
            </div>

            {/* Navigation */}
            <Navbar />

            {/* Main Content */}
            <main className="max-w-7xl mx-auto p-6 relative z-10">
                {/* Header */}
                <div className="text-center mb-8">
                    <h1 className="main-header text-4xl mb-2">
                        Route Optimization Dashboard
                    </h1>
                    <p className="text-gray-400">
                        QUBO/QAOA-based delivery route optimization for Vijayawada
                    </p>
                </div>

                {/* Dashboard Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Map Section */}
                    <div className="lg:col-span-2">
                        <div className="glass-card p-4 h-[600px]">
                            <h2 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                                🗺️ Route Map
                                <span className="text-xs text-gray-500 font-normal ml-2">
                                    Click on the map to add delivery points
                                </span>
                            </h2>
                            <div className="h-[calc(100%-40px)] rounded-xl overflow-hidden">
                                <RouteMap />
                            </div>
                        </div>
                    </div>

                    {/* Sidebar */}
                    <div className="space-y-6">
                        {/* Controls */}
                        <div className="glass-card p-5">
                            <OptimizationPanel />
                        </div>

                        {/* Deliveries */}
                        <div className="glass-card p-5">
                            <DeliveryList />
                        </div>
                    </div>
                </div>

                {/* Results Section */}
                <div className="mt-6">
                    <div className="glass-card p-6">
                        <h2 className="text-xl font-semibold text-white mb-4">
                            📊 Optimization Results
                        </h2>
                        <OptimizationResults />
                    </div>
                </div>

                {/* Footer */}
                <footer className="mt-8 text-center text-gray-500 text-sm">
                    <p>
                        Quantum Traffic Optimizer v1.0.0 | QUBO/QAOA-based Route Optimization | Vijayawada, India
                    </p>
                </footer>
            </main>
        </div>
    );
};

export default Dashboard;

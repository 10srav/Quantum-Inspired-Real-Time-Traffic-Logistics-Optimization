// ================================
// Dashboard Page
// ================================

import Navbar from '../components/Layout/Navbar';
import RouteMap from '../components/Map/RouteMap';
import OptimizationPanel from '../components/Dashboard/OptimizationPanel';
import LocationSearch from '../components/Dashboard/LocationSearch';
import DeliveryList from '../components/Dashboard/DeliveryList';
import OptimizationResults from '../components/Dashboard/OptimizationResults';
import AlgorithmComparison from '../components/Dashboard/AlgorithmComparison';
import useRouteStore from '../stores/routeStore';

const Dashboard = () => {
    const { selectionMode } = useRouteStore();

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
                                Route Map
                                <span className={`text-xs font-normal ml-2 ${selectionMode === 'startingLocation' ? 'text-red-400' : 'text-gray-500'}`}>
                                    {selectionMode === 'startingLocation'
                                        ? '📍 Click on map to set STARTING location'
                                        : 'Click on the map to add delivery locations'}
                                </span>
                            </h2>
                            <div className="h-[calc(100%-40px)] rounded-xl overflow-hidden">
                                <RouteMap />
                            </div>
                        </div>
                    </div>

                    {/* Sidebar */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                        {/* Controls */}
                        <div className="glass-card" style={{ padding: '24px' }}>
                            <OptimizationPanel />
                        </div>

                        {/* Location Search */}
                        <div className="glass-card" style={{ padding: '24px' }}>
                            <LocationSearch />
                        </div>

                        {/* Deliveries */}
                        <div className="glass-card" style={{ padding: '24px' }}>
                            <DeliveryList />
                        </div>
                    </div>
                </div>

                {/* Results Section - Two Column Layout */}
                <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Optimization Results */}
                    <div className="glass-card p-6">
                        <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-3">
                            <span style={{
                                width: '32px',
                                height: '32px',
                                borderRadius: '8px',
                                background: 'linear-gradient(135deg, #8b5cf6, #7c3aed)',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                boxShadow: '0 4px 12px rgba(139,92,246,0.3)'
                            }}>
                                <svg style={{ width: '18px', height: '18px', color: 'white' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                                </svg>
                            </span>
                            Optimization Results
                        </h2>
                        <OptimizationResults />
                    </div>

                    {/* Algorithm Comparison */}
                    <div className="glass-card p-6">
                        <AlgorithmComparison />
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

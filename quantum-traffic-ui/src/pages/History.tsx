// ================================
// History Page
// ================================

import { useState } from 'react';
import Navbar from '../components/Layout/Navbar';
import useRouteStore from '../stores/routeStore';

const History = () => {
    const { routeHistory } = useRouteStore();
    const [selectedRoute, setSelectedRoute] = useState<string | null>(null);

    return (
        <div className="min-h-screen">
            {/* Background Effects */}
            <div className="fixed inset-0 pointer-events-none overflow-hidden">
                <div className="bg-orb w-[600px] h-[600px] bg-primary-500 -top-64 -right-64 opacity-10" />
                <div className="bg-orb w-[500px] h-[500px] bg-accent-500 -bottom-64 -left-64 opacity-10" />
            </div>

            <Navbar />

            <main className="max-w-7xl mx-auto p-6 relative z-10">
                {/* Header */}
                <div className="mb-8">
                    <h1 className="main-header text-4xl mb-2">Route History</h1>
                    <p className="text-gray-400">View your past optimization results</p>
                </div>

                {/* History Table */}
                <div className="glass-card overflow-hidden">
                    <table className="w-full">
                        <thead>
                            <tr className="border-b border-white/10">
                                <th className="text-left p-4 text-gray-400 font-medium">Route ID</th>
                                <th className="text-left p-4 text-gray-400 font-medium">Stops</th>
                                <th className="text-left p-4 text-gray-400 font-medium">Total ETA</th>
                                <th className="text-left p-4 text-gray-400 font-medium">Time</th>
                                <th className="text-left p-4 text-gray-400 font-medium">Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {routeHistory.length === 0 ? (
                                <tr>
                                    <td colSpan={5} className="p-8 text-center text-gray-500">
                                        <div className="text-4xl mb-2">📭</div>
                                        <p>No route history yet</p>
                                        <p className="text-sm mt-1">
                                            Optimize some routes to see them here
                                        </p>
                                    </td>
                                </tr>
                            ) : (
                                routeHistory.map((entry) => (
                                    <tr
                                        key={entry.route_id}
                                        className={`border-b border-white/5 transition-colors cursor-pointer
                      ${selectedRoute === entry.route_id
                                                ? 'bg-primary-500/10'
                                                : 'hover:bg-white/5'
                                            }`}
                                        onClick={() => setSelectedRoute(entry.route_id)}
                                    >
                                        <td className="p-4">
                                            <code className="text-primary-400 text-sm">
                                                {entry.route_id.slice(0, 8)}...
                                            </code>
                                        </td>
                                        <td className="p-4 text-white">
                                            {entry.n_stops} stops
                                        </td>
                                        <td className="p-4 text-white">
                                            {entry.total_eta.toFixed(1)} min
                                        </td>
                                        <td className="p-4 text-gray-400">
                                            {entry.timestamp}
                                        </td>
                                        <td className="p-4">
                                            <button className="text-primary-400 hover:text-primary-300 text-sm">
                                                View Details
                                            </button>
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>

                {/* Stats Summary */}
                {routeHistory.length > 0 && (
                    <div className="mt-6 grid grid-cols-3 gap-4">
                        <div className="metric-card">
                            <div className="value">{routeHistory.length}</div>
                            <div className="label">Total Routes</div>
                        </div>
                        <div className="metric-card">
                            <div className="value">
                                {routeHistory.reduce((sum, r) => sum + r.n_stops, 0)}
                            </div>
                            <div className="label">Total Deliveries</div>
                        </div>
                        <div className="metric-card">
                            <div className="value">
                                {(routeHistory.reduce((sum, r) => sum + r.total_eta, 0) / routeHistory.length).toFixed(1)}
                            </div>
                            <div className="label">Avg ETA (min)</div>
                        </div>
                    </div>
                )}
            </main>
        </div>
    );
};

export default History;

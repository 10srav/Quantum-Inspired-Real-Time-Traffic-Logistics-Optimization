// ================================
// Algorithm Comparison Section
// ================================

import useRouteStore from '../../stores/routeStore';
import ComparisonChart from './ComparisonChart';

const AlgorithmComparison = () => {
    const {
        deliveries,
        compareRoutes,
        isComparing,
        comparisonResult,
        clearComparison,
    } = useRouteStore();

    return (
        <div>
            <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-3">
                <span style={{
                    width: '32px',
                    height: '32px',
                    borderRadius: '8px',
                    background: 'linear-gradient(135deg, #10b981, #059669)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    boxShadow: '0 4px 12px rgba(16,185,129,0.3)'
                }}>
                    <svg style={{ width: '18px', height: '18px', color: 'white' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                </span>
                Algorithm Comparison
            </h2>

            {/* Show comparison results if available */}
            {comparisonResult ? (
                <ComparisonChart
                    data={comparisonResult}
                    onClose={clearComparison}
                    embedded={true}
                />
            ) : (
                <div style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    padding: '40px 20px',
                    background: 'rgba(31, 41, 55, 0.3)',
                    borderRadius: '12px',
                    border: '1px dashed rgba(75, 85, 99, 0.5)',
                }}>
                    <svg
                        style={{ width: '48px', height: '48px', color: '#6b7280', marginBottom: '16px' }}
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                    >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                    <p style={{ color: '#9ca3af', marginBottom: '20px', textAlign: 'center' }}>
                        Compare QAOA, Greedy, Simulated Annealing, and Brute Force algorithms
                    </p>
                    <button
                        onClick={() => compareRoutes(false)}
                        disabled={isComparing || deliveries.length === 0}
                        style={{
                            padding: '14px 28px',
                            borderRadius: '12px',
                            background: deliveries.length === 0
                                ? '#374151'
                                : 'linear-gradient(135deg, #10b981, #059669)',
                            color: 'white',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            gap: '10px',
                            fontSize: '15px',
                            fontWeight: 600,
                            border: 'none',
                            cursor: deliveries.length === 0 ? 'not-allowed' : 'pointer',
                            transition: 'all 0.2s',
                            opacity: deliveries.length === 0 ? 0.6 : 1,
                            boxShadow: deliveries.length === 0 ? 'none' : '0 8px 24px rgba(16,185,129,0.3)',
                        }}
                    >
                        {isComparing ? (
                            <>
                                <div style={{
                                    width: '18px',
                                    height: '18px',
                                    border: '2px solid rgba(255,255,255,0.3)',
                                    borderTopColor: 'white',
                                    borderRadius: '50%',
                                    animation: 'spin 1s linear infinite'
                                }} />
                                Comparing Algorithms...
                            </>
                        ) : (
                            <>
                                <svg style={{ width: '18px', height: '18px' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                                </svg>
                                {deliveries.length === 0
                                    ? 'Add locations first'
                                    : `Compare Algorithms (${deliveries.length} stops)`
                                }
                            </>
                        )}
                    </button>
                </div>
            )}
        </div>
    );
};

export default AlgorithmComparison;

import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Cell,
} from 'recharts';
import type { CompareResult } from '../../types';

interface ComparisonChartProps {
    data: CompareResult;
    onClose: () => void;
    embedded?: boolean;
}

const SOLVER_COLORS: Record<string, string> = {
    greedy: '#6b7280',
    greedy_2opt: '#3b82f6',
    simulated_annealing: '#8b5cf6',
    brute_force: '#10b981',
    qaoa: '#f59e0b',
};

const formatSolverName = (name: string): string => {
    return name
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
};

const ComparisonChart = ({ data, onClose, embedded = false }: ComparisonChartProps) => {
    const successfulSolvers = data.solvers.filter(s => s.success);

    const timeData = successfulSolvers.map(s => ({
        name: formatSolverName(s.name),
        value: s.solve_time,
        color: SOLVER_COLORS[s.name] || '#60a5fa',
    }));

    const costData = successfulSolvers.map(s => ({
        name: formatSolverName(s.name),
        value: s.cost,
        color: SOLVER_COLORS[s.name] || '#60a5fa',
    }));

    const distanceData = successfulSolvers.map(s => ({
        name: formatSolverName(s.name),
        value: s.distance,
        color: SOLVER_COLORS[s.name] || '#60a5fa',
    }));

    const improvementData = Object.entries(data.improvements).map(([key, value]) => ({
        name: formatSolverName(key.replace('_vs_greedy', '')),
        value: value,
        color: value >= 0 ? '#10b981' : '#ef4444',
    }));

    // Content rendering (shared between modal and embedded)
    const chartContent = (
        <>
            {/* Header for embedded mode */}
            {embedded && (
                <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: '20px',
                }}>
                    <p style={{ color: '#9ca3af', fontSize: '14px', margin: 0 }}>
                        {data.problem_size} locations • {data.traffic_level} traffic
                    </p>
                    <button
                        onClick={onClose}
                        style={{
                            background: 'rgba(239,68,68,0.15)',
                            border: '1px solid rgba(239,68,68,0.3)',
                            borderRadius: '8px',
                            padding: '6px 12px',
                            color: '#f87171',
                            cursor: 'pointer',
                            fontSize: '13px',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '6px',
                        }}
                    >
                        <svg width="14" height="14" viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                        </svg>
                        Clear
                    </button>
                </div>
            )}

            {/* Best Solver Badge */}
            {data.best_solver && (
                <div style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '10px 16px',
                    borderRadius: '10px',
                    background: 'rgba(16,185,129,0.15)',
                    border: '1px solid rgba(16,185,129,0.4)',
                    marginBottom: '24px',
                }}>
                    <svg width="16" height="16" viewBox="0 0 20 20" fill="#10b981">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                    <span style={{ color: '#10b981', fontWeight: 500 }}>
                        Best: {formatSolverName(data.best_solver)}
                    </span>
                </div>
            )}

            {/* Charts Grid */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: embedded ? 'repeat(auto-fit, minmax(300px, 1fr))' : 'repeat(auto-fit, minmax(380px, 1fr))',
                gap: '24px',
            }}>
                {/* Execution Time Chart */}
                <div style={{
                    background: 'rgba(31,41,55,0.5)',
                    borderRadius: '12px',
                    padding: '16px',
                    border: '1px solid rgba(75,85,99,0.3)',
                }}>
                    <h4 style={{ color: 'white', fontSize: '14px', fontWeight: 500, marginBottom: '16px' }}>
                        Execution Time (seconds)
                    </h4>
                    <ResponsiveContainer width="100%" height={180}>
                        <BarChart data={timeData} layout="vertical">
                            <CartesianGrid strokeDasharray="3 3" stroke="#374151" horizontal={false} />
                            <XAxis type="number" stroke="#9ca3af" fontSize={11} tickFormatter={(v) => v.toFixed(3)} />
                            <YAxis type="category" dataKey="name" stroke="#9ca3af" fontSize={11} width={100} />
                            <Tooltip
                                contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
                                labelStyle={{ color: '#fff' }}
                                formatter={(value: number) => [`${value.toFixed(4)}s`, 'Time']}
                            />
                            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                                {timeData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={entry.color} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>

                {/* Route Distance Chart */}
                <div style={{
                    background: 'rgba(31,41,55,0.5)',
                    borderRadius: '12px',
                    padding: '16px',
                    border: '1px solid rgba(75,85,99,0.3)',
                }}>
                    <h4 style={{ color: 'white', fontSize: '14px', fontWeight: 500, marginBottom: '16px' }}>
                        Route Distance (meters)
                    </h4>
                    <ResponsiveContainer width="100%" height={180}>
                        <BarChart data={distanceData} layout="vertical">
                            <CartesianGrid strokeDasharray="3 3" stroke="#374151" horizontal={false} />
                            <XAxis type="number" stroke="#9ca3af" fontSize={11} tickFormatter={(v) => v.toFixed(0)} />
                            <YAxis type="category" dataKey="name" stroke="#9ca3af" fontSize={11} width={100} />
                            <Tooltip
                                contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
                                labelStyle={{ color: '#fff' }}
                                formatter={(value: number) => [`${value.toFixed(1)}m`, 'Distance']}
                            />
                            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                                {distanceData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={entry.color} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>

                {/* QUBO Cost Chart */}
                <div style={{
                    background: 'rgba(31,41,55,0.5)',
                    borderRadius: '12px',
                    padding: '16px',
                    border: '1px solid rgba(75,85,99,0.3)',
                }}>
                    <h4 style={{ color: 'white', fontSize: '14px', fontWeight: 500, marginBottom: '16px' }}>
                        QUBO Cost (lower is better)
                    </h4>
                    <ResponsiveContainer width="100%" height={180}>
                        <BarChart data={costData} layout="vertical">
                            <CartesianGrid strokeDasharray="3 3" stroke="#374151" horizontal={false} />
                            <XAxis type="number" stroke="#9ca3af" fontSize={11} />
                            <YAxis type="category" dataKey="name" stroke="#9ca3af" fontSize={11} width={100} />
                            <Tooltip
                                contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
                                labelStyle={{ color: '#fff' }}
                                formatter={(value: number) => [value.toFixed(2), 'Cost']}
                            />
                            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                                {costData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={entry.color} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>

                {/* Improvement vs Greedy Chart */}
                {improvementData.length > 0 && (
                    <div style={{
                        background: 'rgba(31,41,55,0.5)',
                        borderRadius: '12px',
                        padding: '16px',
                        border: '1px solid rgba(75,85,99,0.3)',
                    }}>
                        <h4 style={{ color: 'white', fontSize: '14px', fontWeight: 500, marginBottom: '16px' }}>
                            Improvement vs Greedy (%)
                        </h4>
                        <ResponsiveContainer width="100%" height={180}>
                            <BarChart data={improvementData} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" stroke="#374151" horizontal={false} />
                                <XAxis type="number" stroke="#9ca3af" fontSize={11} domain={['auto', 'auto']} />
                                <YAxis type="category" dataKey="name" stroke="#9ca3af" fontSize={11} width={100} />
                                <Tooltip
                                    contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
                                    labelStyle={{ color: '#fff' }}
                                    formatter={(value: number) => [`${value.toFixed(2)}%`, 'Improvement']}
                                />
                                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                                    {improvementData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.color} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                )}
            </div>

            {/* Solver Details Table */}
            <div style={{
                marginTop: '24px',
                background: 'rgba(31,41,55,0.5)',
                borderRadius: '12px',
                padding: '16px',
                border: '1px solid rgba(75,85,99,0.3)',
                overflowX: 'auto',
            }}>
                <h4 style={{ color: 'white', fontSize: '14px', fontWeight: 500, marginBottom: '16px' }}>
                    Detailed Results
                </h4>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                        <tr style={{ borderBottom: '1px solid #374151' }}>
                            <th style={{ textAlign: 'left', padding: '8px 12px', color: '#9ca3af', fontSize: '12px', fontWeight: 500 }}>Solver</th>
                            <th style={{ textAlign: 'right', padding: '8px 12px', color: '#9ca3af', fontSize: '12px', fontWeight: 500 }}>Time (s)</th>
                            <th style={{ textAlign: 'right', padding: '8px 12px', color: '#9ca3af', fontSize: '12px', fontWeight: 500 }}>Distance (m)</th>
                            <th style={{ textAlign: 'right', padding: '8px 12px', color: '#9ca3af', fontSize: '12px', fontWeight: 500 }}>QUBO Cost</th>
                            <th style={{ textAlign: 'center', padding: '8px 12px', color: '#9ca3af', fontSize: '12px', fontWeight: 500 }}>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {data.solvers.map((solver) => (
                            <tr key={solver.name} style={{ borderBottom: '1px solid rgba(55,65,81,0.5)' }}>
                                <td style={{ padding: '10px 12px' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                        <span style={{
                                            width: '10px',
                                            height: '10px',
                                            borderRadius: '50%',
                                            background: SOLVER_COLORS[solver.name] || '#60a5fa',
                                        }} />
                                        <span style={{ color: 'white', fontSize: '13px' }}>
                                            {formatSolverName(solver.name)}
                                        </span>
                                        {solver.name === data.best_solver && (
                                            <span style={{
                                                fontSize: '10px',
                                                padding: '2px 6px',
                                                borderRadius: '4px',
                                                background: 'rgba(16,185,129,0.2)',
                                                color: '#10b981',
                                            }}>
                                                Best
                                            </span>
                                        )}
                                    </div>
                                </td>
                                <td style={{ textAlign: 'right', padding: '10px 12px', color: '#d1d5db', fontSize: '13px', fontFamily: 'monospace' }}>
                                    {solver.success ? solver.solve_time.toFixed(4) : '-'}
                                </td>
                                <td style={{ textAlign: 'right', padding: '10px 12px', color: '#d1d5db', fontSize: '13px', fontFamily: 'monospace' }}>
                                    {solver.success ? solver.distance.toFixed(1) : '-'}
                                </td>
                                <td style={{ textAlign: 'right', padding: '10px 12px', color: '#d1d5db', fontSize: '13px', fontFamily: 'monospace' }}>
                                    {solver.success ? solver.cost.toFixed(2) : '-'}
                                </td>
                                <td style={{ textAlign: 'center', padding: '10px 12px' }}>
                                    {solver.success ? (
                                        <span style={{ color: '#10b981', fontSize: '12px' }}>Success</span>
                                    ) : (
                                        <span style={{ color: '#f87171', fontSize: '12px' }} title={solver.error || ''}>
                                            Failed
                                        </span>
                                    )}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </>
    );

    // Embedded mode: render inline
    if (embedded) {
        return <div>{chartContent}</div>;
    }

    // Modal mode: render with overlay
    return (
        <div style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0,0,0,0.7)',
            backdropFilter: 'blur(4px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            padding: '20px',
        }}>
            <div style={{
                background: 'linear-gradient(135deg, rgba(17,24,39,0.98), rgba(31,41,55,0.95))',
                borderRadius: '20px',
                border: '1px solid rgba(75,85,99,0.5)',
                maxWidth: '900px',
                width: '100%',
                maxHeight: '90vh',
                overflow: 'auto',
                padding: '24px',
            }}>
                {/* Header */}
                <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: '24px',
                }}>
                    <div>
                        <h2 style={{ color: 'white', fontSize: '20px', fontWeight: 600, margin: 0 }}>
                            Algorithm Comparison
                        </h2>
                        <p style={{ color: '#9ca3af', fontSize: '14px', marginTop: '4px' }}>
                            {data.problem_size} locations • {data.traffic_level} traffic
                        </p>
                    </div>
                    <button
                        onClick={onClose}
                        style={{
                            background: 'rgba(75,85,99,0.5)',
                            border: 'none',
                            borderRadius: '8px',
                            padding: '8px 12px',
                            color: '#9ca3af',
                            cursor: 'pointer',
                            fontSize: '14px',
                        }}
                    >
                        Close
                    </button>
                </div>
                {chartContent}
            </div>
        </div>
    );
};

export default ComparisonChart;

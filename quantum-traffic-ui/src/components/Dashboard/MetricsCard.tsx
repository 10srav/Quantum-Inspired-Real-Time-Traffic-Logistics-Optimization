// ================================
// Metrics Card Component
// ================================

interface MetricsCardProps {
    icon: string;
    label: string;
    value: string | number;
    unit?: string;
    trend?: number; // + for improvement, - for worse
    className?: string;
}

const MetricsCard = ({
    icon,
    label,
    value,
    unit,
    trend,
    className = ''
}: MetricsCardProps) => {
    return (
        <div className={`metric-card ${className}`}>
            <div className="text-3xl mb-3">{icon}</div>
            <div className="value">
                {value}
                {unit && <span className="text-xl ml-1">{unit}</span>}
            </div>
            <div className="label">{label}</div>
            {trend !== undefined && (
                <div className={`mt-2 text-sm font-medium ${trend > 0 ? 'text-green-400' : trend < 0 ? 'text-red-400' : 'text-gray-400'
                    }`}>
                    {trend > 0 ? '↑' : trend < 0 ? '↓' : '→'} {Math.abs(trend).toFixed(1)}%
                </div>
            )}
        </div>
    );
};

export default MetricsCard;

// ================================
// Navbar Component
// ================================

import { Link, useLocation } from 'react-router-dom';
import useAuthStore from '../../stores/authStore';

const Navbar = () => {
    const location = useLocation();
    const { user, logout, isAuthenticated } = useAuthStore();

    const navLinks = [
        { path: '/', label: 'Dashboard', icon: '🗺️' },
        { path: '/history', label: 'History', icon: '📊' },
    ];

    return (
        <nav className="glass-card rounded-none border-b border-white/10 px-6 py-4">
            <div className="max-w-7xl mx-auto flex items-center justify-between">
                {/* Logo */}
                <Link to="/" className="flex items-center gap-3">
                    <span className="text-2xl">🚀</span>
                    <span className="main-header text-xl">Quantum Traffic</span>
                </Link>

                {/* Nav Links */}
                <div className="flex items-center gap-6">
                    {navLinks.map((link) => (
                        <Link
                            key={link.path}
                            to={link.path}
                            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-200
                ${location.pathname === link.path
                                    ? 'bg-primary-500/20 text-primary-400'
                                    : 'text-gray-400 hover:text-white hover:bg-white/5'
                                }`}
                        >
                            <span>{link.icon}</span>
                            <span className="font-medium">{link.label}</span>
                        </Link>
                    ))}
                </div>

                {/* User Menu */}
                <div className="flex items-center gap-4">
                    {isAuthenticated ? (
                        <>
                            <span className="text-sm text-gray-400">
                                👤 {user?.username || 'User'}
                            </span>
                            <button
                                onClick={logout}
                                className="text-sm text-gray-400 hover:text-white transition-colors"
                            >
                                Logout
                            </button>
                        </>
                    ) : (
                        <Link
                            to="/login"
                            className="btn-gradient px-4 py-2 text-sm"
                        >
                            Sign In
                        </Link>
                    )}
                </div>
            </div>
        </nav>
    );
};

export default Navbar;

// ================================
// Login Form Component
// ================================

import { useState, type FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import useAuthStore from '../../stores/authStore';

const LoginForm = () => {
    const navigate = useNavigate();
    const { login, isLoading, error, clearError } = useAuthStore();

    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');

    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();
        clearError();

        const success = await login({ username, password });
        if (success) {
            navigate('/');
        }
    };

    return (
        <div className="min-h-screen flex items-center justify-center p-4">
            {/* Background orbs */}
            <div className="bg-orb w-96 h-96 bg-primary-500 -top-48 -left-48" />
            <div className="bg-orb w-80 h-80 bg-accent-500 -bottom-40 -right-40" />

            <div className="glass-card w-full max-w-md p-8 relative z-10 animate-slide-in">
                {/* Logo */}
                <div className="text-center mb-8">
                    <div className="text-6xl mb-4">🚀</div>
                    <h1 className="main-header text-3xl">Quantum Traffic</h1>
                    <p className="text-gray-400 mt-2">QUBO/QAOA Route Optimization</p>
                </div>

                {/* Form */}
                <form onSubmit={handleSubmit} className="space-y-6">
                    <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                            Username
                        </label>
                        <input
                            type="text"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            className="input-modern"
                            placeholder="Enter username"
                            required
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                            Password
                        </label>
                        <input
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            className="input-modern"
                            placeholder="Enter password"
                            required
                        />
                    </div>

                    {error && (
                        <div className="text-red-400 text-sm bg-red-500/10 p-3 rounded-lg border border-red-500/20">
                            {error}
                        </div>
                    )}

                    <button
                        type="submit"
                        disabled={isLoading}
                        className="btn-gradient w-full disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {isLoading ? (
                            <span className="flex items-center justify-center gap-2">
                                <div className="loading-spinner w-5 h-5" />
                                Signing in...
                            </span>
                        ) : (
                            'Sign In'
                        )}
                    </button>
                </form>

                {/* Demo mode hint */}
                <div className="mt-6 text-center text-sm text-gray-500">
                    <p>Demo mode: Use any credentials to continue</p>
                </div>
            </div>
        </div>
    );
};

export default LoginForm;

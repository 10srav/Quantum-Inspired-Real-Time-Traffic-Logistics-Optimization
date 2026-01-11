// ================================
// Login Page
// ================================

import LoginForm from '../components/Auth/LoginForm';

const Login = () => {
    return (
        <div className="min-h-screen relative overflow-hidden">
            {/* Animated background */}
            <div className="absolute inset-0">
                <div className="absolute inset-0 bg-gradient-to-br from-primary-950 via-gray-900 to-accent-950" />

                {/* Animated orbs */}
                <div className="bg-orb w-96 h-96 bg-primary-500 top-20 left-20" style={{ animationDelay: '0s' }} />
                <div className="bg-orb w-80 h-80 bg-accent-500 bottom-20 right-20" style={{ animationDelay: '2s' }} />
                <div className="bg-orb w-64 h-64 bg-blue-500 top-1/2 left-1/2" style={{ animationDelay: '4s' }} />
            </div>

            {/* Login Form */}
            <div className="relative z-10">
                <LoginForm />
            </div>
        </div>
    );
};

export default Login;

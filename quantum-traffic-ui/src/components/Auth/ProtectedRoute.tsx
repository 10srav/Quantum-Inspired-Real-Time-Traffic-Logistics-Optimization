// ================================
// Protected Route Component
// ================================

// Note: For demo mode, we allow unauthenticated access.
// In production, uncomment the imports and redirect logic below.

// import { Navigate, useLocation } from 'react-router-dom';
// import useAuthStore from '../../stores/authStore';

interface ProtectedRouteProps {
    children: React.ReactNode;
}

const ProtectedRoute = ({ children }: ProtectedRouteProps) => {
    // Demo mode: allow all access
    // const { isAuthenticated } = useAuthStore();
    // const location = useLocation();
    // if (!isAuthenticated) {
    //   return <Navigate to="/login" state={{ from: location }} replace />;
    // }

    return <>{children}</>;
};

export default ProtectedRoute;


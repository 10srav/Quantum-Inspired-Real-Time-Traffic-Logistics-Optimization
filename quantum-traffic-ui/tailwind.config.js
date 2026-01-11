/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    darkMode: 'class',
    theme: {
        extend: {
            colors: {
                primary: {
                    50: '#f0f5ff',
                    100: '#e0ebff',
                    200: '#c7d7fe',
                    300: '#a4bcfd',
                    400: '#7c9afc',
                    500: '#667eea',
                    600: '#4f51d0',
                    700: '#4240a8',
                    800: '#383887',
                    900: '#32356e',
                    950: '#1f1f42',
                },
                accent: {
                    50: '#fdf4ff',
                    100: '#fae8ff',
                    200: '#f5d0fe',
                    300: '#f0abfc',
                    400: '#e879f9',
                    500: '#764ba2',
                    600: '#a855f7',
                    700: '#9333ea',
                    800: '#7e22ce',
                    900: '#581c87',
                    950: '#3b0764',
                },
            },
            fontFamily: {
                sans: ['Inter', 'system-ui', 'sans-serif'],
            },
            animation: {
                'gradient': 'gradient 8s ease infinite',
                'pulse-slow': 'pulse 3s ease-in-out infinite',
                'slide-in': 'slideIn 0.3s ease-out',
            },
            keyframes: {
                gradient: {
                    '0%, 100%': { backgroundPosition: '0% 50%' },
                    '50%': { backgroundPosition: '100% 50%' },
                },
                slideIn: {
                    '0%': { transform: 'translateY(-10px)', opacity: '0' },
                    '100%': { transform: 'translateY(0)', opacity: '1' },
                },
            },
            backgroundImage: {
                'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
                'gradient-conic': 'conic-gradient(var(--tw-gradient-stops))',
            },
            boxShadow: {
                'glass': '0 8px 32px 0 rgba(31, 38, 135, 0.15)',
                'glow': '0 0 20px rgba(102, 126, 234, 0.3)',
            },
        },
    },
    plugins: [],
}

/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        clinical: {
          50: '#f1f6fb',
          100: '#e2edf6',
          200: '#c3daec',
          300: '#93bcdc',
          400: '#5c98c7',
          500: '#387bae',
          600: '#296190',
          700: '#234e74',
          800: '#214361',
          900: '#1f3a52',
          950: '#142639',
        },
        signal: {
          teal: '#0d9488',
          amber: '#b45309',
          rose: '#be123c',
          violet: '#6d28d9',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
        mono: ['JetBrains Mono', 'SFMono-Regular', 'Consolas', 'monospace'],
      },
      keyframes: {
        'fade-up': {
          '0%': { opacity: '0', transform: 'translateY(6px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        shimmer: {
          '100%': { transform: 'translateX(100%)' },
        },
      },
      animation: {
        'fade-up': 'fade-up 0.24s ease-out both',
        shimmer: 'shimmer 1.6s infinite',
      },
    },
  },
  plugins: [],
};

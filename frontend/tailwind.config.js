/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        legal: { 900: '#0f172a', 800: '#1e293b', 700: '#334155', accent: '#2563eb' },
      },
    },
  },
  plugins: [],
}

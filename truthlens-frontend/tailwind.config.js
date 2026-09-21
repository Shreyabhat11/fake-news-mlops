/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: {
          950: '#080B10',
          900: '#0A0E14',
          800: '#0F141C',
          700: '#151B25',
          600: '#1C2430',
          border: 'rgba(232, 234, 237, 0.08)',
          borderHover: 'rgba(232, 234, 237, 0.16)',
        },
        text: {
          primary: '#E8EAED',
          secondary: '#A6AEBC',
          muted: '#6C7686',
        },
        signal: {
          real: '#3DDC97',
          realDim: '#1F5C42',
          fake: '#F2665A',
          fakeDim: '#5C2622',
        },
        brand: {
          DEFAULT: '#7C8CF8',
          dim: '#4A4F8C',
        },
      },
      fontFamily: {
        display: ['"Fraunces"', 'Georgia', 'serif'],
        sans: ['"Inter"', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
      boxShadow: {
        card: '0 1px 0 rgba(255,255,255,0.04) inset, 0 20px 40px -24px rgba(0,0,0,0.6)',
      },
      keyframes: {
        fillBar: {
          '0%': { width: '0%' },
        },
        pulseSoft: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.55' },
        },
        scan: {
          '0%': { backgroundPosition: '0% 0%' },
          '100%': { backgroundPosition: '0% -200%' },
        },
      },
      animation: {
        pulseSoft: 'pulseSoft 1.6s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}

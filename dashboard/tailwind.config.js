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
        cf: {
          blue: '#1a5cff',
          red: '#cc0000',
          green: '#008000',
          gray: '#808080',
          lightgray: '#e1e1e1',
          darkgray: '#cccccc',
          bg: '#ffffff',
          'bg-dark': '#1e1e1e',
          'card-dark': '#2d2d2d',
          'border-dark': '#404040',
        }
      },
      fontFamily: {
        sans: ['Verdana', 'Arial', 'sans-serif'],
        mono: ['Consolas', 'Monaco', 'monospace'],
      },
    },
  },
  plugins: [],
}

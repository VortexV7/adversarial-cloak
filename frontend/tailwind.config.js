/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: { sans: ["Inter", "system-ui", "sans-serif"] },
      colors: {
        ink: { 950: "#070d1a", 900: "#0b1526", 800: "#12203a", 700: "#1b2f4d" },
        teal: { accent: "#2dd4bf" },
      },
      borderRadius: { card: "12px" },
    },
  },
  plugins: [],
};

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:5000", // backend server
        changeOrigin: true,             // makes CORS happy
        secure: false,                  // allow self-signed certs if any
      },
    },
  },
});

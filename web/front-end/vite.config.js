import { defineConfig, loadEnv } from "vite";
import vue from "@vitejs/plugin-vue";

// Nếu back-end chạy ở địa chỉ khác (vd. http://localhost:8000), đặt VITE_API_TARGET trong .env
// để dev server chuyển tiếp /api sang đó (tránh lỗi CORS khi phát triển).
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  return {
    plugins: [vue()],
    server: env.VITE_API_TARGET ? { proxy: { "/api": { target: env.VITE_API_TARGET, changeOrigin: true } } } : {},
  };
});

import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [tailwindcss(), sveltekit()],
	server: {
		proxy: {
			// Proxy API calls to FastAPI during development
			'/api': {
				target: 'http://localhost:8000',
				changeOrigin: true,
			}
		}
	}
});

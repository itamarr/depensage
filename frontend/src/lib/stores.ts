/**
 * Global Svelte stores.
 */
import { writable } from 'svelte/store';

// Current pipeline session ID
export const sessionId = writable<string | null>(null);

// Pipeline status
export const pipelineStatus = writable<string>('idle');

// Categories cache (fetched from backend)
export const categories = writable<Record<string, string[]>>({});

// Auth state
export const isAuthenticated = writable<boolean>(false);

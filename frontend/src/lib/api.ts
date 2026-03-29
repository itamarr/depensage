/**
 * API client for the DepenSage backend.
 */

const BASE = '/api';

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
	const res = await fetch(`${BASE}${path}`, {
		credentials: 'include',
		headers: {
			'Content-Type': 'application/json',
			...((options.headers as Record<string, string>) || {}),
		},
		...options,
	});

	if (res.status === 401) {
		// Redirect to login
		window.location.href = '/login';
		throw new Error('Not authenticated');
	}

	if (!res.ok) {
		const body = await res.json().catch(() => ({ detail: res.statusText }));
		throw new Error(body.detail || `HTTP ${res.status}`);
	}

	return res.json();
}

export async function get<T>(path: string): Promise<T> {
	return request<T>(path);
}

export async function post<T>(path: string, body?: unknown): Promise<T> {
	return request<T>(path, {
		method: 'POST',
		body: body ? JSON.stringify(body) : undefined,
	});
}

export async function put<T>(path: string, body: unknown): Promise<T> {
	return request<T>(path, {
		method: 'PUT',
		body: JSON.stringify(body),
	});
}

export async function del<T>(path: string): Promise<T> {
	return request<T>(path, { method: 'DELETE' });
}

export async function uploadFiles(files: FileList): Promise<{ session_id: string; files: string[] }> {
	const formData = new FormData();
	for (const file of files) {
		formData.append('files', file);
	}

	const res = await fetch(`${BASE}/pipeline/upload`, {
		method: 'POST',
		credentials: 'include',
		body: formData,
	});

	if (res.status === 401) {
		window.location.href = '/login';
		throw new Error('Not authenticated');
	}
	if (!res.ok) {
		const body = await res.json().catch(() => ({ detail: res.statusText }));
		throw new Error(body.detail || `HTTP ${res.status}`);
	}

	return res.json();
}

export function subscribeProgress(
	sessionId: string,
	onEvent: (data: { stage: string; percent: number; error: string | null }) => void,
	onDone: () => void,
): EventSource {
	const source = new EventSource(`${BASE}/pipeline/${sessionId}/progress`);
	source.onmessage = (event) => {
		const data = JSON.parse(event.data);
		onEvent(data);
		if (data.stage === 'complete' || data.stage === 'error') {
			source.close();
			onDone();
		}
	};
	source.onerror = () => {
		source.close();
		onDone();
	};
	return source;
}

export async function login(password: string): Promise<{ ok: boolean; message: string }> {
	const res = await fetch(`${BASE}/system/auth`, {
		method: 'POST',
		credentials: 'include',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ password }),
	});
	return res.json();
}

export async function checkAuth(): Promise<boolean> {
	try {
		// Hit an authenticated endpoint to verify session
		const res = await fetch(`${BASE}/system/config`, { credentials: 'include' });
		return res.ok;
	} catch {
		return false;
	}
}

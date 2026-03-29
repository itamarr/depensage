<script lang="ts">
	import { login } from '$lib/api';
	import { goto } from '$app/navigation';
	import { isAuthenticated } from '$lib/stores';

	let password = $state('');
	let error = $state('');
	let loading = $state(false);

	async function handleLogin() {
		error = '';
		loading = true;
		try {
			const res = await login(password);
			if (res.ok) {
				$isAuthenticated = true;
				goto('/');
			} else {
				error = res.message || 'Wrong password';
			}
		} catch (e: any) {
			error = e.message;
		}
		loading = false;
	}
</script>

<div class="min-h-screen flex items-center justify-center bg-gray-50">
	<div class="bg-white p-8 rounded-lg shadow-lg w-80">
		<h1 class="text-xl font-bold text-gray-800 text-center mb-6">DepenSage</h1>

		<form onsubmit={(e) => { e.preventDefault(); handleLogin(); }}>
			<input
				type="password"
				bind:value={password}
				placeholder="Password"
				class="w-full px-3 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
				autofocus
			/>

			{#if error}
				<p class="text-sm text-red-500 mt-2">{error}</p>
			{/if}

			<button
				type="submit"
				disabled={loading || !password}
				class="w-full mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 font-medium"
			>
				{loading ? 'Logging in...' : 'Login'}
			</button>
		</form>
	</div>
</div>

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

<div class="min-h-screen flex items-center justify-center" style="background: linear-gradient(135deg, #1a2332, #1e3844);">
	<div class="bg-white/95 backdrop-blur p-8 rounded-2xl shadow-2xl w-80 border border-gray-200">
		<div class="text-center mb-6">
			<h1 class="text-2xl font-bold tracking-wide" style="color: #2f6577;">DepenSage</h1>
			<p class="text-xs text-gray-400 mt-1">Household Expense Tracker</p>
		</div>

		<form onsubmit={(e) => { e.preventDefault(); handleLogin(); }}>
			<input
				type="password"
				bind:value={password}
				placeholder="Password"
				class="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:border-transparent text-sm"
				style="--tw-ring-color: #4a9ab4;"
				autofocus
			/>

			{#if error}
				<p class="text-sm text-red-500 mt-2">{error}</p>
			{/if}

			<button
				type="submit"
				disabled={loading || !password}
				class="w-full mt-4 px-4 py-2.5 text-white rounded-lg disabled:opacity-50 font-medium text-sm transition-colors"
				style="background-color: #2f6577;"
				onmouseenter={(e) => e.currentTarget.style.backgroundColor = '#295463'}
				onmouseleave={(e) => e.currentTarget.style.backgroundColor = '#2f6577'}
			>
				{loading ? 'Logging in...' : 'Login'}
			</button>
		</form>
	</div>
</div>

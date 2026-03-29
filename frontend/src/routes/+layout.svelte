<script lang="ts">
	import '../app.css';
	import NavSidebar from '$lib/components/NavSidebar.svelte';
	import { page } from '$app/state';
	import { goto } from '$app/navigation';
	import { isAuthenticated } from '$lib/stores';
	import { checkAuth } from '$lib/api';

	let { children } = $props();
	let checking = $state(true);

	const isLoginPage = $derived(page.url.pathname === '/login');

	// Check auth on mount — redirect to login if not authenticated
	$effect(() => {
		if (isLoginPage) {
			checking = false;
			return;
		}
		checkAuth().then(ok => {
			$isAuthenticated = ok;
			checking = false;
			if (!ok) goto('/login');
		});
	});
</script>

<svelte:head>
	<title>DepenSage</title>
</svelte:head>

{#if checking && !isLoginPage}
	<!-- Auth check in progress -->
	<div class="min-h-screen flex items-center justify-center bg-gray-100">
		<div class="text-gray-400 text-sm">Loading...</div>
	</div>
{:else if isLoginPage}
	{@render children()}
{:else if $isAuthenticated}
	<div class="flex min-h-screen" style="background-color: #e8f0f4;">
		<NavSidebar />
		<main class="flex-1 p-6 overflow-auto">
			{@render children()}
		</main>
	</div>
{/if}

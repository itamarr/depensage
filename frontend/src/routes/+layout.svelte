<script lang="ts">
	import '../app.css';
	import NavSidebar from '$lib/components/NavSidebar.svelte';
	import { page } from '$app/state';

	let { children } = $props();

	const isLoginPage = $derived(page.url.pathname === '/login');
</script>

<svelte:head>
	<title>DepenSage</title>
</svelte:head>

{#if isLoginPage}
	{@render children()}
{:else}
	<div class="flex min-h-screen bg-gray-50">
		<NavSidebar />
		<main class="flex-1 p-6 overflow-auto">
			{@render children()}
		</main>
	</div>
{/if}

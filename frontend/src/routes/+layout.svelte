<script lang="ts">
	import '../app.css';
	import NavSidebar from '$lib/components/NavSidebar.svelte';

	let { children } = $props();
</script>

<svelte:head>
	<title>DepenSage</title>
</svelte:head>

<div class="flex min-h-screen bg-gray-50">
	<NavSidebar />
	<main class="flex-1 p-6 overflow-auto">
		{@render children()}
	</main>
</div>

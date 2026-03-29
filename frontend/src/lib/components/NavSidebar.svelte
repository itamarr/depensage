<script lang="ts">
	import { page } from '$app/state';

	const links = [
		{ href: '/', label: 'Dashboard', icon: '📊' },
		{ href: '/pipeline', label: 'Pipeline', icon: '⚙️' },
		{ href: '/months', label: 'Months', icon: '📅' },
		{ href: '/lookups', label: 'Lookups', icon: '🔍' },
		{ href: '/categories', label: 'Categories', icon: '📂' },
		{ href: '/stats', label: 'Statistics', icon: '📈' },
	];

	let collapsed = $state(false);
</script>

<aside
	class="h-screen flex flex-col flex-shrink-0 shadow-lg"
	style="width: {collapsed ? '4rem' : '14rem'}; background-color: #1a2332; transition: width 0.2s;"
>
	<div class="p-4 flex items-center justify-between min-h-14" style="border-bottom: 1px solid #293545;">
		{#if !collapsed}
			<h1 class="text-lg font-bold tracking-wide" style="color: #e5af42;">DepenSage</h1>
		{/if}
		<button
			onclick={() => collapsed = !collapsed}
			class="p-1 transition-colors"
			style="color: #6b7280;"
			aria-label="Toggle sidebar"
		>
			{#if collapsed}
				<svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 5l7 7-7 7M5 5l7 7-7 7" /></svg>
			{:else}
				<svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 19l-7-7 7-7M19 19l-7-7 7-7" /></svg>
			{/if}
		</button>
	</div>

	<nav class="flex-1 py-4">
		{#each links as link}
			{@const active = page.url.pathname === link.href ||
				(link.href !== '/' && page.url.pathname.startsWith(link.href))}
			<a
				href={link.href}
				class="flex items-center gap-3 px-4 py-2.5 text-sm transition-colors"
				style="color: {active ? '#ebc46c' : '#9ca3af'}; background-color: {active ? 'rgba(38, 70, 82, 0.6)' : 'transparent'}; {active ? 'border-right: 2px solid #e5af42;' : ''}"
				onmouseenter={(e) => { if (!active) { e.currentTarget.style.backgroundColor = '#293545'; e.currentTarget.style.color = '#e5e7eb'; }}}
				onmouseleave={(e) => { if (!active) { e.currentTarget.style.backgroundColor = 'transparent'; e.currentTarget.style.color = '#9ca3af'; }}}
			>
				<span class="text-lg flex-shrink-0">{link.icon}</span>
				{#if !collapsed}
					<span>{link.label}</span>
				{/if}
			</a>
		{/each}
	</nav>

	<div class="p-4 text-xs" style="border-top: 1px solid #293545; color: #4b5563;">
		{#if !collapsed}
			v0.1.0
		{/if}
	</div>
</aside>

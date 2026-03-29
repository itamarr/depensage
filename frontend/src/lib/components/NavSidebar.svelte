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

<div class="sidebar" class:collapsed style="--sidebar-w: {collapsed ? '4rem' : '14rem'}">
	<!-- Header -->
	<div class="sidebar-header">
		{#if !collapsed}
			<span class="sidebar-title">DepenSage</span>
		{/if}
		<button class="collapse-btn" onclick={() => collapsed = !collapsed} aria-label="Toggle sidebar">
			{#if collapsed}&#x276F;{:else}&#x276E;{/if}
		</button>
	</div>

	<!-- Nav links -->
	<nav class="sidebar-nav">
		{#each links as link}
			{@const active = page.url.pathname === link.href ||
				(link.href !== '/' && page.url.pathname.startsWith(link.href))}
			<a href={link.href} class="nav-link" class:active>
				<span class="nav-icon">{link.icon}</span>
				{#if !collapsed}
					<span class="nav-label">{link.label}</span>
				{/if}
			</a>
		{/each}
	</nav>

	<!-- Footer -->
	<div class="sidebar-footer">
		{#if !collapsed}
			<span>v0.1.0</span>
		{/if}
	</div>
</div>

<style>
	.sidebar {
		width: var(--sidebar-w);
		min-height: 100vh;
		background: #1a2332;
		display: flex;
		flex-direction: column;
		flex-shrink: 0;
		transition: width 0.2s ease;
		box-shadow: 2px 0 8px rgba(0,0,0,0.15);
	}

	.sidebar-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 1rem;
		border-bottom: 1px solid #293545;
		min-height: 3.5rem;
	}

	.sidebar-title {
		font-size: 1.125rem;
		font-weight: 700;
		color: #e5af42;
		letter-spacing: 0.025em;
	}

	.collapse-btn {
		color: #6b7280;
		padding: 0.25rem;
		cursor: pointer;
		background: none;
		border: none;
		font-size: 1rem;
		line-height: 1;
		transition: color 0.15s;
	}
	.collapse-btn:hover {
		color: #e5af42;
	}

	.sidebar-nav {
		flex: 1;
		padding: 1rem 0;
	}

	.nav-link {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		padding: 0.625rem 1rem;
		font-size: 0.875rem;
		color: #9ca3af;
		text-decoration: none;
		transition: background-color 0.15s, color 0.15s;
	}
	.nav-link:hover {
		background: #293545;
		color: #e5e7eb;
	}
	.nav-link.active {
		background: rgba(38, 70, 82, 0.6);
		color: #ebc46c;
		border-right: 2px solid #e5af42;
	}

	.nav-icon {
		font-size: 1.125rem;
		flex-shrink: 0;
	}

	.sidebar-footer {
		padding: 1rem;
		border-top: 1px solid #293545;
		font-size: 0.75rem;
		color: #4b5563;
	}
</style>

<script lang="ts">
	import { get } from '$lib/api';
	import ErrorBanner from '$lib/components/ErrorBanner.svelte';

	type MonthEntry = { month: string; year: number };
	let months = $state<MonthEntry[]>([]);
	let loading = $state(true);
	let error = $state('');

	$effect(() => {
		get<{ months: MonthEntry[] }>('/months/')
			.then(data => { months = data.months; loading = false; })
			.catch(e => { error = e.message; loading = false; });
	});

	// Group by year
	const byYear = $derived(() => {
		const groups: Record<number, string[]> = {};
		for (const m of months) {
			if (!groups[m.year]) groups[m.year] = [];
			groups[m.year].push(m.month);
		}
		return groups;
	});

	const monthOrder = ['January','February','March','April','May','June',
		'July','August','September','October','November','December'];
</script>

<div class="max-w-4xl">
	<h1 class="text-2xl font-bold text-primary-800 mb-6">Months</h1>

	{#if error}
		<ErrorBanner message={error} ondismiss={() => error = ''} />
	{/if}

	{#if loading}
		<p class="text-gray-400 text-sm">Loading...</p>
	{:else}
		{#each Object.entries(byYear()).sort((a, b) => Number(b[0]) - Number(a[0])) as [year, monthNames]}
			<div class="mb-6">
				<h2 class="text-lg font-semibold text-primary-700 mb-3">{year}</h2>
				<div class="grid grid-cols-3 md:grid-cols-4 gap-2">
					{#each monthOrder as m}
						{@const exists = monthNames.includes(m)}
						{#if exists}
							<a
								href="/months/{year}-{m}"
								class="p-3 rounded-lg text-center text-sm font-medium transition-shadow hover:shadow-md"
								style="background: white; border: 1px solid #b3dbe9; color: #2f6577;"
							>{m}</a>
						{:else}
							<div class="p-3 rounded-lg text-center text-sm text-gray-300 bg-gray-50">{m}</div>
						{/if}
					{/each}
				</div>
			</div>
		{/each}
	{/if}
</div>

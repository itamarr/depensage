<script lang="ts">
	import { get } from '$lib/api';

	let categories = $state<Record<string, string[]>>({});
	let loading = $state(true);
	let error = $state('');

	$effect(() => {
		get<{ categories: Record<string, string[]> }>('/categories/')
			.then(data => { categories = data.categories; loading = false; })
			.catch(e => { error = e.message; loading = false; });
	});

	const catNames = $derived(Object.keys(categories));
</script>

<div class="max-w-5xl">
	<h1 class="text-2xl font-bold text-primary-800 mb-6">Categories</h1>

	{#if error}
		<div class="mb-4 p-3 bg-red-50 border border-red-200 rounded text-sm text-red-700">{error}</div>
	{/if}

	{#if loading}
		<p class="text-gray-400 text-sm">Loading categories...</p>
	{:else}
		<div class="bg-white rounded-xl shadow-sm p-4 overflow-x-auto" style="border: 1px solid #b3dbe9;">
			<div class="grid gap-4" style="grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));">
				{#each catNames as cat}
					<div class="rounded-lg p-3" style="background: #f0f7fa; border: 1px solid #d9edf4;">
						<h3 class="text-sm font-semibold text-primary-700 rtl mb-2 pb-1"
							style="border-bottom: 1px solid #b3dbe9;">{cat}</h3>
						{#if categories[cat].length > 0}
							<ul class="space-y-0.5">
								{#each categories[cat] as sub}
									<li class="text-xs text-gray-600 rtl">{sub}</li>
								{/each}
							</ul>
						{:else}
							<p class="text-xs text-gray-400 italic">No subcategories</p>
						{/if}
					</div>
				{/each}
			</div>
		</div>
	{/if}
</div>

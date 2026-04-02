<script lang="ts">
	import { get, put } from '$lib/api';

	let categories = $state<Record<string, string[]>>({});
	let loading = $state(true);
	let saving = $state(false);
	let error = $state('');
	let dirty = $state(false);

	// Add category form
	let newCatName = $state('');
	// Add subcategory: track which category is being added to
	let addingSubTo = $state<string | null>(null);
	let newSubName = $state('');

	$effect(() => {
		get<{ categories: Record<string, string[]> }>('/categories/')
			.then(data => { categories = data.categories; loading = false; })
			.catch(e => { error = e.message; loading = false; });
	});

	const catNames = $derived(Object.keys(categories));

	function addCategory() {
		if (!newCatName.trim()) return;
		categories[newCatName.trim()] = [];
		categories = { ...categories };
		newCatName = '';
		dirty = true;
	}

	function removeCategory(cat: string) {
		if (!confirm(`Remove category "${cat}" and all its subcategories?`)) return;
		delete categories[cat];
		categories = { ...categories };
		dirty = true;
	}

	function addSubcategory(cat: string) {
		if (!newSubName.trim()) return;
		categories[cat] = [...categories[cat], newSubName.trim()];
		categories = { ...categories };
		newSubName = '';
		addingSubTo = null;
		dirty = true;
	}

	function removeSubcategory(cat: string, sub: string) {
		categories[cat] = categories[cat].filter(s => s !== sub);
		categories = { ...categories };
		dirty = true;
	}

	async function saveAll() {
		saving = true; error = '';
		try {
			await put('/categories/', { categories });
			dirty = false;
		} catch (e: any) { error = e.message; }
		saving = false;
	}
</script>

<div class="max-w-5xl">
	<div class="flex items-center justify-between mb-6">
		<h1 class="text-2xl font-bold text-primary-800">Categories</h1>
		{#if dirty}
			<button
				onclick={saveAll}
				disabled={saving}
				class="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 text-sm font-medium"
			>{saving ? 'Saving...' : 'Save Changes'}</button>
		{/if}
	</div>

	{#if error}
		<div class="mb-4 p-3 bg-red-50 border border-red-200 rounded text-sm text-red-700">{error}</div>
	{/if}

	{#if loading}
		<p class="text-gray-400 text-sm">Loading categories...</p>
	{:else}
		<!-- Add category -->
		<div class="mb-4 flex items-center gap-2">
			<input
				bind:value={newCatName}
				placeholder="New category name..."
				class="border rounded px-3 py-1.5 text-sm rtl"
				style="border-color: #b3dbe9;"
				onkeydown={(e) => { if (e.key === 'Enter') addCategory(); }}
			/>
			<button
				onclick={addCategory}
				disabled={!newCatName.trim()}
				class="px-3 py-1.5 bg-primary-600 text-white rounded text-sm hover:bg-primary-700 disabled:opacity-50"
			>+ Add Category</button>
		</div>

		<div class="bg-white rounded-xl shadow-sm p-4 overflow-x-auto" style="border: 1px solid #b3dbe9;">
			<div class="grid gap-4" style="grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));">
				{#each catNames as cat}
					<div class="rounded-lg p-3" style="background: #f0f7fa; border: 1px solid #d9edf4;">
						<div class="flex items-center justify-between mb-2 pb-1" style="border-bottom: 1px solid #b3dbe9;">
							<h3 class="text-sm font-semibold text-primary-700 rtl">{cat}</h3>
							<button
								onclick={() => removeCategory(cat)}
								class="text-red-400 hover:text-red-600 text-xs"
								title="Remove category"
							>✕</button>
						</div>
						<ul class="space-y-0.5">
							{#each categories[cat] as sub}
								<li class="text-xs text-gray-600 rtl flex items-center justify-between group">
									<span>{sub}</span>
									<button
										onclick={() => removeSubcategory(cat, sub)}
										class="text-red-400 hover:text-red-600 text-xs opacity-0 group-hover:opacity-100"
									>✕</button>
								</li>
							{/each}
						</ul>
						{#if addingSubTo === cat}
							<div class="mt-2 flex gap-1">
								<input
									bind:value={newSubName}
									placeholder="Subcategory..."
									class="text-xs border rounded px-1.5 py-0.5 flex-1 rtl"
									autofocus
									onkeydown={(e) => { if (e.key === 'Enter') addSubcategory(cat); if (e.key === 'Escape') addingSubTo = null; }}
								/>
								<button onclick={() => addSubcategory(cat)} class="text-xs text-green-600">✓</button>
							</div>
						{:else}
							<button
								onclick={() => { addingSubTo = cat; newSubName = ''; }}
								class="mt-2 text-xs text-primary-500 hover:text-primary-700"
							>+ subcategory</button>
						{/if}
					</div>
				{/each}
			</div>
		</div>

		{#if dirty}
			<p class="mt-3 text-xs text-amber-600">Unsaved changes — click "Save Changes" to write to the Categories sheet.</p>
		{/if}
	{/if}
</div>

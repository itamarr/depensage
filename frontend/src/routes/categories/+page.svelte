<script lang="ts">
	import { get, put, post } from '$lib/api';

	let categories = $state<Record<string, string[]>>({});
	let original = $state<Record<string, string[]>>({});  // snapshot for detecting renames
	let loading = $state(true);
	let saving = $state(false);
	let error = $state('');
	let success = $state('');
	let dirty = $state(false);

	// Track renames: old_name → new_name
	let catRenames = $state<Record<string, string>>({});
	let subRenames = $state<Record<string, Record<string, string>>>({});

	// Add category
	let newCatName = $state('');
	// Add subcategory
	let addingSubTo = $state<string | null>(null);
	let newSubName = $state('');
	// Rename inline
	let renamingCat = $state<string | null>(null);
	let renameCatValue = $state('');
	let renamingSub = $state<{ cat: string; sub: string } | null>(null);
	let renameSubValue = $state('');

	$effect(() => {
		get<{ categories: Record<string, string[]> }>('/categories/')
			.then(data => {
				categories = data.categories;
				original = JSON.parse(JSON.stringify(data.categories));
				loading = false;
			})
			.catch(e => { error = e.message; loading = false; });
	});

	const catNames = $derived(Object.keys(categories));

	function markDirty() { dirty = true; success = ''; }

	function addCategory() {
		if (!newCatName.trim()) return;
		categories[newCatName.trim()] = [];
		categories = { ...categories };
		newCatName = '';
		markDirty();
	}

	function removeCategory(cat: string) {
		if (!confirm(`Remove category "${cat}" and all its subcategories?`)) return;
		delete categories[cat];
		categories = { ...categories };
		markDirty();
	}

	function startRenameCat(cat: string) {
		renamingCat = cat;
		renameCatValue = cat;
	}

	function finishRenameCat(oldName: string) {
		const newName = renameCatValue.trim();
		renamingCat = null;
		if (!newName || newName === oldName) return;
		if (newName in categories) { error = `Category "${newName}" already exists`; return; }

		// Rebuild with new key preserving order
		const newCats: Record<string, string[]> = {};
		for (const [k, v] of Object.entries(categories)) {
			newCats[k === oldName ? newName : k] = v;
		}
		categories = newCats;

		// Track rename for propagation
		// If oldName was itself a rename, chain: original → newName
		const origName = Object.entries(catRenames).find(([, v]) => v === oldName)?.[0] || oldName;
		if (origName in original) {
			catRenames[origName] = newName;
			catRenames = { ...catRenames };
		}
		markDirty();
	}

	function addSubcategory(cat: string) {
		if (!newSubName.trim()) return;
		categories[cat] = [...categories[cat], newSubName.trim()];
		categories = { ...categories };
		newSubName = '';
		addingSubTo = null;
		markDirty();
	}

	function removeSubcategory(cat: string, sub: string) {
		categories[cat] = categories[cat].filter(s => s !== sub);
		categories = { ...categories };
		markDirty();
	}

	function startRenameSub(cat: string, sub: string) {
		renamingSub = { cat, sub };
		renameSubValue = sub;
	}

	function finishRenameSub(cat: string, oldSub: string) {
		const newSub = renameSubValue.trim();
		renamingSub = null;
		if (!newSub || newSub === oldSub) return;

		categories[cat] = categories[cat].map(s => s === oldSub ? newSub : s);
		categories = { ...categories };

		// Track sub rename
		if (!subRenames[cat]) subRenames[cat] = {};
		subRenames[cat][oldSub] = newSub;
		subRenames = { ...subRenames };
		markDirty();
	}

	async function saveAll() {
		saving = true; error = ''; success = '';
		try {
			const resp = await put<any>('/categories/', {
				categories,
				renames: catRenames,
				sub_renames: subRenames,
			});
			dirty = false;
			original = JSON.parse(JSON.stringify(categories));

			const parts = [`${resp.categories} categories saved`];
			if (resp.template_updated) parts.push('template updated');

			// Propagate renames to month data if any
			const hasRenames = Object.keys(catRenames).length > 0 ||
				Object.values(subRenames).some(v => Object.keys(v).length > 0);

			if (hasRenames) {
				const doPropagate = confirm(
					'Category names were changed. Update all expenses in the current year to match?'
				);
				if (doPropagate) {
					const propResp = await post<any>('/categories/propagate', {
						renames: catRenames,
						sub_renames: subRenames,
					});
					parts.push(`${propResp.cell_updates} cells + ${propResp.lookup_updates} lookups updated`);
				}
			}

			success = parts.join('. ');
			catRenames = {};
			subRenames = {};
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
	{#if success}
		<div class="mb-4 p-3 bg-green-50 border border-green-200 rounded text-sm text-green-700">{success}</div>
	{/if}

	{#if loading}
		<p class="text-gray-400 text-sm">Loading categories...</p>
	{:else}
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
							{#if renamingCat === cat}
								<input
									bind:value={renameCatValue}
									class="text-sm font-semibold text-primary-700 rtl border rounded px-1 py-0.5 w-full"
									autofocus
									onkeydown={(e) => { if (e.key === 'Enter') finishRenameCat(cat); if (e.key === 'Escape') renamingCat = null; }}
									onblur={() => finishRenameCat(cat)}
								/>
							{:else}
								<h3
									class="text-sm font-semibold text-primary-700 rtl cursor-pointer hover:underline"
									title="Click to rename"
									ondblclick={() => startRenameCat(cat)}
								>{cat}</h3>
							{/if}
							<button
								onclick={() => removeCategory(cat)}
								class="text-red-400 hover:text-red-600 text-xs ml-1 flex-shrink-0"
								title="Remove category"
							>✕</button>
						</div>
						<ul class="space-y-0.5">
							{#each categories[cat] as sub}
								<li class="text-xs text-gray-600 rtl flex items-center justify-between group">
									{#if renamingSub?.cat === cat && renamingSub?.sub === sub}
										<input
											bind:value={renameSubValue}
											class="text-xs border rounded px-1 py-0.5 w-full rtl"
											autofocus
											onkeydown={(e) => { if (e.key === 'Enter') finishRenameSub(cat, sub); if (e.key === 'Escape') renamingSub = null; }}
											onblur={() => finishRenameSub(cat, sub)}
										/>
									{:else}
										<span
											class="cursor-pointer hover:underline"
											title="Double-click to rename"
											ondblclick={() => startRenameSub(cat, sub)}
										>{sub}</span>
										<button
											onclick={() => removeSubcategory(cat, sub)}
											class="text-red-400 hover:text-red-600 text-xs opacity-0 group-hover:opacity-100"
										>✕</button>
									{/if}
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
			<p class="mt-3 text-xs text-amber-600">
				Unsaved changes. Double-click a name to rename it.
				{#if Object.keys(catRenames).length > 0 || Object.values(subRenames).some(v => Object.keys(v).length > 0)}
					Renames will be tracked for propagation to month data.
				{/if}
			</p>
		{/if}

		<p class="mt-4 text-xs text-gray-400">
			Categories are saved to the current year's spreadsheet and the template. Double-click any name to rename.
		</p>
	{/if}
</div>

<script lang="ts">
	import { get, post, del } from '$lib/api';

	type Entry = { key: string; category: string; subcategory: string };
	type PatternEntry = { prefix: string; category: string; subcategory: string };

	let activeTab = $state<'cc' | 'bank' | 'income'>('cc');
	let exact = $state<Entry[]>([]);
	let patterns = $state<PatternEntry[]>([]);
	let loading = $state(false);
	let error = $state('');
	let search = $state('');

	// Add form
	let showAdd = $state(false);
	let addKey = $state('');
	let addCategory = $state('');
	let addSubcategory = $state('');
	let addType = $state<'exact' | 'pattern'>('exact');

	async function loadTab(tab: typeof activeTab) {
		activeTab = tab;
		loading = true; error = '';
		try {
			const data = await get<{ exact: Entry[]; patterns: PatternEntry[] }>(`/lookups/${tab}`);
			exact = data.exact;
			patterns = data.patterns;
		} catch (e: any) { error = e.message; }
		loading = false;
	}

	$effect(() => { loadTab('cc'); });

	const filteredExact = $derived(
		search ? exact.filter(e =>
			e.key.includes(search) || e.category.includes(search) || e.subcategory.includes(search)
		) : exact
	);

	async function handleAdd() {
		error = '';
		try {
			if (addType === 'exact') {
				await post(`/lookups/${activeTab}/exact`, {
					key: addKey, category: addCategory, subcategory: addSubcategory,
				});
			} else {
				await post(`/lookups/${activeTab}/pattern`, {
					prefix: addKey, category: addCategory, subcategory: addSubcategory,
				});
			}
			showAdd = false; addKey = ''; addCategory = ''; addSubcategory = '';
			await loadTab(activeTab);
		} catch (e: any) { error = e.message; }
	}

	async function handleDelete(key: string) {
		if (!confirm(`Delete "${key}"?`)) return;
		try {
			await del(`/lookups/${activeTab}/exact/${encodeURIComponent(key)}`);
			await loadTab(activeTab);
		} catch (e: any) { error = e.message; }
	}

	async function handleDeletePattern(index: number) {
		if (!confirm('Delete this pattern?')) return;
		try {
			await del(`/lookups/${activeTab}/pattern/${index}`);
			await loadTab(activeTab);
		} catch (e: any) { error = e.message; }
	}
</script>

<div class="max-w-5xl">
	<h1 class="text-2xl font-bold text-primary-800 mb-6">Lookup Tables</h1>

	<!-- Tabs -->
	<div class="flex gap-1 mb-4">
		{#each [['cc', 'CC Merchants'], ['bank', 'Bank Actions'], ['income', 'Income']] as [tab, label]}
			<button
				onclick={() => loadTab(tab as typeof activeTab)}
				class="px-4 py-2 rounded-t-lg text-sm font-medium transition-colors
					{activeTab === tab ? 'bg-white text-primary-700 shadow-sm' : 'text-gray-500 hover:text-gray-700'}"
				style={activeTab === tab ? 'border: 1px solid #b3dbe9; border-bottom: none;' : ''}
			>{label}</button>
		{/each}
	</div>

	{#if error}
		<div class="mb-4 p-3 bg-red-50 border border-red-200 rounded text-sm text-red-700">{error}</div>
	{/if}

	<div class="bg-white rounded-xl shadow-sm p-4" style="border: 1px solid #b3dbe9;">
		{#if loading}
			<p class="text-gray-400 text-sm py-8 text-center">Loading...</p>
		{:else}
			<!-- Toolbar -->
			<div class="flex items-center justify-between mb-4">
				<input
					bind:value={search}
					placeholder="Search..."
					class="border rounded px-3 py-1.5 text-sm w-64"
				/>
				<button
					onclick={() => showAdd = !showAdd}
					class="px-3 py-1.5 bg-primary-600 text-white rounded text-sm hover:bg-primary-700"
				>{showAdd ? 'Cancel' : '+ Add'}</button>
			</div>

			<!-- Add form -->
			{#if showAdd}
				<div class="mb-4 p-3 rounded" style="background: #f0f7fa; border: 1px solid #b3dbe9;">
					<div class="flex gap-2 items-end flex-wrap">
						<label class="text-xs text-gray-600">
							Type
							<select bind:value={addType} class="block border rounded px-2 py-1 text-sm mt-0.5">
								<option value="exact">Exact match</option>
								<option value="pattern">Prefix pattern</option>
							</select>
						</label>
						<label class="text-xs text-gray-600">
							{addType === 'exact' ? 'Name' : 'Prefix'}
							<input bind:value={addKey} class="block border rounded px-2 py-1 text-sm mt-0.5 rtl" />
						</label>
						<label class="text-xs text-gray-600">
							Category
							<input bind:value={addCategory} class="block border rounded px-2 py-1 text-sm mt-0.5 rtl" />
						</label>
						<label class="text-xs text-gray-600">
							{activeTab === 'income' ? 'Comments' : 'Subcategory'}
							<input bind:value={addSubcategory} class="block border rounded px-2 py-1 text-sm mt-0.5 rtl" />
						</label>
						<button
							onclick={handleAdd}
							disabled={!addKey || !addCategory}
							class="px-3 py-1 bg-green-600 text-white rounded text-sm hover:bg-green-700 disabled:opacity-50"
						>Save</button>
					</div>
				</div>
			{/if}

			<!-- Exact matches -->
			<h3 class="text-sm font-medium text-gray-600 mb-2">Exact Matches ({filteredExact.length})</h3>
			<div class="overflow-x-auto mb-6">
				<table class="w-full text-sm" dir="rtl">
					<thead style="background: #f0f7fa;">
						<tr>
							<th class="px-2 py-1.5 text-right text-xs font-medium text-gray-600">Name</th>
							<th class="px-2 py-1.5 text-right text-xs font-medium text-gray-600">Category</th>
							<th class="px-2 py-1.5 text-right text-xs font-medium text-gray-600">
								{activeTab === 'income' ? 'Comments' : 'Subcategory'}
							</th>
							<th class="px-2 py-1.5 w-16" dir="ltr"></th>
						</tr>
					</thead>
					<tbody>
						{#each filteredExact as entry}
							<tr class="border-t hover:bg-gray-50">
								<td class="px-2 py-1 text-xs">{entry.key}</td>
								<td class="px-2 py-1 text-xs">{entry.category}</td>
								<td class="px-2 py-1 text-xs">{entry.subcategory}</td>
								<td class="px-2 py-1" dir="ltr">
									<button onclick={() => handleDelete(entry.key)}
										class="text-red-400 hover:text-red-600 text-xs">delete</button>
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>

			<!-- Patterns -->
			{#if patterns.length > 0}
				<h3 class="text-sm font-medium text-gray-600 mb-2">Prefix Patterns ({patterns.length})</h3>
				<table class="w-full text-sm" dir="rtl">
					<thead style="background: #f0f7fa;">
						<tr>
							<th class="px-2 py-1.5 text-right text-xs font-medium text-gray-600">Prefix</th>
							<th class="px-2 py-1.5 text-right text-xs font-medium text-gray-600">Category</th>
							<th class="px-2 py-1.5 text-right text-xs font-medium text-gray-600">
								{activeTab === 'income' ? 'Comments' : 'Subcategory'}
							</th>
							<th class="px-2 py-1.5 w-16" dir="ltr"></th>
						</tr>
					</thead>
					<tbody>
						{#each patterns as pat, i}
							<tr class="border-t hover:bg-gray-50">
								<td class="px-2 py-1 text-xs">{pat.prefix}*</td>
								<td class="px-2 py-1 text-xs">{pat.category}</td>
								<td class="px-2 py-1 text-xs">{pat.subcategory}</td>
								<td class="px-2 py-1" dir="ltr">
									<button onclick={() => handleDeletePattern(i)}
										class="text-red-400 hover:text-red-600 text-xs">delete</button>
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			{/if}
		{/if}
	</div>
</div>

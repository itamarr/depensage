<script lang="ts">
	import { get, post, put, del } from '$lib/api';
	import CategoryPicker from '$lib/components/CategoryPicker.svelte';
	import ErrorBanner from '$lib/components/ErrorBanner.svelte';

	type Entry = { key: string; category: string; subcategory: string };
	type PatternEntry = { prefix: string; category: string; subcategory: string };

	let activeTab = $state<'cc' | 'bank' | 'income'>('cc');
	let exact = $state<Entry[]>([]);
	let patterns = $state<PatternEntry[]>([]);
	let loading = $state(false);
	let error = $state('');
	let search = $state('');
	let categories = $state<Record<string, string[]>>({});

	// Inline editing
	let editingKey = $state<string | null>(null);
	let editCat = $state('');
	let editSub = $state('');

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

	$effect(() => {
		loadTab('cc');
		get<{ categories: Record<string, string[]> }>('/categories/')
			.then(data => categories = data.categories)
			.catch(() => {});
	});

	const useDropdowns = $derived(activeTab !== 'income');

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

	function startEdit(entry: Entry) {
		editingKey = entry.key;
		editCat = entry.category;
		editSub = entry.subcategory;
	}

	async function saveEdit(key: string) {
		error = '';
		try {
			await put(`/lookups/${activeTab}/exact/${encodeURIComponent(key)}`, {
				key, category: editCat, subcategory: editSub,
			});
			editingKey = null;
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
		<ErrorBanner message={error} ondismiss={() => error = ''} />
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
						{#if useDropdowns && Object.keys(categories).length > 0}
							<label class="text-xs text-gray-600">
								Category / Subcategory
								<div class="mt-0.5">
									<CategoryPicker
										{categories}
										value={addCategory}
										subValue={addSubcategory}
										onchange={(cat, sub) => { addCategory = cat; addSubcategory = sub; }}
									/>
								</div>
							</label>
						{:else}
							<label class="text-xs text-gray-600">
								Category
								<input bind:value={addCategory} class="block border rounded px-2 py-1 text-sm mt-0.5 rtl" />
							</label>
							<label class="text-xs text-gray-600">
								{activeTab === 'income' ? 'Comments' : 'Subcategory'}
								<input bind:value={addSubcategory} class="block border rounded px-2 py-1 text-sm mt-0.5 rtl" />
							</label>
						{/if}
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
							{#if editingKey === entry.key}
								<tr class="border-t" style="background: #f0f7fa;">
									<td class="px-2 py-1 text-xs">{entry.key}</td>
									{#if useDropdowns && Object.keys(categories).length > 0}
										<td class="px-2 py-1" colspan="2">
											<CategoryPicker
												{categories}
												value={editCat}
												subValue={editSub}
												onchange={(cat, sub) => { editCat = cat; editSub = sub; }}
											/>
										</td>
									{:else}
										<td class="px-2 py-1"><input bind:value={editCat} class="text-xs border rounded px-1 py-0.5 w-full" style="direction:rtl;" /></td>
										<td class="px-2 py-1"><input bind:value={editSub} class="text-xs border rounded px-1 py-0.5 w-full" style="direction:rtl;" /></td>
									{/if}
									<td class="px-2 py-1" dir="ltr">
										<button onclick={() => saveEdit(entry.key)} class="text-green-600 hover:text-green-800 text-xs mr-1">save</button>
										<button onclick={() => editingKey = null} class="text-gray-400 hover:text-gray-600 text-xs">cancel</button>
									</td>
								</tr>
							{:else}
								<tr class="border-t hover:bg-gray-50">
									<td class="px-2 py-1 text-xs">{entry.key}</td>
									<td class="px-2 py-1 text-xs">{entry.category}</td>
									<td class="px-2 py-1 text-xs">{entry.subcategory}</td>
									<td class="px-2 py-1" dir="ltr">
										<button onclick={() => startEdit(entry)} class="text-primary-600 hover:text-primary-800 text-xs mr-1">edit</button>
										<button onclick={() => handleDelete(entry.key)} class="text-red-400 hover:text-red-600 text-xs">delete</button>
									</td>
								</tr>
							{/if}
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

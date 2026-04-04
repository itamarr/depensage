<script lang="ts">
	import { get, put } from '$lib/api';
	import ErrorBanner from '$lib/components/ErrorBanner.svelte';
	import CategoryPicker from '$lib/components/CategoryPicker.svelte';

	type BudgetLine = {
		category: string; subcategory: string;
		budget_amount: number; carry_status: string; row_number: number;
		_deleted?: boolean; _new?: boolean;
	};

	let lines = $state<BudgetLine[]>([]);
	let averages = $state<Record<string, number>>({});
	let monthCount = $state(0);
	let year = $state(0);
	let loading = $state(true);
	let saving = $state(false);
	let error = $state('');
	let success = $state('');
	let dirty = $state(false);

	// Original state for dirty/revert
	let originalLines = $state<string>('');

	// Categories for picker
	let categories = $state<Record<string, string[]>>({});

	// Add form
	let showAdd = $state(false);
	let addCategory = $state('');
	let addSubcategory = $state('');
	let addAmount = $state(0);
	let addFlag = $state('');

	// Drag and drop
	let dragIdx = $state<number | null>(null);
	let dragOverIdx = $state<number | null>(null);
	let reordered = $state(false);

	$effect(() => {
		Promise.all([
			get<any>('/budget/'),
			get<{ categories: Record<string, string[]> }>('/categories/'),
		]).then(([data, catData]) => {
			lines = data.lines;
			averages = data.averages;
			monthCount = data.month_count;
			year = data.year;
			originalLines = JSON.stringify(data.lines);
			categories = catData.categories;
			loading = false;
		}).catch(e => { error = e.message; loading = false; });
	});

	function markDirty() {
		dirty = JSON.stringify(lines.filter(l => !l._deleted)) !== originalLines
			|| lines.some(l => l._deleted || l._new) || reordered;
		success = '';
	}

	function handleDragStart(idx: number) { dragIdx = idx; }
	function handleDragOver(e: DragEvent, idx: number) { e.preventDefault(); dragOverIdx = idx; }
	function handleDragEnd() { dragIdx = null; dragOverIdx = null; }
	function handleDrop(targetIdx: number) {
		if (dragIdx === null || dragIdx === targetIdx) return;
		const item = lines[dragIdx];
		const newLines = lines.filter((_, i) => i !== dragIdx);
		newLines.splice(targetIdx, 0, item);
		lines = newLines;
		dragIdx = null; dragOverIdx = null;
		reordered = true;
		markDirty();
	}

	function getAvg(line: BudgetLine): number {
		if (line.subcategory) {
			return averages[`${line.category}/${line.subcategory}`] || 0;
		}
		// For categories with subcats in budget, check direct key
		const direct = averages[line.category];
		if (direct) return direct;
		// Sum subcategory averages
		let sum = 0;
		for (const [key, val] of Object.entries(averages)) {
			if (key.startsWith(line.category + '/')) sum += val;
		}
		return sum;
	}

	function handleAmountChange(line: BudgetLine, val: string) {
		line.budget_amount = parseFloat(val) || 0;
		lines = [...lines];
		markDirty();
	}

	function handleFlagChange(line: BudgetLine, val: string) {
		line.carry_status = val;
		lines = [...lines];
		markDirty();
	}

	function deleteLine(line: BudgetLine) {
		if (line._new) {
			lines = lines.filter(l => l !== line);
		} else {
			line._deleted = true;
			lines = [...lines];
		}
		markDirty();
	}

	function restoreLine(line: BudgetLine) {
		line._deleted = false;
		lines = [...lines];
		markDirty();
	}

	function addLine() {
		if (!addCategory.trim()) return;
		lines = [...lines, {
			category: addCategory.trim(),
			subcategory: addSubcategory.trim(),
			budget_amount: addAmount,
			carry_status: addFlag,
			row_number: 0,  // Will be assigned by backend
			_new: true,
		}];
		addCategory = ''; addSubcategory = ''; addAmount = 0; addFlag = '';
		showAdd = false;
		markDirty();
	}

	function revertAll() {
		const orig = JSON.parse(originalLines) as BudgetLine[];
		lines = orig;
		dirty = false;
		success = '';
	}

	const activeLines = $derived(lines.filter(l => !l._deleted));

	const totalBudget = $derived(
		activeLines.filter(l => l.category !== 'חסכון').reduce((sum, l) => sum + l.budget_amount, 0)
	);

	const totalAvg = $derived(
		activeLines.filter(l => l.category !== 'חסכון').reduce((sum, l) => sum + getAvg(l), 0)
	);

	async function saveBudget() {
		saving = true; error = ''; success = '';
		try {
			const orig = JSON.parse(originalLines) as BudgetLine[];
			const origMap = new Map(orig.map(l => [l.row_number, l]));

			// Updates: changed amount or flag
			const updates = lines
				.filter(l => !l._deleted && !l._new && l.row_number > 0)
				.filter(l => {
					const o = origMap.get(l.row_number);
					return o && (o.budget_amount !== l.budget_amount || o.carry_status !== l.carry_status);
				})
				.map(l => ({
					row_number: l.row_number,
					budget_amount: l.budget_amount,
					carry_status: l.carry_status,
				}));

			// Deletions
			const deletions = lines.filter(l => l._deleted && l.row_number > 0).map(l => l.row_number);

			// Additions
			const additions = lines.filter(l => l._new).map(l => ({
				category: l.category,
				subcategory: l.subcategory,
				budget_amount: l.budget_amount,
				carry_status: l.carry_status,
			}));

			// If reordered, send full rewrite instead of incremental updates
			let payload: any;
			if (reordered || deletions.length > 0 || additions.length > 0) {
				// Full rewrite: send all active lines in current order
				const fullRewrite = lines.filter(l => !l._deleted).map(l => ({
					category: l.category,
					subcategory: l.subcategory,
					budget_amount: l.budget_amount,
					carry_status: l.carry_status,
				}));
				payload = { updates: [], deletions: [], additions: [], full_rewrite: fullRewrite };
			} else {
				payload = { updates, deletions, additions };
			}

			const resp = await put<any>('/budget/', payload);

			const parts = [`${resp.cells_written} cells written`];
			if (resp.template_updated) parts.push('template updated');
			success = parts.join(', ');

			// Reload fresh data from server to get correct row numbers
			const freshData = await get<any>('/budget/');
			lines = freshData.lines;
			averages = freshData.averages;
			originalLines = JSON.stringify(freshData.lines);
			dirty = false; reordered = false;
		} catch (e: any) { error = e.message; }
		saving = false;
	}

	function fmtNum(n: number) { return n.toLocaleString(undefined, { maximumFractionDigits: 0 }); }
</script>

<div class="max-w-4xl">
	<div class="flex items-center justify-between mb-2">
		<h1 class="text-2xl font-bold text-primary-800">Budget Planning</h1>
		<div class="flex gap-2">
			{#if dirty}
				<button onclick={revertAll}
					class="px-3 py-1.5 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 text-sm">
					Revert All
				</button>
				<button onclick={saveBudget} disabled={saving}
					class="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 text-sm font-medium">
					{saving ? 'Saving...' : 'Save to Template'}
				</button>
			{/if}
		</div>
	</div>
	{#if year}
		<p class="text-sm text-gray-500 mb-6">
			Editing Month Template for {year}.
			{#if monthCount > 0}Averages based on {monthCount} month{monthCount !== 1 ? 's' : ''} of data.{/if}
			Changes apply to future months only.
		</p>
	{/if}

	{#if error}
		<ErrorBanner message={error} ondismiss={() => error = ''} />
	{/if}
	{#if success}
		<div class="mb-4 p-3 bg-green-50 border border-green-200 rounded text-sm text-green-700">{success}</div>
	{/if}

	{#if loading}
		<p class="text-gray-400 text-sm py-8 text-center">Loading budget...</p>
	{:else}
		<!-- Add button -->
		<div class="mb-3 flex items-center justify-end">
			<button onclick={() => showAdd = !showAdd}
				class="px-3 py-1.5 bg-primary-600 text-white rounded text-sm hover:bg-primary-700">
				{showAdd ? 'Cancel' : '+ Add Line'}
			</button>
		</div>

		{#if showAdd}
			<div class="mb-4 p-3 rounded" style="background: #f0f7fa; border: 1px solid #b3dbe9;">
				<div class="flex gap-2 items-end flex-wrap">
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
					<label class="text-xs text-gray-600">
						Budget
						<input type="number" bind:value={addAmount} class="block border rounded px-2 py-1 text-sm mt-0.5 w-24" />
					</label>
					<label class="text-xs text-gray-600">
						Flag
						<select bind:value={addFlag} class="block border rounded px-2 py-1 text-sm mt-0.5">
							<option value="">—</option>
							<option value="CARRY">CARRY</option>
							<option value="IGNORE">IGNORE</option>
						</select>
					</label>
					<button onclick={addLine} disabled={!addCategory.trim()}
						class="px-3 py-1 bg-green-600 text-white rounded text-sm hover:bg-green-700 disabled:opacity-50">
						Add
					</button>
				</div>
			</div>
		{/if}

		<div class="bg-white rounded-xl shadow-sm p-4" style="border: 1px solid #b3dbe9;">
			<table class="w-full text-sm" dir="rtl">
				<thead style="background: #f0f7fa;">
					<tr>
						<th class="w-8" dir="ltr"></th>
						<th class="px-3 py-2 text-right text-xs font-medium text-gray-600">Category</th>
						<th class="px-3 py-2 text-right text-xs font-medium text-gray-600">Subcategory</th>
						<th class="px-3 py-2 text-left text-xs font-medium text-gray-600" style="width: 110px;">Budget</th>
						<th class="px-3 py-2 text-left text-xs font-medium text-gray-600">Avg Spent</th>
						<th class="px-3 py-2 text-left text-xs font-medium text-gray-600">Diff</th>
						<th class="px-3 py-2 text-center text-xs font-medium text-gray-600" style="width: 90px;">Flag</th>
						<th class="px-3 py-2 w-12" dir="ltr"></th>
					</tr>
				</thead>
				<tbody>
					{#each lines as line, i}
						{@const avg = getAvg(line)}
						{@const diff = line.budget_amount - avg}
						{@const isSavings = line.category === 'חסכון'}
						<!-- svelte-ignore a11y_no_static_element_interactions -->
						<tr
							class="border-t {line._deleted ? 'opacity-40 line-through' : ''} {line._new ? 'bg-green-50' : isSavings ? 'bg-gray-50 text-gray-400' : 'hover:bg-gray-50'} {dragOverIdx === i && dragIdx !== i ? 'border-t-2 border-t-primary-400' : ''}"
							draggable="true"
							ondragstart={() => handleDragStart(i)}
							ondragover={(e) => handleDragOver(e, i)}
							ondragend={handleDragEnd}
							ondrop={() => handleDrop(i)}
						>
							<td class="px-1 text-center cursor-grab text-gray-300 hover:text-gray-500" dir="ltr" style="width:24px;">⠿</td>
						<td class="px-3 py-1.5 text-xs font-medium">{line.category}</td>
							<td class="px-3 py-1.5 text-xs">{line.subcategory}</td>
							<td class="px-3 py-1.5" dir="ltr">
								{#if isSavings}
									<span class="text-xs text-gray-400 italic">auto</span>
								{:else if line._deleted}
									<span class="text-xs">{fmtNum(line.budget_amount)}</span>
								{:else}
									<input
										type="number"
										value={line.budget_amount}
										class="text-xs border rounded px-2 py-1 w-full text-right"
										style="border-color: #d1d5db;"
										onchange={(e) => handleAmountChange(line, (e.target as HTMLInputElement).value)}
									/>
								{/if}
							</td>
							<td class="px-3 py-1.5 text-xs text-left" dir="ltr">
								{#if avg > 0}{fmtNum(avg)}{:else}—{/if}
							</td>
							<td class="px-3 py-1.5 text-xs text-left {diff < 0 ? 'text-red-500' : diff > 0 ? 'text-green-600' : ''}" dir="ltr">
								{#if !isSavings && avg > 0}
									{diff > 0 ? '+' : ''}{fmtNum(diff)}
								{/if}
							</td>
							<td class="px-3 py-1.5 text-center" dir="ltr">
								{#if isSavings || line._deleted}
									{#if line.carry_status}
										<span class="text-xs">{line.carry_status}</span>
									{/if}
								{:else}
									<select
										value={line.carry_status}
										class="text-xs border rounded px-1 py-0.5"
										style="border-color: #d1d5db;"
										onchange={(e) => handleFlagChange(line, (e.target as HTMLSelectElement).value)}
									>
										<option value="">—</option>
										<option value="CARRY">CARRY</option>
										<option value="IGNORE">IGNORE</option>
									</select>
								{/if}
							</td>
							<td class="px-3 py-1.5" dir="ltr">
								{#if !isSavings}
									{#if line._deleted}
										<button onclick={() => restoreLine(line)}
											class="text-xs text-green-600 hover:text-green-800">undo</button>
									{:else}
										<button onclick={() => deleteLine(line)}
											class="text-xs text-red-400 hover:text-red-600">✕</button>
									{/if}
								{/if}
							</td>
						</tr>
					{/each}

					<!-- Totals -->
					<tr class="border-t bg-gray-100 font-bold">
						<td></td>
						<td class="px-3 py-2 text-xs" colspan="2">Total (excl. savings)</td>
						<td class="px-3 py-2 text-xs text-left" dir="ltr">{fmtNum(totalBudget)}</td>
						<td class="px-3 py-2 text-xs text-left" dir="ltr">{fmtNum(totalAvg)}</td>
						<td class="px-3 py-2 text-xs text-left {totalBudget - totalAvg < 0 ? 'text-red-500' : 'text-green-600'}" dir="ltr">
							{totalBudget - totalAvg > 0 ? '+' : ''}{fmtNum(totalBudget - totalAvg)}
						</td>
						<td colspan="2"></td>
					</tr>
				</tbody>
			</table>
		</div>

		{#if dirty}
			<p class="mt-3 text-xs text-amber-600">
				Unsaved changes. Click "Save to Template" to apply, or "Revert All" to undo.
			</p>
		{/if}

		<p class="mt-4 text-xs text-gray-400">
			Budget is read from the Month Template. Changes are saved to both the current year's template
			and the master template spreadsheet. Savings budget is auto-calculated (income minus other budgets).
		</p>
	{/if}
</div>

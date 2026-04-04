<script lang="ts">
	import { get, put } from '$lib/api';
	import ErrorBanner from '$lib/components/ErrorBanner.svelte';

	type BudgetLine = {
		category: string; subcategory: string;
		budget_amount: number; carry_status: string; row_number: number;
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

	// Track original values for dirty detection
	let originalAmounts = $state<Record<number, number>>({});

	$effect(() => {
		get<any>('/budget/')
			.then(data => {
				lines = data.lines;
				averages = data.averages;
				monthCount = data.month_count;
				year = data.year;
				originalAmounts = {};
				for (const l of data.lines) {
					originalAmounts[l.row_number] = l.budget_amount;
				}
				loading = false;
			})
			.catch(e => { error = e.message; loading = false; });
	});

	function getAvg(line: BudgetLine): number {
		if (line.subcategory) {
			return averages[`${line.category}/${line.subcategory}`] || 0;
		}
		return averages[line.category] || 0;
	}

	function handleAmountChange(line: BudgetLine, val: string) {
		line.budget_amount = parseFloat(val) || 0;
		lines = [...lines];
		dirty = Object.entries(originalAmounts).some(
			([row, orig]) => {
				const current = lines.find(l => l.row_number === Number(row));
				return current && current.budget_amount !== orig;
			}
		);
		success = '';
	}

	const totalBudget = $derived(
		lines.filter(l => l.category !== 'חסכון').reduce((sum, l) => sum + l.budget_amount, 0)
	);

	const totalAvg = $derived(
		lines.filter(l => l.category !== 'חסכון').reduce((sum, l) => sum + getAvg(l), 0)
	);

	async function saveBudget() {
		saving = true; error = ''; success = '';
		try {
			const updates = lines
				.filter(l => l.budget_amount !== originalAmounts[l.row_number])
				.map(l => ({ row_number: l.row_number, budget_amount: l.budget_amount }));

			if (updates.length === 0) {
				success = 'No changes to save';
				saving = false;
				return;
			}

			const resp = await put<any>('/budget/', { updates });
			dirty = false;
			for (const l of lines) {
				originalAmounts[l.row_number] = l.budget_amount;
			}
			const parts = [`${resp.lines_updated} budget lines updated`];
			if (resp.template_updated) parts.push('template updated');
			success = parts.join(', ');
		} catch (e: any) { error = e.message; }
		saving = false;
	}

	function fmtNum(n: number) { return n.toLocaleString(undefined, { maximumFractionDigits: 0 }); }
</script>

<div class="max-w-4xl">
	<div class="flex items-center justify-between mb-2">
		<h1 class="text-2xl font-bold text-primary-800">Budget Planning</h1>
		{#if dirty}
			<button
				onclick={saveBudget}
				disabled={saving}
				class="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 text-sm font-medium"
			>{saving ? 'Saving...' : 'Save to Template'}</button>
		{/if}
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
		<div class="bg-white rounded-xl shadow-sm p-4" style="border: 1px solid #b3dbe9;">
			<table class="w-full text-sm" dir="rtl">
				<thead style="background: #f0f7fa;">
					<tr>
						<th class="px-3 py-2 text-right text-xs font-medium text-gray-600">Category</th>
						<th class="px-3 py-2 text-right text-xs font-medium text-gray-600">Subcategory</th>
						<th class="px-3 py-2 text-left text-xs font-medium text-gray-600" style="width: 120px;">Budget</th>
						<th class="px-3 py-2 text-left text-xs font-medium text-gray-600">Avg Spent</th>
						<th class="px-3 py-2 text-left text-xs font-medium text-gray-600">Diff</th>
						<th class="px-3 py-2 text-center text-xs font-medium text-gray-600" style="width: 70px;">Flag</th>
					</tr>
				</thead>
				<tbody>
					{#each lines as line}
						{@const avg = getAvg(line)}
						{@const diff = line.budget_amount - avg}
						{@const isSavings = line.category === 'חסכון'}
						<tr class="border-t {isSavings ? 'bg-gray-50 text-gray-400' : 'hover:bg-gray-50'}">
							<td class="px-3 py-1.5 text-xs font-medium">{line.category}</td>
							<td class="px-3 py-1.5 text-xs">{line.subcategory}</td>
							<td class="px-3 py-1.5" dir="ltr">
								{#if isSavings}
									<span class="text-xs text-gray-400 italic">auto</span>
								{:else}
									<input
										type="number"
										value={line.budget_amount}
										class="text-xs border rounded px-2 py-1 w-full text-right {line.budget_amount !== originalAmounts[line.row_number] ? 'border-amber-400 bg-amber-50' : ''}"
										style="border-color: {line.budget_amount !== originalAmounts[line.row_number] ? '#f59e0b' : '#d1d5db'};"
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
							<td class="px-3 py-1.5 text-center">
								{#if line.carry_status === 'CARRY'}
									<span class="text-xs px-1 rounded" style="background: #fdf8ed; color: #b87420;">CARRY</span>
								{:else if line.carry_status === 'IGNORE'}
									<span class="text-xs px-1 rounded" style="background: #f0f0f0; color: #6b7280;">IGNORE</span>
								{/if}
							</td>
						</tr>
					{/each}

					<!-- Totals row (excluding savings) -->
					<tr class="border-t bg-gray-100 font-bold">
						<td class="px-3 py-2 text-xs" colspan="2">Total (excl. savings)</td>
						<td class="px-3 py-2 text-xs text-left" dir="ltr">{fmtNum(totalBudget)}</td>
						<td class="px-3 py-2 text-xs text-left" dir="ltr">{fmtNum(totalAvg)}</td>
						<td class="px-3 py-2 text-xs text-left {totalBudget - totalAvg < 0 ? 'text-red-500' : 'text-green-600'}" dir="ltr">
							{totalBudget - totalAvg > 0 ? '+' : ''}{fmtNum(totalBudget - totalAvg)}
						</td>
						<td></td>
					</tr>
				</tbody>
			</table>
		</div>

		{#if dirty}
			<p class="mt-3 text-xs text-amber-600">
				Unsaved changes (highlighted in amber). Click "Save to Template" to apply.
			</p>
		{/if}

		<p class="mt-4 text-xs text-gray-400">
			Budget is read from the Month Template. Changes are saved to both the current year's template
			and the master template spreadsheet. Savings budget is auto-calculated (income minus other budgets).
		</p>
	{/if}
</div>

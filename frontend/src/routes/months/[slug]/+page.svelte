<script lang="ts">
	import { page } from '$app/state';
	import { get } from '$lib/api';

	// Parse slug: "2026-January"
	const slug = $derived(page.params.slug);
	const year = $derived(Number(slug.split('-')[0]));
	const month = $derived(slug.split('-')[1]);

	type Expense = { index: number; business_name: string; notes: string; subcategory: string; amount: string; category: string; date: string; status: string };
	type Income = { index: number; comments: string; amount: string; category: string; date: string };
	type BudgetLine = { category: string; subcategory: string; budget_amount: number; accumulated: number; remaining: number; carry_flag: boolean; carry_status: string; row_number: number };
	type SavingsLine = { goal_name: string; target: number; accumulated: number; incoming: number; outgoing: number; total: number };

	let activeTab = $state<'expenses' | 'budget' | 'income'>('expenses');
	let expenses = $state<Expense[]>([]);
	let income = $state<Income[]>([]);
	let budgetLines = $state<BudgetLine[]>([]);
	let savingsLines = $state<SavingsLine[]>([]);
	let savingsBudget = $state<number | null>(null);
	let incomeTotal = $state<number | null>(null);
	let sheetUrl = $state('');
	let loading = $state(false);
	let error = $state('');

	// Sort state
	let sortKey = $state('');
	let sortAsc = $state(true);

	function toggleSort(key: string) {
		if (sortKey === key) {
			sortAsc = !sortAsc;
		} else {
			sortKey = key;
			sortAsc = true;
		}
	}

	function sortIndicator(key: string): string {
		if (sortKey !== key) return '';
		return sortAsc ? ' ▲' : ' ▼';
	}

	function sorted<T>(items: T[], key: string): T[] {
		if (!sortKey) return items;
		return [...items].sort((a: any, b: any) => {
			let va = a[key], vb = b[key];
			// Try numeric comparison
			const na = parseFloat(va), nb = parseFloat(vb);
			if (!isNaN(na) && !isNaN(nb)) {
				return sortAsc ? na - nb : nb - na;
			}
			// String comparison
			va = String(va || ''); vb = String(vb || '');
			return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
		});
	}

	async function loadTab(tab: typeof activeTab) {
		activeTab = tab;
		loading = true; error = '';
		sortKey = ''; sortAsc = true; // Reset sort when switching tabs
		try {
			if (tab === 'expenses') {
				const data = await get<{ expenses: Expense[] }>(`/months/${year}/${month}/expenses`);
				expenses = data.expenses;
			} else if (tab === 'budget') {
				const data = await get<any>(`/months/${year}/${month}/budget`);
				budgetLines = data.budget_lines;
				savingsLines = data.savings_lines;
				savingsBudget = data.savings_budget;
				incomeTotal = data.income_total;
			} else {
				const data = await get<{ income: Income[] }>(`/months/${year}/${month}/income`);
				income = data.income;
			}
		} catch (e: any) { error = e.message; }
		loading = false;
	}

	$effect(() => {
		loadTab('expenses');
		get<{ url: string }>(`/months/${year}/${month}/link`)
			.then(data => sheetUrl = data.url)
			.catch(() => {});
	});

	function fmtNum(n: number) { return n.toLocaleString(undefined, { maximumFractionDigits: 2 }); }
</script>

<div class="max-w-5xl">
	<div class="flex items-center justify-between mb-6">
		<div>
			<a href="/months" class="text-sm text-gray-400 hover:text-primary-600">&larr; All Months</a>
			<h1 class="text-2xl font-bold text-primary-800">{month} {year}</h1>
		</div>
		{#if sheetUrl}
			<a href={sheetUrl} target="_blank" rel="noopener"
				class="px-3 py-1.5 text-sm rounded border transition-colors"
				style="color: #2f6577; border-color: #b3dbe9;"
			>Open in Google Sheets ↗</a>
		{/if}
	</div>

	<!-- Tabs -->
	<div class="flex gap-1 mb-4">
		{#each [['expenses', 'Expenses'], ['budget', 'Budget & Savings'], ['income', 'Income']] as [tab, label]}
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

		{:else if activeTab === 'expenses'}
			<div class="overflow-x-auto">
				<!-- Column order: B(Business) C(Notes) D(Subcat) E(Amount) F(Category) G(Date) H(Status) -->
				<table class="w-full text-sm">
					<thead style="background: #f0f7fa;">
						<tr>
							<th class="sortable-th rtl" onclick={() => toggleSort('business_name')}>Business{sortIndicator('business_name')}</th>
							<th class="sortable-th rtl" onclick={() => toggleSort('notes')}>Notes{sortIndicator('notes')}</th>
							<th class="sortable-th rtl" onclick={() => toggleSort('subcategory')}>Subcat{sortIndicator('subcategory')}</th>
							<th class="sortable-th text-right" onclick={() => toggleSort('amount')}>Amount{sortIndicator('amount')}</th>
							<th class="sortable-th rtl" onclick={() => toggleSort('category')}>Category{sortIndicator('category')}</th>
							<th class="sortable-th" onclick={() => toggleSort('date')}>Date{sortIndicator('date')}</th>
							<th class="sortable-th" onclick={() => toggleSort('status')}>Status{sortIndicator('status')}</th>
						</tr>
					</thead>
					<tbody>
						{#each sorted(expenses, sortKey) as exp}
							<tr class="border-t hover:bg-gray-50">
								<td class="px-2 py-1 text-xs rtl">{exp.business_name}</td>
								<td class="px-2 py-1 text-xs rtl">{exp.notes}</td>
								<td class="px-2 py-1 text-xs rtl">{exp.subcategory}</td>
								<td class="px-2 py-1 text-xs text-right">{exp.amount}</td>
								<td class="px-2 py-1 text-xs rtl">{exp.category}</td>
								<td class="px-2 py-1 text-xs whitespace-nowrap">{exp.date}</td>
								<td class="px-2 py-1">
									{#if exp.status === 'CC'}
										<span class="text-xs px-1 rounded" style="background: #d9edf4; color: #2f6577;">CC</span>
									{:else if exp.status === 'BANK'}
										<span class="text-xs bg-green-100 text-green-700 px-1 rounded">BANK</span>
									{:else}
										<span class="text-xs text-gray-400">pending</span>
									{/if}
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
				{#if expenses.length === 0}
					<p class="text-gray-400 text-sm py-4 text-center">No expenses recorded.</p>
				{/if}
			</div>

		{:else if activeTab === 'budget'}
			<div class="space-y-6">
				<div>
					<h3 class="text-sm font-medium text-primary-600 mb-2">Budget</h3>
					{#if savingsBudget != null}
						<p class="text-xs text-gray-500 mb-2">Savings budget: {fmtNum(savingsBudget)} ₪
							{#if incomeTotal != null} (from income: {fmtNum(incomeTotal)} ₪){/if}
						</p>
					{/if}
					<div class="overflow-x-auto">
						<!-- Column order: B(Remaining) D(Budget) E(Accumulated) F(Subcat) G(Category) H(Flag) -->
						<table class="w-full text-sm">
							<thead style="background: #f0f7fa;">
								<tr>
									<th class="sortable-th text-right" onclick={() => toggleSort('remaining')}>Remaining{sortIndicator('remaining')}</th>
									<th class="sortable-th text-right" onclick={() => toggleSort('budget_amount')}>Budget{sortIndicator('budget_amount')}</th>
									<th class="sortable-th text-right" onclick={() => toggleSort('accumulated')}>Accumulated{sortIndicator('accumulated')}</th>
									<th class="sortable-th rtl" onclick={() => toggleSort('subcategory')}>Subcat{sortIndicator('subcategory')}</th>
									<th class="sortable-th rtl" onclick={() => toggleSort('category')}>Category{sortIndicator('category')}</th>
									<th class="sortable-th text-center" onclick={() => toggleSort('carry_status')}>Flag{sortIndicator('carry_status')}</th>
								</tr>
							</thead>
							<tbody>
								{#each sorted(budgetLines, sortKey) as bl}
									<tr class="border-t hover:bg-gray-50">
										<td class="px-2 py-1 text-xs text-right font-medium
											{bl.remaining < 0 ? 'text-red-500' : bl.remaining > bl.budget_amount ? 'text-green-600' : ''}">
											{fmtNum(bl.remaining)}
										</td>
										<td class="px-2 py-1 text-xs text-right">{fmtNum(bl.budget_amount)}</td>
										<td class="px-2 py-1 text-xs text-right">{bl.accumulated ? fmtNum(bl.accumulated) : ''}</td>
										<td class="px-2 py-1 text-xs rtl">{bl.subcategory}</td>
										<td class="px-2 py-1 text-xs rtl font-medium">{bl.category}</td>
										<td class="px-2 py-1 text-center">
											{#if bl.carry_status === 'CARRY'}
												<span class="text-xs px-1 rounded" style="background: #fdf8ed; color: #b87420;">CARRY</span>
											{:else if bl.carry_status === 'IGNORE'}
												<span class="text-xs px-1 rounded" style="background: #f0f0f0; color: #6b7280;">IGNORE</span>
											{/if}
										</td>
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
				</div>

				<div>
					<h3 class="text-sm font-medium text-primary-600 mb-2">Savings Goals</h3>
					<div class="overflow-x-auto">
						<!-- Column order: B(Target) C(Total) D(Outgoing) E(Incoming) F(Accumulated) G(Goal) -->
						<table class="w-full text-sm">
							<thead style="background: #f0f7fa;">
								<tr>
									<th class="sortable-th text-right" onclick={() => toggleSort('target')}>Target{sortIndicator('target')}</th>
									<th class="sortable-th text-right" onclick={() => toggleSort('total')}>Total{sortIndicator('total')}</th>
									<th class="sortable-th text-right" onclick={() => toggleSort('outgoing')}>Outgoing{sortIndicator('outgoing')}</th>
									<th class="sortable-th text-right" onclick={() => toggleSort('incoming')}>Incoming{sortIndicator('incoming')}</th>
									<th class="sortable-th text-right" onclick={() => toggleSort('accumulated')}>Accumulated{sortIndicator('accumulated')}</th>
									<th class="sortable-th rtl" onclick={() => toggleSort('goal_name')}>Goal{sortIndicator('goal_name')}</th>
								</tr>
							</thead>
							<tbody>
								{#each sorted(savingsLines, sortKey) as sl}
									<tr class="border-t hover:bg-gray-50">
										<td class="px-2 py-1 text-xs text-right">{sl.target ? fmtNum(sl.target) : ''}</td>
										<td class="px-2 py-1 text-xs text-right font-medium">{fmtNum(sl.total)}</td>
										<td class="px-2 py-1 text-xs text-right">{sl.outgoing ? fmtNum(sl.outgoing) : ''}</td>
										<td class="px-2 py-1 text-xs text-right">{sl.incoming ? fmtNum(sl.incoming) : ''}</td>
										<td class="px-2 py-1 text-xs text-right">{sl.accumulated ? fmtNum(sl.accumulated) : ''}</td>
										<td class="px-2 py-1 text-xs rtl font-medium">{sl.goal_name}</td>
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
				</div>
			</div>

		{:else if activeTab === 'income'}
			<div class="overflow-x-auto">
				<!-- Column order: D(Comments) E(Amount) F(Category) G(Date) -->
				<table class="w-full text-sm">
					<thead style="background: #f0f7fa;">
						<tr>
							<th class="sortable-th rtl" onclick={() => toggleSort('comments')}>Comments{sortIndicator('comments')}</th>
							<th class="sortable-th text-right" onclick={() => toggleSort('amount')}>Amount{sortIndicator('amount')}</th>
							<th class="sortable-th rtl" onclick={() => toggleSort('category')}>Category{sortIndicator('category')}</th>
							<th class="sortable-th" onclick={() => toggleSort('date')}>Date{sortIndicator('date')}</th>
						</tr>
					</thead>
					<tbody>
						{#each sorted(income, sortKey) as inc}
							<tr class="border-t hover:bg-gray-50">
								<td class="px-2 py-1 text-xs rtl">{inc.comments}</td>
								<td class="px-2 py-1 text-xs text-right">{inc.amount}</td>
								<td class="px-2 py-1 text-xs rtl">{inc.category}</td>
								<td class="px-2 py-1 text-xs whitespace-nowrap">{inc.date}</td>
							</tr>
						{/each}
					</tbody>
				</table>
				{#if income.length === 0}
					<p class="text-gray-400 text-sm py-4 text-center">No income recorded.</p>
				{/if}
			</div>
		{/if}
	</div>
</div>

<style>
	.sortable-th {
		padding: 0.375rem 0.5rem;
		font-size: 0.75rem;
		font-weight: 500;
		color: #4b5563;
		cursor: pointer;
		user-select: none;
		white-space: nowrap;
	}
	.sortable-th:hover {
		color: #2f6577;
		background: #e8f0f4;
	}
</style>

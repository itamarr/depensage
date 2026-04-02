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

	async function loadTab(tab: typeof activeTab) {
		activeTab = tab;
		loading = true; error = '';
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
				<table class="w-full text-sm">
					<thead style="background: #f0f7fa;">
						<!-- Column order matches spreadsheet B→H -->
						<tr>
							<th class="px-2 py-1.5 text-right text-xs font-medium text-gray-600">Status</th>
							<th class="px-2 py-1.5 text-left text-xs font-medium text-gray-600">Date</th>
							<th class="px-2 py-1.5 text-left text-xs font-medium text-gray-600 rtl">Category</th>
							<th class="px-2 py-1.5 text-right text-xs font-medium text-gray-600">Amount</th>
							<th class="px-2 py-1.5 text-left text-xs font-medium text-gray-600 rtl">Subcat</th>
							<th class="px-2 py-1.5 text-left text-xs font-medium text-gray-600 rtl">Notes</th>
							<th class="px-2 py-1.5 text-left text-xs font-medium text-gray-600 rtl">Business</th>
						</tr>
					</thead>
					<tbody>
						{#each expenses as exp}
							<tr class="border-t hover:bg-gray-50">
								<td class="px-2 py-1">
									{#if exp.status === 'CC'}
										<span class="text-xs px-1 rounded" style="background: #d9edf4; color: #2f6577;">CC</span>
									{:else if exp.status === 'BANK'}
										<span class="text-xs bg-green-100 text-green-700 px-1 rounded">BANK</span>
									{:else}
										<span class="text-xs text-gray-400">pending</span>
									{/if}
								</td>
								<td class="px-2 py-1 text-xs whitespace-nowrap">{exp.date}</td>
								<td class="px-2 py-1 text-xs rtl">{exp.category}</td>
								<td class="px-2 py-1 text-xs text-right">{exp.amount}</td>
								<td class="px-2 py-1 text-xs rtl">{exp.subcategory}</td>
								<td class="px-2 py-1 text-xs rtl">{exp.notes}</td>
								<td class="px-2 py-1 text-xs rtl">{exp.business_name}</td>
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
				<!-- Budget lines -->
				<div>
					<h3 class="text-sm font-medium text-primary-600 mb-2">Budget</h3>
					{#if savingsBudget != null}
						<p class="text-xs text-gray-500 mb-2">Savings budget: {fmtNum(savingsBudget)} ₪
							{#if incomeTotal != null} (from income: {fmtNum(incomeTotal)} ₪){/if}
						</p>
					{/if}
					<div class="overflow-x-auto">
						<!-- Column order matches spreadsheet B→H -->
						<table class="w-full text-sm">
							<thead style="background: #f0f7fa;">
								<tr>
									<th class="px-2 py-1.5 text-right text-xs font-medium text-gray-600">Remaining</th>
									<th class="px-2 py-1.5 text-right text-xs font-medium text-gray-600">Budget</th>
									<th class="px-2 py-1.5 text-right text-xs font-medium text-gray-600">Accumulated</th>
									<th class="px-2 py-1.5 text-left text-xs font-medium text-gray-600 rtl">Subcat</th>
									<th class="px-2 py-1.5 text-left text-xs font-medium text-gray-600 rtl">Category</th>
									<th class="px-2 py-1.5 text-center text-xs font-medium text-gray-600">Flag</th>
								</tr>
							</thead>
							<tbody>
								{#each budgetLines as bl}
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

				<!-- Savings lines -->
				<div>
					<h3 class="text-sm font-medium text-primary-600 mb-2">Savings Goals</h3>
					<div class="overflow-x-auto">
						<!-- Column order matches spreadsheet A→G -->
						<table class="w-full text-sm">
							<thead style="background: #f0f7fa;">
								<tr>
									<th class="px-2 py-1.5 text-right text-xs font-medium text-gray-600">Target</th>
									<th class="px-2 py-1.5 text-right text-xs font-medium text-gray-600">Total</th>
									<th class="px-2 py-1.5 text-right text-xs font-medium text-gray-600">Outgoing</th>
									<th class="px-2 py-1.5 text-right text-xs font-medium text-gray-600">Incoming</th>
									<th class="px-2 py-1.5 text-right text-xs font-medium text-gray-600">Accumulated</th>
									<th class="px-2 py-1.5 text-left text-xs font-medium text-gray-600 rtl">Goal</th>
								</tr>
							</thead>
							<tbody>
								{#each savingsLines as sl}
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
				<table class="w-full text-sm">
					<thead style="background: #f0f7fa;">
						<tr>
							<th class="px-2 py-1.5 text-left text-xs font-medium text-gray-600">Date</th>
							<th class="px-2 py-1.5 text-left text-xs font-medium text-gray-600 rtl">Comments</th>
							<th class="px-2 py-1.5 text-right text-xs font-medium text-gray-600">Amount</th>
							<th class="px-2 py-1.5 text-left text-xs font-medium text-gray-600 rtl">Category</th>
						</tr>
					</thead>
					<tbody>
						{#each income as inc}
							<tr class="border-t hover:bg-gray-50">
								<td class="px-2 py-1 text-xs whitespace-nowrap">{inc.date}</td>
								<td class="px-2 py-1 text-xs rtl">{inc.comments}</td>
								<td class="px-2 py-1 text-xs text-right">{inc.amount}</td>
								<td class="px-2 py-1 text-xs rtl">{inc.category}</td>
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

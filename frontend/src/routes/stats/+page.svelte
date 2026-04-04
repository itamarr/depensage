<script lang="ts">
	import { get } from '$lib/api';
	import ErrorBanner from '$lib/components/ErrorBanner.svelte';
	import { onMount } from 'svelte';
	import { Chart, registerables } from 'chart.js';

	Chart.register(...registerables);

	type ExpenseRow = { category: string; subcategory: string; total: number; average: number; is_total?: boolean; is_grand?: boolean };
	type IncomeRow = { category: string; details: string; total: number; average: number; is_grand?: boolean };
	type MonthData = { month: string; expenses: number; income: number; savings_budget: number };
	type SavingsGoal = { goal_name: string; target: number; total: number; progress: number };

	let error = $state('');
	let loading = $state(true);
	let year = $state(0);
	let availableYears = $state<number[]>([]);
	let monthCount = $state(0);

	let expenseRows = $state<ExpenseRow[]>([]);
	let budget = $state<Record<string, number>>({});
	let incomeRows = $state<IncomeRow[]>([]);
	let monthlyData = $state<MonthData[]>([]);
	let savingsGoals = $state<SavingsGoal[]>([]);
	let savingsMonth = $state('');

	// Collapsible sections
	let showBudget = $state(true);
	let showPie = $state(true);
	let showTrends = $state(true);
	let showIncome = $state(true);
	let showSavings = $state(true);

	// Collapsible categories in budget table
	let expandedCats = $state<Set<string>>(new Set());

	function toggleCat(cat: string) {
		const s = new Set(expandedCats);
		if (s.has(cat)) s.delete(cat); else s.add(cat);
		expandedCats = s;
	}

	// Chart canvases
	let budgetChartEl: HTMLCanvasElement;
	let pieChartEl: HTMLCanvasElement;
	let trendChartEl: HTMLCanvasElement;
	let savingsChartEl: HTMLCanvasElement;
	let charts: Chart[] = [];

	async function loadStats(targetYear?: number) {
		loading = true; error = '';
		try {
			const qs = targetYear ? `?year=${targetYear}` : '';
			const [yearsResp, expResp, incResp, monthResp, savResp] = await Promise.all([
				get<any>('/stats/years'),
				get<any>(`/stats/expenses${qs}`),
				get<any>(`/stats/income${qs}`),
				get<any>(`/stats/monthly${qs}`),
				get<any>(`/stats/savings${qs}`),
			]);
			availableYears = yearsResp.years;
			year = expResp.year;
			monthCount = expResp.month_count;
			expenseRows = expResp.rows;
			budget = expResp.budget || {};
			incomeRows = incResp.rows;
			monthlyData = monthResp.months;
			savingsGoals = savResp.goals;
			savingsMonth = savResp.month || '';
		} catch (e: any) { error = e.message; }
		loading = false;
	}

	onMount(() => loadStats());

	$effect(() => {
		if (loading) return;
		charts.forEach(c => c.destroy());
		charts = [];
		setTimeout(() => {
			renderBudgetChart();
			renderPieChart();
			renderTrendChart();
			renderSavingsChart();
		}, 50);
	});

	function getBudgetForCategory(cat: string): number {
		// Direct category budget
		if (budget[cat]) return budget[cat];
		// Sum per-subcategory budgets (e.g., שונות/מזון בחוץ + שונות/ביגוד + ...)
		let sum = 0;
		for (const [key, val] of Object.entries(budget)) {
			if (key.startsWith(cat + '/')) sum += val;
		}
		return sum;
	}

	// Budget vs Actual bar chart — category totals with budget comparison (exclude savings)
	function renderBudgetChart() {
		if (!budgetChartEl || !expenseRows.length) return;
		const cats = expenseRows.filter(r => r.is_total && !r.is_grand && r.category !== 'חסכון');
		const budgetVals = cats.map(r => getBudgetForCategory(r.category));
		charts.push(new Chart(budgetChartEl, {
			type: 'bar',
			data: {
				labels: cats.map(r => r.category),
				datasets: [
					{ label: 'Monthly Budget', data: budgetVals, backgroundColor: '#b3dbe9' },
					{ label: 'Monthly Avg Spent', data: cats.map(r => r.average), backgroundColor: '#e5af42' },
				],
			},
			options: {
				responsive: true,
				plugins: { legend: { position: 'top' } },
				scales: { y: { beginAtZero: true } },
			},
		}));
	}

	// Pie chart — only category totals, no Grand Total or subcategories
	function renderPieChart() {
		if (!pieChartEl || !expenseRows.length) return;
		const cats = expenseRows.filter(r => r.is_total && !r.is_grand && r.total > 0 && r.category !== 'חסכון');
		const colors = ['#4a9ab4','#e5af42','#2f6577','#ebc46c','#7cc0d6','#b87420','#99561d','#3a7d94','#d4952a','#295463','#f2da9e','#264652','#68391c','#1e3844'];
		charts.push(new Chart(pieChartEl, {
			type: 'pie',
			data: {
				labels: cats.map(r => r.category),
				datasets: [{ data: cats.map(r => r.total), backgroundColor: colors.slice(0, cats.length) }],
			},
			options: { responsive: true, plugins: { legend: { position: 'right' } } },
		}));
	}

	function renderTrendChart() {
		if (!trendChartEl || !monthlyData.length) return;
		charts.push(new Chart(trendChartEl, {
			type: 'line',
			data: {
				labels: monthlyData.map(m => m.month.slice(0, 3)),
				datasets: [
					{ label: 'Expenses', data: monthlyData.map(m => m.expenses), borderColor: '#e5af42', backgroundColor: '#e5af4233', fill: true },
					{ label: 'Income', data: monthlyData.map(m => m.income), borderColor: '#4a9ab4', backgroundColor: '#4a9ab433', fill: true },
				],
			},
			options: { responsive: true, plugins: { legend: { position: 'top' } }, scales: { y: { beginAtZero: true } } },
		}));
	}

	// Savings chart — vertical bars (not horizontal)
	function renderSavingsChart() {
		if (!savingsChartEl || !savingsGoals.length) return;
		const goals = savingsGoals.filter(g => g.target > 0);
		charts.push(new Chart(savingsChartEl, {
			type: 'bar',
			data: {
				labels: goals.map(g => g.goal_name),
				datasets: [
					{ label: 'Current', data: goals.map(g => g.total), backgroundColor: '#4a9ab4' },
					{ label: 'Target', data: goals.map(g => g.target), backgroundColor: '#e5e7eb' },
				],
			},
			options: {
				responsive: true,
				plugins: { legend: { position: 'top' } },
				scales: { y: { beginAtZero: true } },
			},
		}));
	}

	function fmtNum(n: number) { return n.toLocaleString(undefined, { maximumFractionDigits: 0 }); }

	// Group expense rows by category for collapsible display
	type CatGroup = { category: string; total: number; average: number; budget: number; subcats: ExpenseRow[] };
	const catGroups = $derived((): CatGroup[] => {
		const groups: CatGroup[] = [];
		let current: CatGroup | null = null;

		for (const row of expenseRows) {
			if (row.is_grand) continue;
			if (row.category === 'חסכון') continue;  // Exclude savings
			if (row.is_total) {
				if (current) {
					current.total = row.total;
					current.average = row.average;
					groups.push(current);
					current = null;
				}
				continue;
			}
			if (row.category && !row.subcategory) {
				// Category with no subcategory (single row)
				if (current) groups.push(current);
				current = { category: row.category, total: row.total, average: row.average, budget: budget[row.category] || 0, subcats: [] };
			} else if (row.category && row.subcategory) {
				// First subcategory of a new category
				if (current) groups.push(current);
				current = { category: row.category, total: 0, average: 0, budget: 0, subcats: [] };
				current.subcats.push(row);
			} else if (!row.category && row.subcategory && current) {
				// Additional subcategory
				current.subcats.push(row);
			}
		}
		if (current) groups.push(current);

		// Fill budget for groups:
		// - If category has a direct budget entry (no subcats in budget), use it
		// - If category has per-subcat budget entries (like שונות), sum them
		for (const g of groups) {
			if (g.budget === 0) {
				// Check direct category budget first
				const directBudget = budget[g.category];
				if (directBudget) {
					g.budget = directBudget;
				} else if (g.subcats.length > 0) {
					// Sum per-subcat budgets
					g.budget = g.subcats.reduce((sum, s) => sum + (budget[`${g.category}/${s.subcategory}`] || 0), 0);
				}
			}
		}
		return groups;
	});

	const grandTotal = $derived(expenseRows.find(r => r.is_grand));
</script>

<div class="max-w-5xl">
	<div class="flex items-center justify-between mb-2">
		<h1 class="text-2xl font-bold text-primary-800">Statistics</h1>
		{#if availableYears.length > 1}
			<select
				value={year}
				onchange={(e) => loadStats(Number((e.target as HTMLSelectElement).value))}
				class="border rounded px-3 py-1.5 text-sm"
				style="border-color: #b3dbe9;"
			>
				{#each availableYears as y}
					<option value={y}>{y}</option>
				{/each}
			</select>
		{/if}
	</div>
	{#if year}
		<p class="text-sm text-gray-500 mb-6">{year} — {monthCount} month{monthCount !== 1 ? 's' : ''} of data</p>
	{/if}

	{#if error}
		<ErrorBanner message={error} ondismiss={() => error = ''} />
	{/if}

	{#if loading}
		<p class="text-gray-400 text-sm py-8 text-center">Loading statistics...</p>
	{:else}
		<!-- Section 1: Budget vs Actual -->
		<div class="mb-6 bg-white rounded-xl shadow-sm" style="border: 1px solid #b3dbe9;">
			<button onclick={() => showBudget = !showBudget} class="w-full text-left p-4 flex items-center justify-between">
				<h2 class="text-lg font-semibold text-primary-700">Budget vs Actual</h2>
				<span class="text-gray-400">{showBudget ? '▼' : '▶'}</span>
			</button>
			{#if showBudget}
				<div class="px-4 pb-4 space-y-4">
					{#if catGroups().length > 0}
						<div style="max-height: 400px;"><canvas bind:this={budgetChartEl}></canvas></div>
						<div class="overflow-x-auto">
							<table class="w-full text-sm" dir="rtl">
								<thead style="background: #f0f7fa;">
									<tr>
										<th class="px-2 py-1.5 text-right text-xs font-medium text-gray-600">Category</th>
										<th class="px-2 py-1.5 text-left text-xs font-medium text-gray-600">Budget</th>
										<th class="px-2 py-1.5 text-left text-xs font-medium text-gray-600">Total</th>
										<th class="px-2 py-1.5 text-left text-xs font-medium text-gray-600">Monthly Avg</th>
									</tr>
								</thead>
								<tbody>
									{#each catGroups() as group}
										<tr
											class="border-t font-medium cursor-pointer hover:bg-gray-50"
											onclick={() => { if (group.subcats.length > 0) toggleCat(group.category); }}
										>
											<td class="px-2 py-1.5 text-xs">
												{#if group.subcats.length > 0}
													<span class="text-gray-400 mr-1">{expandedCats.has(group.category) ? '▼' : '▶'}</span>
												{/if}
												{group.category}
											</td>
											<td class="px-2 py-1.5 text-xs text-left">{group.budget ? fmtNum(group.budget) : '—'}</td>
											<td class="px-2 py-1.5 text-xs text-left">{fmtNum(group.total)}</td>
											<td class="px-2 py-1.5 text-xs text-left {group.average > group.budget && group.budget > 0 ? 'text-red-500' : ''}">{fmtNum(group.average)}</td>
										</tr>
										{#if expandedCats.has(group.category)}
											{#each group.subcats as sub}
												<tr class="border-t bg-gray-50">
													<td class="px-2 py-1 text-xs pr-6">{sub.subcategory}</td>
													<td class="px-2 py-1 text-xs text-left">{budget[`${group.category}/${sub.subcategory}`] ? fmtNum(budget[`${group.category}/${sub.subcategory}`]) : '—'}</td>
													<td class="px-2 py-1 text-xs text-left">{fmtNum(sub.total)}</td>
													<td class="px-2 py-1 text-xs text-left">{fmtNum(sub.average)}</td>
												</tr>
											{/each}
										{/if}
									{/each}
									{#if grandTotal}
										<tr class="border-t bg-gray-100 font-bold">
											<td class="px-2 py-1.5 text-xs">Grand Total</td>
											<td class="px-2 py-1.5 text-xs text-left"></td>
											<td class="px-2 py-1.5 text-xs text-left">{fmtNum(grandTotal.total)}</td>
											<td class="px-2 py-1.5 text-xs text-left">{fmtNum(grandTotal.average)}</td>
										</tr>
									{/if}
								</tbody>
							</table>
						</div>
					{:else}
						<p class="text-sm text-gray-400">No expense data available.</p>
					{/if}
				</div>
			{/if}
		</div>

		<!-- Section 2: Spending by Category (Pie) -->
		<div class="mb-6 bg-white rounded-xl shadow-sm" style="border: 1px solid #b3dbe9;">
			<button onclick={() => showPie = !showPie} class="w-full text-left p-4 flex items-center justify-between">
				<h2 class="text-lg font-semibold text-primary-700">Spending by Category</h2>
				<span class="text-gray-400">{showPie ? '▼' : '▶'}</span>
			</button>
			{#if showPie}
				<div class="px-4 pb-4">
					<div style="max-width: 500px; margin: 0 auto;"><canvas bind:this={pieChartEl}></canvas></div>
				</div>
			{/if}
		</div>

		<!-- Section 3: Monthly Trends -->
		<div class="mb-6 bg-white rounded-xl shadow-sm" style="border: 1px solid #b3dbe9;">
			<button onclick={() => showTrends = !showTrends} class="w-full text-left p-4 flex items-center justify-between">
				<h2 class="text-lg font-semibold text-primary-700">Monthly Trends</h2>
				<span class="text-gray-400">{showTrends ? '▼' : '▶'}</span>
			</button>
			{#if showTrends}
				<div class="px-4 pb-4">
					{#if monthlyData.length > 0}
						<canvas bind:this={trendChartEl}></canvas>
					{:else}
						<p class="text-sm text-gray-400">No monthly data.</p>
					{/if}
				</div>
			{/if}
		</div>

		<!-- Section 4: Income Summary -->
		<div class="mb-6 bg-white rounded-xl shadow-sm" style="border: 1px solid #b3dbe9;">
			<button onclick={() => showIncome = !showIncome} class="w-full text-left p-4 flex items-center justify-between">
				<h2 class="text-lg font-semibold text-primary-700">Income Summary</h2>
				<span class="text-gray-400">{showIncome ? '▼' : '▶'}</span>
			</button>
			{#if showIncome}
				<div class="px-4 pb-4">
					{#if incomeRows.length > 0}
						<table class="w-full text-sm" dir="rtl">
							<thead style="background: #f0f7fa;">
								<tr>
									<th class="px-2 py-1.5 text-right text-xs font-medium text-gray-600">Category</th>
									<th class="px-2 py-1.5 text-right text-xs font-medium text-gray-600">Details</th>
									<th class="px-2 py-1.5 text-left text-xs font-medium text-gray-600">Total</th>
									<th class="px-2 py-1.5 text-left text-xs font-medium text-gray-600">Monthly Avg</th>
								</tr>
							</thead>
							<tbody>
								{#each incomeRows as row}
									<tr class="border-t {row.is_grand ? 'bg-gray-100 font-bold' : 'hover:bg-gray-50'}">
										<td class="px-2 py-1 text-xs">{row.category}</td>
										<td class="px-2 py-1 text-xs">{row.details}</td>
										<td class="px-2 py-1 text-xs text-left">{fmtNum(row.total)}</td>
										<td class="px-2 py-1 text-xs text-left">{fmtNum(row.average)}</td>
									</tr>
								{/each}
							</tbody>
						</table>
					{:else}
						<p class="text-sm text-gray-400">No income data available.</p>
					{/if}
				</div>
			{/if}
		</div>

		<!-- Section 5: Savings Overview -->
		<div class="mb-6 bg-white rounded-xl shadow-sm" style="border: 1px solid #b3dbe9;">
			<button onclick={() => showSavings = !showSavings} class="w-full text-left p-4 flex items-center justify-between">
				<h2 class="text-lg font-semibold text-primary-700">
					Savings Overview
					{#if savingsMonth}<span class="text-sm font-normal text-gray-400 ml-2">(as of {savingsMonth})</span>{/if}
				</h2>
				<span class="text-gray-400">{showSavings ? '▼' : '▶'}</span>
			</button>
			{#if showSavings}
				<div class="px-4 pb-4 space-y-4">
					{#if savingsGoals.length > 0}
						<canvas bind:this={savingsChartEl}></canvas>
						<table class="w-full text-sm" dir="rtl">
							<thead style="background: #f0f7fa;">
								<tr>
									<th class="px-2 py-1.5 text-right text-xs font-medium text-gray-600">Goal</th>
									<th class="px-2 py-1.5 text-left text-xs font-medium text-gray-600">Target</th>
									<th class="px-2 py-1.5 text-left text-xs font-medium text-gray-600">Current</th>
									<th class="px-2 py-1.5 text-left text-xs font-medium text-gray-600">Progress</th>
								</tr>
							</thead>
							<tbody>
								{#each savingsGoals as goal}
									<tr class="border-t hover:bg-gray-50">
										<td class="px-2 py-1 text-xs font-medium">{goal.goal_name}</td>
										<td class="px-2 py-1 text-xs text-left">{goal.target ? fmtNum(goal.target) : '—'}</td>
										<td class="px-2 py-1 text-xs text-left">{fmtNum(goal.total)}</td>
										<td class="px-2 py-1 text-xs text-left">
											{#if goal.target > 0}
												<div class="flex items-center gap-2" style="direction:ltr;">
													<div class="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
														<div class="h-full rounded-full" style="width: {Math.min(goal.progress, 100)}%; background: {goal.progress >= 100 ? '#22c55e' : '#4a9ab4'};"></div>
													</div>
													<span class="text-xs {goal.progress >= 100 ? 'text-green-600' : ''}">{goal.progress}%</span>
												</div>
											{/if}
										</td>
									</tr>
								{/each}
							</tbody>
						</table>
					{:else}
						<p class="text-sm text-gray-400">No savings data.</p>
					{/if}
				</div>
			{/if}
		</div>
	{/if}
</div>

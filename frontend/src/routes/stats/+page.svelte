<script lang="ts">
	import { get } from '$lib/api';
	import ErrorBanner from '$lib/components/ErrorBanner.svelte';
	import { onMount } from 'svelte';
	import { Chart, registerables } from 'chart.js';

	Chart.register(...registerables);

	type ExpenseRow = { category: string; subcategory: string; total: number; average: number };
	type IncomeRow = { category: string; total: number; average: number };
	type MonthData = { month: string; expenses: number; income: number; savings_budget: number };
	type SavingsGoal = { goal_name: string; target: number; total: number; progress: number };

	let error = $state('');
	let loading = $state(true);
	let year = $state(0);
	let monthCount = $state(0);

	let expenseRows = $state<ExpenseRow[]>([]);
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

	// Chart canvases
	let budgetChartEl: HTMLCanvasElement;
	let pieChartEl: HTMLCanvasElement;
	let trendChartEl: HTMLCanvasElement;
	let savingsChartEl: HTMLCanvasElement;

	let charts: Chart[] = [];

	onMount(async () => {
		try {
			const [expResp, incResp, monthResp, savResp] = await Promise.all([
				get<any>('/stats/expenses'),
				get<any>('/stats/income'),
				get<any>('/stats/monthly'),
				get<any>('/stats/savings'),
			]);
			year = expResp.year;
			monthCount = expResp.month_count;
			expenseRows = expResp.rows;
			incomeRows = incResp.rows;
			monthlyData = monthResp.months;
			savingsGoals = savResp.goals;
			savingsMonth = savResp.month || '';
		} catch (e: any) { error = e.message; }
		loading = false;
	});

	// Render charts after data loads and elements mount
	$effect(() => {
		if (loading) return;
		// Cleanup old charts
		charts.forEach(c => c.destroy());
		charts = [];

		setTimeout(() => {
			renderBudgetChart();
			renderPieChart();
			renderTrendChart();
			renderSavingsChart();
		}, 50);
	});

	function renderBudgetChart() {
		if (!budgetChartEl || !expenseRows.length) return;
		// Top-level categories only (rows where category is non-empty and subcategory is empty or it's a total)
		const cats = expenseRows.filter(r => r.category && !r.subcategory && r.category !== 'Grand Total');
		charts.push(new Chart(budgetChartEl, {
			type: 'bar',
			data: {
				labels: cats.map(r => r.category),
				datasets: [
					{ label: 'Total', data: cats.map(r => r.total), backgroundColor: '#4a9ab4' },
					{ label: 'Monthly Avg', data: cats.map(r => r.average), backgroundColor: '#e5af42' },
				],
			},
			options: {
				responsive: true,
				plugins: { legend: { position: 'top' } },
				scales: { y: { beginAtZero: true } },
			},
		}));
	}

	function renderPieChart() {
		if (!pieChartEl || !expenseRows.length) return;
		const cats = expenseRows.filter(r => r.category && !r.subcategory && r.total > 0 && r.category !== 'Grand Total');
		const colors = ['#4a9ab4','#e5af42','#2f6577','#ebc46c','#7cc0d6','#b87420','#99561d','#3a7d94','#d4952a','#295463','#f2da9e','#264652','#68391c','#1e3844'];
		charts.push(new Chart(pieChartEl, {
			type: 'pie',
			data: {
				labels: cats.map(r => r.category),
				datasets: [{
					data: cats.map(r => r.total),
					backgroundColor: colors.slice(0, cats.length),
				}],
			},
			options: {
				responsive: true,
				plugins: { legend: { position: 'right' } },
			},
		}));
	}

	function renderTrendChart() {
		if (!trendChartEl || !monthlyData.length) return;
		charts.push(new Chart(trendChartEl, {
			type: 'line',
			data: {
				labels: monthlyData.map(m => m.month.slice(0, 3)),
				datasets: [
					{
						label: 'Expenses', data: monthlyData.map(m => m.expenses),
						borderColor: '#e5af42', backgroundColor: '#e5af4233', fill: true,
					},
					{
						label: 'Income', data: monthlyData.map(m => m.income),
						borderColor: '#4a9ab4', backgroundColor: '#4a9ab433', fill: true,
					},
				],
			},
			options: {
				responsive: true,
				plugins: { legend: { position: 'top' } },
				scales: { y: { beginAtZero: true } },
			},
		}));
	}

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
				indexAxis: 'y',
				plugins: { legend: { position: 'top' } },
				scales: { x: { beginAtZero: true } },
			},
		}));
	}

	function fmtNum(n: number) { return n.toLocaleString(undefined, { maximumFractionDigits: 0 }); }
</script>

<div class="max-w-5xl">
	<h1 class="text-2xl font-bold text-primary-800 mb-2">Statistics</h1>
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
					{#if expenseRows.length > 0}
						<div style="max-height: 400px;"><canvas bind:this={budgetChartEl}></canvas></div>
						<div class="overflow-x-auto">
							<table class="w-full text-sm" dir="rtl">
								<thead style="background: #f0f7fa;">
									<tr>
										<th class="px-2 py-1.5 text-right text-xs font-medium text-gray-600">Category</th>
										<th class="px-2 py-1.5 text-right text-xs font-medium text-gray-600">Subcategory</th>
										<th class="px-2 py-1.5 text-left text-xs font-medium text-gray-600">Total</th>
										<th class="px-2 py-1.5 text-left text-xs font-medium text-gray-600">Monthly Avg</th>
									</tr>
								</thead>
								<tbody>
									{#each expenseRows as row}
										<tr class="border-t {row.category && !row.subcategory ? 'font-medium' : ''} {row.category === 'Grand Total' ? 'bg-gray-100 font-bold' : 'hover:bg-gray-50'}">
											<td class="px-2 py-1 text-xs">{row.category}</td>
											<td class="px-2 py-1 text-xs">{row.subcategory}</td>
											<td class="px-2 py-1 text-xs text-left">{fmtNum(row.total)}</td>
											<td class="px-2 py-1 text-xs text-left">{fmtNum(row.average)}</td>
										</tr>
									{/each}
								</tbody>
							</table>
						</div>
					{:else}
						<p class="text-sm text-gray-400">No expense data available.</p>
					{/if}
				</div>
			{/if}
		</div>

		<!-- Section 2: Category Breakdown (Pie) -->
		<div class="mb-6 bg-white rounded-xl shadow-sm" style="border: 1px solid #b3dbe9;">
			<button onclick={() => showPie = !showPie} class="w-full text-left p-4 flex items-center justify-between">
				<h2 class="text-lg font-semibold text-primary-700">Spending by Category</h2>
				<span class="text-gray-400">{showPie ? '▼' : '▶'}</span>
			</button>
			{#if showPie}
				<div class="px-4 pb-4">
					{#if expenseRows.length > 0}
						<div style="max-width: 500px; margin: 0 auto;"><canvas bind:this={pieChartEl}></canvas></div>
					{:else}
						<p class="text-sm text-gray-400">No data.</p>
					{/if}
				</div>
			{/if}
		</div>

		<!-- Section 3: Month-over-Month Trends -->
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
									<th class="px-2 py-1.5 text-left text-xs font-medium text-gray-600">Total</th>
									<th class="px-2 py-1.5 text-left text-xs font-medium text-gray-600">Monthly Avg</th>
								</tr>
							</thead>
							<tbody>
								{#each incomeRows as row}
									<tr class="border-t hover:bg-gray-50">
										<td class="px-2 py-1 text-xs">{row.category}</td>
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
												<div class="flex items-center gap-2">
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

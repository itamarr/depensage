<script lang="ts">
	import { get, del } from '$lib/api';

	type RunMonth = { month: string; year: number; written: number; income_written: number };
	type RunEntry = {
		timestamp: string;
		files: string[];
		spreadsheet_year: number;
		total_parsed: number;
		months: RunMonth[];
		status: string;
	};

	let runs = $state<RunEntry[]>([]);
	let loadingRuns = $state(true);

	$effect(() => {
		get<{ runs: RunEntry[] }>('/pipeline/history')
			.then(data => { runs = data.runs.reverse(); loadingRuns = false; })
			.catch(() => { loadingRuns = false; });
	});

	function fmtTime(iso: string): string {
		try {
			const d = new Date(iso);
			return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
				+ ' ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
		} catch { return iso; }
	}

	function monthsSummary(months: RunMonth[]): string {
		return months.map(m => `${m.month.slice(0, 3)} ${m.year}`).join(', ');
	}
</script>

<div class="max-w-4xl">
	<h1 class="text-2xl font-bold text-primary-800 mb-6">Dashboard</h1>

	<div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
		<a
			href="/pipeline"
			class="block p-6 rounded-xl shadow-sm hover:shadow-md transition-shadow group"
			style="background: white; border: 1px solid #b3dbe9;"
		>
			<div class="text-2xl mb-2 group-hover:scale-110 transition-transform inline-block">⚙️</div>
			<h2 class="text-lg font-semibold text-primary-700">Process Statements</h2>
			<p class="text-sm text-gray-500 mt-1">Upload CC or bank statements and run the pipeline</p>
		</a>

		<a
			href="/months"
			class="block p-6 rounded-xl shadow-sm hover:shadow-md transition-shadow group"
			style="background: white; border: 1px solid #b3dbe9;"
		>
			<div class="text-2xl mb-2 group-hover:scale-110 transition-transform inline-block">📅</div>
			<h2 class="text-lg font-semibold text-primary-700">View Months</h2>
			<p class="text-sm text-gray-500 mt-1">Browse expenses, budget, and savings by month</p>
		</a>

		<a
			href="/stats"
			class="block p-6 rounded-xl shadow-sm hover:shadow-md transition-shadow group"
			style="background: white; border: 1px solid #b3dbe9;"
		>
			<div class="text-2xl mb-2 group-hover:scale-110 transition-transform inline-block">📈</div>
			<h2 class="text-lg font-semibold text-primary-700">Statistics</h2>
			<p class="text-sm text-gray-500 mt-1">Budget vs actual, trends, savings progress</p>
		</a>
	</div>

	<div class="rounded-xl shadow-sm p-6" style="background: white; border: 1px solid #b3dbe9;">
		<div class="flex items-center justify-between mb-4">
			<h2 class="text-lg font-semibold text-primary-700">Recent Runs</h2>
			{#if runs.length > 0}
				<button
					onclick={async () => {
						if (!confirm('Clear all run history?')) return;
						await del('/pipeline/history');
						runs = [];
					}}
					class="text-xs text-red-400 hover:text-red-600"
				>Clear history</button>
			{/if}
		</div>
		{#if loadingRuns}
			<p class="text-sm text-gray-400">Loading...</p>
		{:else if runs.length === 0}
			<p class="text-sm text-gray-400">No pipeline runs yet. Upload statements to get started.</p>
		{:else}
			<div class="overflow-x-auto">
				<table class="w-full text-sm">
					<thead style="background: #f0f7fa;">
						<tr>
							<th class="px-3 py-2 text-left text-xs font-medium text-gray-600">Time</th>
							<th class="px-3 py-2 text-left text-xs font-medium text-gray-600">Files</th>
							<th class="px-3 py-2 text-left text-xs font-medium text-gray-600">Months</th>
							<th class="px-3 py-2 text-right text-xs font-medium text-gray-600">Parsed</th>
							<th class="px-3 py-2 text-center text-xs font-medium text-gray-600">Status</th>
						</tr>
					</thead>
					<tbody>
						{#each runs as run}
							<tr class="border-t hover:bg-gray-50">
								<td class="px-3 py-1.5 text-xs whitespace-nowrap">{fmtTime(run.timestamp)}</td>
								<td class="px-3 py-1.5 text-xs">{run.files.join(', ')}</td>
								<td class="px-3 py-1.5 text-xs">{monthsSummary(run.months)}</td>
								<td class="px-3 py-1.5 text-xs text-right">{run.total_parsed}</td>
								<td class="px-3 py-1.5 text-center">
									<span class="text-xs px-1.5 py-0.5 rounded bg-green-100 text-green-700">{run.status}</span>
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}
	</div>
</div>

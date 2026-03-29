<script lang="ts">
	import FileUpload from '$lib/components/FileUpload.svelte';
	import ProgressBar from '$lib/components/ProgressBar.svelte';
	import { uploadFiles, post, get, subscribeProgress } from '$lib/api';
	import { sessionId } from '$lib/stores';

	type StagedResult = {
		total_parsed: number;
		in_process_skipped: number;
		classified: number;
		unclassified: number;
		unclassified_merchants: string[];
		months: MonthSummary[];
		has_writes: boolean;
	};

	type MonthSummary = {
		month: string;
		year: number;
		is_new: boolean;
		new_expenses: number;
		duplicates: number;
		new_income: number;
		income_duplicates: number;
		bank_balance: number | null;
		savings_allocations: number;
		savings_warning: string | null;
		carryover_updates: number;
	};

	type MonthDetail = {
		month: string;
		year: number;
		expenses: { index: number; business_name: string; category: string; subcategory: string; amount: string; date: string; status: string }[];
		income: { index: number; comments: string; category: string; amount: string; date: string }[];
		savings_allocations: { goal_name: string; allocated: number; preset_incoming: number }[];
		savings_warning: string | null;
		bank_balance: number | null;
		carryover_updates: number;
	};

	// Wizard state
	let step = $state(1); // 1=upload, 2=running, 3=review, 4=confirm
	let uploadedFiles = $state<string[]>([]);
	let spreadsheetKey = $state('');
	let progressStage = $state('');
	let progressPercent = $state(0);
	let progressError = $state<string | null>(null);
	let stagedResult = $state<StagedResult | null>(null);
	let selectedMonth = $state<MonthDetail | null>(null);
	let commitResult = $state<any>(null);
	let loading = $state(false);
	let error = $state('');

	// Config
	let spreadsheets = $state<Record<string, { year: number; default: boolean }>>({});

	// Load config on mount
	$effect(() => {
		get<{ spreadsheets: Record<string, { year: number; default: boolean }> }>('/system/config')
			.then(cfg => {
				spreadsheets = cfg.spreadsheets;
				// Auto-select default
				const defaultKey = Object.entries(cfg.spreadsheets).find(([, v]) => v.default)?.[0];
				if (defaultKey) spreadsheetKey = defaultKey;
			})
			.catch(() => {});
	});

	async function handleUpload(files: FileList) {
		error = '';
		loading = true;
		try {
			const res = await uploadFiles(files);
			$sessionId = res.session_id;
			uploadedFiles = res.files;
		} catch (e: any) {
			error = e.message;
		}
		loading = false;
	}

	async function runPipeline() {
		if (!$sessionId || !spreadsheetKey) return;
		error = '';
		step = 2;
		progressStage = 'starting';
		progressPercent = 0;
		progressError = null;

		try {
			await post(`/pipeline/${$sessionId}/run`, { spreadsheet_key: spreadsheetKey });
			subscribeProgress(
				$sessionId,
				(data) => {
					progressStage = data.stage;
					progressPercent = data.percent;
					progressError = data.error;
				},
				async () => {
					if (progressError) {
						error = progressError;
						step = 1;
						return;
					}
					// Load result
					try {
						stagedResult = await get<StagedResult>(`/pipeline/${$sessionId}/result`);
						step = 3;
					} catch (e: any) {
						error = e.message;
						step = 1;
					}
				}
			);
		} catch (e: any) {
			error = e.message;
			step = 1;
		}
	}

	async function loadMonthDetail(month: string, year: number) {
		if (!$sessionId) return;
		try {
			selectedMonth = await get<MonthDetail>(`/pipeline/${$sessionId}/months/${month}/${year}`);
		} catch (e: any) {
			error = e.message;
		}
	}

	async function handleCommit() {
		if (!$sessionId) return;
		loading = true;
		error = '';
		try {
			commitResult = await post(`/pipeline/${$sessionId}/commit`);
			step = 4;
		} catch (e: any) {
			error = e.message;
		}
		loading = false;
	}

	function reset() {
		step = 1;
		uploadedFiles = [];
		stagedResult = null;
		selectedMonth = null;
		commitResult = null;
		error = '';
		$sessionId = null;
	}
</script>

<div class="max-w-5xl">
	<h1 class="text-2xl font-bold text-primary-800 mb-6">Process Statements</h1>

	<!-- Step indicator -->
	<div class="flex items-center gap-2 mb-8 text-sm">
		{#each [{ n: 1, label: 'Upload' }, { n: 2, label: 'Processing' }, { n: 3, label: 'Review' }, { n: 4, label: 'Done' }] as s}
			<div class="flex items-center gap-2">
				<span class="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold
					{step >= s.n ? 'bg-primary-600 text-white' : 'bg-gray-200 text-gray-500'}">
					{s.n}
				</span>
				<span class="{step >= s.n ? 'text-gray-800' : 'text-gray-400'}">{s.label}</span>
			</div>
			{#if s.n < 4}
				<div class="flex-1 h-px bg-gray-300"></div>
			{/if}
		{/each}
	</div>

	{#if error}
		<div class="mb-4 p-3 bg-red-50 border border-red-200 rounded text-sm text-red-700">
			{error}
			<button onclick={() => error = ''} class="ml-2 underline">dismiss</button>
		</div>
	{/if}

	<!-- Step 1: Upload -->
	{#if step === 1}
		<FileUpload onFilesSelected={handleUpload} />

		{#if uploadedFiles.length > 0}
			<div class="mt-4 p-4 bg-white rounded-xl shadow-sm border border-gray-200">
				<h3 class="font-medium text-gray-700 mb-2">Uploaded files</h3>
				<ul class="text-sm text-gray-600 space-y-1">
					{#each uploadedFiles as f}
						<li>📄 {f}</li>
					{/each}
				</ul>

				<div class="mt-4 flex items-center gap-4">
					<label class="text-sm text-gray-600">
						Spreadsheet:
						<select
							bind:value={spreadsheetKey}
							class="ml-2 border rounded px-2 py-1 text-sm"
						>
							{#each Object.entries(spreadsheets) as [key, info]}
								<option value={key}>{key} ({info.year})</option>
							{/each}
						</select>
					</label>

					<button
						onclick={runPipeline}
						disabled={!spreadsheetKey || loading}
						class="px-4 py-2 bg-primary-600 text-white rounded hover:bg-primary-700 disabled:opacity-50 text-sm font-medium"
					>
						Run Pipeline
					</button>
				</div>
			</div>
		{/if}
	{/if}

	<!-- Step 2: Processing -->
	{#if step === 2}
		<div class="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
			<h2 class="text-lg font-medium text-gray-700 mb-4">Processing...</h2>
			<ProgressBar stage={progressStage} percent={progressPercent} error={progressError} />
		</div>
	{/if}

	<!-- Step 3: Review -->
	{#if step === 3 && stagedResult}
		<div class="space-y-4">
			<!-- Summary -->
			<div class="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
				<h2 class="text-lg font-medium text-gray-700 mb-3">Pipeline Summary</h2>
				<div class="grid grid-cols-4 gap-4 text-center">
					<div>
						<div class="text-2xl font-bold text-gray-800">{stagedResult.total_parsed}</div>
						<div class="text-xs text-gray-500">Parsed</div>
					</div>
					<div>
						<div class="text-2xl font-bold text-green-600">{stagedResult.classified}</div>
						<div class="text-xs text-gray-500">Classified</div>
					</div>
					<div>
						<div class="text-2xl font-bold {stagedResult.unclassified > 0 ? 'text-red-500' : 'text-gray-800'}">{stagedResult.unclassified}</div>
						<div class="text-xs text-gray-500">Unknown</div>
					</div>
					<div>
						<div class="text-2xl font-bold text-gray-400">{stagedResult.in_process_skipped}</div>
						<div class="text-xs text-gray-500">Skipped</div>
					</div>
				</div>
			</div>

			<!-- Per-month details -->
			{#each stagedResult.months as m}
				<div class="bg-white rounded-xl shadow-sm border border-gray-200 p-4">
					<button
						onclick={() => loadMonthDetail(m.month, m.year)}
						class="w-full text-left"
					>
						<div class="flex items-center justify-between">
							<h3 class="font-medium text-gray-700">
								{m.month} {m.year}
								{#if m.is_new}<span class="ml-2 text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded">NEW</span>{/if}
							</h3>
							<div class="flex gap-4 text-xs text-gray-500">
								{#if m.new_expenses}<span>{m.new_expenses} expenses</span>{/if}
								{#if m.new_income}<span>{m.new_income} income</span>{/if}
								{#if m.carryover_updates}<span>{m.carryover_updates} carryover</span>{/if}
								{#if m.savings_allocations}<span>{m.savings_allocations} savings</span>{/if}
							</div>
						</div>
					</button>

					{#if m.savings_warning}
						<div class="mt-2 text-xs text-amber-600 bg-amber-50 p-2 rounded">
							⚠️ {m.savings_warning}
						</div>
					{/if}

					<!-- Expanded month detail -->
					{#if selectedMonth?.month === m.month && selectedMonth?.year === m.year}
						<div class="mt-4 border-t pt-4 space-y-4">
							{#if selectedMonth.expenses.length > 0}
								<div>
									<h4 class="text-sm font-medium text-gray-600 mb-2">Expenses ({selectedMonth.expenses.length})</h4>
									<div class="overflow-x-auto">
										<table class="w-full text-sm">
											<thead class="bg-gray-50">
												<tr>
													<th class="px-2 py-1 text-left">Date</th>
													<th class="px-2 py-1 text-left rtl">Business</th>
													<th class="px-2 py-1 text-right">Amount</th>
													<th class="px-2 py-1 text-left rtl">Category</th>
													<th class="px-2 py-1 text-left rtl">Subcat</th>
													<th class="px-2 py-1 text-left">Status</th>
												</tr>
											</thead>
											<tbody>
												{#each selectedMonth.expenses as exp}
													<tr class="border-t {!exp.category ? 'bg-red-50' : ''}">
														<td class="px-2 py-1 whitespace-nowrap">{exp.date}</td>
														<td class="px-2 py-1 rtl">{exp.business_name}</td>
														<td class="px-2 py-1 text-right">{exp.amount}</td>
														<td class="px-2 py-1 rtl {!exp.category ? 'text-red-500 font-medium' : ''}">{exp.category || '—'}</td>
														<td class="px-2 py-1 rtl">{exp.subcategory || ''}</td>
														<td class="px-2 py-1">
															{#if exp.status === 'CC'}
																<span class="text-xs bg-blue-100 text-blue-700 px-1 rounded">CC</span>
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
									</div>
								</div>
							{/if}

							{#if selectedMonth.income.length > 0}
								<div>
									<h4 class="text-sm font-medium text-gray-600 mb-2">Income ({selectedMonth.income.length})</h4>
									<table class="w-full text-sm">
										<thead class="bg-gray-50">
											<tr>
												<th class="px-2 py-1 text-left">Date</th>
												<th class="px-2 py-1 text-left rtl">Comments</th>
												<th class="px-2 py-1 text-right">Amount</th>
												<th class="px-2 py-1 text-left rtl">Category</th>
											</tr>
										</thead>
										<tbody>
											{#each selectedMonth.income as inc}
												<tr class="border-t">
													<td class="px-2 py-1">{inc.date}</td>
													<td class="px-2 py-1 rtl">{inc.comments}</td>
													<td class="px-2 py-1 text-right">{inc.amount}</td>
													<td class="px-2 py-1 rtl">{inc.category}</td>
												</tr>
											{/each}
										</tbody>
									</table>
								</div>
							{/if}

							{#if selectedMonth.savings_allocations.length > 0}
								<div>
									<h4 class="text-sm font-medium text-gray-600 mb-2">Savings Allocations</h4>
									<table class="w-full text-sm">
										<thead class="bg-gray-50">
											<tr>
												<th class="px-2 py-1 text-left rtl">Goal</th>
												<th class="px-2 py-1 text-right">Preset</th>
												<th class="px-2 py-1 text-right">Allocated</th>
											</tr>
										</thead>
										<tbody>
											{#each selectedMonth.savings_allocations as alloc}
												<tr class="border-t">
													<td class="px-2 py-1 rtl">{alloc.goal_name}</td>
													<td class="px-2 py-1 text-right">{alloc.preset_incoming.toLocaleString()}</td>
													<td class="px-2 py-1 text-right font-medium">{alloc.allocated.toLocaleString()}</td>
												</tr>
											{/each}
										</tbody>
									</table>
								</div>
							{/if}

							{#if selectedMonth.bank_balance != null}
								<div class="text-sm">
									<span class="text-gray-600">Bank balance:</span>
									<span class="font-medium">{selectedMonth.bank_balance.toLocaleString()} ₪</span>
								</div>
							{/if}
						</div>
					{/if}
				</div>
			{/each}

			<!-- Unclassified merchants warning -->
			{#if stagedResult.unclassified_merchants.length > 0}
				<div class="bg-amber-50 border border-amber-200 rounded-lg p-4">
					<h3 class="font-medium text-amber-800 mb-2">Unclassified Merchants ({stagedResult.unclassified_merchants.length})</h3>
					<div class="flex flex-wrap gap-2">
						{#each stagedResult.unclassified_merchants as name}
							<span class="text-xs bg-amber-100 text-amber-700 px-2 py-1 rounded rtl">{name}</span>
						{/each}
					</div>
					<p class="text-xs text-amber-600 mt-2">These will be written with empty categories. You can classify them in Phase 2.</p>
				</div>
			{/if}

			<!-- Actions -->
			<div class="flex gap-3">
				<button
					onclick={handleCommit}
					disabled={!stagedResult.has_writes || loading}
					class="px-5 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50 font-medium"
				>
					{loading ? 'Committing...' : 'Commit to Spreadsheet'}
				</button>
				<button
					onclick={reset}
					class="px-5 py-2 bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
				>
					Discard
				</button>
			</div>
		</div>
	{/if}

	<!-- Step 4: Done -->
	{#if step === 4 && commitResult}
		<div class="bg-white rounded-xl shadow-sm border border-green-200 p-6">
			<div class="text-center">
				<div class="text-4xl mb-3">✅</div>
				<h2 class="text-xl font-bold text-green-700 mb-4">Committed Successfully</h2>

				<div class="text-sm text-gray-600 space-y-1">
					{#each commitResult.months as m}
						<div>
							{m.month} {m.year}: {m.written} expenses, {m.income_written} income
						</div>
					{/each}
				</div>

				<button
					onclick={reset}
					class="mt-6 px-5 py-2 bg-primary-600 text-white rounded hover:bg-primary-700 font-medium"
				>
					Process More Statements
				</button>
			</div>
		</div>
	{/if}
</div>

<script lang="ts">
	import FileUpload from '$lib/components/FileUpload.svelte';
	import ProgressBar from '$lib/components/ProgressBar.svelte';
	import CategoryPicker from '$lib/components/CategoryPicker.svelte';
	import ErrorBanner from '$lib/components/ErrorBanner.svelte';
	import { uploadFiles, post, get, put, subscribeProgress } from '$lib/api';
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
		month: string; year: number; is_new: boolean;
		new_expenses: number; duplicates: number;
		new_income: number; income_duplicates: number;
		bank_balance: number | null; savings_allocations: number;
		savings_warning: string | null; carryover_updates: number;
	};

	type MonthDetail = {
		month: string; year: number;
		expenses: ExpRow[];
		income: IncRow[];
		savings_allocations: { goal_name: string; allocated: number; preset_incoming: number }[];
		savings_warning: string | null;
		bank_balance: number | null;
		carryover_updates: number;
	};

	type ExpRow = {
		index: number; business_name: string; category: string;
		subcategory: string; amount: string; date: string; status: string;
	};
	type IncRow = {
		index: number; comments: string; category: string;
		amount: string; date: string;
	};
	type Change = {
		month: string; row_type: string; source: string; lookup_key: string;
		old_category: string; new_category: string;
		old_subcategory: string; new_subcategory: string;
		date: string;
	};

	// Wizard steps: 1=upload, 2=running, 3=review/edit, 4=changes, 5=done
	let step = $state(1);

	// No navigation warning needed — state restores from server session
	let pendingFiles = $state<File[]>([]);  // accumulated locally before upload
	let spreadsheetKey = $state('');
	let progressStage = $state('');
	let progressPercent = $state(0);
	let progressError = $state<string | null>(null);
	let stagedResult = $state<StagedResult | null>(null);
	let selectedMonth = $state<MonthDetail | null>(null);
	let categories = $state<Record<string, string[]>>({});
	let changes = $state<Change[]>([]);
	let selectedChanges = $state<Set<number>>(new Set());
	let commitResult = $state<any>(null);
	let loading = $state(false);
	let error = $state('');
	let editingMonth = $state<string | null>(null);
	let spreadsheets = $state<Record<string, { year: number; default: boolean }>>({});

	$effect(() => {
		get<{ spreadsheets: Record<string, { year: number; default: boolean }> }>('/system/config')
			.then(cfg => {
				spreadsheets = cfg.spreadsheets;
				const defaultKey = Object.entries(cfg.spreadsheets).find(([, v]) => v.default)?.[0];
				if (defaultKey) spreadsheetKey = defaultKey;
			})
			.catch(() => {});

		// Restore from existing server session if available
		if ($sessionId && step === 1) {
			get<StagedResult>(`/pipeline/${$sessionId}/result`)
				.then(async (result) => {
					stagedResult = result;
					const catResp = await get<{ categories: Record<string, string[]> }>(`/staging/${$sessionId}/categories`);
					categories = catResp.categories;
					step = 3;
				})
				.catch(() => { $sessionId = null; }); // Session expired, start fresh
		}
	});

	function handleFilesAdded(files: FileList) {
		// Accumulate files — allow multiple drops/selections
		const existing = new Set(pendingFiles.map(f => f.name));
		for (const f of files) {
			if (!existing.has(f.name)) {
				pendingFiles = [...pendingFiles, f];
			}
		}
	}

	function removeFile(name: string) {
		pendingFiles = pendingFiles.filter(f => f.name !== name);
	}

	async function runPipeline() {
		if (!pendingFiles.length || !spreadsheetKey) return;

		// Upload all accumulated files at once
		error = ''; loading = true;
		try {
			const fileList = new DataTransfer();
			for (const f of pendingFiles) fileList.items.add(f);
			const res = await uploadFiles(fileList.files);
			$sessionId = res.session_id;
		} catch (e: any) { error = e.message; loading = false; return; }
		loading = false;

		if (!$sessionId) return;
		error = ''; step = 2;
		progressStage = 'starting'; progressPercent = 0; progressError = null;
		try {
			await post(`/pipeline/${$sessionId}/run`, { spreadsheet_key: spreadsheetKey });
			subscribeProgress($sessionId, (data) => {
				progressStage = data.stage;
				progressPercent = data.percent;
				progressError = data.error;
			}, async () => {
				if (progressError) { error = progressError; step = 1; return; }
				try {
					stagedResult = await get<StagedResult>(`/pipeline/${$sessionId}/result`);
					const catResp = await get<{ categories: Record<string, string[]> }>(`/staging/${$sessionId}/categories`);
					categories = catResp.categories;
					step = 3;
				} catch (e: any) { error = e.message; step = 1; }
			});
		} catch (e: any) { error = e.message; step = 1; }
	}

	async function loadMonthDetail(month: string, year: number) {
		if (!$sessionId) return;
		const key = `${month}-${year}`;
		pipeSortKey = ''; pipeSortAsc = true;
		if (editingMonth === key) { editingMonth = null; selectedMonth = null; return; }
		try {
			selectedMonth = await get<MonthDetail>(`/pipeline/${$sessionId}/months/${month}/${year}`);
			editingMonth = key;
		} catch (e: any) { error = e.message; }
	}

	async function saveExpenseEdit(month: string, year: number, index: number, category: string, subcategory: string) {
		if (!$sessionId) return;
		try {
			await put(`/staging/${$sessionId}/months/${month}/${year}`, {
				expenses: [{ index, category, subcategory }], income: [],
			});
			// Update local state
			if (selectedMonth) {
				const exp = selectedMonth.expenses.find(e => e.index === index);
				if (exp) { exp.category = category; exp.subcategory = subcategory; }
				selectedMonth = { ...selectedMonth };
			}
		} catch (e: any) { error = e.message; }
	}

	async function saveIncomeEdit(month: string, year: number, index: number, category: string, comments: string) {
		if (!$sessionId) return;
		try {
			await put(`/staging/${$sessionId}/months/${month}/${year}`, {
				expenses: [], income: [{ index, category, comments }],
			});
			if (selectedMonth) {
				const inc = selectedMonth.income.find(e => e.index === index);
				if (inc) { inc.category = category; inc.comments = comments; }
				selectedMonth = { ...selectedMonth };
			}
		} catch (e: any) { error = e.message; }
	}

	async function saveSavingsEdit(month: string, year: number, goalName: string, allocated: number) {
		if (!$sessionId) return;
		try {
			await put(`/staging/${$sessionId}/months/${month}/${year}`, {
				expenses: [], income: [], savings: [{ goal_name: goalName, allocated }],
			});
			if (selectedMonth) {
				const alloc = selectedMonth.savings_allocations.find(a => a.goal_name === goalName);
				if (alloc) alloc.allocated = allocated;
				selectedMonth = { ...selectedMonth };
			}
		} catch (e: any) { error = e.message; }
	}

	async function saveBankBalance(month: string, year: number, balance: number) {
		if (!$sessionId) return;
		try {
			await put(`/staging/${$sessionId}/months/${month}/${year}`, {
				expenses: [], income: [], bank_balance: balance,
			});
			if (selectedMonth) {
				selectedMonth.bank_balance = balance;
				selectedMonth = { ...selectedMonth };
			}
		} catch (e: any) { error = e.message; }
	}

	async function reviewChanges() {
		if (!$sessionId) return;
		try {
			const resp = await get<{ changes: Change[] }>(`/staging/${$sessionId}/changes`);
			changes = resp.changes;
			selectedChanges = new Set(changes.map((_, i) => i));
			if (changes.length > 0) {
				pipeSortKey = ''; pipeSortAsc = true;
				step = 4;
			} else {
				await handleCommit();
			}
		} catch (e: any) { error = e.message; }
	}

	async function applyChangesAndCommit() {
		if (!$sessionId) return;
		loading = true; error = '';
		try {
			const selected = changes.filter((_, i) => selectedChanges.has(i));
			if (selected.length > 0) {
				await put(`/staging/${$sessionId}/lookup-updates`, { changes: selected });
			}
			await handleCommit();
		} catch (e: any) { error = e.message; }
		loading = false;
	}

	async function handleCommit() {
		if (!$sessionId) return;
		loading = true; error = '';
		try {
			commitResult = await post(`/pipeline/${$sessionId}/commit`);
			step = 5;
		} catch (e: any) { error = e.message; }
		loading = false;
	}

	function reset() {
		step = 1; pendingFiles = []; stagedResult = null; selectedMonth = null;
		editingMonth = null; commitResult = null; changes = []; error = '';
		$sessionId = null;
	}

	// Pipeline table sorting
	let pipeSortKey = $state('');
	let pipeSortAsc = $state(true);

	function pipeSortToggle(key: string) {
		if (pipeSortKey === key) pipeSortAsc = !pipeSortAsc;
		else { pipeSortKey = key; pipeSortAsc = true; }
	}

	function pipeSortInd(key: string): string {
		if (pipeSortKey !== key) return '';
		return pipeSortAsc ? ' ▲' : ' ▼';
	}

	function pipeSorted<T>(items: T[]): T[] {
		if (!pipeSortKey) return items;
		return [...items].sort((a: any, b: any) => {
			let va = a[pipeSortKey], vb = b[pipeSortKey];
			const na = parseFloat(va), nb = parseFloat(vb);
			if (!isNaN(na) && !isNaN(nb)) return pipeSortAsc ? na - nb : nb - na;
			va = String(va || ''); vb = String(vb || '');
			return pipeSortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
		});
	}

	const stepLabels = [
		{ n: 1, label: 'Upload' }, { n: 2, label: 'Processing' },
		{ n: 3, label: 'Review & Edit' }, { n: 4, label: 'Lookup Updates' },
		{ n: 5, label: 'Done' },
	];
</script>

<div class="max-w-5xl">
	<h1 class="text-2xl font-bold text-primary-800 mb-6">Process Statements</h1>

	<!-- Step indicator -->
	<div class="flex items-center gap-2 mb-8 text-sm">
		{#each stepLabels as s}
			<div class="flex items-center gap-2">
				<span class="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold
					{step >= s.n ? 'bg-primary-600 text-white' : 'bg-gray-200 text-gray-500'}">
					{s.n}
				</span>
				<span class="{step >= s.n ? 'text-gray-800' : 'text-gray-400'}">{s.label}</span>
			</div>
			{#if s.n < 5}<div class="flex-1 h-px bg-gray-300"></div>{/if}
		{/each}
	</div>

	{#if error}
		<ErrorBanner message={error} ondismiss={() => error = ''} />
	{/if}

	<!-- Step 1: Upload -->
	{#if step === 1}
		<FileUpload onFilesSelected={handleFilesAdded} />
		{#if pendingFiles.length > 0}
			<div class="mt-4 p-4 bg-white rounded-xl shadow-sm" style="border: 1px solid #b3dbe9;">
				<h3 class="font-medium text-gray-700 mb-2">Files to process ({pendingFiles.length})</h3>
				<ul class="text-sm text-gray-600 space-y-1">
					{#each pendingFiles as f}
						<li class="flex items-center justify-between">
							<span>📄 {f.name}</span>
							<button onclick={() => removeFile(f.name)} class="text-red-400 hover:text-red-600 text-xs ml-2">remove</button>
						</li>
					{/each}
				</ul>
				<p class="text-xs text-gray-400 mt-2">Drop more files or click the zone above to add more.</p>
				<div class="mt-4 flex items-center gap-4">
					<label class="text-sm text-gray-600">
						Spreadsheet:
						<select bind:value={spreadsheetKey} class="ml-2 border rounded px-2 py-1 text-sm">
							{#each Object.entries(spreadsheets) as [key, info]}
								<option value={key}>{key} ({info.year})</option>
							{/each}
						</select>
					</label>
					<button onclick={runPipeline} disabled={!spreadsheetKey || loading}
						class="px-4 py-2 bg-primary-600 text-white rounded hover:bg-primary-700 disabled:opacity-50 text-sm font-medium">
						{loading ? 'Uploading...' : 'Run Pipeline'}
					</button>
				</div>
			</div>
		{/if}
	{/if}

	<!-- Step 2: Processing -->
	{#if step === 2}
		<div class="bg-white rounded-xl shadow-sm p-6" style="border: 1px solid #b3dbe9;">
			<h2 class="text-lg font-medium text-gray-700 mb-4">Processing...</h2>
			<ProgressBar stage={progressStage} percent={progressPercent} error={progressError} />
		</div>
	{/if}

	<!-- Step 3: Review & Edit -->
	{#if step === 3 && stagedResult}
		<div class="space-y-4">
			<!-- Summary -->
			<div class="bg-white rounded-xl shadow-sm p-6" style="border: 1px solid #b3dbe9;">
				<h2 class="text-lg font-medium text-primary-700 mb-3">Pipeline Summary</h2>
				<div class="grid grid-cols-4 gap-4 text-center">
					<div><div class="text-2xl font-bold text-gray-800">{stagedResult.total_parsed}</div><div class="text-xs text-gray-500">Parsed</div></div>
					<div><div class="text-2xl font-bold text-green-600">{stagedResult.classified}</div><div class="text-xs text-gray-500">Classified</div></div>
					<div><div class="text-2xl font-bold {stagedResult.unclassified > 0 ? 'text-red-500' : 'text-gray-800'}">{stagedResult.unclassified}</div><div class="text-xs text-gray-500">Unknown</div></div>
					<div><div class="text-2xl font-bold text-gray-400">{stagedResult.in_process_skipped}</div><div class="text-xs text-gray-500">Skipped</div></div>
				</div>
			</div>

			<!-- Per-month expandable cards -->
			{#each stagedResult.months as m}
				<div class="bg-white rounded-xl shadow-sm" style="border: 1px solid #b3dbe9;">
					<button onclick={() => loadMonthDetail(m.month, m.year)} class="w-full text-left p-4">
						<div class="flex items-center justify-between">
							<h3 class="font-medium text-primary-700">
								{m.month} {m.year}
								{#if m.is_new}<span class="ml-2 text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded">NEW</span>{/if}
							</h3>
							<div class="flex gap-4 text-xs text-gray-500">
								{#if m.new_expenses}<span>{m.new_expenses} expenses</span>{/if}
								{#if m.duplicates}
									<span class="{m.new_expenses === 0 ? 'text-amber-600 font-medium' : 'text-gray-400'}">{m.duplicates} dupes</span>
								{/if}
								{#if m.new_income}<span>{m.new_income} income</span>{/if}
								{#if m.carryover_updates}<span>{m.carryover_updates} carryover</span>{/if}
								{#if m.savings_allocations}<span>{m.savings_allocations} savings</span>{/if}
							</div>
						</div>
						{#if m.savings_warning}
							<div class="mt-2 text-xs text-amber-600 bg-amber-50 p-2 rounded">⚠️ {m.savings_warning}</div>
						{/if}
					</button>

					<!-- Expanded month detail with editing -->
					{#if editingMonth === `${m.month}-${m.year}` && selectedMonth}
						<div class="border-t p-4 space-y-4">
							{#if m.new_expenses === 0 && m.duplicates > 0}
								<div class="p-3 bg-amber-50 border border-amber-200 rounded-lg text-sm text-amber-700 text-center font-medium">
									All {m.duplicates} expenses were duplicates — nothing new to write.
								</div>
							{/if}
							{#if selectedMonth.expenses.length > 0}
								<div>
									<h4 class="text-sm font-medium text-primary-600 mb-2">Expenses ({selectedMonth.expenses.length})</h4>
									<div class="overflow-x-auto">
										<table class="w-full text-sm">
											<thead style="background: #f0f7fa;">
												<tr>
													<th class="sortable-th" onclick={() => pipeSortToggle('date')}>Date{pipeSortInd('date')}</th>
													<th class="sortable-th rtl" onclick={() => pipeSortToggle('business_name')}>Business{pipeSortInd('business_name')}</th>
													<th class="sortable-th text-right" onclick={() => pipeSortToggle('amount')}>Amount{pipeSortInd('amount')}</th>
													<th class="sortable-th" onclick={() => pipeSortToggle('category')}>Category{pipeSortInd('category')}</th>
													<th class="sortable-th" onclick={() => pipeSortToggle('status')}>Status{pipeSortInd('status')}</th>
												</tr>
											</thead>
											<tbody>
												{#each pipeSorted(selectedMonth.expenses) as exp}
													<tr class="border-t {!exp.category ? 'bg-red-50' : 'hover:bg-gray-50'}">
														<td class="px-2 py-1 whitespace-nowrap text-xs">{exp.date}</td>
														<td class="px-2 py-1 rtl text-xs">{exp.business_name}</td>
														<td class="px-2 py-1 text-right text-xs">{exp.amount}</td>
														<td class="px-2 py-1">
															<CategoryPicker
																{categories}
																value={exp.category}
																subValue={exp.subcategory}
																onchange={(cat, sub) => saveExpenseEdit(m.month, m.year, exp.index, cat, sub)}
															/>
														</td>
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
									</div>
								</div>
							{/if}

							{#if selectedMonth.income.length > 0}
								<div>
									<h4 class="text-sm font-medium text-primary-600 mb-2">Income ({selectedMonth.income.length})</h4>
									<table class="w-full text-sm">
										<thead style="background: #f0f7fa;">
											<tr>
												<th class="sortable-th" onclick={() => pipeSortToggle('date')}>Date{pipeSortInd('date')}</th>
												<th class="sortable-th rtl" onclick={() => pipeSortToggle('comments')}>Comments{pipeSortInd('comments')}</th>
												<th class="sortable-th text-right" onclick={() => pipeSortToggle('amount')}>Amount{pipeSortInd('amount')}</th>
												<th class="sortable-th" onclick={() => pipeSortToggle('category')}>Category{pipeSortInd('category')}</th>
											</tr>
										</thead>
										<tbody>
											{#each pipeSorted(selectedMonth.income) as inc}
												<tr class="border-t hover:bg-gray-50">
													<td class="px-2 py-1 text-xs">{inc.date}</td>
													<td class="px-2 py-1">
														<input
															type="text"
															value={inc.comments}
															class="text-xs border rounded px-1.5 py-0.5 w-full rtl"
															style="border-color: #d1d5db;"
															onchange={(e) => saveIncomeEdit(m.month, m.year, inc.index, inc.category, (e.target as HTMLInputElement).value)}
														/>
													</td>
													<td class="px-2 py-1 text-right text-xs">{inc.amount}</td>
													<td class="px-2 py-1">
														<input
															type="text"
															value={inc.category}
															class="text-xs border rounded px-1.5 py-0.5 w-24 rtl"
															style="border-color: #d1d5db;"
															onchange={(e) => saveIncomeEdit(m.month, m.year, inc.index, (e.target as HTMLInputElement).value, inc.comments)}
														/>
													</td>
												</tr>
											{/each}
										</tbody>
									</table>
								</div>
							{/if}

							{#if selectedMonth.savings_allocations.length > 0}
								<div>
									<h4 class="text-sm font-medium text-primary-600 mb-2">Savings Allocations</h4>
									<table class="w-full text-sm">
										<thead style="background: #f0f7fa;">
											<tr>
												<th class="px-2 py-1.5 text-left text-xs font-medium text-gray-600 rtl">Goal</th>
												<th class="px-2 py-1.5 text-right text-xs font-medium text-gray-600">Preset</th>
												<th class="px-2 py-1.5 text-right text-xs font-medium text-gray-600">Allocated</th>
											</tr>
										</thead>
										<tbody>
											{#each selectedMonth.savings_allocations as alloc}
												<tr class="border-t">
													<td class="px-2 py-1 rtl text-xs">{alloc.goal_name}</td>
													<td class="px-2 py-1 text-right text-xs">{alloc.preset_incoming.toLocaleString()}</td>
													<td class="px-2 py-1 text-right">
														<input
															type="number"
															value={alloc.allocated}
															class="text-xs border rounded px-1.5 py-0.5 w-20 text-right"
															style="border-color: #d1d5db;"
															onchange={(e) => saveSavingsEdit(m.month, m.year, alloc.goal_name, parseFloat((e.target as HTMLInputElement).value) || 0)}
														/>
													</td>
												</tr>
											{/each}
										</tbody>
									</table>
								</div>
							{/if}

							{#if selectedMonth.bank_balance != null}
								<div class="text-sm flex items-center gap-2">
									<span class="text-gray-600">Bank balance:</span>
									<input
										type="number"
										value={selectedMonth.bank_balance}
										class="text-sm border rounded px-2 py-0.5 w-32 text-right font-medium"
										style="border-color: #d1d5db;"
										onchange={(e) => saveBankBalance(m.month, m.year, parseFloat((e.target as HTMLInputElement).value) || 0)}
									/>
									<span class="text-gray-400">₪</span>
								</div>
							{/if}
						</div>
					{/if}
				</div>
			{/each}

			<!-- Actions -->
			<div class="flex gap-3 pt-2">
				<button onclick={reviewChanges} disabled={!stagedResult.has_writes || loading}
					class="px-5 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 font-medium">
					{loading ? 'Checking...' : 'Review & Commit'}
				</button>
				<button onclick={reset}
					class="px-5 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300">
					Discard
				</button>
			</div>
		</div>
	{/if}

	<!-- Step 4: Lookup Updates -->
	{#if step === 4}
		<div class="space-y-4">
			<div class="bg-white rounded-xl shadow-sm p-6" style="border: 1px solid #b3dbe9;">
				<h2 class="text-lg font-medium text-primary-700 mb-4">
					Lookup Changes ({changes.length})
				</h2>
				<p class="text-sm text-gray-500 mb-4">
					These classification changes will update the lookup tables for future runs.
					Uncheck any you don't want to save.
				</p>
				<table class="w-full text-sm">
					<thead style="background: #f0f7fa;">
						<tr>
							<th class="px-2 py-1.5 w-8">
								<input type="checkbox" checked={selectedChanges.size === changes.length}
									onchange={() => {
										if (selectedChanges.size === changes.length) selectedChanges = new Set();
										else selectedChanges = new Set(changes.map((_, i) => i));
									}} />
							</th>
							<th class="px-2 py-1.5 text-left text-xs font-medium text-gray-600">Source</th>
							<th class="px-2 py-1.5 text-left text-xs font-medium text-gray-600 rtl">Name</th>
							<th class="px-2 py-1.5 text-left text-xs font-medium text-gray-600">Date</th>
							<th class="px-2 py-1.5 text-left text-xs font-medium text-gray-600 rtl">Old</th>
							<th class="px-2 py-1.5 text-left text-xs font-medium text-gray-600 rtl">New</th>
						</tr>
					</thead>
					<tbody>
						{#each changes as c, i}
							<tr class="border-t hover:bg-gray-50">
								<td class="px-2 py-1">
									<input type="checkbox" checked={selectedChanges.has(i)}
										onchange={() => {
											const s = new Set(selectedChanges);
											if (s.has(i)) s.delete(i); else s.add(i);
											selectedChanges = s;
										}} />
								</td>
								<td class="px-2 py-1 text-xs">
									<span class="px-1 rounded" style="background: #d9edf4; color: #2f6577;">
										{c.source.toUpperCase()}
									</span>
								</td>
								<td class="px-2 py-1 text-xs rtl">{c.lookup_key}</td>
								<td class="px-2 py-1 text-xs whitespace-nowrap">{c.date}</td>
								<td class="px-2 py-1 text-xs rtl text-gray-400">
									{c.old_category || '(empty)'}{c.old_subcategory ? ` / ${c.old_subcategory}` : ''}
								</td>
								<td class="px-2 py-1 text-xs rtl font-medium">
									{c.new_category}{c.new_subcategory ? ` / ${c.new_subcategory}` : ''}
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>

			<div class="flex gap-3">
				<button onclick={applyChangesAndCommit} disabled={loading}
					class="px-5 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 font-medium">
					{loading ? 'Committing...' : `Apply ${selectedChanges.size} Changes & Commit`}
				</button>
				<button onclick={() => step = 3}
					class="px-5 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300">
					Back to Review
				</button>
			</div>
		</div>
	{/if}

	<!-- Step 5: Done -->
	{#if step === 5 && commitResult}
		<div class="bg-white rounded-xl shadow-sm p-6" style="border: 1px solid #b3dbe9;">
			<div class="text-center">
				<div class="text-4xl mb-3">✅</div>
				<h2 class="text-xl font-bold text-green-700 mb-4">Committed Successfully</h2>
				<div class="text-sm text-gray-600 space-y-1">
					{#each commitResult.months as m}
						<div>{m.month} {m.year}: {m.written} expenses, {m.income_written} income</div>
					{/each}
				</div>
				<button onclick={reset}
					class="mt-6 px-5 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 font-medium">
					Process More Statements
				</button>
			</div>
		</div>
	{/if}
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

<script lang="ts">
	import { get, post, put, del } from '$lib/api';

	type SpreadsheetEntry = { id: string; year: number | null; default: boolean };

	let spreadsheets = $state<Record<string, SpreadsheetEntry>>({});
	let loading = $state(true);
	let error = $state('');
	let success = $state('');

	// Add form
	let showAdd = $state(false);
	let addKey = $state('');
	let addId = $state('');
	let addYear = $state<number | null>(null);
	let addDefault = $state(false);

	// Password form
	let newPassword = $state('');
	let confirmPassword = $state('');

	$effect(() => {
		loadSpreadsheets();
	});

	async function loadSpreadsheets() {
		loading = true; error = '';
		try {
			const data = await get<{ spreadsheets: Record<string, SpreadsheetEntry> }>('/system/spreadsheets');
			spreadsheets = data.spreadsheets;
		} catch (e: any) { error = e.message; }
		loading = false;
	}

	async function handleAdd() {
		error = ''; success = '';
		try {
			await post(`/system/spreadsheets/${encodeURIComponent(addKey)}`, {
				spreadsheet_id: addId, year: addYear, default: addDefault,
			});
			showAdd = false; addKey = ''; addId = ''; addYear = null; addDefault = false;
			success = 'Spreadsheet added';
			await loadSpreadsheets();
		} catch (e: any) { error = e.message; }
	}

	async function handleSetDefault(key: string) {
		error = ''; success = '';
		const entry = spreadsheets[key];
		try {
			await put(`/system/spreadsheets/${encodeURIComponent(key)}`, {
				spreadsheet_id: '', year: entry.year, default: true,
			});
			success = `${key} set as default`;
			await loadSpreadsheets();
		} catch (e: any) { error = e.message; }
	}

	async function handleRemove(key: string) {
		if (!confirm(`Remove spreadsheet "${key}"?`)) return;
		error = ''; success = '';
		try {
			await del(`/system/spreadsheets/${encodeURIComponent(key)}`);
			success = `${key} removed`;
			await loadSpreadsheets();
		} catch (e: any) { error = e.message; }
	}

	async function handlePasswordChange() {
		if (newPassword !== confirmPassword) { error = 'Passwords do not match'; return; }
		if (!newPassword) { error = 'Password cannot be empty'; return; }
		error = ''; success = '';
		try {
			await post('/system/password', { password: newPassword });
			newPassword = ''; confirmPassword = '';
			success = 'Password updated';
		} catch (e: any) { error = e.message; }
	}
</script>

<div class="max-w-3xl">
	<h1 class="text-2xl font-bold text-primary-800 mb-6">Settings</h1>

	{#if error}
		<div class="mb-4 p-3 bg-red-50 border border-red-200 rounded text-sm text-red-700">{error}</div>
	{/if}
	{#if success}
		<div class="mb-4 p-3 bg-green-50 border border-green-200 rounded text-sm text-green-700">{success}</div>
	{/if}

	<!-- Spreadsheets -->
	<div class="bg-white rounded-xl shadow-sm p-6 mb-6" style="border: 1px solid #b3dbe9;">
		<div class="flex items-center justify-between mb-4">
			<h2 class="text-lg font-semibold text-primary-700">Spreadsheets</h2>
			<button
				onclick={() => showAdd = !showAdd}
				class="px-3 py-1.5 bg-primary-600 text-white rounded text-sm hover:bg-primary-700"
			>{showAdd ? 'Cancel' : '+ Add'}</button>
		</div>

		{#if showAdd}
			<div class="mb-4 p-3 rounded" style="background: #f0f7fa; border: 1px solid #b3dbe9;">
				<div class="grid grid-cols-2 gap-3">
					<label class="text-xs text-gray-600">
						Config Key
						<input bind:value={addKey} placeholder="e.g. 2027" class="block w-full border rounded px-2 py-1 text-sm mt-0.5" />
					</label>
					<label class="text-xs text-gray-600">
						Spreadsheet ID
						<input bind:value={addId} placeholder="Google Sheets ID" class="block w-full border rounded px-2 py-1 text-sm mt-0.5" />
					</label>
					<label class="text-xs text-gray-600">
						Year
						<input type="number" bind:value={addYear} placeholder="2027" class="block w-full border rounded px-2 py-1 text-sm mt-0.5" />
					</label>
					<label class="text-xs text-gray-600 flex items-end gap-2 pb-1">
						<input type="checkbox" bind:checked={addDefault} />
						Set as default for this year
					</label>
				</div>
				<button
					onclick={handleAdd}
					disabled={!addKey || !addId}
					class="mt-3 px-4 py-1.5 bg-green-600 text-white rounded text-sm hover:bg-green-700 disabled:opacity-50"
				>Add Spreadsheet</button>
			</div>
		{/if}

		{#if loading}
			<p class="text-gray-400 text-sm">Loading...</p>
		{:else}
			<table class="w-full text-sm">
				<thead style="background: #f0f7fa;">
					<tr>
						<th class="px-3 py-2 text-left text-xs font-medium text-gray-600">Key</th>
						<th class="px-3 py-2 text-left text-xs font-medium text-gray-600">ID</th>
						<th class="px-3 py-2 text-center text-xs font-medium text-gray-600">Year</th>
						<th class="px-3 py-2 text-center text-xs font-medium text-gray-600">Default</th>
						<th class="px-3 py-2 w-24"></th>
					</tr>
				</thead>
				<tbody>
					{#each Object.entries(spreadsheets) as [key, entry]}
						<tr class="border-t hover:bg-gray-50">
							<td class="px-3 py-2 text-xs font-medium">{key}</td>
							<td class="px-3 py-2 text-xs text-gray-500">{entry.id}</td>
							<td class="px-3 py-2 text-xs text-center">{entry.year || '—'}</td>
							<td class="px-3 py-2 text-center">
								{#if entry.default}
									<span class="text-xs px-1.5 py-0.5 rounded bg-green-100 text-green-700">default</span>
								{:else}
									<button onclick={() => handleSetDefault(key)} class="text-xs text-primary-500 hover:text-primary-700">set default</button>
								{/if}
							</td>
							<td class="px-3 py-2">
								<button onclick={() => handleRemove(key)} class="text-xs text-red-400 hover:text-red-600">remove</button>
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		{/if}
	</div>

	<!-- Password -->
	<div class="bg-white rounded-xl shadow-sm p-6" style="border: 1px solid #b3dbe9;">
		<h2 class="text-lg font-semibold text-primary-700 mb-4">Change Password</h2>
		<div class="flex gap-3 items-end">
			<label class="text-xs text-gray-600">
				New Password
				<input type="password" bind:value={newPassword} class="block border rounded px-2 py-1 text-sm mt-0.5 w-48" />
			</label>
			<label class="text-xs text-gray-600">
				Confirm
				<input type="password" bind:value={confirmPassword} class="block border rounded px-2 py-1 text-sm mt-0.5 w-48"
					onkeydown={(e) => { if (e.key === 'Enter') handlePasswordChange(); }}
				/>
			</label>
			<button
				onclick={handlePasswordChange}
				disabled={!newPassword || !confirmPassword}
				class="px-4 py-1.5 bg-primary-600 text-white rounded text-sm hover:bg-primary-700 disabled:opacity-50"
			>Update</button>
		</div>
	</div>
</div>

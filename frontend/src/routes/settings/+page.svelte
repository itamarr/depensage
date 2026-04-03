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

	// Edit form
	let editKey = $state<string | null>(null);
	let editNewKey = $state('');
	let editId = $state('');
	let editYear = $state<number | null>(null);
	let editDefault = $state(false);

	// Password
	let newPassword = $state('');
	let confirmPassword = $state('');

	$effect(() => { loadSpreadsheets(); });

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

	function startEdit(key: string) {
		const entry = spreadsheets[key];
		editKey = key;
		editNewKey = key;
		editId = entry.id.replace('...', '');  // truncated, user may paste full ID
		editYear = entry.year;
		editDefault = entry.default;
	}

	async function saveEdit(oldKey: string) {
		error = ''; success = '';
		const newKey = editNewKey.trim();
		if (!newKey) { error = 'Key cannot be empty'; return; }

		try {
			if (newKey !== oldKey) {
				// Rename: create new, delete old
				await post(`/system/spreadsheets/${encodeURIComponent(newKey)}`, {
					spreadsheet_id: editId || '', year: editYear, default: editDefault,
				});
				await del(`/system/spreadsheets/${encodeURIComponent(oldKey)}`);
				success = `Renamed ${oldKey} → ${newKey}`;
			} else {
				await put(`/system/spreadsheets/${encodeURIComponent(oldKey)}`, {
					spreadsheet_id: editId || '', year: editYear, default: editDefault,
				});
				success = `${oldKey} updated`;
			}
			editKey = null;
			await loadSpreadsheets();
		} catch (e: any) { error = e.message; }
	}

	async function handleSetDefault(key: string) {
		const entry = spreadsheets[key];
		error = ''; success = '';
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
						<input bind:value={addId} placeholder="Paste from URL" class="block w-full border rounded px-2 py-1 text-sm mt-0.5 font-mono text-xs" />
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
						<th class="px-3 py-2 w-28"></th>
					</tr>
				</thead>
				<tbody>
					{#each Object.entries(spreadsheets) as [key, entry]}
						{#if editKey === key}
							<tr class="border-t" style="background: #f0f7fa;">
								<td class="px-3 py-2">
									<input bind:value={editNewKey} class="text-xs border rounded px-1 py-0.5 w-full font-medium" />
								</td>
								<td class="px-3 py-2">
									<input bind:value={editId} class="text-xs border rounded px-1 py-0.5 w-full font-mono" placeholder="Spreadsheet ID" />
								</td>
								<td class="px-3 py-2">
									<input type="number" bind:value={editYear} class="text-xs border rounded px-1 py-0.5 w-16 text-center" />
								</td>
								<td class="px-3 py-2 text-center">
									<input type="checkbox" bind:checked={editDefault} />
								</td>
								<td class="px-3 py-2">
									<button onclick={() => saveEdit(key)} class="text-xs text-green-600 hover:text-green-800 mr-1">save</button>
									<button onclick={() => editKey = null} class="text-xs text-gray-400 hover:text-gray-600">cancel</button>
								</td>
							</tr>
						{:else}
							<tr class="border-t hover:bg-gray-50">
								<td class="px-3 py-2 text-xs font-medium">{key}</td>
								<td class="px-3 py-2 text-xs text-gray-500 font-mono">{entry.id}</td>
								<td class="px-3 py-2 text-xs text-center">{entry.year || '—'}</td>
								<td class="px-3 py-2 text-center">
									{#if entry.default}
										<span class="text-xs px-1.5 py-0.5 rounded bg-green-100 text-green-700">default</span>
									{:else}
										<button onclick={() => handleSetDefault(key)} class="text-xs text-primary-500 hover:text-primary-700">set default</button>
									{/if}
								</td>
								<td class="px-3 py-2">
									<button onclick={() => startEdit(key)} class="text-xs text-primary-600 hover:text-primary-800 mr-1">edit</button>
									<button onclick={() => handleRemove(key)} class="text-xs text-red-400 hover:text-red-600">remove</button>
								</td>
							</tr>
						{/if}
					{/each}
				</tbody>
			</table>
		{/if}

		<div class="mt-4 p-3 rounded text-xs text-gray-500" style="background: #f8fafb;">
			<strong>How to find the Spreadsheet ID:</strong> Open the spreadsheet in Google Sheets.
			The URL looks like <code class="text-xs">docs.google.com/spreadsheets/d/<strong>SPREADSHEET_ID</strong>/edit</code>.
			Copy the long string between <code>/d/</code> and <code>/edit</code>.
		</div>
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

<script lang="ts">
	let { stage, percent, error }: { stage: string; percent: number; error: string | null } = $props();

	const stages = [
		{ key: 'starting', label: 'Starting' },
		{ key: 'parsing', label: 'Parsing files' },
		{ key: 'classifying', label: 'Classifying' },
		{ key: 'complete', label: 'Complete' },
	];

	const currentIdx = $derived(stages.findIndex(s => s.key === stage));
</script>

<div class="space-y-3">
	<!-- Progress bar -->
	<div class="h-2 bg-gray-200 rounded-full overflow-hidden">
		<div
			class="h-full transition-all duration-500 rounded-full {error ? 'bg-red-500' : 'bg-primary-500'}"
			style="width: {percent}%"
		></div>
	</div>

	<!-- Stage labels -->
	<div class="flex justify-between text-xs text-gray-500">
		{#each stages as s, i}
			<span class="{i <= currentIdx ? 'text-primary-600 font-medium' : ''}">{s.label}</span>
		{/each}
	</div>

	{#if error}
		<div class="p-3 bg-red-50 border border-red-200 rounded text-sm text-red-700">
			{error}
		</div>
	{/if}
</div>

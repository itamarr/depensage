<script lang="ts">
	let {
		categories,
		value,
		subValue = '',
		onchange,
	}: {
		categories: Record<string, string[]>;
		value: string;
		subValue?: string;
		onchange: (cat: string, sub: string) => void;
	} = $props();

	const catNames = $derived(Object.keys(categories));
	const subcats = $derived(value ? (categories[value] || []) : []);

	function handleCatChange(e: Event) {
		const cat = (e.target as HTMLSelectElement).value;
		onchange(cat, '');
	}

	function handleSubChange(e: Event) {
		const sub = (e.target as HTMLSelectElement).value;
		onchange(value, sub);
	}
</script>

<div class="flex gap-1">
	<select
		class="cat-select"
		{value}
		onchange={handleCatChange}
	>
		<option value="">--</option>
		{#each catNames as cat}
			<option value={cat}>{cat}</option>
		{/each}
	</select>

	{#if subcats.length > 0}
		<select
			class="cat-select"
			value={subValue}
			onchange={handleSubChange}
		>
			<option value="">--</option>
			{#each subcats as sub}
				<option value={sub}>{sub}</option>
			{/each}
		</select>
	{/if}
</div>

<style>
	.cat-select {
		font-size: 0.75rem;
		padding: 0.125rem 0.25rem;
		border: 1px solid #d1d5db;
		border-radius: 0.25rem;
		background: white;
		direction: rtl;
		max-width: 8rem;
	}
	.cat-select:focus {
		outline: none;
		border-color: #4a9ab4;
		box-shadow: 0 0 0 1px #4a9ab4;
	}
</style>

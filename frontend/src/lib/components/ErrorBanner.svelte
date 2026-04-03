<script lang="ts">
	let { message, ondismiss, timeout = 10000 }: {
		message: string;
		ondismiss: () => void;
		timeout?: number;
	} = $props();

	$effect(() => {
		if (message && timeout > 0) {
			const timer = setTimeout(ondismiss, timeout);
			return () => clearTimeout(timer);
		}
	});
</script>

{#if message}
	<div class="mb-4 p-3 bg-red-50 border border-red-200 rounded text-sm text-red-700 flex items-center justify-between">
		<span>{message}</span>
		<button onclick={ondismiss} class="ml-2 text-red-400 hover:text-red-600 text-xs">dismiss</button>
	</div>
{/if}

<script lang="ts">
	let { onFilesSelected }: { onFilesSelected: (files: FileList) => void } = $props();
	let dragOver = $state(false);
	let fileInput: HTMLInputElement;

	function handleDrop(e: DragEvent) {
		e.preventDefault();
		e.stopPropagation();
		dragOver = false;
		if (e.dataTransfer?.files.length) {
			onFilesSelected(e.dataTransfer.files);
		}
	}

	function handleDragOver(e: DragEvent) {
		e.preventDefault();
		e.stopPropagation();
		dragOver = true;
	}

	function handleSelect(e: Event) {
		const input = e.target as HTMLInputElement;
		if (input.files?.length) {
			onFilesSelected(input.files);
		}
	}

	function handleClick(e: MouseEvent) {
		e.stopPropagation();
		fileInput.click();
	}
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
	class="border-2 border-dashed rounded-xl p-8 text-center transition-colors cursor-pointer
		{dragOver ? 'border-primary-400 bg-primary-50' : 'border-gray-300 hover:border-primary-300'}"
	ondragover={handleDragOver}
	ondragenter={handleDragOver}
	ondragleave={() => dragOver = false}
	ondrop={handleDrop}
	onclick={handleClick}
	onkeydown={(e) => { if (e.key === 'Enter') fileInput.click(); }}
	role="button"
	tabindex="0"
>
	<input
		bind:this={fileInput}
		type="file"
		accept=".xlsx"
		multiple
		class="hidden"
		onchange={handleSelect}
	/>
	<div class="text-4xl mb-3">📄</div>
	<p class="text-gray-600 font-medium">Drop .xlsx files here or click to browse</p>
	<p class="text-sm text-gray-400 mt-1">CC statements and bank transcripts</p>
</div>

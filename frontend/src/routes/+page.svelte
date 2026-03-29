<script lang="ts">
	// Dashboard - will show summary cards and run history
</script>

<div class="max-w-4xl">
	<h1 class="text-2xl font-bold text-gray-800 mb-6">Dashboard</h1>

	<div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
		<!-- Quick actions -->
		<a
			href="/pipeline"
			class="block p-6 bg-white rounded-lg shadow hover:shadow-md transition-shadow border border-gray-200"
		>
			<div class="text-2xl mb-2">⚙️</div>
			<h2 class="text-lg font-semibold text-gray-700">Process Statements</h2>
			<p class="text-sm text-gray-500 mt-1">Upload CC or bank statements and run the pipeline</p>
		</a>

		<a
			href="/months"
			class="block p-6 bg-white rounded-lg shadow hover:shadow-md transition-shadow border border-gray-200"
		>
			<div class="text-2xl mb-2">📅</div>
			<h2 class="text-lg font-semibold text-gray-700">View Months</h2>
			<p class="text-sm text-gray-500 mt-1">Browse expenses, budget, and savings by month</p>
		</a>
	</div>

	<!-- Run history placeholder -->
	<div class="bg-white rounded-lg shadow border border-gray-200 p-6">
		<h2 class="text-lg font-semibold text-gray-700 mb-4">Recent Runs</h2>
		<p class="text-sm text-gray-400">No pipeline runs yet. Upload statements to get started.</p>
	</div>
</div>

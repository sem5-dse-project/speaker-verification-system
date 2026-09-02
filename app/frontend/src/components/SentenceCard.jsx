import { RefreshCcw } from 'lucide-react'

function SentenceCard({ sentence, onGenerateSentence }) {
  return (
    <section className="card p-6 sm:p-8">
      <div className="mb-4 flex items-center justify-between gap-4">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-subtle">
          Sentence to Read
        </h2>
        <button
          type="button"
          onClick={onGenerateSentence}
          className="inline-flex items-center gap-2 rounded-xl bg-brand-50 px-4 py-2 text-sm font-semibold text-brand-800 transition-all duration-200 hover:bg-brand-100 hover:text-brand-900 focus-ring dark:bg-surface-800 dark:text-brand-300 dark:hover:bg-surface-700 dark:hover:text-brand-200"
        >
          <RefreshCcw className="h-4 w-4" />
          Generate New Sentence
        </button>
      </div>

      <p className="rounded-xl bg-brand-50/80 p-5 text-lg leading-relaxed text-slate-800 ring-1 ring-brand-100 dark:bg-surface-800/80 dark:text-slate-100 dark:ring-brand-900/25 sm:text-xl">
        "{sentence}"
      </p>
    </section>
  )
}

export default SentenceCard

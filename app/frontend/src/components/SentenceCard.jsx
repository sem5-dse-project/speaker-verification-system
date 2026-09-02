import { RefreshCcw } from 'lucide-react'

function SentenceCard({ sentence, onGenerateSentence }) {
  return (
    <section className="card p-6 sm:p-8">
      <div className="mb-4 flex items-center justify-between gap-4">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
          Sentence to Read
        </h2>
        <button
          type="button"
          onClick={onGenerateSentence}
          className="inline-flex items-center gap-2 rounded-xl bg-slate-100 px-4 py-2 text-sm font-semibold text-slate-700 transition-all duration-200 hover:bg-blue-50 hover:text-blue-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-300 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700 dark:hover:text-blue-300"
        >
          <RefreshCcw className="h-4 w-4" />
          Generate New Sentence
        </button>
      </div>

      <p className="rounded-xl bg-slate-50 p-5 text-lg leading-relaxed text-slate-800 ring-1 ring-slate-200 dark:bg-slate-800/80 dark:text-slate-100 dark:ring-slate-700 sm:text-xl">
        "{sentence}"
      </p>
    </section>
  )
}

export default SentenceCard

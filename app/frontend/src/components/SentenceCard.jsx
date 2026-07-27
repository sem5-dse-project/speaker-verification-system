import { RefreshCcw } from 'lucide-react'

function SentenceCard({ sentence, onGenerateSentence }) {
  return (
    <section className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-200 sm:p-8">
      <div className="mb-4 flex items-center justify-between gap-4">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
          Sentence to Read
        </h2>
        <button
          type="button"
          onClick={onGenerateSentence}
          className="inline-flex items-center gap-2 rounded-xl bg-slate-100 px-4 py-2 text-sm font-semibold text-slate-700 transition-all duration-200 hover:bg-blue-50 hover:text-blue-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-300"
        >
          <RefreshCcw className="h-4 w-4" />
          Generate New Sentence
        </button>
      </div>

      <p className="rounded-xl bg-slate-50 p-5 text-lg leading-relaxed text-slate-800 ring-1 ring-slate-200 sm:text-xl">
        "{sentence}"
      </p>
    </section>
  )
}

export default SentenceCard

const getStrengthFromPassword = (password) => {
  let score = 0

  if (password.length >= 8) score += 1
  if (/[A-Z]/.test(password)) score += 1
  if (/[a-z]/.test(password)) score += 1
  if (/\d/.test(password)) score += 1

  if (score <= 1) {
    return { label: 'Weak', level: 1, color: 'bg-rose-500' }
  }

  if (score <= 3) {
    return { label: 'Medium', level: 2, color: 'bg-amber-500' }
  }

  return { label: 'Strong', level: 3, color: 'bg-emerald-500' }
}

function PasswordStrength({ password }) {
  const strength = getStrengthFromPassword(password)

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs">
        <span className="font-medium text-slate-500">Password strength</span>
        <span className="font-semibold text-slate-700">{strength.label}</span>
      </div>

      <div className="grid grid-cols-3 gap-2">
        {[1, 2, 3].map((segment) => (
          <div
            key={segment}
            className={[
              'h-1.5 rounded-full transition-colors duration-300',
              segment <= strength.level ? strength.color : 'bg-slate-200',
            ].join(' ')}
          />
        ))}
      </div>
    </div>
  )
}

export default PasswordStrength
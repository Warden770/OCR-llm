import { useEffect, useMemo, useRef, useState } from 'react'
import * as pdfjsLib from 'pdfjs-dist'
import pdfjsWorker from 'pdfjs-dist/build/pdf.worker.min.mjs?url'
import { renderAsync as renderDocx } from 'docx-preview'
import { authRequired, authenticatedFetch, getAccessToken, signOut, supabase } from './supabaseClient'

const fetch = authenticatedFetch

function AuthScreen({ onSignedIn, mode, onModeChange }) {
  const isSignUp = mode === 'signup'
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  function changeMode(nextMode) {
    setShowPassword(false)
    setShowConfirmPassword(false)
    onModeChange(nextMode)
  }

  async function handleSubmit(event) {
    event.preventDefault()
    setBusy(true)
    setError('')

    const normalizedName = String(name || '').trim()
    const trimmedEmail = String(email || '').trim()
    const normalizedEmail = trimmedEmail.toLowerCase()
    if (isSignUp && normalizedName.length < 2) {
      setError('Name must be at least 2 characters long.')
      setBusy(false)
      return
    }
    if (isSignUp && normalizedName.length > 100) {
      setError('Name must be 100 characters or fewer.')
      setBusy(false)
      return
    }
    if (!normalizedEmail) {
      setError('Email is required.')
      setBusy(false)
      return
    }
    if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(normalizedEmail)) {
      setError('Enter a valid email address.')
      setBusy(false)
      return
    }
    if (!password) {
      setError('Password is required.')
      setBusy(false)
      return
    }
    if (isSignUp) {
      if (!confirmPassword) {
        setError('Please confirm your password.')
        setBusy(false)
        return
      }
      if (password !== confirmPassword) {
        setError('Passwords do not match.')
        setBusy(false)
        return
      }
      if (password.length < 8) {
        setError('Password must be at least 8 characters long.')
        setBusy(false)
        return
      }
      try {
        const { data, error: signUpError } = await supabase.auth.signUp({
          email: normalizedEmail,
          password,
          options: { data: { name: normalizedName } },
        })
        if (signUpError) {
          setError(signUpError.message || 'Unable to create your account.')
        } else if (data?.session) {
          onSignedIn?.(normalizedName)
        } else {
          setError('Check your email to confirm your account before signing in.')
        }
      } catch {
        setError('Unable to contact the authentication service. Please try again.')
      } finally {
        setBusy(false)
      }
      return
    }

    try {
      const { error: signInError } = await supabase.auth.signInWithPassword({ email: normalizedEmail, password })
      if (signInError) {
        setError(signInError.message || 'Invalid email or password.')
      } else {
        onSignedIn?.()
      }
    } catch {
      setError('Unable to contact the authentication service. Please try again.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="auth-root">
      {/* ── LEFT: form panel ── */}
      <div className="auth-form-panel">
        <div className="auth-card page-fade-in">
          {/* Logo */}
          <div className="auth-logo">
            <div className="auth-logo-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                <path d="M7 3h7l5 5v13H7z" />
                <path d="M14 3v6h6" />
                <path d="M10 12h5M10 15.5h3" strokeLinecap="round" />
              </svg>
            </div>
            <div>
              <div className="auth-logo-label">OCR System</div>
            </div>
          </div>

          {/* Headline */}
          <div className="auth-headline">
            <h1 className="auth-title">{isSignUp ? 'Create account' : 'Welcome back'}</h1>
            <p className="auth-subtitle">
              {isSignUp
                ? 'Start extracting text from your documents.'
                : 'Sign in to your secure workspace.'}
            </p>
          </div>

          {/* Mode toggle */}
          <div className="auth-toggle">
            <button
              type="button"
              onClick={() => changeMode('signin')}
              className={`auth-toggle-btn${!isSignUp ? ' auth-toggle-btn--active' : ''}`}
            >
              Sign In
            </button>
            <button
              type="button"
              onClick={() => changeMode('signup')}
              className={`auth-toggle-btn${isSignUp ? ' auth-toggle-btn--active' : ''}`}
            >
              Sign Up
            </button>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="auth-form">
            {isSignUp ? (
              <div className="auth-field">
                <label className="auth-label" htmlFor="auth-name">Full Name</label>
                <div className="auth-input-wrap">
                  <svg className="auth-input-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
                    <circle cx="12" cy="8" r="4" />
                    <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7" strokeLinecap="round" />
                  </svg>
                  <input
                    id="auth-name"
                    className="auth-input"
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    autoComplete="name"
                    maxLength={100}
                    placeholder="Jane Smith"
                    required
                  />
                </div>
              </div>
            ) : null}

            <div className="auth-field">
              <label className="auth-label" htmlFor="auth-email">Email address</label>
              <div className="auth-input-wrap">
                <svg className="auth-input-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
                  <rect x="3" y="5" width="18" height="14" rx="2" />
                  <path d="M3 7l9 6 9-6" />
                </svg>
                <input
                  id="auth-email"
                  className="auth-input"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  autoComplete="email"
                  placeholder="you@example.com"
                  required
                />
              </div>
            </div>

            <div className="auth-field">
              <label className="auth-label" htmlFor="auth-password">Password</label>
              <div className="auth-input-wrap">
                <svg className="auth-input-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
                  <rect x="5" y="11" width="14" height="10" rx="2" />
                  <path d="M8 11V7a4 4 0 0 1 8 0v4" />
                </svg>
                <input
                  id="auth-password"
                  className="auth-input"
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete={isSignUp ? 'new-password' : 'current-password'}
                  placeholder="••••••••"
                  required
                />
                <button
                  type="button"
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                  aria-pressed={showPassword}
                  onClick={() => setShowPassword((v) => !v)}
                  className="auth-eye-btn"
                >
                  {showPassword ? (
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
                      <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94" />
                      <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19" />
                      <line x1="1" y1="1" x2="23" y2="23" />
                    </svg>
                  ) : (
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
                      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                      <circle cx="12" cy="12" r="3" />
                    </svg>
                  )}
                </button>
              </div>
            </div>

            {isSignUp ? (
              <div className="auth-field">
                <label className="auth-label" htmlFor="auth-confirm">Confirm Password</label>
                <div className="auth-input-wrap">
                  <svg className="auth-input-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
                    <rect x="5" y="11" width="14" height="10" rx="2" />
                    <path d="M8 11V7a4 4 0 0 1 8 0v4" />
                  </svg>
                  <input
                    id="auth-confirm"
                    className="auth-input"
                    type={showConfirmPassword ? 'text' : 'password'}
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    autoComplete="new-password"
                    placeholder="••••••••"
                    required
                  />
                  <button
                    type="button"
                    aria-label={showConfirmPassword ? 'Hide confirm password' : 'Show confirm password'}
                    aria-pressed={showConfirmPassword}
                    onClick={() => setShowConfirmPassword((v) => !v)}
                    className="auth-eye-btn"
                  >
                    {showConfirmPassword ? (
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
                        <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94" />
                        <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19" />
                        <line x1="1" y1="1" x2="23" y2="23" />
                      </svg>
                    ) : (
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
                        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                        <circle cx="12" cy="12" r="3" />
                      </svg>
                    )}
                  </button>
                </div>
              </div>
            ) : null}

            {error ? <p className="auth-error">{error}</p> : null}

            <button className="auth-submit-btn" disabled={busy} type="submit">
              {busy ? (
                <span className="auth-submit-spinner" />
              ) : null}
              {busy
                ? (isSignUp ? 'Creating account…' : 'Signing in…')
                : (isSignUp ? 'Create Account' : 'Sign In')}
            </button>
          </form>

          {!isSignUp ? (
            <p className="auth-footer">
              Don&apos;t have an account?{' '}
              <button type="button" onClick={() => changeMode('signup')} className="auth-footer-link">
                Create one
              </button>
            </p>
          ) : (
            <p className="auth-footer">
              Already have an account?{' '}
              <button type="button" onClick={() => changeMode('signin')} className="auth-footer-link">
                Sign in
              </button>
            </p>
          )}
        </div>
      </div>

      {/* ── RIGHT: visual panel ── */}
      <div className="auth-visual-panel" aria-hidden="true">
        {/* Decorative blobs */}
        <div className="auth-blob auth-blob--1" />
        <div className="auth-blob auth-blob--2" />
        <div className="auth-blob auth-blob--3" />

        {/* Floating document cards */}
        <div className="auth-doc-stack">
          {/* Main doc */}
          <div className="auth-doc auth-doc--main">
            <div className="auth-doc-topbar">
              <div className="auth-doc-dot" style={{background:'#f87171'}} />
              <div className="auth-doc-dot" style={{background:'#fbbf24'}} />
              <div className="auth-doc-dot" style={{background:'#34d399'}} />
            </div>
            <div className="auth-doc-lines">
              <div className="auth-doc-line auth-doc-line--title" />
              <div className="auth-doc-line" style={{width:'88%'}} />
              <div className="auth-doc-line" style={{width:'75%'}} />
              <div className="auth-doc-line" style={{width:'92%'}} />
              <div className="auth-doc-line" style={{width:'60%'}} />
              <div style={{height:'0.75rem'}} />
              <div className="auth-doc-line" style={{width:'80%'}} />
              <div className="auth-doc-line" style={{width:'95%'}} />
              <div className="auth-doc-line" style={{width:'70%'}} />
            </div>
            {/* Scan line animation */}
            <div className="auth-scan-line" />
          </div>

          {/* Result chip */}
          <div className="auth-chip auth-chip--result">
            <div className="auth-chip-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
                <polyline points="20 6 9 17 4 12" />
              </svg>
            </div>
            <div>
              <div className="auth-chip-title">Text extracted</div>
              <div className="auth-chip-sub">98.4% accuracy</div>
            </div>
          </div>

          {/* Stats chip */}
          <div className="auth-chip auth-chip--stats">
            <div className="auth-chip-icon auth-chip-icon--alt">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 3v18h18" />
                <path d="M7 16l4-4 4 4 4-6" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
            <div>
              <div className="auth-chip-title">1 240 pages</div>
              <div className="auth-chip-sub">processed today</div>
            </div>
          </div>
        </div>

        {/* Caption */}
        <div className="auth-visual-caption">
          <p className="auth-visual-caption-title">Intelligent OCR</p>
          <p className="auth-visual-caption-sub">Extract, search & manage document text at scale</p>
        </div>
      </div>
    </main>
  )
}

function AuthConfigurationScreen() {
  return (
    <main className="flex min-h-screen items-center justify-center bg-slate-100 px-4">
      <section className="w-full max-w-xl space-y-4 rounded-2xl bg-white p-8 shadow-xl">
        <h1 className="text-2xl font-semibold text-slate-900">Authentication is not configured</h1>
        <p className="text-sm leading-6 text-slate-600">
          The backend requires authentication, but the frontend has no Supabase configuration. Set
          <code className="mx-1 rounded bg-slate-100 px-1.5 py-0.5">VITE_SUPABASE_URL</code>
          and
          <code className="mx-1 rounded bg-slate-100 px-1.5 py-0.5">VITE_SUPABASE_PUBLISHABLE_KEY</code>
          and rebuild the frontend.
        </p>
        <button
          type="button"
          onClick={() => window.location.reload()}
          className="rounded-xl bg-slate-900 px-4 py-3 font-semibold text-white hover:bg-slate-800"
        >
          Reload
        </button>
      </section>
    </main>
  )
}

function ProfilePage({ name, userEmail, userId, createdAt, updatedAt, lastSignInAt, emailConfirmed, documentCount, processedCount, favoriteCount, onSaveName, onBackToLibrary }) {
  const [editing, setEditing] = useState(false)
  const [draftName, setDraftName] = useState(name)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const initials = String(name || userEmail || 'User')
    .split(/[\s@]+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0])
    .join('')
    .toUpperCase()

  useEffect(() => setDraftName(name), [name])

  async function handleSave(event) {
    event.preventDefault()
    const normalizedName = draftName.trim()
    if (normalizedName.length < 2 || normalizedName.length > 100) {
      setError('Name must be between 2 and 100 characters.')
      return
    }
    setBusy(true)
    setError('')
    try {
      await onSaveName(normalizedName)
      setEditing(false)
    } catch (saveError) {
      setError(saveError.message || 'Unable to save your profile.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="mx-auto max-w-6xl px-4 pb-12 pt-8 sm:px-6 lg:px-8">
      <header className="flex flex-wrap items-center justify-between gap-5 border-b border-slate-200 pb-7">
        <div className="flex min-w-0 items-center gap-4">
          <div className="flex h-16 w-16 shrink-0 items-center justify-center rounded-full bg-cyan-50 text-xl font-semibold text-cyan-900">{initials}</div>
          <div className="min-w-0">
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Workspace account</p>
            <h1 className="mt-1 text-2xl font-semibold text-slate-900">Your profile</h1>
            <p className="mt-1 truncate text-sm text-slate-600">{userEmail || 'Email unavailable'}</p>
          </div>
        </div>
        <button type="button" onClick={onBackToLibrary} className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 transition hover:bg-slate-50">
          <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true"><path d="m14.5 5-7 7 7 7M8 12h13" strokeLinecap="round" strokeLinejoin="round" /></svg>
          Back to library
        </button>
      </header>

      <section aria-label="Workspace summary" className="grid grid-cols-3 divide-x divide-slate-200 border-b border-slate-200 py-5">
        <div className="pr-3 sm:px-5 sm:first:pl-0"><p className="text-xs font-medium text-slate-500">Documents</p><p className="mt-1 text-2xl font-semibold text-slate-900">{documentCount}</p></div>
        <div className="px-3 sm:px-5"><p className="text-xs font-medium text-slate-500">Processed</p><p className="mt-1 text-2xl font-semibold text-emerald-700">{processedCount}</p></div>
        <div className="pl-3 sm:px-5"><p className="text-xs font-medium text-slate-500">Favorites</p><p className="mt-1 text-2xl font-semibold text-amber-700">{favoriteCount}</p></div>
      </section>

      <div className="grid gap-10 py-8 md:grid-cols-2 md:gap-12">
        <section className="space-y-5 md:border-r md:border-slate-200 md:pr-10">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">Personal details</h2>
            <p className="mt-1 text-sm text-slate-500">Manage the name shown in your workspace.</p>
          </div>
          <div className="flex items-center justify-between gap-4 border-y border-slate-200 py-4">
            <div className="min-w-0">
              <p className="text-xs font-medium text-slate-500">Display name</p>
              <p className="mt-1 truncate text-sm font-semibold text-slate-900">{name}</p>
            </div>
            {!editing ? <button type="button" onClick={() => setEditing(true)} className="shrink-0 rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50">Edit name</button> : null}
          </div>
          {editing ? (
            <form onSubmit={handleSave} className="space-y-3">
              <label className="block space-y-1.5 text-sm font-medium text-slate-700">
                <span>Name</span>
                <input autoFocus maxLength={100} value={draftName} onChange={(event) => setDraftName(event.target.value)} className="w-full rounded-lg border border-slate-300 px-3 py-2.5 text-slate-900 outline-none focus:border-cyan-600 focus:ring-2 focus:ring-cyan-100" required />
              </label>
              {error ? <p role="alert" className="text-sm text-rose-700">{error}</p> : null}
              <div className="flex gap-2">
                <button disabled={busy} className="rounded-lg bg-slate-900 px-4 py-2.5 text-sm font-semibold text-white disabled:opacity-50">{busy ? 'Saving...' : 'Save changes'}</button>
                <button type="button" onClick={() => { setEditing(false); setDraftName(name); setError('') }} className="rounded-lg border border-slate-200 px-4 py-2.5 text-sm font-semibold text-slate-700">Cancel</button>
              </div>
            </form>
          ) : null}
        </section>

        <section className="space-y-5">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">Account details</h2>
            <p className="mt-1 text-sm text-slate-500">Security and account history.</p>
          </div>
          <dl className="divide-y divide-slate-200 border-y border-slate-200">
            <div className="flex items-start justify-between gap-4 py-3"><dt className="text-sm text-slate-500">Email status</dt><dd className={`text-right text-sm font-medium ${emailConfirmed ? 'text-emerald-700' : 'text-amber-700'}`}>{emailConfirmed ? 'Verified' : 'Pending verification'}</dd></div>
            <div className="flex items-start justify-between gap-4 py-3"><dt className="text-sm text-slate-500">Member since</dt><dd className="text-right text-sm font-medium text-slate-900">{createdAt ? formatDate(createdAt) : 'Not available'}</dd></div>
            <div className="flex items-start justify-between gap-4 py-3"><dt className="text-sm text-slate-500">Last sign in</dt><dd className="text-right text-sm font-medium text-slate-900">{lastSignInAt ? formatDate(lastSignInAt) : 'Not available'}</dd></div>
            <div className="flex items-start justify-between gap-4 py-3"><dt className="text-sm text-slate-500">Profile updated</dt><dd className="text-right text-sm font-medium text-slate-900">{updatedAt ? formatDate(updatedAt) : 'Not available'}</dd></div>
            <div className="space-y-1 py-3"><dt className="text-sm text-slate-500">Account ID</dt><dd className="break-all font-mono text-xs text-slate-700">{userId || 'Not available'}</dd></div>
          </dl>
        </section>
      </div>
    </main>
  )
}

function CompleteProfilePage({ initialName, error, onSubmit, onSignOut }) {
  const [name, setName] = useState(initialName)
  const [busy, setBusy] = useState(false)
  const [validationError, setValidationError] = useState('')

  useEffect(() => setName(initialName), [initialName])

  async function handleSubmit(event) {
    event.preventDefault()
    const normalizedName = name.trim()
    if (normalizedName.length < 2 || normalizedName.length > 100) {
      setValidationError('Name must be between 2 and 100 characters.')
      return
    }
    setBusy(true)
    setValidationError('')
    try {
      await onSubmit(normalizedName)
    } catch (submitError) {
      setValidationError(submitError.message || 'Unable to save your profile.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center bg-slate-100 px-4">
      <form onSubmit={handleSubmit} className="w-full max-w-md space-y-5 rounded-2xl bg-white p-8 shadow-xl">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">Profile</p>
          <h1 className="mt-2 text-2xl font-semibold text-slate-900">Complete Your Profile</h1>
        </div>
        <label className="block space-y-1.5 text-sm font-medium text-slate-700">
          <span>Name</span>
          <input autoFocus maxLength={100} value={name} onChange={(event) => setName(event.target.value)} className="w-full rounded-xl border border-slate-200 px-4 py-3 text-slate-900 outline-none focus:border-blue-400" required />
        </label>
        {validationError || error ? <p role="alert" className="text-sm text-rose-600">{validationError || error}</p> : null}
        <button disabled={busy} className="w-full rounded-xl bg-slate-900 px-4 py-3 font-semibold text-white disabled:opacity-50">
          {busy ? 'Saving...' : 'Continue'}
        </button>
        <button type="button" onClick={onSignOut} className="w-full rounded-xl border border-slate-200 px-4 py-3 font-semibold text-slate-700">
          Sign Out
        </button>
      </form>
    </main>
  )
}

const DEFAULT_PAGE = 'landing'
const DEFAULT_FILTER = 'recent'
const DEFAULT_ZOOM = 100
pdfjsLib.GlobalWorkerOptions.workerSrc = pdfjsWorker

function parseInitialRoute() {
  const url = new URL(window.location.href)
  const page = url.searchParams.get('page') || DEFAULT_PAGE
  const documentId = Number(url.searchParams.get('documentId') || 0) || null
  const attributeId = Number(url.searchParams.get('attributeId') || 0) || null
  return { page, documentId, attributeId }
}

function formatDate(value) {
  if (!value) {
    return 'Unknown date'
  }
  const date = new Date(value)
  return new Intl.DateTimeFormat('en', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  }).format(date)
}

function formatRelativeTime(value) {
  if (!value) {
    return 'Just now'
  }
  const date = new Date(value)
  const delta = Date.now() - date.getTime()
  const minutes = Math.max(1, Math.round(delta / 60000))
  if (minutes < 60) {
    return `${minutes} min ago`
  }
  const hours = Math.round(minutes / 60)
  if (hours < 24) {
    return `${hours} hr ago`
  }
  const days = Math.round(hours / 24)
  return `${days} d ago`
}

function confidenceTone(score) {
  if (score >= 0.9) return 'bg-emerald-50 text-emerald-700 border-emerald-200'
  if (score >= 0.7) return 'bg-amber-50 text-amber-700 border-amber-200'
  if (score >= 0.5) return 'bg-orange-50 text-orange-700 border-orange-200'
  return 'bg-rose-50 text-rose-700 border-rose-200'
}

function confidenceToPercent(score) {
  const value = Number(score || 0)
  if (!Number.isFinite(value)) {
    return 0
  }
  return value > 1 ? value : value * 100
}

function confidenceBorderTone(score) {
  if (score >= 0.9) return 'border-emerald-300 bg-emerald-50/35'
  if (score >= 0.75) return 'border-amber-300 bg-amber-50/35'
  return 'border-rose-300 bg-rose-50/35'
}

function statusTone(status) {
  const normalized = String(status || '').toUpperCase()
  if (normalized === 'PROCESSED') return 'bg-emerald-50 text-emerald-700 border-emerald-200'
  if (normalized === 'PROCESSING') return 'bg-blue-50 text-blue-700 border-blue-200'
  if (normalized === 'QUEUED') return 'bg-slate-100 text-slate-700 border-slate-200'
  if (normalized === 'FAILED') return 'bg-rose-50 text-rose-700 border-rose-200'
  return 'bg-slate-100 text-slate-600 border-slate-200'
}

function pageTitle(page, documentData) {
  if (page === 'profile') return 'Profile'
  if (page === 'library') return 'Library'
  if (page === 'processing') return 'Processing Document'
  if (page === 'viewer') {
    return documentData ? `Reviewing ${documentData.name}` : 'Document Viewer'
  }
  return 'Document Dashboard'
}

function describeApiError(detail, fallback) {
  if (typeof detail === 'string' && detail.trim()) return detail
  if (Array.isArray(detail) && detail.length) {
    // FastAPI validation errors arrive as a list of {loc, msg} objects.
    return detail.map((item) => [Array.isArray(item?.loc) ? item.loc.slice(1).join('.') : '', item?.msg].filter(Boolean).join(': ')).join('; ') || fallback
  }
  return fallback
}

function readTableStructure(attribute) {
  if (attribute?.table_structure && typeof attribute.table_structure === 'object') {
    return attribute.table_structure
  }
  try {
    const parsed = JSON.parse(attribute?.extracted_value || '')
    if (parsed?.kind !== 'table') return null
    return parsed.table && typeof parsed.table === 'object' ? parsed.table : parsed
  } catch {
    return null
  }
}

function normalizeDocumentAttributes(attributes) {
  return attributes.map((attribute) => {
    const tableData = readTableStructure(attribute)
    return {
      ...attribute,
      kind: tableData ? 'table' : attribute.kind,
      table_data: tableData,
      source: attribute.source || 'model',
      saved: attribute.saved ?? Boolean(attribute.id),
      draft_label: attribute.entity_type_label,
      draft_value: attribute.extracted_value,
      draft_validation_status: attribute.validation_status,
      original_label: attribute.entity_type_label,
      original_value: attribute.extracted_value,
      original_validation_status: attribute.validation_status,
    }
  })
}

function isStructuredTable(table) {
  const rows = Array.isArray(table?.rows) ? table.rows : []
  const columns = Array.isArray(table?.columns) ? table.columns : []
  if (table?.source === 'gemini' || table?.source === 'claude') {
    // The model judged this region to be a table from its borders, cell boundaries and layout, so it is not
    // second-guessed by counting rows or filled cells: form-like tables have few records, merged cells and
    // long paragraphs, and are still tables.
    const cellCount = rows.reduce((sum, row) => sum + (row.cells || []).length, 0)
    return rows.length >= 1 && columns.length >= 1 && cellCount >= 2
  }
  if (rows.length < 2 || columns.length < 2) {
    return false
  }
  const multiCellRows = rows.filter((row) => (row.cells || []).filter((cell) => String(cell.text || '').trim()).length >= 2).length
  return multiCellRows >= Math.max(2, Math.ceil(rows.length * 0.5))
}

function collectDocumentTables(ocrPages, attributes) {
  const tablesById = new Map()
  for (const page of ocrPages || []) {
    for (const table of page.tables || []) {
      if (isStructuredTable(table)) {
        tablesById.set(table.id, table)
      }
    }
  }
  for (const attribute of attributes || []) {
    const table = attribute.table_data || (attribute.kind === 'table' ? readTableStructure(attribute) : null)
    if (table && table.id && isStructuredTable(table)) {
      tablesById.set(table.id, table)
    }
  }
  return [...tablesById.values()].sort((left, right) =>
    Number(left.page_number) - Number(right.page_number)
    || Number(left.bbox?.[1] || 0) - Number(right.bbox?.[1] || 0)
    || Number(left.bbox?.[0] || 0) - Number(right.bbox?.[0] || 0),
  )
}

function isValidBox(box) {
  return Array.isArray(box) && box.length === 4 && box.every((value) => Number.isFinite(Number(value)))
}

// The model returns at most one approximate region per attribute (no word tokens), so the page viewer draws
// each attribute's own boxes. Attributes without boxes stay selectable from the side panel.
function buildAttributeBoxIndex(attributes, pages) {
  const boxToAttribute = new Map()
  const attributeToBoxes = new Map()
  const boxesByPage = new Map()

  for (const attribute of attributes || []) {
    const attributeKey = attribute.id ?? attribute.temp_id
    if (attributeKey == null) {
      continue
    }
    const boxKeys = []

    for (const page of pages || []) {
      const pageSpan = (attribute.page_spans || []).find((span) => Number(span.page_number) === Number(page.page_number))
      const boxes = pageSpan
        ? (Array.isArray(pageSpan.bounding_boxes) ? pageSpan.bounding_boxes : [])
        : (Number(attribute.page_number) === Number(page.page_number) || !attribute.page_number
          ? (Array.isArray(attribute.bounding_boxes) ? attribute.bounding_boxes : [])
          : [])
      boxes.forEach((box, index) => {
        if (!isValidBox(box)) {
          return
        }
        const boxKey = `${page.page_number}:${attributeKey}:${index}`
        boxToAttribute.set(boxKey, attributeKey)
        boxKeys.push(boxKey)
        const entry = {
          key: boxKey,
          attributeKey,
          box: box.map(Number),
          score: Number(attribute.confidence_score || 0),
          label: attribute.draft_label ?? attribute.entity_type_label ?? '',
          value: attribute.draft_value ?? attribute.extracted_value ?? '',
        }
        const pageEntries = boxesByPage.get(Number(page.page_number)) || []
        pageEntries.push(entry)
        boxesByPage.set(Number(page.page_number), pageEntries)
      })
    }

    attributeToBoxes.set(attributeKey, boxKeys)
  }

  return { boxToAttribute, attributeToBoxes, boxesByPage }
}

function getDocumentSearchText(document) {
  if (!document) {
    return ''
  }

  const attributeText = Array.isArray(document.attribute_summaries)
    ? document.attribute_summaries
        .map((attribute) => `${attribute.label || ''} ${attribute.value || ''} ${attribute.validation_status || ''}`)
        .join(' ')
    : ''

  return `${document.name || ''} ${document.format || ''} ${document.status || ''} ${attributeText}`.toLowerCase()
}

function mapStatusProgress(status) {
  const normalized = String(status || '').toUpperCase()
  if (normalized === 'QUEUED') return 35
  if (normalized === 'PROCESSING') return 67
  if (normalized === 'PROCESSED') return 100
  if (normalized === 'FAILED') return 100
  return 20
}

function resolveApiAssetUrl(path, apiBaseUrl) {
  if (!path) {
    return ''
  }
  if (/^https?:\/\//i.test(path)) {
    return path
  }
  const base = String(apiBaseUrl || '').replace(/\/+$/, '')
  const normalizedPath = path.startsWith('/') ? path : `/${path}`
  return `${base}${normalizedPath}`
}

function AuthenticatedPageImage({ src, alt, className, children }) {
  const [imageUrl, setImageUrl] = useState('')
  const [loadFailed, setLoadFailed] = useState(false)

  useEffect(() => {
    let active = true
    let objectUrl = ''
    setImageUrl('')
    setLoadFailed(false)

    async function loadImage() {
      try {
        const response = await authenticatedFetch(src)
        if (!response.ok) {
          throw new Error('Page image request failed')
        }
        const blob = await response.blob()
        const nextObjectUrl = URL.createObjectURL(blob)
        if (!active) {
          URL.revokeObjectURL(nextObjectUrl)
          return
        }
        objectUrl = nextObjectUrl
        setImageUrl(objectUrl)
      } catch {
        if (active) setLoadFailed(true)
      }
    }

    void loadImage()
    return () => {
      active = false
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [src])

  if (loadFailed) {
    return (
      <div className="flex min-h-[480px] w-full items-center justify-center rounded-xl border border-dashed border-slate-300 bg-white text-sm text-slate-500">
        Source page preview is unavailable.
      </div>
    )
  }

  if (!imageUrl) {
    return (
      <div role="status" className="flex min-h-[480px] w-full items-center justify-center rounded-xl border border-slate-200 bg-white text-sm text-slate-500">
        Loading source page...
      </div>
    )
  }

  return (
    <>
      <img src={imageUrl} alt={alt} className={className} draggable="false" />
      {children}
    </>
  )
}

function LoadingSkeleton({ className = '' }) {
  return <div className={`skeleton rounded-2xl ${className}`} />
}
function EffectRegister({ popupRef, onClose }) {
  useEffect(() => {
    function handleMouseDown(event) {
      try {
        if (!popupRef?.current) return
        if (!popupRef.current.contains(event.target)) onClose()
      } catch {
        // Ignore detached popup references during unmount.
      }
    }

    function handleKey(event) {
      if (event.key === 'Escape') onClose()
    }

    document.addEventListener('mousedown', handleMouseDown)
    document.addEventListener('keydown', handleKey)
    return () => {
      document.removeEventListener('mousedown', handleMouseDown)
      document.removeEventListener('keydown', handleKey)
    }
  }, [popupRef, onClose])

  return null
}

function TopBar({ pageLabel, documentData, onNavigateLanding, searchQuery, setSearchQuery, searchMode, setSearchMode, onRefresh, serverResults, serverPopupOpen, onCloseSearchResults, onSelectSearchResult, userEmail, onOpenProfile, onOpenLibrary, onSignOut }) {
  const searchPopupRef = useRef(null)
  const searchModeRef = useRef(null)
  const accountMenuRef = useRef(null)
  const [searchModeOpen, setSearchModeOpen] = useState(false)
  const [accountMenuOpen, setAccountMenuOpen] = useState(false)

  return (
    <header className="sticky top-0 z-30 border-b border-slate-200/80 bg-white/95 backdrop-blur-xl">
      <EffectRegister popupRef={searchPopupRef} onClose={onCloseSearchResults} />
      <EffectRegister popupRef={searchModeRef} onClose={() => setSearchModeOpen(false)} />
      <EffectRegister popupRef={accountMenuRef} onClose={() => setAccountMenuOpen(false)} />
      <div className="mx-auto flex max-w-[1600px] flex-col gap-3 px-4 py-3 sm:px-6 lg:flex-row lg:items-center lg:justify-between lg:px-8 lg:py-4">
        <button type="button" onClick={onNavigateLanding} className="flex min-w-0 items-center gap-3 text-left">
          <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl border border-slate-200 bg-gradient-to-br from-blue-600 to-cyan-500 text-white shadow-sm">
            <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8">
              <path d="M7 3h7l5 5v13H7z" />
              <path d="M14 3v6h6" />
            </svg>
          </div>
          <div className="min-w-0">
            <p className="text-sm font-semibold tracking-wide text-slate-900">OCR Studio</p>
            <p className="truncate text-xs text-slate-500">Premium document extraction workspace</p>
          </div>
        </button>

        <div className="flex shrink-0 items-center justify-end gap-2 sm:gap-3">
          {pageLabel === 'Document Dashboard' || pageLabel === 'Library' ? (
            <div ref={searchPopupRef} className="relative flex max-w-full items-center gap-2 rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2">
              <div ref={searchModeRef} className="relative shrink-0">
                <button
                  type="button"
                  aria-label="Choose search type"
                  aria-haspopup="menu"
                  aria-expanded={searchModeOpen}
                  onClick={() => setSearchModeOpen((open) => !open)}
                  className="flex h-9 w-24 items-center justify-between gap-2 rounded-lg border border-slate-200 bg-white px-3 text-sm font-medium text-slate-700 shadow-sm transition hover:border-slate-300 hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                >
                  {searchMode === 'documents' ? 'Docs' : 'Attrs'}
                  <svg viewBox="0 0 20 20" className={`h-4 w-4 text-slate-500 transition-transform ${searchModeOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true"><path d="m5 7.5 5 5 5-5" strokeLinecap="round" strokeLinejoin="round" /></svg>
                </button>
                {searchModeOpen ? (
                  <div role="menu" className="absolute left-0 top-full z-50 mt-2 w-64 overflow-hidden rounded-xl border border-slate-200 bg-white p-1.5 shadow-xl">
                    {[
                      { value: 'documents', label: 'Documents', description: 'Search file names and extracted fields' },
                      { value: 'attributes', label: 'Attributes', description: 'Search extracted values and labels' },
                    ].map((mode) => (
                      <button
                        key={mode.value}
                        type="button"
                        role="menuitemradio"
                        aria-checked={searchMode === mode.value}
                        onClick={() => {
                          setSearchMode?.(mode.value)
                          setSearchModeOpen(false)
                        }}
                        className={`flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left transition ${searchMode === mode.value ? 'bg-cyan-50 text-cyan-950' : 'text-slate-700 hover:bg-slate-50'}`}
                      >
                        <span className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${searchMode === mode.value ? 'bg-white text-cyan-700' : 'bg-slate-100 text-slate-500'}`}>
                          <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
                            {mode.value === 'documents' ? <><path d="M6 3.5h8l4 4v13H6z" /><path d="M14 3.5v4h4M9 13h6M9 16h6" strokeLinecap="round" /></> : <><path d="M4 7h16M4 12h10M4 17h7" strokeLinecap="round" /><circle cx="18" cy="16" r="3" /></>}
                          </svg>
                        </span>
                        <span className="min-w-0 flex-1">
                          <span className="block text-sm font-semibold">{mode.label}</span>
                          <span className="mt-0.5 block text-xs text-slate-500">{mode.description}</span>
                        </span>
                        {searchMode === mode.value ? <svg viewBox="0 0 20 20" className="h-4 w-4 shrink-0 text-cyan-700" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><path d="m4 10 4 4 8-8" strokeLinecap="round" strokeLinejoin="round" /></svg> : null}
                      </button>
                    ))}
                  </div>
                ) : null}
              </div>
              <input type="search" value={searchQuery} onChange={(event) => setSearchQuery?.(event.target.value)} placeholder={searchMode === 'documents' ? 'Search documents' : 'Search attributes'} className="w-32 bg-transparent text-sm text-slate-900 outline-none placeholder:text-slate-400 sm:w-48 lg:w-72" />
              {serverPopupOpen && serverResults?.length ? (
                <div className="absolute right-0 top-full z-50 mt-2 w-[min(720px,calc(100vw-2rem))] overflow-hidden rounded-xl border border-slate-200 bg-white shadow-xl">
                  <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3 text-sm font-semibold">
                    <span>Search results</span>
                    <button type="button" title="Close" onClick={onCloseSearchResults} className="rounded px-2 py-1 text-sm text-slate-500 hover:bg-slate-50">✕</button>
                  </div>
                  <div className="max-h-80 overflow-auto">
                    {serverResults.slice(0, 8).map((item) => (
                      <button type="button" key={`${item.document_id || item.id}-${item.id}`} onClick={() => onSelectSearchResult(item)} className="block w-full border-b px-4 py-3 text-left hover:bg-slate-50">
                        <div className="flex items-center justify-between">
                          <div className="text-sm font-medium text-slate-900">{item.entity_type_label || item.name}</div>
                          <div className="text-xs text-slate-500">{item.document_id ? `Doc ${item.document_id}` : item.timestamp ? new Date(item.timestamp).toLocaleDateString() : ''}</div>
                        </div>
                        <div className="mt-1 text-sm text-slate-500">{item.extracted_value || item.document_type || ''}</div>
                      </button>
                    ))}
                  </div>
                </div>
              ) : null}
            </div>
          ) : documentData ? (
            <>
              <button type="button" onClick={onNavigateLanding} className="rounded-full border border-slate-200 bg-white px-4 py-2 text-sm text-slate-600 transition hover:bg-slate-50">Back</button>
              <button type="button" onClick={() => onRefresh?.()} className="rounded-full border border-slate-200 bg-white px-4 py-2 text-sm text-slate-600 transition hover:bg-slate-50">Refresh</button>
            </>
          ) : null}

          <div ref={accountMenuRef} className="relative">
            <button
              type="button"
              aria-label="Open account menu"
              aria-haspopup="menu"
              aria-expanded={accountMenuOpen}
              onClick={() => setAccountMenuOpen((open) => !open)}
              className="flex items-center gap-2 rounded-full border border-slate-200 bg-white py-1.5 pl-1.5 pr-2.5 text-sm font-medium text-slate-700 transition hover:border-slate-300 hover:bg-slate-50"
            >
              <span className="flex h-8 w-8 items-center justify-center rounded-full bg-cyan-50 text-xs font-semibold text-cyan-800">
                {(userEmail?.[0] || 'P').toUpperCase()}
              </span>
              <span className="hidden max-w-36 truncate sm:inline">{userEmail ? userEmail.split('@')[0] : 'Account'}</span>
              <svg viewBox="0 0 20 20" className={`h-4 w-4 transition-transform ${accountMenuOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
                <path d="m5 7.5 5 5 5-5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
            {accountMenuOpen ? (
              <div role="menu" className="absolute right-0 top-full z-50 mt-2 w-56 overflow-hidden rounded-xl border border-slate-200 bg-white py-1 shadow-xl">
                {userEmail ? <p className="truncate border-b border-slate-100 px-4 py-3 text-xs text-slate-500">{userEmail}</p> : null}
                <button type="button" role="menuitem" onClick={() => { setAccountMenuOpen(false); onOpenProfile?.() }} className="flex w-full items-center gap-3 px-4 py-2.5 text-left text-sm text-slate-700 hover:bg-slate-50">
                  <svg viewBox="0 0 24 24" className="h-4 w-4 text-slate-500" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true"><circle cx="12" cy="8" r="3.5" /><path d="M4.5 20a7.5 7.5 0 0 1 15 0" strokeLinecap="round" /></svg>
                  Profile
                </button>
                <button type="button" role="menuitem" onClick={() => { setAccountMenuOpen(false); onOpenLibrary?.() }} className="flex w-full items-center gap-3 px-4 py-2.5 text-left text-sm text-slate-700 hover:bg-slate-50">
                  <svg viewBox="0 0 24 24" className="h-4 w-4 text-slate-500" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true"><path d="M3.5 6.5h6l2 2H20a1.5 1.5 0 0 1 1.5 1.5v8a1.5 1.5 0 0 1-1.5 1.5h-17A1.5 1.5 0 0 1 1.5 18v-10a1.5 1.5 0 0 1 2-1.5Z" /></svg>
                  Library
                </button>
                <div className="my-1 border-t border-slate-100" />
                <button type="button" role="menuitem" onClick={() => { setAccountMenuOpen(false); onSignOut?.() }} className="flex w-full items-center gap-3 px-4 py-2.5 text-left text-sm text-rose-700 hover:bg-rose-50">
                  <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true"><path d="M10 17l5-5-5-5M15 12H3" strokeLinecap="round" strokeLinejoin="round" /><path d="M12 3h6a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-6" strokeLinecap="round" /></svg>
                  Sign out
                </button>
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </header>
  )
}

function LandingPage({
  documents,
  documentsLoading,
  searchQuery,
  setSearchQuery,
  searchMode,
  setSearchMode,
  historyFilter,
  setHistoryFilter,
  favoriteIds,
  toggleFavorite,
  onOpenDocument,
  onUploadDocument,
  loadError,
  apiBaseUrl,
  onServerResults,
  serverResults,
  serverLoading,
  serverError,
  serverTotal,
  libraryOnly = false,
  allDocuments = documents,
  onOpenLibrary,
  onBackToDashboard,
}) {
  const [dragActive, setDragActive] = useState(false)
  const fileInputRef = useRef(null)
  const serverPopupRef = useRef(null)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(12)
  const [showFilters, setShowFilters] = useState(false)
  const [docTypes, setDocTypes] = useState([])
  const [attributeNames, setAttributeNames] = useState([])
  const [filters, setFilters] = useState({ document_type: '', status: '', start_date: '', end_date: '', attribute_name: '', min_conf: '', max_conf: '', hitl_corrected: '' })

  useEffect(() => {
    let active = true
    let retryTimer
    async function loadOptions(attempt = 0) {
      try {
        const [dtResp, anResp] = await Promise.all([
          fetch(`${apiBaseUrl}/api/v1/search/document-types`),
          fetch(`${apiBaseUrl}/api/v1/search/attribute-names`),
        ])
        const [dtJson, anJson] = await Promise.all([dtResp.json(), anResp.json()])
        if (!active) return
        setDocTypes(dtJson.document_types || [])
        setAttributeNames(anJson.attribute_names || [])
      } catch {
        if (active && attempt < 10) retryTimer = window.setTimeout(() => loadOptions(attempt + 1), 1500)
      }
    }
    loadOptions()
    return () => {
      active = false
      if (retryTimer) window.clearTimeout(retryTimer)
    }
  }, [apiBaseUrl])

  useEffect(() => {
    const interval = setInterval(() => {
      if (page === 'landing') void refreshDocuments().catch(() => {})
    }, 15000)
    return () => clearInterval(interval)
  }, [page, apiBaseUrl])

  useEffect(() => {
    const requestKey = JSON.stringify({ query: searchQuery.trim().toLowerCase(), mode: searchMode, page, pageSize, filters })
    const hasSearchCriteria = Boolean(searchQuery.trim() || Object.values(filters).some(Boolean))
    if (!hasSearchCriteria) {
      onServerResults?.({ results: [], total: 0, mode: searchMode, q: '', requestKey })
      return undefined
    }

    onServerResults?.({ results: [], total: 0, mode: searchMode, q: searchQuery, loading: true, requestKey })
    const timeout = setTimeout(() => {
      async function doSearch() {
        try {
          const params = new URLSearchParams()
          if (searchQuery) params.set('q', searchQuery)
          params.set('page', String(page))
          params.set('page_size', String(pageSize))
          if (searchMode === 'documents') {
            if (filters.document_type) params.set('document_type', filters.document_type)
            if (filters.status) params.set('status', filters.status)
            if (filters.start_date) params.set('start_date', filters.start_date)
            if (filters.end_date) params.set('end_date', filters.end_date)
            const response = await fetch(`${apiBaseUrl}/api/v1/search/documents?${params}`)
            const data = await response.json()
            if (!response.ok) throw new Error(data.detail || 'Search failed')
            onServerResults?.({ results: data.documents || [], total: data.total || 0, mode: 'documents', q: searchQuery, requestKey })
          } else {
            if (filters.attribute_name) params.set('attribute_name', filters.attribute_name)
            if (filters.min_conf) params.set('min_conf', filters.min_conf)
            if (filters.max_conf) params.set('max_conf', filters.max_conf)
            if (filters.document_type) params.set('document_type', filters.document_type)
            if (filters.hitl_corrected) params.set('hitl_corrected', filters.hitl_corrected)
            const response = await fetch(`${apiBaseUrl}/api/v1/search/attributes?${params}`)
            const data = await response.json()
            if (!response.ok) throw new Error(data.detail || 'Search failed')
            onServerResults?.({ results: data.attributes || [], total: data.total || 0, mode: 'attributes', q: searchQuery, requestKey })
          }
        } catch (error) {
          onServerResults?.({ results: [], total: 0, mode: searchMode, q: searchQuery, error: error.message, requestKey })
        }
      }
      void doSearch()
    }, 350)
    return () => clearTimeout(timeout)
  }, [searchQuery, page, pageSize, searchMode, filters, apiBaseUrl])

  useEffect(() => setPage(1), [searchQuery, searchMode])

  const handleFileSelected = (event) => {
    const file = event.target.files?.[0]
    if (file) {
      onUploadDocument(file)
    }
    event.target.value = ''
  }

  const filteredHistory = documents.filter((document) => {
    if (historyFilter === 'favorites') {
      return favoriteIds.includes(document.id)
    }
    return true
  })

  // Recent history should not be filtered by the global search input —
  // search results are shown in the popup. Keep recent documents stable.
  const recentDocuments = historyFilter === 'recent' ? filteredHistory.slice(0, 10) : filteredHistory
  const processedCount = allDocuments.filter((document) => String(document.status || '').toUpperCase() === 'PROCESSED').length
  const favoriteCount = allDocuments.filter((document) => favoriteIds.includes(document.id)).length

  return (
    <div className={libraryOnly ? 'mx-auto min-h-[calc(100vh-72px)] max-w-7xl px-4 pb-10 pt-8 sm:px-6 lg:px-8' : 'mx-auto grid min-h-[calc(100vh-140px)] max-w-[1800px] gap-6 px-4 pb-8 pt-6 lg:px-8'}>
      <section className={`${libraryOnly ? 'hidden' : ''} grid gap-8 rounded-[28px] border border-slate-200 bg-white px-4 py-8 shadow-[0_10px_30px_rgba(15,23,42,0.05)] lg:grid-cols-1 lg:px-12`}>
        <div className="w-full">
          <div className="grid gap-8">
            <div className="space-y-5">
              <div className="inline-flex items-center gap-2 rounded-full border border-blue-100 bg-blue-50 px-4 py-2 text-xs font-semibold uppercase tracking-[0.24em] text-blue-700">
                Fast OCR review
              </div>
              <h2 className="max-w-2xl text-4xl font-semibold tracking-tight text-slate-900 sm:text-5xl">
                A refined workspace for document extraction and human review.
              </h2>
              <p className="max-w-2xl text-base leading-7 text-slate-500">
                Upload PDF, DOCX, DOC, or scanned image files and move from ingestion to review in a single, polished flow.
              </p>

              <div
                onDragOver={(event) => {
                  event.preventDefault()
                  setDragActive(true)
                }}
                onDragLeave={() => setDragActive(false)}
                onDrop={(event) => {
                  event.preventDefault()
                  setDragActive(false)
                  const file = event.dataTransfer.files?.[0]
                  if (file) {
                    onUploadDocument(file)
                  }
                }}
                className={`rounded-[30px] border-2 border-dashed bg-slate-50 p-6 transition ${dragActive ? 'border-blue-400 bg-blue-50/60' : 'border-slate-200'}`}
              >
                <div className="grid gap-5 md:grid-cols-[auto_1fr] md:items-center">
                  <div className="flex h-20 w-20 items-center justify-center rounded-3xl bg-white text-blue-600 shadow-sm">
                    <svg viewBox="0 0 24 24" className="h-10 w-10" fill="none" stroke="currentColor" strokeWidth="1.6">
                      <path d="M7 3h7l5 5v13H7z" />
                      <path d="M14 3v6h6" />
                      <path d="M8 13h8" />
                      <path d="M8 16h8" />
                    </svg>
                  </div>

                  <div className="space-y-4">
                    <div>
                      <h3 className="text-xl font-semibold text-slate-900">Drop a document here</h3>
                      <p className="mt-2 text-sm leading-6 text-slate-500">
                        Drag and drop a file, or choose one from your device. The next step opens automatically.
                      </p>
                    </div>

                    <div className="flex flex-wrap items-center gap-3">
                      <button
                        type="button"
                        onClick={() => fileInputRef.current?.click()}
                        className="rounded-full bg-slate-900 px-5 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-slate-800"
                      >
                        Choose File
                      </button>
                      <span className="text-sm text-slate-500">PDF, DOCX, DOC, PNG, JPG, TIFF, BMP</span>
                    </div>

                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".pdf,.docx,.doc,.jpeg,.jpg,.png,.tiff,.tif,.bmp"
                      onChange={handleFileSelected}
                      className="hidden"
                    />
                  </div>
                </div>
              </div>
            </div>

            <div className="space-y-4 rounded-[30px] border border-slate-200 bg-slate-50 p-6 shadow-[0_10px_30px_rgba(15,23,42,0.04)]">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">Workspace preview</p>
                  <p className="mt-1 text-sm text-slate-500">Minimal, spacious, and built for speed</p>
                </div>
                <div className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs text-slate-500">Live</div>
              </div>

              <div className="grid gap-3 sm:grid-cols-2">
                <div className="rounded-3xl border border-slate-200 bg-white p-4 shadow-sm">
                  <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Pipeline</p>
                  <div className="mt-4 space-y-3">
                    {['Upload', 'OCR', 'Layout', 'Attributes'].map((item, index) => (
                      <div key={item} className="flex items-center gap-3">
                        <span className="flex h-7 w-7 items-center justify-center rounded-full bg-blue-50 text-xs font-semibold text-blue-700">
                          {index + 1}
                        </span>
                        <span className="text-sm text-slate-600">{item}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="rounded-3xl border border-slate-200 bg-white p-4 shadow-sm">
                  <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Quality</p>
                  <div className="mt-4 space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-slate-600">High confidence</span>
                      <span className="text-sm font-semibold text-emerald-700">91%</span>
                    </div>
                    <div className="h-2 rounded-full bg-slate-100">
                      <div className="h-2 w-[91%] rounded-full bg-emerald-500" />
                    </div>
                    <div className="rounded-2xl bg-slate-50 px-3 py-3 text-sm text-slate-500">
                      Smooth editing, instant search, and synchronized highlights.
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className={`${libraryOnly ? 'border-0 p-0 shadow-none' : 'rounded-[28px] border border-slate-200 bg-white p-5 shadow-[0_10px_30px_rgba(15,23,42,0.05)]'} overflow-hidden`}>
        <div className="space-y-5">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
            <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">Library</p>
            <h1 className="mt-2 text-2xl font-semibold text-slate-900">{libraryOnly ? 'Your document library' : 'Processed documents'}</h1>
            <p className="mt-2 text-sm leading-6 text-slate-500">{libraryOnly ? 'Browse, revisit, and organize every document in your workspace.' : 'Search by filename, extracted fields, or document metadata.'}</p>
            </div>
            {libraryOnly ? (
              <button type="button" onClick={onBackToDashboard} className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 transition hover:border-slate-300 hover:bg-slate-50">
                <svg viewBox="0 0 20 20" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true"><path d="M15.5 10h-12m5 5-5-5 5-5" strokeLinecap="round" strokeLinejoin="round" /></svg>
                Back to dashboard
              </button>
            ) : (
              <button type="button" onClick={onOpenLibrary} className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 transition hover:border-slate-300 hover:bg-slate-50">
                Open library
                <svg viewBox="0 0 20 20" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true"><path d="M3.5 10h12m-5-5 5 5-5 5" strokeLinecap="round" strokeLinejoin="round" /></svg>
              </button>
            )}
          </div>

          {libraryOnly ? (
            <div className="grid grid-cols-3 divide-x divide-slate-200 border-y border-slate-200 py-4">
              <div className="px-3 first:pl-0 sm:px-5"><p className="text-xs font-medium text-slate-500">All documents</p><p className="mt-1 text-xl font-semibold text-slate-900">{allDocuments.length}</p></div>
              <div className="px-3 sm:px-5"><p className="text-xs font-medium text-slate-500">Processed</p><p className="mt-1 text-xl font-semibold text-emerald-700">{processedCount}</p></div>
              <div className="px-3 sm:px-5"><p className="text-xs font-medium text-slate-500">Favorites</p><p className="mt-1 text-xl font-semibold text-amber-700">{favoriteCount}</p></div>
            </div>
          ) : null}

          {showFilters ? (
            <div className="mt-3 grid gap-2 sm:grid-cols-2">
              {searchMode === 'documents' ? (
                <>
                  <select value={filters.document_type} onChange={(e) => setFilters((f) => ({ ...f, document_type: e.target.value }))} className="rounded-md border bg-white px-2 py-1 text-sm">
                    <option value="">All document types</option>
                    {docTypes.map((dt) => (<option key={dt} value={dt}>{dt}</option>))}
                  </select>
                  <select value={filters.status} onChange={(e) => setFilters((f) => ({ ...f, status: e.target.value }))} className="rounded-md border bg-white px-2 py-1 text-sm">
                    <option value="">All statuses</option>
                    <option value="QUEUED">QUEUED</option>
                    <option value="PROCESSING">PROCESSING</option>
                    <option value="PROCESSED">PROCESSED</option>
                    <option value="FAILED">FAILED</option>
                  </select>
                  <input type="date" value={filters.start_date} onChange={(e) => setFilters((f) => ({ ...f, start_date: e.target.value }))} className="rounded-md border bg-white px-2 py-1 text-sm" />
                  <input type="date" value={filters.end_date} onChange={(e) => setFilters((f) => ({ ...f, end_date: e.target.value }))} className="rounded-md border bg-white px-2 py-1 text-sm" />
                </>
              ) : (
                <>
                  <select value={filters.attribute_name} onChange={(e) => setFilters((f) => ({ ...f, attribute_name: e.target.value }))} className="rounded-md border bg-white px-2 py-1 text-sm">
                    <option value="">All attribute names</option>
                    {attributeNames.map((an) => (<option key={an} value={an}>{an}</option>))}
                  </select>
                  <input type="number" step="0.01" min="0" max="1" placeholder="min confidence" value={filters.min_conf} onChange={(e) => setFilters((f) => ({ ...f, min_conf: e.target.value }))} className="rounded-md border bg-white px-2 py-1 text-sm" />
                  <input type="number" step="0.01" min="0" max="1" placeholder="max confidence" value={filters.max_conf} onChange={(e) => setFilters((f) => ({ ...f, max_conf: e.target.value }))} className="rounded-md border bg-white px-2 py-1 text-sm" />
                  <select value={filters.hitl_corrected} onChange={(e) => setFilters((f) => ({ ...f, hitl_corrected: e.target.value }))} className="rounded-md border bg-white px-2 py-1 text-sm">
                    <option value="">All</option>
                    <option value="true">HITL corrected</option>
                    <option value="false">Original only</option>
                  </select>
                </>
              )}
            </div>
          ) : null}

          <div className="flex flex-wrap gap-2">
            {[
              { key: 'recent', label: 'Recent' },
              { key: 'favorites', label: 'Favorites' },
              { key: 'all', label: 'All Documents' },
            ].map((item) => (
              <button
                key={item.key}
                type="button"
                onClick={() => setHistoryFilter(item.key)}
                className={`rounded-full px-4 py-2 text-sm transition ${historyFilter === item.key ? 'bg-slate-900 text-white shadow-sm' : 'border border-slate-200 bg-white text-slate-600 hover:bg-slate-50'}`}
              >
                {item.label}
              </button>
            ))}
          </div>

            <div className="rounded-2xl border border-slate-200 bg-slate-50/70 p-3">
            <div className="flex items-center justify-between px-1 pb-3">
              <div>
                <p className="text-sm font-semibold text-slate-900">Recent history</p>
                <p className="text-xs text-slate-500">Click any item to open the result</p>
              </div>
              <span className="rounded-full bg-white px-2.5 py-1 text-xs text-slate-500 shadow-sm">{recentDocuments.length}</span>
            </div>

            <div className={`${libraryOnly ? 'max-h-none' : 'max-h-[calc(100vh-260px)]'} space-y-2 overflow-auto pr-2`}>
              {documentsLoading ? (
                <div className="space-y-2">
                  {Array.from({ length: 5 }).map((_, index) => (
                    <div key={index} className="flex items-center gap-3 rounded-2xl border border-slate-200 bg-white p-3">
                      <LoadingSkeleton className="skeleton-circle h-11 w-11" />
                      <div className="flex-1 space-y-2">
                        <LoadingSkeleton className="h-3 w-4/5" />
                        <LoadingSkeleton className="h-3 w-2/5" />
                      </div>
                    </div>
                  ))}
                </div>
              ) : recentDocuments.length ? (
                recentDocuments.map((document) => (
                  <div
                    key={document.id}
                    role="button"
                    tabIndex={0}
                    onClick={() => onOpenDocument(document)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault()
                        onOpenDocument(document)
                      }
                    }}
                    className="group flex w-full cursor-pointer items-center gap-3 rounded-2xl border border-transparent bg-white p-3 text-left transition hover:border-slate-200 hover:shadow-sm"
                  >
                    <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-slate-200 bg-slate-50 text-slate-400 transition group-hover:bg-blue-50 group-hover:text-blue-600">
                      <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8">
                        <path d="M7 3h7l5 5v13H7z" />
                        <path d="M14 3v6h6" />
                      </svg>
                    </div>

                    <div className="min-w-0 flex-1">
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0">
                          <p className="truncate text-sm font-semibold text-slate-900">{document.name}</p>
                          <p className="text-xs text-slate-500">{formatDate(document.timestamp)} · {document.page_count} page{document.page_count === 1 ? '' : 's'}</p>
                        </div>
                        <button
                          type="button"
                          onClick={(event) => {
                            event.stopPropagation()
                            toggleFavorite(document.id)
                          }}
                          className={`rounded-full border px-2 py-1 text-xs transition ${favoriteIds.includes(document.id) ? 'border-amber-200 bg-amber-50 text-amber-600' : 'border-slate-200 bg-white text-slate-400 hover:text-amber-500'}`}
                        >
                          ★
                        </button>
                      </div>

                      <div className="mt-2 flex items-center gap-2">
                        <span className={`rounded-full border px-2.5 py-1 text-[11px] font-medium ${statusTone(document.status)}`}>{document.status}</span>
                        <span className="text-xs text-slate-500">Updated {formatRelativeTime(document.timestamp)}</span>
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="rounded-2xl border border-dashed border-slate-200 bg-white px-4 py-10 text-center text-sm text-slate-500">
                  {searchQuery ? 'No results match your search.' : historyFilter === 'favorites' ? 'No favorites yet. Star a document to keep it here.' : historyFilter === 'all' ? 'No processed documents yet.' : 'No recent documents yet.'}
                </div>
              )}

              {/* Pagination controls for server results */}
              {serverTotal > 0 && (searchQuery.trim() || Object.values(filters).some(Boolean)) ? (
                <div className="mt-4 flex items-center justify-between">
                  <div className="text-sm text-slate-600">Showing {Math.min(page * pageSize, serverTotal)} of {serverTotal}</div>
                  <div className="flex items-center gap-2">
                    <button disabled={page <= 1} type="button" onClick={() => setPage((p) => Math.max(1, p - 1))} className="rounded-full border px-3 py-1 text-sm">
                      Prev
                    </button>
                    <button disabled={page * pageSize >= serverTotal} type="button" onClick={() => setPage((p) => p + 1)} className="rounded-full border px-3 py-1 text-sm">
                      Next
                    </button>
                  </div>
                </div>
              ) : null}
            </div>
          </div>

          {loadError ? (
            <div className="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">{loadError}</div>
          ) : null}
        </div>
      </section>
    </div>
  )
}

function UploadPagePreview({ preview, fileName }) {
  const docxRef = useRef(null)
  const canvasRef = useRef(null)
  const pageRef = useRef(null)
  const [previewError, setPreviewError] = useState('')
  const extension = String(fileName || '').split('.').pop()?.toLowerCase()
  const mimeType = preview?.type || ''
  const isImage = mimeType.startsWith('image/')
  const isPdf = mimeType === 'application/pdf' || extension === 'pdf'
  const isDocx = mimeType === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' || extension === 'docx'
  const isDocument = isPdf || isDocx || extension === 'doc' || extension === 'docm'

  useEffect(() => {
    let active = true
    setPreviewError('')

    async function renderPreview() {
      if (!preview?.url || !isPdf || !canvasRef.current) return
      try {
        const pdf = await pdfjsLib.getDocument({ url: preview.url }).promise
        const page = await pdf.getPage(1)
        const container = pageRef.current
        const containerWidth = Math.max(container?.clientWidth || 190, 160)
        const containerHeight = Math.max(container?.clientHeight || 264, 220)
        const baseViewport = page.getViewport({ scale: 1 })
        const scale = Math.min(
          (containerWidth - 16) / baseViewport.width,
          (containerHeight - 16) / baseViewport.height,
        )
        const viewport = page.getViewport({ scale })
        const canvas = canvasRef.current
        canvas.width = viewport.width
        canvas.height = viewport.height
        canvas.style.width = `${viewport.width}px`
        canvas.style.height = `${viewport.height}px`
        if (active) await page.render({ canvasContext: canvas.getContext('2d'), viewport }).promise
      } catch (error) {
        if (active) setPreviewError(error.message || 'Unable to render the first page.')
      }
    }

    async function renderWordPreview() {
      if (!preview?.url || !isDocx || !docxRef.current) return
      try {
        const response = await window.fetch(preview.url)
        const blob = await response.blob()
        await renderDocx(blob, docxRef.current, undefined, { inWrapper: true, breakPages: true })
        const pages = docxRef.current.querySelectorAll('.docx-wrapper > section.docx')
        pages.forEach((page, index) => {
          page.style.display = index === 0 ? 'block' : 'none'
        })
        const firstPage = pages[0]
        if (!firstPage) return
        await new Promise((resolve) => window.requestAnimationFrame(resolve))
        const availableWidth = Math.max(docxRef.current.clientWidth - 16, 1)
        const availableHeight = Math.max(docxRef.current.clientHeight - 16, 1)
        const pageWidth = firstPage.offsetWidth
        const pageHeight = firstPage.offsetHeight
        const scale = Math.min(availableWidth / pageWidth, availableHeight / pageHeight, 1)
        firstPage.style.transform = `scale(${scale})`
        firstPage.style.transformOrigin = 'center center'
      } catch (error) {
        if (active) setPreviewError(error.message || 'Unable to render the first page.')
      }
    }

    void renderPreview()
    void renderWordPreview()
    return () => {
      active = false
      if (docxRef.current) docxRef.current.replaceChildren()
    }
  }, [isDocx, isPdf, preview?.url])

  return (
    <div className="document-preview-shell">
      <div className="document-preview-page">
        {preview?.url && isImage ? (
          <div className="document-preview-page-inner">
            <img src={preview.url} alt="First page preview" className="document-preview-image" />
          </div>
        ) : null}
        {preview?.url && isPdf ? <div ref={pageRef} className="document-preview-page-inner"><canvas ref={canvasRef} className="document-preview-pdf" aria-label="First page preview" /></div> : null}
        {preview?.url && isDocx ? <div ref={docxRef} className="document-preview-docx" aria-label="First page preview" /> : null}
        {!preview?.url || (!isImage && !isPdf && !isDocx) || previewError ? (
          <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center">
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-blue-50 text-blue-700">
              <svg viewBox="0 0 24 24" className="h-7 w-7" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="M6 3.75h8l4 4v12.5H6z" /><path d="M14 3.75v4h4M8.5 12h7M8.5 15.5h5" /></svg>
            </div>
            <p className="text-sm font-semibold text-slate-700">{extension === 'doc' || extension === 'docm' ? 'Word document' : 'First page preview'}</p>
            <p className="text-xs leading-5 text-slate-500">{previewError || 'The file is ready for OCR. A first-page image preview is unavailable for this format.'}</p>
          </div>
        ) : null}
      </div>
      <div className="document-scan-line" aria-hidden="true" />
      <div className="document-preview-label">{isDocument ? 'Page 1' : 'Image preview'}</div>
    </div>
  )
}

function ProcessingPage({ activity, progress, etaLabel, documentData, processingError, onRetry, onCancel, onBackToLibrary }) {
  return (
    <div className="flex min-h-[calc(100vh-82px)] items-center justify-center px-4 py-8 lg:px-8">
      <div className="w-full max-w-5xl rounded-[32px] border border-slate-200 bg-white px-6 py-10 shadow-[0_10px_30px_rgba(15,23,42,0.05)] sm:px-10">
        <div className="grid gap-10 lg:grid-cols-1 lg:items-center">
          <div className="space-y-8 text-center lg:text-left">
            <div className="space-y-4">
              <p className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">Processing</p>
              <h1 className="text-4xl font-semibold tracking-tight text-slate-900 sm:text-5xl">Your document is being prepared.</h1>
              <p className="mx-auto max-w-2xl text-base leading-7 text-slate-500 lg:mx-0">
                OCR, layout detection, and attribute extraction are running in the background. We’ll move to the review workspace automatically when the results are ready.
              </p>
              {processingError ? (
                <div className="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
                  {processingError}
                </div>
              ) : null}
            </div>

            <div className="w-full space-y-4">
              <div role="progressbar" aria-label="Document processing progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(progress)} className="w-full rounded-full bg-slate-100 p-1">
                <div className="h-3 rounded-full bg-gradient-to-r from-blue-600 to-cyan-500 transition-all duration-500" style={{ width: `${Math.min(100, Math.max(0, progress))}%` }} />
              </div>
              <div className="flex flex-wrap items-center justify-between gap-3 text-sm text-slate-500">
                <span>{activity}</span>
                <span>{Math.round(progress)}%</span>
              </div>
              <div className="flex flex-wrap items-center gap-3 text-sm text-slate-500">
                <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1">{etaLabel}</span>
                {documentData ? (
                  <span className={`rounded-full border px-3 py-1 ${statusTone(documentData.status)}`}>{documentData.status}</span>
                ) : null}
              </div>
            </div>

            <div className="flex flex-wrap items-center justify-center gap-3 lg:justify-start">
              {processingError ? (
                <button
                  type="button"
                  onClick={onRetry}
                  className="rounded-full bg-blue-600 px-5 py-3 text-sm font-semibold text-white transition hover:bg-blue-700"
                >
                  Retry status check
                </button>
              ) : null}
              <button
                type="button"
                onClick={onBackToLibrary}
                className="rounded-full border border-slate-200 bg-white px-5 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
              >
                Back to library
              </button>
              <button
                type="button"
                onClick={onCancel}
                className="rounded-full border border-rose-200 bg-rose-50 px-5 py-3 text-sm font-semibold text-rose-700 transition hover:bg-rose-100"
              >
                Cancel
              </button>
            </div>
          </div>

        </div>
      </div>
    </div>
  )
}

function CanceledDocumentPage({ documentData, onBackToLibrary, onProcessAgain, processingError }) {
  return (
    <div className="flex min-h-[calc(100vh-82px)] items-center justify-center px-4 py-8 lg:px-8">
      <section className="w-full max-w-3xl rounded-[32px] border border-slate-200 bg-white px-6 py-12 text-center shadow-[0_10px_30px_rgba(15,23,42,0.05)] sm:px-12">
        <div className="mx-auto flex h-20 w-20 items-center justify-center rounded-[24px] border border-amber-200 bg-amber-50 text-amber-700">
          <svg viewBox="0 0 24 24" className="h-10 w-10" fill="none" stroke="currentColor" strokeWidth="1.8">
            <path d="M6 6l12 12M18 6 6 18" />
          </svg>
        </div>
        <p className="mt-6 text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">Processing canceled</p>
        <h1 className="mt-3 text-3xl font-semibold tracking-tight text-slate-900 sm:text-4xl">{documentData?.name || 'Document'} was not processed.</h1>
        <p className="mx-auto mt-4 max-w-xl text-base leading-7 text-slate-500">The source file is still available. Start processing again whenever you are ready.</p>
        {processingError ? <p className="mx-auto mt-4 max-w-xl rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">{processingError}</p> : null}
        <div className="mt-8 flex flex-wrap justify-center gap-3">
          <button type="button" onClick={onBackToLibrary} className="rounded-full border border-slate-200 bg-white px-5 py-3 text-sm font-semibold text-slate-700 hover:bg-slate-50">Back to library</button>
          <button type="button" onClick={onProcessAgain} className="rounded-full bg-slate-900 px-5 py-3 text-sm font-semibold text-white hover:bg-slate-800">Process again</button>
        </div>
      </section>
    </div>
  )
}

function ReconstructedTableCard({
  table,
  selected,
  selectedCellKeys,
  onToggleTable,
  onToggleCell,
  onEditCell,
  onToggleHeader,
  onEditTitle,
  onEditCaption,
  onCompare,
  onAddRow,
  onRemoveRow,
  onAddColumn,
  onRemoveColumn,
  onMergeCells,
  onSplitCell,
  onSaveChanges,
  saveStatus,
}) {
  const rows = table.rows || []
  const tableConfidenceTone = confidenceBorderTone(Number(table.confidence_score || 0))
  const columnWidths = table.columns || []
  const columnTotal = columnWidths.reduce((sum, width) => sum + Number(width || 0), 0) || columnWidths.length || 1
  const tableHeight = Math.max(1, Number(table.bbox?.[3] || 0) - Number(table.bbox?.[1] || 0))
  const displayHeight = Math.max(190, rows.length * 42)
  const selectedCells = rows.flatMap((row) => row.cells || []).filter((cell) => selectedCellKeys.has(`${table.id}:${cell.id}`))
  const canMerge = selectedCells.length > 1
    && selectedCells.every((cell) => Number(cell.rowspan || 1) === 1)
    && new Set(selectedCells.map((cell) => cell.row_index)).size === 1
    && selectedCells.map((cell) => cell.column_index).sort((left, right) => left - right).every((column, index, columns) => index === 0 || column === columns[index - 1] + 1)

  return (
    <section className={`overflow-hidden rounded-[20px] border-2 ${tableConfidenceTone}`}>
      <header className="space-y-3 border-b border-slate-200 px-4 py-3">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-[150px] flex-1">
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
              {table.table_type === 'form' ? 'Form-like table' : table.style?.layout_source === 'claude' ? 'Detected table' : 'Reconstructed table'}
              {' · Page '}{table.page_number}
            </p>
            <input
              aria-label="Table title"
              value={table.title || 'Table'}
              onChange={(event) => onEditTitle(table.id, event.target.value)}
              className="mt-1 w-full bg-transparent text-base font-semibold text-slate-900 outline-none focus:border-b focus:border-slate-400"
            />
          </div>
          <label className="flex items-center gap-2 pt-1 text-xs font-medium text-slate-600">
            <input type="checkbox" checked={selected} onChange={() => onToggleTable(table.id)} />
            Select table
          </label>
        </div>
        <input
          aria-label="Table caption"
          value={table.caption || ''}
          onChange={(event) => onEditCaption(table.id, event.target.value)}
          placeholder="Caption or section heading"
          className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs text-slate-700 outline-none focus:border-slate-400"
        />
        <div className="flex flex-wrap gap-1.5">
          <button type="button" onClick={() => onCompare(table)} className="rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-100">Compare on page</button>
          {onSaveChanges ? (
            <button
              type="button"
              onClick={() => onSaveChanges(table)}
              disabled={saveStatus === 'saving'}
              className="rounded-lg border border-emerald-600 bg-emerald-600 px-2.5 py-1.5 text-xs font-semibold text-white hover:bg-emerald-700 disabled:opacity-60"
            >
              {saveStatus === 'saving' ? 'Saving...' : saveStatus === 'saved' ? 'Saved' : 'Save table changes'}
            </button>
          ) : null}
          <button type="button" onClick={() => onAddRow(table.id)} className="rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-100">Add row</button>
          <button type="button" onClick={() => onRemoveRow(table.id)} disabled={rows.length <= 1} className="rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-xs font-semibold text-slate-700 disabled:opacity-40">Remove last row</button>
          <button type="button" onClick={() => onAddColumn(table.id)} className="rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-100">Add column</button>
          <button type="button" onClick={() => onRemoveColumn(table.id)} disabled={columnWidths.length <= 1} className="rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-xs font-semibold text-slate-700 disabled:opacity-40">Remove last column</button>
          <button type="button" onClick={() => onMergeCells(table.id)} disabled={!canMerge} className="rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-xs font-semibold text-slate-700 disabled:opacity-40">Merge selected</button>
          <button type="button" onClick={() => onSplitCell(table.id, selectedCells[0]?.id)} disabled={selectedCells.length !== 1 || (Number(selectedCells[0]?.colspan || 1) <= 1 && Number(selectedCells[0]?.rowspan || 1) <= 1)} className="rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-xs font-semibold text-slate-700 disabled:opacity-40">Split selected</button>
        </div>
      </header>

      <div className="overflow-x-auto overflow-y-hidden bg-white p-3">
        <table className="min-w-full border-collapse text-left text-xs text-slate-800" style={{ tableLayout: 'fixed' }}>
          <colgroup>
            {columnWidths.map((width, index) => <col key={`${table.id}-column-${index}`} style={{ width: `${Number(width || 0) / columnTotal * 100}%` }} />)}
          </colgroup>
          <tbody>
            {rows.map((row, rowIndex) => {
              const rowHeight = Math.max(32, Number(row.height || 0) / tableHeight * displayHeight)
              return (
                <tr key={row.id || `${table.id}-row-${rowIndex}`} style={{ height: `${rowHeight}px` }}>
                  {(row.cells || []).map((cell) => {
                    const cellKey = `${table.id}:${cell.id}`
                    const CellTag = cell.is_header ? 'th' : 'td'
                    const isSelected = selectedCellKeys.has(cellKey)
                    const borderColor = table.style?.bordered ? '#94a3b8' : '#cbd5e1'
                    return (
                      <CellTag
                        key={cell.id}
                        rowSpan={Math.max(1, Number(cell.rowspan || 1))}
                        colSpan={Math.max(1, Number(cell.colspan || 1))}
                        className={`align-top p-1.5 ${isSelected ? 'bg-cyan-100 ring-2 ring-inset ring-cyan-600' : cell.is_header ? 'bg-slate-100' : 'bg-white'}`}
                        style={{
                          border: table.style?.bordered ? `${Math.max(1, Number(table.style.border_width || 1))}px solid ${borderColor}` : '0',
                          borderBottom: table.style?.bordered ? undefined : `1px dashed ${borderColor}`,
                          textAlign: cell.alignment || 'left',
                          fontSize: `${11 * Number(cell.font_scale || 1)}px`,
                        }}
                        onClick={() => onToggleCell(table.id, cell.id)}
                      >
                        <div className="flex h-full min-w-0 items-start gap-1.5">
                          <input
                            aria-label={`Select cell ${rowIndex + 1}, ${Number(cell.column_index || 0) + 1}`}
                            type="checkbox"
                            checked={isSelected}
                            onClick={(event) => event.stopPropagation()}
                            onChange={() => onToggleCell(table.id, cell.id)}
                            className="mt-1 shrink-0"
                          />
                          <div className="min-w-0 flex-1">
                            <textarea
                              aria-label={`Cell ${rowIndex + 1}, ${Number(cell.column_index || 0) + 1}`}
                              value={cell.text || ''}
                              rows={Math.min(14, Math.max(1, String(cell.text || '').split('\n').length))}
                              ref={(element) => {
                                // Grow with the content so long paragraphs in form-like cells are fully visible.
                                if (element) {
                                  element.style.height = 'auto'
                                  element.style.height = `${element.scrollHeight}px`
                                }
                              }}
                              onClick={(event) => event.stopPropagation()}
                              onChange={(event) => onEditCell(table.id, cell.id, event.target.value)}
                              className={`w-full min-w-[52px] resize-y overflow-hidden rounded-sm bg-transparent p-0.5 text-xs leading-4 outline-none focus:bg-white focus:ring-1 focus:ring-cyan-600 ${cell.bold || cell.is_header ? 'font-semibold' : 'font-normal'}`}
                            />
                            <button
                              type="button"
                              onClick={(event) => { event.stopPropagation(); onToggleHeader(table.id, cell.id) }}
                              className="mt-1 rounded px-1 py-0.5 text-[10px] font-semibold uppercase text-slate-500 hover:bg-slate-100"
                              title="Toggle header emphasis"
                            >
                              {cell.is_header ? 'Header' : 'Body'}
                            </button>
                          </div>
                        </div>
                      </CellTag>
                    )
                  })}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <footer className="flex flex-wrap items-center justify-between gap-2 border-t border-slate-200 px-4 py-2 text-[11px] text-slate-500">
        <span>{rows.length} rows · {columnWidths.length} columns · {Math.round(Number(table.confidence_score || 0) * 100)}% layout confidence</span>
        {table.uncertain_fields?.length ? <span title={table.uncertain_fields.join(', ')}>Needs visual check: {table.uncertain_fields.slice(0, 2).join(', ')}</span> : <span>Cell geometry detected from page rules</span>}
      </footer>
    </section>
  )
}

function ViewerPage({
  apiBaseUrl,
  documentData,
  attributes,
  ocrPages,
  viewerLoading,
  viewerError,
  viewerSearch,
  setViewerSearch,
  zoom,
  setZoom,
  selectedAttributeId,
  setSelectedAttributeId,
  onBackToLibrary,
  onUpdateAttribute,
  onResetAttribute,
  onSaveAttribute,
  onSaveSelectedAttributes,
  onRefreshDocument,
}) {
  const boxRefs = useRef(new Map())
  const pageRefs = useRef(new Map())
  const [editingAttributeId, setEditingAttributeId] = useState(null)
  const [reviewAttributes, setReviewAttributes] = useState(attributes)
  const [selectedReviewKeys, setSelectedReviewKeys] = useState(() => new Set())
  const [manualLabel, setManualLabel] = useState('')
  const [manualValue, setManualValue] = useState('')
  const [manualPageNumber, setManualPageNumber] = useState(1)
  const [reviewError, setReviewError] = useState('')
  const [savingSelected, setSavingSelected] = useState(false)
  const [activePanelTab, setActivePanelTab] = useState('tables')
  const [selectedTableKeys, setSelectedTableKeys] = useState(() => new Set())
  const [selectedCellKeys, setSelectedCellKeys] = useState(() => new Set())
  const [selectedTextFieldKeys, setSelectedTextFieldKeys] = useState(() => new Set())
  const initialTables = useMemo(() => collectDocumentTables(ocrPages, attributes), [attributes, ocrPages])
  const [detectedTables, setDetectedTables] = useState(initialTables)
  const [tableSaveStatus, setTableSaveStatus] = useState({})

  // Tables that were already saved as attributes can be edited here and saved back in place.
  const savedTableAttributes = useMemo(() => {
    const byTableId = new Map()
    for (const attribute of attributes || []) {
      if (attribute.id && attribute.table_data?.id) {
        byTableId.set(attribute.table_data.id, attribute)
      }
    }
    return byTableId
  }, [attributes])

  const saveTableChanges = async (table) => {
    const attribute = savedTableAttributes.get(table.id)
    if (!attribute) {
      return
    }
    setReviewError('')
    setTableSaveStatus((current) => ({ ...current, [table.id]: 'saving' }))
    try {
      const response = await fetch(`${apiBaseUrl}/api/v1/attributes/correct`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          document_id: attribute.document_id ?? documentData?.id,
          corrections: [
            {
              attribute_id: attribute.id,
              original_prediction: attribute.extracted_value,
              corrected_value: JSON.stringify({ kind: 'table', version: 1, table: { ...table, kind: 'table', version: 1 } }),
              corrected_label: table.title || attribute.entity_type_label,
              validation_status: attribute.validation_status,
            },
          ],
        }),
      })
      if (!response.ok) {
        const result = await response.json().catch(() => ({}))
        throw new Error(result.detail || 'Failed to save table changes.')
      }
      setTableSaveStatus((current) => ({ ...current, [table.id]: 'saved' }))
      await onRefreshDocument?.()
    } catch (error) {
      setTableSaveStatus((current) => ({ ...current, [table.id]: undefined }))
      setReviewError(error.message || 'Failed to save table changes.')
    }
  }

  useEffect(() => {
    setDetectedTables(initialTables)
    setSelectedTableKeys(new Set())
    setSelectedCellKeys(new Set())
  }, [initialTables])

  const textFieldEntries = useMemo(() => (reviewAttributes || []).filter((attribute) => attribute.kind !== 'table'), [reviewAttributes])

  const { boxToAttribute, attributeToBoxes, boxesByPage } = useMemo(
    () => buildAttributeBoxIndex(reviewAttributes, ocrPages),
    [reviewAttributes, ocrPages],
  )

  const isDraftReview = reviewAttributes.some((attribute) => attribute.saved === false)
  const selectedAttributeKeys = isDraftReview ? selectedReviewKeys : new Set(selectedAttributeId == null ? [] : [selectedAttributeId])
  const viewerQuery = viewerSearch.trim().toLowerCase()

  useEffect(() => {
    setReviewAttributes(attributes)
    setSelectedReviewKeys(new Set())
    setEditingAttributeId(null)
    setReviewError('')
    setManualLabel('')
    setManualValue('')
    setManualPageNumber(1)
  }, [attributes])

  useEffect(() => {
    if (isDraftReview || !selectedAttributeId) {
      return
    }
    const firstBoxKey = (attributeToBoxes.get(selectedAttributeId) || [])[0]
    if (!firstBoxKey) {
      return
    }
    const element = boxRefs.current.get(firstBoxKey)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }
  }, [attributeToBoxes, isDraftReview, selectedAttributeId])

  const handleBoxClick = (boxKey) => {
    const attributeId = boxToAttribute.get(boxKey)
    if (attributeId) {
      if (isDraftReview) {
        setSelectedReviewKeys((current) => {
          const next = new Set(current)
          if (next.has(attributeId)) {
            next.delete(attributeId)
          } else {
            next.add(attributeId)
          }
          return next
        })
      } else {
        setSelectedAttributeId(attributeId)
      }
    }
  }

  const getAttributeKey = (attribute) => attribute.id ?? attribute.temp_id

  const updateReviewAttribute = (attributeKey, field, value) => {
    setReviewAttributes((current) =>
      current.map((attribute) => (getAttributeKey(attribute) === attributeKey ? { ...attribute, [field]: value } : attribute)),
    )
  }

  const updateTable = (tableId, transform) => {
    setDetectedTables((current) => current.map((table) => (table.id === tableId ? transform(table) : table)))
  }

  const toggleTableSelection = (tableId) => {
    setSelectedTableKeys((current) => {
      const next = new Set(current)
      if (next.has(tableId)) next.delete(tableId)
      else next.add(tableId)
      return next
    })
  }

  const toggleCellSelection = (tableId, cellId) => {
    const key = `${tableId}:${cellId}`
    setSelectedCellKeys((current) => {
      const next = new Set(current)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  const editTableCell = (tableId, cellId, text) => {
    updateTable(tableId, (table) => ({
      ...table,
      rows: table.rows.map((row) => ({
        ...row,
        cells: row.cells.map((cell) => cell.id === cellId ? { ...cell, text, corrected: text !== cell.original_text } : cell),
      })),
    }))
  }

  const toggleTableHeader = (tableId, cellId) => {
    updateTable(tableId, (table) => ({
      ...table,
      rows: table.rows.map((row) => ({
        ...row,
        cells: row.cells.map((cell) => cell.id === cellId ? { ...cell, is_header: !cell.is_header, bold: !cell.is_header } : cell),
      })),
    }))
  }

  const addTableRow = (tableId) => {
    updateTable(tableId, (table) => {
      const rowIndex = table.rows.length
      const columnCount = Math.max(1, table.columns.length)
      const previousHeight = Number(table.rows.at(-1)?.height || 35)
      const y0 = Number(table.bbox?.[3] || 0)
      const x0 = Number(table.bbox?.[0] || 0)
      const tableWidth = Math.max(1, Number(table.bbox?.[2] || 1000) - x0)
      const columnTotal = table.columns.reduce((sum, width) => sum + Number(width || 0), 0) || columnCount
      let currentX = x0
      const cells = Array.from({ length: columnCount }, (_, columnIndex) => ({
        ...(() => {
          const share = Number(table.columns[columnIndex] || 1) / columnTotal
          const cellX0 = currentX
          currentX = columnIndex === columnCount - 1 ? x0 + tableWidth : currentX + tableWidth * share
          return {
            id: `edit-r${rowIndex}-c${columnIndex}-${Date.now()}`,
            row_index: rowIndex,
            column_index: columnIndex,
            text: '',
            original_text: '',
            bbox: [cellX0, y0, currentX, Math.min(1000, y0 + previousHeight)],
            rowspan: 1,
            colspan: 1,
            is_header: false,
            bold: false,
            alignment: 'left',
            confidence_score: 0,
            token_indices: [],
            uncertain: true,
          }
        })(),
      }))
      return { ...table, rows: [...table.rows, { id: `edit-row-${Date.now()}`, index: rowIndex, height: previousHeight, cells }], uncertain_fields: [...new Set([...(table.uncertain_fields || []), 'edited_row_geometry'])] }
    })
  }

  const removeTableRow = (tableId) => {
    updateTable(tableId, (table) => ({ ...table, rows: table.rows.slice(0, -1).map((row, index) => ({ ...row, index })) }))
  }

  const addTableColumn = (tableId) => {
    updateTable(tableId, (table) => {
      const columnIndex = table.columns.length
      const columnWidth = table.columns.length ? table.columns[table.columns.length - 1] : 100
      return {
        ...table,
        columns: [...table.columns, columnWidth],
        rows: table.rows.map((row, rowIndex) => ({
          ...row,
          cells: [...row.cells, {
            id: `edit-r${rowIndex}-c${columnIndex}-${Date.now()}`,
            row_index: rowIndex,
            column_index: columnIndex,
            text: '',
            original_text: '',
            bbox: [Number(table.bbox?.[2] || 0), Number(table.bbox?.[1] || 0), Math.min(1000, Number(table.bbox?.[2] || 0) + columnWidth), Number(table.bbox?.[3] || 0)],
            rowspan: 1,
            colspan: 1,
            is_header: rowIndex === 0,
            bold: rowIndex === 0,
            alignment: 'left',
            confidence_score: 0,
            token_indices: [],
            uncertain: true,
          }],
        })),
        uncertain_fields: [...new Set([...(table.uncertain_fields || []), 'edited_column_geometry'])],
      }
    })
  }

  const removeTableColumn = (tableId) => {
    updateTable(tableId, (table) => ({
      ...table,
      columns: table.columns.slice(0, -1),
      rows: table.rows.map((row) => ({ ...row, cells: row.cells.filter((cell) => Number(cell.column_index || 0) < table.columns.length - 1) })),
    }))
  }

  const mergeSelectedTableCells = (tableId) => {
    const table = detectedTables.find((item) => item.id === tableId)
    if (!table) return
    const selected = table.rows.flatMap((row) => row.cells || []).filter((cell) => selectedCellKeys.has(`${tableId}:${cell.id}`))
    if (selected.length < 2) return
    const rowIndex = selected[0].row_index
    const ordered = selected.sort((left, right) => left.column_index - right.column_index)
    const row = table.rows.find((item) => item.index === rowIndex || item.cells.some((cell) => cell.id === ordered[0].id))
    if (!row || ordered.some((cell, index) => Number(cell.rowspan || 1) !== 1 || (index > 0 && cell.column_index !== ordered[index - 1].column_index + 1))) return
    const mergedCell = {
      ...ordered[0],
      id: `merged-${Date.now()}`,
      text: ordered.map((cell) => cell.text || '').join('\n'),
      original_text: ordered.map((cell) => cell.original_text || '').join('\n'),
      colspan: ordered.reduce((sum, cell) => sum + Number(cell.colspan || 1), 0),
      bbox: [
        Math.min(...ordered.map((cell) => cell.bbox[0])),
        Math.min(...ordered.map((cell) => cell.bbox[1])),
        Math.max(...ordered.map((cell) => cell.bbox[2])),
        Math.max(...ordered.map((cell) => cell.bbox[3])),
      ],
      merged_parts: ordered,
      uncertain: true,
    }
    updateTable(tableId, (currentTable) => ({
      ...currentTable,
      rows: currentTable.rows.map((currentRow) => currentRow === row ? {
        ...currentRow,
        cells: [...currentRow.cells.filter((cell) => !ordered.some((selectedCell) => selectedCell.id === cell.id)), mergedCell].sort((left, right) => left.column_index - right.column_index),
      } : currentRow),
      uncertain_fields: [...new Set([...(currentTable.uncertain_fields || []), 'manual_cell_merge'])],
    }))
    setSelectedCellKeys(new Set([`${tableId}:${mergedCell.id}`]))
  }

  const splitTableCell = (tableId, cellId) => {
    updateTable(tableId, (table) => {
      const sourceRowIndex = table.rows.findIndex((row) => row.cells.some((item) => item.id === cellId))
      const sourceCell = table.rows[sourceRowIndex]?.cells.find((item) => item.id === cellId)
      if (!sourceCell) return table
      const rowspan = Math.max(1, Number(sourceCell.rowspan || 1))
      const colspan = Math.max(1, Number(sourceCell.colspan || 1))
      if (rowspan <= 1 && colspan <= 1) return table
      const mergedParts = Array.isArray(sourceCell.merged_parts) ? sourceCell.merged_parts : []
      const originalMergedText = mergedParts.map((part) => part.text || '').join('\n')
      const hasCorrection = Boolean(sourceCell.corrected) || (mergedParts.length > 0 && sourceCell.text !== originalMergedText)
      const cellWidth = Math.max(1, Number(sourceCell.bbox[2]) - Number(sourceCell.bbox[0])) / colspan
      const cellHeight = Math.max(1, Number(sourceCell.bbox[3]) - Number(sourceCell.bbox[1])) / rowspan
      const startRow = Number(sourceCell.row_index ?? sourceRowIndex)
      const startColumn = Number(sourceCell.column_index || 0)
      const replacementsByRow = new Map()
      for (let rowOffset = 0; rowOffset < rowspan; rowOffset += 1) {
        const replacements = []
        for (let columnOffset = 0; columnOffset < colspan; columnOffset += 1) {
          const part = rowOffset === 0 && !hasCorrection ? mergedParts[columnOffset] : null
          replacements.push({
            ...(part || sourceCell),
            id: part?.id || `split-${Date.now()}-${rowOffset}-${columnOffset}`,
            row_index: startRow + rowOffset,
            column_index: startColumn + columnOffset,
            text: part ? part.text : (rowOffset === 0 && columnOffset === 0 ? sourceCell.text : ''),
            original_text: part ? part.original_text : (rowOffset === 0 && columnOffset === 0 ? sourceCell.original_text : ''),
            corrected: hasCorrection && rowOffset === 0 && columnOffset === 0,
            rowspan: 1,
            colspan: 1,
            merged_parts: undefined,
            bbox: [
              Number(sourceCell.bbox[0]) + cellWidth * columnOffset,
              Number(sourceCell.bbox[1]) + cellHeight * rowOffset,
              columnOffset === colspan - 1 ? sourceCell.bbox[2] : Number(sourceCell.bbox[0]) + cellWidth * (columnOffset + 1),
              rowOffset === rowspan - 1 ? sourceCell.bbox[3] : Number(sourceCell.bbox[1]) + cellHeight * (rowOffset + 1),
            ],
          })
        }
        replacementsByRow.set(startRow + rowOffset, replacements)
      }
      return {
        ...table,
        rows: table.rows.map((row, rowIndex) => {
          const logicalRow = Number(row.index ?? rowIndex)
          const rowReplacements = replacementsByRow.get(logicalRow)
          if (!rowReplacements) return row
          const remainingCells = row.cells.filter((cell) => cell.id !== cellId && !(
            Number(cell.column_index || 0) >= startColumn
            && Number(cell.column_index || 0) < startColumn + colspan
          ))
          return { ...row, cells: [...remainingCells, ...rowReplacements].sort((left, right) => left.column_index - right.column_index) }
        }),
        uncertain_fields: [...new Set([...(table.uncertain_fields || []), 'manual_cell_split'])],
      }
    })
  }

  const compareTableOnPage = (table) => {
    setSelectedTableKeys((current) => new Set([...current, table.id]))
    // Scroll to the table's own region; without a box, fall back to the page that holds it.
    const element = boxRefs.current.get(`table:${table.id}`) || pageRefs.current.get(Number(table.page_number))
    element?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }

  const toggleReviewSelection = (attributeKey) => {
    setSelectedReviewKeys((current) => {
      const next = new Set(current)
      if (next.has(attributeKey)) {
        next.delete(attributeKey)
      } else {
        next.add(attributeKey)
      }
      return next
    })
  }

  const allReviewAttributesSelected = reviewAttributes.length > 0 && reviewAttributes.every((attribute) => selectedReviewKeys.has(getAttributeKey(attribute)))

  const toggleSelectAllAttributes = () => {
    if (allReviewAttributesSelected) {
      setSelectedReviewKeys(new Set())
      return
    }
    setSelectedReviewKeys(new Set(reviewAttributes.map((attribute) => getAttributeKey(attribute))))
  }

  const addManualAttribute = () => {
    const label = manualLabel.trim()
    const value = manualValue.trim()
    if (!label || !value) {
      setReviewError('Manual attribute name and value are required.')
      return
    }

    const tempId = `manual-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
    setReviewAttributes((current) => [
      ...current,
      {
        temp_id: tempId,
        entity_type_label: label,
        extracted_value: value,
        draft_label: label,
        draft_value: value,
        draft_validation_status: 'PENDING',
        original_label: label,
        original_value: value,
        original_validation_status: 'PENDING',
        confidence_score: 0,
        bounding_boxes: [],
        page_number: manualPageNumber,
        source: 'manual',
        saved: false,
      },
    ])
    setSelectedReviewKeys((current) => new Set([...current, tempId]))
    setManualLabel('')
    setManualValue('')
    setReviewError('')
  }

  const selectedTableCount = detectedTables.filter((table) => (
    selectedTableKeys.has(table.id)
    || (table.rows || []).some((row) => (row.cells || []).some((cell) => selectedCellKeys.has(`${table.id}:${cell.id}`)))
  )).length
  const selectedItemCount = reviewAttributes.filter((attribute) => selectedReviewKeys.has(getAttributeKey(attribute))).length
    + selectedTableCount
    + textFieldEntries.filter((attribute) => selectedTextFieldKeys.has(getAttributeKey(attribute)) && !selectedReviewKeys.has(getAttributeKey(attribute))).length

  const saveSelectedReviewAttributes = async () => {
    const selectedAttributes = reviewAttributes.filter((attribute) => selectedReviewKeys.has(getAttributeKey(attribute)))
    const selectedTableItems = detectedTables.flatMap((table) => {
      const tableCells = (table.rows || []).flatMap((row) => row.cells || [])
      const hasSelectedCells = tableCells.some((cell) => selectedCellKeys.has(`${table.id}:${cell.id}`))
      if (!selectedTableKeys.has(table.id) && !hasSelectedCells) return []
      const tableText = table.rows.map((row) => (row.cells || []).map((cell) => cell.text || '').join('\t')).join('\n')
      return [{
        temp_id: `table-${table.id}`,
        kind: 'table',
        metadata: { ...table, kind: 'table', version: 1 },
        source: 'model',
        entity_type_label: table.title || 'Table',
        extracted_value: tableText || table.title || 'Table',
        page_number: table.page_number || 1,
        page_id: table.page_id,
        bounding_boxes: table.bounding_boxes || [table.bbox].filter(Boolean),
        page_spans: [{ page_number: table.page_number || 1, bounding_boxes: table.bounding_boxes || [table.bbox].filter(Boolean) }],
        confidence_score: table.confidence_score || 0,
        draft_label: table.title || 'Table',
        draft_value: tableText || table.title || 'Table',
      }]
    })
    const selectedFieldItems = textFieldEntries.filter((attribute) => selectedTextFieldKeys.has(getAttributeKey(attribute)) && !selectedReviewKeys.has(getAttributeKey(attribute))).map((attribute) => ({
      ...attribute,
      temp_id: attribute.temp_id || `field-${attribute.id || Date.now()}`,
      draft_label: attribute.draft_label || attribute.entity_type_label,
      draft_value: attribute.draft_value || attribute.extracted_value,
    }))

    const combinedSelection = [...selectedAttributes, ...selectedTableItems, ...selectedFieldItems]
    if (!combinedSelection.length) {
      setReviewError('Select at least one table or field to save.')
      return
    }

    setSavingSelected(true)
    setReviewError('')
    try {
      await onSaveSelectedAttributes?.(combinedSelection)
      setSelectedReviewKeys(new Set())
      setSelectedTableKeys(new Set())
      setSelectedCellKeys(new Set())
      setSelectedTextFieldKeys(new Set())
    } catch (error) {
      setReviewError(error.message)
    } finally {
      setSavingSelected(false)
    }
  }

  const handleEditAttribute = (attributeId) => {
    setSelectedAttributeId(attributeId)
    setEditingAttributeId(attributeId)
  }

  const handleSaveAttribute = async (attributeId) => {
    setReviewError('')
    try {
      await onSaveAttribute(attributeId)
      setEditingAttributeId((current) => (current === attributeId ? null : current))
    } catch (error) {
      setReviewError(error.message || 'Failed to save attribute.')
    }
  }

  const filteredAttributes = useMemo(() => {
    if (!viewerQuery) {
      return reviewAttributes
    }
    return reviewAttributes.filter((attribute) => attribute.kind !== 'table').filter((attribute) => {
      const haystack = `${attribute.draft_label} ${attribute.draft_value} ${attribute.draft_validation_status}`.toLowerCase()
      return haystack.includes(viewerQuery)
    })
  }, [reviewAttributes, viewerQuery])

  if (viewerLoading) {
    return (
      <div className="grid gap-6 px-4 py-8 lg:grid-cols-[minmax(0,1.5fr)_minmax(340px,0.9fr)] lg:px-8">
        <div className="space-y-4 rounded-[28px] border border-slate-200 bg-white p-5 shadow-[0_10px_30px_rgba(15,23,42,0.05)]">
          <LoadingSkeleton className="h-12 w-2/5" />
          <LoadingSkeleton className="h-[42rem] w-full" />
        </div>
        <div className="space-y-4 rounded-[28px] border border-slate-200 bg-white p-5 shadow-[0_10px_30px_rgba(15,23,42,0.05)]">
          <LoadingSkeleton className="h-12 w-3/5" />
          {Array.from({ length: 4 }).map((_, index) => (
            <LoadingSkeleton key={index} className="h-40 w-full" />
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="grid gap-6 px-4 pb-8 pt-6 lg:h-[calc(100dvh-6rem)] lg:min-h-0 lg:grid-cols-[minmax(0,1.5fr)_minmax(340px,0.9fr)] lg:overflow-hidden lg:px-8">
      <section className="overflow-hidden rounded-[30px] border border-slate-200 bg-white shadow-[0_10px_30px_rgba(15,23,42,0.05)] lg:flex lg:min-h-0 lg:flex-col">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 px-5 py-4 sm:px-6">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">Document Viewer</p>
            <h1 className="mt-1 text-xl font-semibold text-slate-900">{documentData?.name || 'Document'}</h1>
            <p className="text-sm text-slate-500">
              {documentData?.page_count || ocrPages.length || 0} page{(documentData?.page_count || ocrPages.length || 0) === 1 ? '' : 's'} · {documentData?.status || 'Unknown'}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <div className="flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-600">
              <span>Zoom</span>
              <button type="button" onClick={() => setZoom(Math.max(80, zoom - 10))} className="rounded-full bg-white px-2 py-1 text-slate-500 shadow-sm">−</button>
              <span className="min-w-[3rem] text-center font-semibold text-slate-900">{zoom}%</span>
              <button type="button" onClick={() => setZoom(Math.min(150, zoom + 10))} className="rounded-full bg-white px-2 py-1 text-slate-500 shadow-sm">+</button>
              <button type="button" onClick={() => setZoom(100)} className="rounded-full bg-white px-3 py-1 text-slate-500 shadow-sm">Fit</button>
            </div>
          </div>
        </div>

        <div className="bg-slate-50/60 p-4 sm:p-6 lg:min-h-0 lg:flex-1 lg:overflow-y-auto">
          <div className="space-y-5">
            {ocrPages.length ? (
              ocrPages.map((page) => (
                <article
                  key={page.page_number}
                  ref={(element) => {
                    if (element) {
                      pageRefs.current.set(Number(page.page_number), element)
                    } else {
                      pageRefs.current.delete(Number(page.page_number))
                    }
                  }}
                  className="overflow-hidden rounded-[28px] border border-slate-200 bg-white shadow-sm"
                >
                  <div className="flex items-center justify-between border-b border-slate-200 px-5 py-4">
                    <div>
                      <p className="text-sm font-semibold text-slate-900">Page {page.page_number}</p>
                      <p className="text-xs text-slate-500">Original source page; attribute regions appear when Claude located them</p>
                    </div>
                  </div>

                  <div className="p-5 sm:p-6">
                    <div className="overflow-x-auto overflow-y-hidden rounded-2xl border border-slate-200 bg-slate-100 p-3">
                      <div className="relative mx-auto" style={{ width: `${zoom}%`, minWidth: '680px' }}>
                        {page.image_url ? (
                          <AuthenticatedPageImage
                            src={resolveApiAssetUrl(page.image_url, apiBaseUrl)}
                            alt={`Source page ${page.page_number}`}
                            className="block w-full rounded-xl border border-slate-200 bg-white"
                          />
                        ) : (
                          <div className="flex min-h-[480px] w-full items-center justify-center rounded-xl border border-dashed border-slate-300 bg-white text-sm text-slate-500">
                            Source page image is unavailable.
                          </div>
                        )}

                        {page.image_url ? (
                          <div className="absolute inset-0 overflow-hidden rounded-xl">
                            {(detectedTables || [])
                              .filter((table) => Number(table.page_number) === Number(page.page_number) && isValidBox(table.bbox))
                              .map((table) => {
                                const [x0, y0, x1, y1] = table.bbox
                                const isSelectedTable = selectedTableKeys.has(table.id)
                                return (
                                  <div
                                    key={`table:${table.id}`}
                                    ref={(element) => {
                                      if (element) {
                                        boxRefs.current.set(`table:${table.id}`, element)
                                      } else {
                                        boxRefs.current.delete(`table:${table.id}`)
                                      }
                                    }}
                                    className={`pointer-events-none absolute rounded ${isSelectedTable ? 'token-highlight-active border-2 border-blue-500' : ''}`}
                                    style={{
                                      left: `${(x0 / 1000) * 100}%`,
                                      top: `${(y0 / 1000) * 100}%`,
                                      width: `${Math.max(0.5, ((x1 - x0) / 1000) * 100)}%`,
                                      height: `${Math.max(0.7, ((y1 - y0) / 1000) * 100)}%`,
                                    }}
                                  />
                                )
                              })}
                            {(detectedTables || [])
                              .filter((table) => Number(table.page_number) === Number(page.page_number))
                              .flatMap((table) => (table.rows || []).flatMap((row) => row.cells || []).map((cell) => ({ table, cell })))
                              .filter(({ table, cell }) => selectedCellKeys.has(`${table.id}:${cell.id}`) && isValidBox(cell.bbox))
                              .map(({ table, cell }) => {
                                const [x0, y0, x1, y1] = cell.bbox
                                return (
                                  <div
                                    key={`cell:${table.id}:${cell.id}`}
                                    className="token-highlight-active pointer-events-none absolute rounded border-2 border-blue-500"
                                    style={{
                                      left: `${(x0 / 1000) * 100}%`,
                                      top: `${(y0 / 1000) * 100}%`,
                                      width: `${Math.max(0.5, ((x1 - x0) / 1000) * 100)}%`,
                                      height: `${Math.max(0.7, ((y1 - y0) / 1000) * 100)}%`,
                                    }}
                                  />
                                )
                              })}
                            {(boxesByPage.get(Number(page.page_number)) || []).map((entry) => {
                              const [x0, y0, x1, y1] = entry.box
                              const isSelectedAttribute = selectedAttributeKeys.has(entry.attributeKey)
                              const isSearchHit = viewerQuery && `${entry.label} ${entry.value}`.toLowerCase().includes(viewerQuery)
                              const boxClasses = [
                                'absolute rounded border-2 transition-all cursor-pointer',
                                confidenceBorderTone(entry.score),
                                'hover:bg-blue-200/45',
                                isSearchHit ? 'token-highlight' : '',
                                isSelectedAttribute ? 'token-highlight-active ring-2 ring-blue-400' : '',
                              ]
                                .filter(Boolean)
                                .join(' ')

                              return (
                                <button
                                  type="button"
                                  key={entry.key}
                                  ref={(element) => {
                                    if (element) {
                                      boxRefs.current.set(entry.key, element)
                                    } else {
                                      boxRefs.current.delete(entry.key)
                                    }
                                  }}
                                  onClick={() => handleBoxClick(entry.key)}
                                  className={boxClasses}
                                  style={{
                                    left: `${(x0 / 1000) * 100}%`,
                                    top: `${(y0 / 1000) * 100}%`,
                                    width: `${Math.max(0.5, ((x1 - x0) / 1000) * 100)}%`,
                                    height: `${Math.max(0.7, ((y1 - y0) / 1000) * 100)}%`,
                                  }}
                                  title={`${entry.label}: ${entry.value}`}
                                  aria-label={`Select attribute ${entry.label}`}
                                />
                              )
                            })}
                          </div>
                        ) : null}
                      </div>

                      <p className="mt-3 text-xs text-slate-500">
                        Click a highlighted region (when shown) to select its attribute. Regions are approximate.
                      </p>
                      {page.text ? (
                        <details className="mt-3 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600">
                          <summary className="cursor-pointer font-semibold text-slate-700">Page transcription</summary>
                          <pre className="mt-2 max-h-64 select-text overflow-auto whitespace-pre-wrap font-sans">{page.text}</pre>
                        </details>
                      ) : null}
                    </div>
                  </div>
                </article>
              ))
            ) : (
              <div className="rounded-[28px] border border-dashed border-slate-200 bg-white px-6 py-16 text-center text-sm text-slate-500">
                No page text is available for this document yet.
              </div>
            )}
          </div>
        </div>
      </section>

      <aside className="overflow-hidden rounded-[30px] border border-slate-200 bg-white shadow-[0_10px_30px_rgba(15,23,42,0.05)] lg:flex lg:min-h-0 lg:flex-col">
        <div className="border-b border-slate-200 px-5 py-4 sm:px-6 lg:shrink-0">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">Attributes</p>
          <h2 className="mt-1 text-xl font-semibold text-slate-900">Extracted fields</h2>
          <p className="mt-1 text-sm text-slate-500">Edit labels and values, then save changes back to the database.</p>
          {isDraftReview ? (
            <p className="mt-2 rounded-2xl border border-blue-200 bg-blue-50 px-3 py-2 text-sm text-blue-700">
              Review mode: select the extracted attributes you want to save, or add a manual attribute below.
            </p>
          ) : null}
        </div>

        <div className="space-y-4 overflow-y-visible p-4 sm:p-6 lg:min-h-0 lg:flex-1 lg:overflow-y-auto">
          <section className="rounded-[20px] border border-slate-200 bg-slate-50/80 p-4">
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Confidence legend</p>
            <div className="mt-3 grid gap-2 text-xs text-slate-700">
              <div className="flex items-center gap-2">
                <span className="inline-block h-3 w-3 rounded-full border border-emerald-300 bg-emerald-100" />
                <span>High confidence: 90% and above</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="inline-block h-3 w-3 rounded-full border border-amber-300 bg-amber-100" />
                <span>Medium confidence: 75% - 89.9%</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="inline-block h-3 w-3 rounded-full border border-rose-300 bg-rose-100" />
                <span>Low confidence: below 75%</span>
              </div>
            </div>
          </section>

          {isDraftReview ? (
            <label className="flex cursor-pointer items-center gap-3 rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm font-semibold text-slate-700">
              <input
                type="checkbox"
                checked={allReviewAttributesSelected}
                onChange={toggleSelectAllAttributes}
                className="h-4 w-4 accent-slate-900"
              />
              Select all attributes
            </label>
          ) : null}

          <div className="flex flex-wrap items-center gap-3">
            <label className="flex min-w-[220px] flex-1 items-center gap-2 rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3">
              <svg viewBox="0 0 24 24" className="h-4 w-4 shrink-0 text-slate-400" fill="none" stroke="currentColor" strokeWidth="1.8">
                <path d="M10.5 6.5a4 4 0 1 1 0 8 4 4 0 0 1 0-8Z" />
                <path d="M21 21l-4.3-4.3" />
              </svg>
              <input
                value={viewerSearch}
                onChange={(event) => setViewerSearch(event.target.value)}
                placeholder="Search within this document"
                className="w-full bg-transparent text-sm text-slate-900 outline-none placeholder:text-slate-400"
              />
            </label>
            <button
              type="button"
              onClick={() => setViewerSearch('')}
              className="rounded-full border border-slate-200 bg-white px-4 py-3 text-sm text-slate-600 transition hover:bg-slate-50"
            >
              Clear search
            </button>
          </div>

          {viewerError ? (
            <div className="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">{viewerError}</div>
          ) : null}
          {!isDraftReview && reviewError ? (
            <div className="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">{reviewError}</div>
          ) : null}

          <div className="rounded-[22px] border border-slate-200 bg-slate-50 p-1">
            <div className="flex gap-1">
              <button type="button" onClick={() => setActivePanelTab('tables')} className={`flex-1 rounded-xl px-3 py-2 text-sm font-semibold transition ${activePanelTab === 'tables' ? 'bg-slate-900 text-white' : 'text-slate-600 hover:bg-white'}`}>
                Tables
              </button>
              <button type="button" onClick={() => setActivePanelTab('text-fields')} className={`flex-1 rounded-xl px-3 py-2 text-sm font-semibold transition ${activePanelTab === 'text-fields' ? 'bg-slate-900 text-white' : 'text-slate-600 hover:bg-white'}`}>
                Text Fields
              </button>
            </div>
          </div>

          {activePanelTab === 'tables' ? (
            <section className="space-y-4">
              {detectedTables.length ? (
                detectedTables.map((table) => (
                  <ReconstructedTableCard
                    key={table.id}
                    table={table}
                    selected={selectedTableKeys.has(table.id)}
                    selectedCellKeys={selectedCellKeys}
                    onSaveChanges={!isDraftReview && savedTableAttributes.has(table.id) ? saveTableChanges : undefined}
                    saveStatus={tableSaveStatus[table.id]}
                    onToggleTable={toggleTableSelection}
                    onToggleCell={toggleCellSelection}
                    onEditCell={editTableCell}
                    onToggleHeader={toggleTableHeader}
                    onEditTitle={(tableId, title) => updateTable(tableId, (current) => ({ ...current, title }))}
                    onEditCaption={(tableId, caption) => updateTable(tableId, (current) => ({ ...current, caption }))}
                    onCompare={compareTableOnPage}
                    onAddRow={addTableRow}
                    onRemoveRow={removeTableRow}
                    onAddColumn={addTableColumn}
                    onRemoveColumn={removeTableColumn}
                    onMergeCells={mergeSelectedTableCells}
                    onSplitCell={splitTableCell}
                  />
                ))
              ) : (
                <div className="rounded-[24px] border border-dashed border-slate-200 bg-slate-50 px-4 py-12 text-center text-sm text-slate-500">
                  No tables were detected on these pages.
                </div>
              )}
            </section>
          ) : (
            <section className="space-y-4">
              {textFieldEntries.length ? (
                textFieldEntries.map((attribute) => {
                  const fieldKey = getAttributeKey(attribute)
                  const selected = selectedTextFieldKeys.has(fieldKey)
                  return (
                    <section key={fieldKey} className={`rounded-[24px] border-2 p-4 ${confidenceBorderTone(Number(attribute.confidence_score || 0))} ${selected ? 'bg-blue-50 ring-2 ring-blue-200' : ''}`}>
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Text field</p>
                          <p className="mt-1 text-base font-semibold text-slate-900">{attribute.draft_label}</p>
                        </div>
                        <label className="flex items-center gap-2 text-xs text-slate-600">
                          <input
                            type="checkbox"
                            checked={selected}
                            onChange={() => {
                              setSelectedTextFieldKeys((current) => {
                                const next = new Set(current)
                                if (next.has(fieldKey)) next.delete(fieldKey)
                                else next.add(fieldKey)
                                return next
                              })
                            }}
                          />
                          Select
                        </label>
                      </div>

                      <div className="mt-3 space-y-3">
                        <label className="block space-y-2">
                          <span className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Label</span>
                          <input
                            value={attribute.draft_label}
                            onChange={(event) => updateReviewAttribute(fieldKey, 'draft_label', event.target.value)}
                            className="w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none"
                          />
                        </label>
                        <label className="block space-y-2">
                          <span className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Value</span>
                          <textarea
                            value={attribute.draft_value}
                            rows={3}
                            onChange={(event) => updateReviewAttribute(fieldKey, 'draft_value', event.target.value)}
                            className="w-full resize-y rounded-2xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none"
                          />
                        </label>
                      </div>
                    </section>
                  )
                })
              ) : (
                <div className="rounded-[24px] border border-dashed border-slate-200 bg-slate-50 px-4 py-12 text-center text-sm text-slate-500">
                  No text fields were extracted for this document yet.
                </div>
              )}
            </section>
          )}

          {activePanelTab === 'text-fields' ? (
            <>
              {isDraftReview ? (
            <section className="rounded-[24px] border border-slate-200 bg-slate-50 p-4">
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Add Attribute</p>
              <div className="mt-3 grid gap-3">
                <input
                  value={manualLabel}
                  onChange={(event) => setManualLabel(event.target.value)}
                  placeholder="Attribute name"
                  className="rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 outline-none"
                />
                <textarea
                  value={manualValue}
                  onChange={(event) => setManualValue(event.target.value)}
                  placeholder="Attribute value"
                  rows={3}
                  className="rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 outline-none"
                />
                <div className="flex items-center gap-3">
                  <label className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Page</label>
                  <input
                    type="number"
                    min="1"
                    value={manualPageNumber}
                    onChange={(event) => setManualPageNumber(Number(event.target.value) || 1)}
                    className="w-24 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none"
                  />
                  <button type="button" onClick={addManualAttribute} className="rounded-full bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800">
                    + Add Attribute
                  </button>
                </div>
              </div>
            </section>
              ) : null}

              {filteredAttributes.length ? (
            filteredAttributes.map((attribute) => {
              const attributeKey = getAttributeKey(attribute)
              const isSelected = isDraftReview ? selectedReviewKeys.has(attributeKey) : selectedAttributeId === attribute.id
              const isEditing = !isDraftReview && editingAttributeId === attribute.id
              const borderTone = confidenceBorderTone(Number(attribute.confidence_score || 0))

              return (
                <section
                  key={attributeKey}
                  onMouseEnter={() => {
                    if (!isDraftReview && attribute.id) {
                      setSelectedAttributeId(attribute.id)
                    }
                  }}
                  className={`rounded-[24px] border-2 p-4 transition ${borderTone} ${isSelected ? 'ring-2 ring-blue-200 shadow-sm' : 'hover:shadow-sm'}`}
                >
                  <div>
                    <div className="flex items-center justify-between gap-3">
                      <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Attribute</p>
                      {isDraftReview ? (
                        <label className="flex items-center gap-2 text-xs text-slate-500">
                          <input
                            type="checkbox"
                            checked={isSelected}
                            onChange={() => toggleReviewSelection(attributeKey)}
                          />
                          Save
                        </label>
                      ) : null}
                    </div>
                    <p className="mt-1 text-base font-semibold text-slate-900">{attribute.draft_label}</p>
                    {attribute.source ? <p className="mt-1 text-xs uppercase tracking-[0.18em] text-slate-400">Source: {attribute.source}</p> : null}
                  </div>

                  <div className="mt-4">
                    <label className="block space-y-2">
                      <span className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Value</span>
                      <textarea
                        value={attribute.draft_value}
                        rows={attribute.draft_value?.includes('\n') ? 5 : 2}
                        onChange={(event) => {
                          if (isDraftReview) {
                            updateReviewAttribute(attributeKey, 'draft_value', event.target.value)
                          } else {
                            onUpdateAttribute(attribute.id, 'draft_value', event.target.value)
                          }
                        }}
                        readOnly={!isEditing && !isDraftReview}
                        className={`w-full resize-y rounded-2xl border px-4 py-3 text-sm text-slate-900 outline-none transition ${isEditing || isDraftReview ? 'border-blue-300 bg-white' : 'border-slate-200 bg-white/80'}`}
                      />
                    </label>
                  </div>

                  {isDraftReview ? (
                    <div className="mt-4">
                      <label className="block space-y-2">
                        <span className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Label</span>
                        <input
                          value={attribute.draft_label}
                          onChange={(event) => updateReviewAttribute(attributeKey, 'draft_label', event.target.value)}
                          className="w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 outline-none"
                        />
                      </label>
                    </div>
                  ) : null}

                  {!isDraftReview ? (
                    <div className="mt-4 flex flex-wrap gap-2">
                      <button
                        type="button"
                        onClick={() => handleEditAttribute(attribute.id)}
                        className="rounded-full border border-slate-200 bg-white px-4 py-2 text-sm text-slate-600 transition hover:bg-slate-50"
                      >
                        Edit
                      </button>
                      <button
                        type="button"
                        onClick={() => handleSaveAttribute(attribute.id)}
                        disabled={!isEditing}
                        className="rounded-full bg-slate-900 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-400"
                      >
                        Save
                      </button>
                    </div>
                  ) : null}
                </section>
              )
            })
              ) : (
            <div className="rounded-[24px] border border-dashed border-slate-200 bg-slate-50 px-4 py-14 text-center text-sm text-slate-500">
              {viewerQuery ? 'No attributes match the current search.' : 'No attributes were extracted for this document yet.'}
            </div>
              )}
            </>
          ) : null}

          {isDraftReview ? (
            <div className="sticky bottom-0 rounded-[24px] border border-slate-200 bg-white p-4 shadow-sm">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="text-sm text-slate-600">
                  {selectedItemCount} selected
                </div>
                <button
                  type="button"
                  onClick={saveSelectedReviewAttributes}
                  disabled={savingSelected || !selectedItemCount}
                  className="rounded-full bg-slate-900 px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-400"
                >
                  {savingSelected ? 'Saving...' : 'Save Selected'}
                </button>
              </div>
              {reviewError ? <p className="mt-3 text-sm text-rose-600">{reviewError}</p> : null}
            </div>
          ) : null}
        </div>
      </aside>
    </div>
  )
}

function AttributePage({
  apiBaseUrl,
  attributeDetail,
  attributeLoading,
  attributeError,
  onBackToLibrary,
  onOpenDocument,
  onUpdateAttribute,
  onResetAttribute,
  onSaveAttribute,
  onDeleteAttribute,
  savedAttributes,
  showSavedPanel,
  onCloseSavedPanel,
}) {
  if (attributeLoading) {
    return (
    <>
      {showSavedPanel ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
          <div className="w-[780px] rounded-2xl bg-white p-8 shadow-lg">
            <h3 className="text-lg font-semibold">Saved attributes</h3>
            <p className="mt-2 text-sm text-slate-600">The following attributes were saved successfully:</p>
            <ul className="mt-4 max-h-56 overflow-auto space-y-2">
              {savedAttributes.map((a, i) => (
                <li key={i} className="rounded-md border border-slate-100 bg-slate-50 px-3 py-2 text-sm">
                  <div className="font-medium">{a.label || '—'}</div>
                  <div className="text-slate-600">{a.value || '—'}</div>
                  <div className="text-xs text-slate-400">Page: {a.page_number}</div>
                </li>
              ))}
            </ul>
            <div className="mt-6 flex justify-end gap-3">
              <button onClick={() => onCloseSavedPanel?.()} className="rounded-full border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-slate-700">Close</button>
              <button onClick={() => { onCloseSavedPanel?.(); onBackToLibrary?.(); }} className="rounded-full bg-slate-900 px-4 py-2 text-sm font-semibold text-white">Back to home</button>
            </div>
          </div>
        </div>
      ) : null}

      <div className="mx-auto grid min-h-[calc(100vh-140px)] max-w-[1200px] gap-6 px-4 pb-8 pt-6 lg:px-8">
        <div className="rounded-[30px] border border-slate-200 bg-white p-6 shadow-[0_10px_30px_rgba(15,23,42,0.05)]">
          <LoadingSkeleton className="h-8 w-1/3" />
          <LoadingSkeleton className="mt-4 h-56 w-full" />
        </div>
      </div>
    </>
    )
  }

  if (attributeError) {
    return (
      <div className="mx-auto grid min-h-[calc(100vh-140px)] max-w-[1200px] gap-6 px-4 pb-8 pt-6 lg:px-8">
        <div className="rounded-[30px] border border-rose-200 bg-rose-50 px-6 py-10 text-sm text-rose-700">
          <div className="font-semibold">Unable to load attribute</div>
          <div className="mt-2">{attributeError}</div>
          <button type="button" onClick={onBackToLibrary} className="mt-4 rounded-full bg-white px-4 py-2 font-semibold text-rose-700 shadow-sm">
            Back to library
          </button>
        </div>
      </div>
    )
  }

  if (!attributeDetail) {
    return (
      <div className="mx-auto grid min-h-[calc(100vh-140px)] max-w-[1200px] gap-6 px-4 pb-8 pt-6 lg:px-8">
        <div className="rounded-[30px] border border-slate-200 bg-white p-6 shadow-[0_10px_30px_rgba(15,23,42,0.05)] text-sm text-slate-500">
          No attribute selected.
        </div>
      </div>
    )
  }

  const confidencePercent = confidenceToPercent(attributeDetail.confidence_score)
  const normalizedConfidence = confidencePercent / 100
  const tone = confidenceTone(normalizedConfidence)
  const borderTone = confidenceBorderTone(normalizedConfidence)

  return (
    <div className="mx-auto grid min-h-[calc(100vh-140px)] max-w-[1400px] gap-6 px-4 pb-8 pt-6 lg:grid-cols-[minmax(0,1.1fr)_minmax(360px,0.9fr)] lg:px-8">
      <section className="rounded-[30px] border border-slate-200 bg-white p-6 shadow-[0_10px_30px_rgba(15,23,42,0.05)]">
        <div className="flex flex-wrap items-start justify-between gap-4 border-b border-slate-200 pb-5">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">Attribute detail</p>
            <h1 className="mt-2 text-2xl font-semibold text-slate-900">{attributeDetail.entity_type_label || 'Attribute'}</h1>
            <div className="mt-2 flex flex-wrap items-center gap-2 text-sm text-slate-500">
              <span>From document:</span>
              <button type="button" onClick={() => onOpenDocument?.(attributeDetail.document_id)} className="font-semibold text-blue-700 hover:underline">
                {attributeDetail.document_name || `Document ${attributeDetail.document_id}`}
              </button>
              <span>· Page {attributeDetail.page_number || '?'}</span>
            </div>
          </div>
          <div className={`rounded-full border px-3 py-1 text-xs font-semibold ${tone}`}>{attributeDetail.validation_status || 'PENDING'}</div>
        </div>

        <div className="mt-6 grid gap-4">
          <div className="rounded-[24px] border border-slate-200 bg-slate-50 p-4">
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Confidence</div>
            <div className="mt-2 text-lg font-semibold text-slate-900">{confidencePercent.toFixed(confidencePercent >= 10 ? 0 : 1)}%</div>
          </div>

          <div className={`rounded-[24px] border-2 p-4 ${borderTone}`}>
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Extracted value</div>
            <textarea
              value={attributeDetail.draft_value || ''}
              onChange={(event) => onUpdateAttribute(attributeDetail.id, 'draft_value', event.target.value)}
              rows={8}
              className="mt-3 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 outline-none"
            />
          </div>

          <div className={`rounded-[24px] border-2 p-4 ${borderTone}`}>
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Label</div>
            <input
              value={attributeDetail.draft_label || ''}
              onChange={(event) => onUpdateAttribute(attributeDetail.id, 'draft_label', event.target.value)}
              className="mt-3 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 outline-none"
            />
          </div>

          <div className="flex flex-wrap gap-3 pt-2">
            <button type="button" onClick={() => onSaveAttribute(attributeDetail.id)} className="rounded-full bg-slate-900 px-5 py-3 text-sm font-semibold text-white hover:bg-slate-800">
              Save changes
            </button>
            <button type="button" onClick={() => onResetAttribute(attributeDetail.id)} className="rounded-full border border-slate-200 bg-white px-5 py-3 text-sm font-semibold text-slate-700 hover:bg-slate-50">
              Reset
            </button>
            <button type="button" onClick={() => onDeleteAttribute(attributeDetail.id)} className="rounded-full border border-rose-200 bg-rose-50 px-5 py-3 text-sm font-semibold text-rose-700 hover:bg-rose-100">
              Delete
            </button>
            <button type="button" onClick={onBackToLibrary} className="rounded-full border border-slate-200 bg-white px-5 py-3 text-sm font-semibold text-slate-700 hover:bg-slate-50">
              Back to library
            </button>
          </div>
        </div>
      </section>

      <aside className="rounded-[30px] border border-slate-200 bg-white p-6 shadow-[0_10px_30px_rgba(15,23,42,0.05)]">
        <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">Source</p>
        <h2 className="mt-2 text-xl font-semibold text-slate-900">Document information</h2>
        <div className="mt-4 space-y-3 text-sm text-slate-600">
          <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3">
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">Document name</div>
            <div className="mt-1 font-semibold text-slate-900">{attributeDetail.document_name || 'Unknown document'}</div>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3">
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">Document status</div>
            <div className="mt-1 font-semibold text-slate-900">{attributeDetail.document_status || 'Unknown'}</div>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3">
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">Page</div>
            <div className="mt-1 font-semibold text-slate-900">{attributeDetail.page_number || '?'}</div>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3">
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">Attribute id</div>
            <div className="mt-1 font-semibold text-slate-900">#{attributeDetail.id}</div>
          </div>
        </div>
      </aside>
    </div>
  )
}

export default function App() {
  // Prefer explicit Vite env override. Otherwise derive from the current origin
  // and default to port 8001 for local development (where backend runs).
  const envBase = import.meta.env.VITE_API_BASE_URL
  const defaultPort = '8000'
  const originBase = typeof window !== 'undefined' ? `${window.location.protocol}//${window.location.hostname}` : ''
  const apiBaseUrl = envBase || (originBase ? `${originBase}:${defaultPort}` : `http://localhost:${defaultPort}`)
  const [session, setSession] = useState(null)
  const [sessionLoading, setSessionLoading] = useState(true)
  const [authMode, setAuthMode] = useState('signin')
  const [profileOpen, setProfileOpen] = useState(false)
  const [profile, setProfile] = useState(null)
  const [profileName, setProfileName] = useState('')
  const [profileUserId, setProfileUserId] = useState(null)
  const [profileLoading, setProfileLoading] = useState(false)
  const [profileError, setProfileError] = useState('')
  const [profileReloadKey, setProfileReloadKey] = useState(0)
  useEffect(() => {
    if (!supabase) {
      setSessionLoading(false)
      return undefined
    }
    let active = true
    supabase.auth.getSession().then(({ data }) => {
      if (active) setSession(data.session)
    }).catch(() => {
      if (active) setSession(null)
    }).finally(() => {
      if (active) setSessionLoading(false)
    })
    const { data: listener } = supabase.auth.onAuthStateChange((_event, nextSession) => {
      // Coming back to the tab re-announces the same login; keep the current session object so nothing reloads.
      setSession((current) => (
        current && nextSession && current.user?.id === nextSession.user?.id && current.access_token === nextSession.access_token
          ? current
          : nextSession
      ))
      setSessionLoading(false)
      if (!nextSession) {
        setProfileOpen(false)
        setProfile(null)
        setProfileName('')
        setProfileUserId(null)
        setProfileError('')
        setPage('landing')
        setActiveDocumentId(null)
        setActiveAttributeId(null)
        setDocuments([])
        setViewerBundle(null)
        setSavedAttributes([])
        setCurrentAttribute(null)
        setAttributeDetail(null)
        setServerResults([])
        setServerTotal(0)
      }
    })
    return () => {
      active = false
      listener.subscription.unsubscribe()
    }
  }, [])
  const currentUserEmail = session?.user?.email || null
  const createdAt = session?.user?.created_at || null
  async function handleSignOut() {
    await signOut()
    setSession(null)
    setProfileOpen(false)
    setProfile(null)
    setPage('landing')
  }

  async function saveProfileName(name) {
    const response = await fetch(`${apiBaseUrl}/api/v1/profile`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    })
    const result = await response.json()
    if (!response.ok) throw new Error(result.detail || 'Unable to save your profile.')
    setProfile(result.profile)
    setProfileName(result.profile.name)
    setProfileUserId(result.profile.user_id)
    setProfileError('')
  }

  useEffect(() => {
    if (!supabase || !session?.user?.id) {
      setProfile(null)
      setProfileName('')
      setProfileUserId(null)
      setProfileLoading(false)
      return undefined
    }

    let active = true
    setProfileLoading(true)
    setProfileUserId(null)
    setProfileError('')
    async function loadProfile() {
      try {
        const response = await fetch(`${apiBaseUrl}/api/v1/profile`)
        const result = await response.json()
        if (!response.ok) throw new Error(result.detail || 'Unable to load your profile.')
        if (!active) return
        setProfile(result.profile)
        setProfileName(result.profile?.name || String(session.user.user_metadata?.name || '').trim())
      } catch (error) {
        if (active) {
          setProfile(null)
          setProfileError(error.message || 'Unable to load your profile.')
        }
      } finally {
        if (active) {
          setProfileUserId(session.user.id)
          setProfileLoading(false)
        }
      }
    }
    void loadProfile()
    return () => {
      active = false
    }
  }, [session?.user?.id, apiBaseUrl, profileReloadKey])

  const initialRoute = useMemo(() => parseInitialRoute(), [])
  const [page, setPage] = useState(initialRoute.page)
  const [activeDocumentId, setActiveDocumentId] = useState(initialRoute.documentId)
  const [activeAttributeId, setActiveAttributeId] = useState(initialRoute.attributeId)
  const [documents, setDocuments] = useState([])
  const [documentsLoading, setDocumentsLoading] = useState(true)
  const [documentsError, setDocumentsError] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [searchMode, setSearchMode] = useState('documents')
  const [historyFilter, setHistoryFilter] = useState(DEFAULT_FILTER)
  const [savedAttributes, setSavedAttributes] = useState([])
  const [showSavedPanel, setShowSavedPanel] = useState(false)
  const [savedToast, setSavedToast] = useState('')
  const [favoriteIds, setFavoriteIds] = useState(() => {
    try {
      const raw = window.localStorage.getItem('ocr-studio-favorites')
      const parsed = raw ? JSON.parse(raw) : []
      return Array.isArray(parsed) ? parsed.filter((id) => Number.isInteger(id)) : []
    } catch {
      return []
    }
  })
  const [uploadPreview, setUploadPreview] = useState(null)
  const [uploadName, setUploadName] = useState('')
  const [activity, setActivity] = useState('Preparing upload...')
  const [processingProgress, setProcessingProgress] = useState(0)
  const [processingDocument, setProcessingDocument] = useState(null)
  const [uploadError, setUploadError] = useState('')
  const [processingError, setProcessingError] = useState('')
  const [processingRetryKey, setProcessingRetryKey] = useState(0)
  const [viewerLoading, setViewerLoading] = useState(false)
  const [viewerError, setViewerError] = useState('')
  const [viewerBundle, setViewerBundle] = useState(null)
  const [serverResults, setServerResults] = useState([])
  const [serverTotal, setServerTotal] = useState(0)
  const [serverLoading, setServerLoading] = useState(false)
  const [serverError, setServerError] = useState('')
  const [serverPopupOpen, setServerPopupOpen] = useState(false)
  const latestServerRequestKeyRef = useRef('')
  const [currentAttribute, setCurrentAttribute] = useState(null)
  const [attributeDetail, setAttributeDetail] = useState(null)
  const [attributeDetailLoading, setAttributeDetailLoading] = useState(false)
  const [attributeDetailError, setAttributeDetailError] = useState('')

  const handleServerResults = (payload) => {
    try {
      if (payload.loading || !payload.q) {
        latestServerRequestKeyRef.current = payload.requestKey
      }
      const matchesCurrentRequest = payload.requestKey === latestServerRequestKeyRef.current

      if (matchesCurrentRequest) {
        setServerResults(payload.results || [])
        setServerTotal(payload.total || 0)
        setServerError(payload.error || '')
        setServerLoading(Boolean(payload.loading))
        setServerPopupOpen(Boolean(payload.q && ((payload.results && payload.results.length > 0) || payload.error)))
      }
    } catch (err) {
      // ignore
    }
  }
  const [viewerSearch, setViewerSearch] = useState('')
  const [zoom, setZoom] = useState(DEFAULT_ZOOM)
  const [selectedAttributeId, setSelectedAttributeId] = useState(null)
  const uploadXhrRef = useRef(null)
  const processingAbortRef = useRef(false)

  // When the user types in the global search or toggles include-unapproved,
  // ensure we show the landing/library view so results are visible.
  const handleSetSearchQuery = (value) => {
    setSearchQuery(value)
    if (page !== 'landing') setPage('landing')
  }

  // include_unapproved behavior removed — saved attributes are discoverable by default

  const currentDocument = useMemo(
    () => documents.find((document) => document.id === activeDocumentId) || null,
    [activeDocumentId, documents],
  )

  const documentData = viewerBundle?.document || currentDocument
  const viewerAttributes = viewerBundle?.attributes || []
  const ocrPages = viewerBundle?.ocr?.pages || []
  const pageLabel = pageTitle(page, documentData)

  useEffect(() => {
    const url = new URL(window.location.href)
    url.searchParams.set('page', page)
    if (activeDocumentId) {
      url.searchParams.set('documentId', String(activeDocumentId))
    } else {
      url.searchParams.delete('documentId')
    }
    if (page === 'attribute' && activeAttributeId) {
      url.searchParams.set('attributeId', String(activeAttributeId))
    } else {
      url.searchParams.delete('attributeId')
    }
    window.history.replaceState({}, '', `${url.pathname}${url.search}${url.hash}`)
  }, [activeAttributeId, activeDocumentId, page])

  useEffect(() => {
    let active = true
    let retryTimer

    async function loadDocuments(attempt = 0) {
      setDocumentsLoading(true)
      setDocumentsError('')
      try {
        const response = await fetch(`${apiBaseUrl}/api/v1/documents?limit=200`)
        const result = await response.json()
        if (!response.ok) {
          throw new Error(result.detail || `Failed to load documents (${response.status})`)
        }
        if (active) {
          setDocuments(result.documents || [])
        }
      } catch (error) {
        if (active) {
          if (attempt < 15) {
            retryTimer = window.setTimeout(() => loadDocuments(attempt + 1), 1500)
          } else {
            setDocumentsError(error.message)
          }
        }
      } finally {
        if (active) {
          setDocumentsLoading(false)
        }
      }
    }

    loadDocuments()
    return () => {
      active = false
      if (retryTimer) window.clearTimeout(retryTimer)
    }
  }, [apiBaseUrl])

  useEffect(() => {
    if (page !== 'processing' || !activeDocumentId) {
      processingAbortRef.current = true
      return undefined
    }

    let active = true
    const processingDocumentId = activeDocumentId
    const isCurrent = () => active && !processingAbortRef.current && activeDocumentId === processingDocumentId
    processingAbortRef.current = false
    setProcessingError('')
    let timeoutId = null

    async function pollStatus() {
      if (!isCurrent()) {
        return
      }

      try {
        const response = await fetch(`${apiBaseUrl}/api/v1/documents/${processingDocumentId}`)
        const result = await response.json()
        if (!isCurrent()) {
          return
        }
        if (!response.ok) {
          throw new Error(result.detail || `Failed to load document ${processingDocumentId}`)
        }

        setProcessingDocument(result)
        setProcessingProgress((current) => Math.max(current, mapStatusProgress(result.status)))

        const normalized = String(result.status || '').toUpperCase()
        if (normalized === 'QUEUED') {
          setActivity('Queued for extraction...')
        } else if (normalized === 'PROCESSING') {
          const progress = result.processing_progress
          setActivity(
            progress?.total_pages
              ? `Reading the document... ${progress.pages_read} of ${progress.total_pages} pages read`
              : 'Reading the document...',
          )
        }
        if (normalized === 'PROCESSED') {
          setProcessingProgress(100)
          setActivity('Finalizing results...')
          await loadViewerBundle(processingDocumentId, isCurrent)
          if (!isCurrent()) {
            return
          }
          await refreshDocuments(isCurrent)
          if (isCurrent()) {
            setPage('viewer')
          }
          return
        }

        if (normalized === 'FAILED') {
          setProcessingProgress(100)
          setActivity('Processing failed')
          const failedProgress = result.processing_progress
          const resumeNote = failedProgress?.pages_read
            ? ` ${failedProgress.pages_read} of ${failedProgress.total_pages} pages were already read and will not be read again when you press Retry.`
            : ''
          setProcessingError(
            (result.processing_error
              ? `Document processing failed: ${result.processing_error}`
              : 'Document processing failed. Please try again or upload a different file.') + resumeNote,
          )
          return
        }

        timeoutId = window.setTimeout(pollStatus, 1800)
      } catch (error) {
        if (isCurrent()) {
          setProcessingError(`Unable to check processing status. Retrying... (${error.message})`)
          timeoutId = window.setTimeout(pollStatus, 3000)
        }
      }
    }

    pollStatus()
    return () => {
      active = false
      processingAbortRef.current = true
      if (timeoutId) {
        window.clearTimeout(timeoutId)
      }
    }
  }, [activeDocumentId, apiBaseUrl, page, processingRetryKey])

  function retryProcessing() {
    setProcessingError('')
    setActivity('Retrying status check...')
    setProcessingRetryKey((current) => current + 1)
  }

  useEffect(() => {
    if (!uploadPreview) {
      return undefined
    }

    return () => {
      URL.revokeObjectURL(uploadPreview.url)
    }
  }, [uploadPreview])

  async function refreshDocuments(isCurrent = () => true) {
    const response = await fetch(`${apiBaseUrl}/api/v1/documents?limit=200`)
    const result = await response.json()
    if (!isCurrent()) {
      return
    }
    if (!response.ok) {
      throw new Error(result.detail || `Failed to load documents (${response.status})`)
    }
    setDocuments(result.documents || [])
  }

  async function loadViewerBundle(documentId, isCurrent = () => true) {
    if (!isCurrent()) {
      return
    }
    setViewerLoading(true)
    setViewerError('')
    try {
      const [documentResponse, attributesResponse, ocrResponse] = await Promise.all([
        fetch(`${apiBaseUrl}/api/v1/documents/${documentId}`),
        fetch(`${apiBaseUrl}/api/v1/documents/${documentId}/attributes`),
        fetch(`${apiBaseUrl}/api/v1/documents/${documentId}/ocr`),
      ])

      const [documentJson, attributesJson, ocrJson] = await Promise.all([
        documentResponse.json(),
        attributesResponse.json(),
        ocrResponse.json(),
      ])

      if (!documentResponse.ok) {
        throw new Error(documentJson.detail || `Document ${documentId} not found`)
      }
      if (!attributesResponse.ok) {
        throw new Error(attributesJson.detail || `Attributes for document ${documentId} not found`)
      }
      if (!ocrResponse.ok) {
        throw new Error(ocrJson.detail || `OCR data for document ${documentId} not found`)
      }

      if (!isCurrent()) {
        return
      }
      setViewerBundle({
        document: documentJson,
        attributes: normalizeDocumentAttributes(attributesJson.attributes || []),
        ocr: ocrJson,
      })
      setSelectedAttributeId((attributesJson.attributes || [])[0]?.id || null)
      setZoom(DEFAULT_ZOOM)
      setViewerSearch('')
      setViewerError('')
    } catch (error) {
      if (isCurrent()) {
        setViewerError(error.message)
        setViewerBundle(null)
      }
    } finally {
      if (isCurrent()) {
        setViewerLoading(false)
      }
    }
  }

  function openDocument(document) {
    setActiveDocumentId(document.id)
    setActiveAttributeId(null)
    setUploadError('')
    setProcessingError('')
    setViewerError('')
    setSelectedAttributeId(null)

    const normalizedStatus = String(document.status || '').toUpperCase()
    if (normalizedStatus === 'PROCESSED') {
      setPage('viewer')
      void loadViewerBundle(document.id)
      return
    }

    if (normalizedStatus === 'CANCELED') {
      setPage('canceled')
      return
    }

    setPage('processing')
    setProcessingDocument(document)
    setActivity('Preparing document...')
    setProcessingProgress(mapStatusProgress(document.status))
  }

  async function processDocumentAgain() {
    if (!activeDocumentId) return
    setProcessingError('')
    try {
      const response = await fetch(`${apiBaseUrl}/api/v1/documents/${activeDocumentId}/reprocess`, { method: 'POST' })
      const result = await response.json()
      if (!response.ok) throw new Error(result.detail || 'Failed to restart processing')
      await refreshDocuments()
      setProcessingDocument((current) => ({ ...(current || documentData || {}), status: 'QUEUED' }))
      setPage('processing')
      setActivity('Queued for extraction...')
      setProcessingProgress(0)
      setProcessingRetryKey((current) => current + 1)
    } catch (error) {
      setProcessingError(error.message)
    }
  }

  async function openDocumentById(documentId) {
    const selected = documents.find((document) => document.id === documentId)
    if (selected) {
      openDocument(selected)
      return
    }

    const response = await fetch(`${apiBaseUrl}/api/v1/documents/${documentId}`)
    const result = await response.json()
    if (!response.ok) {
      throw new Error(result.detail || `Document ${documentId} not found`)
    }
    openDocument(result)
  }

  async function loadAttributeDetail(attributeId) {
    setAttributeDetailLoading(true)
    setAttributeDetailError('')
    try {
      const response = await fetch(`${apiBaseUrl}/api/v1/attributes/${attributeId}`)
      const result = await response.json()
      if (!response.ok) {
        throw new Error(result.detail || `Attribute ${attributeId} not found`)
      }

      const detail = result.attribute || null
      if (!detail) {
        throw new Error(`Attribute ${attributeId} not found`)
      }

      setAttributeDetail({
        ...detail,
        draft_label: detail.entity_type_label,
        draft_value: detail.extracted_value,
        draft_validation_status: detail.validation_status,
        original_label: detail.entity_type_label,
        original_value: detail.extracted_value,
        original_validation_status: detail.validation_status,
      })
    } catch (error) {
      setAttributeDetail(null)
      setAttributeDetailError(error.message)
    } finally {
      setAttributeDetailLoading(false)
    }
  }

  function toggleFavorite(documentId) {
    setFavoriteIds((current) => {
      const next = current.includes(documentId) ? current.filter((id) => id !== documentId) : [...current, documentId]
      window.localStorage.setItem('ocr-studio-favorites', JSON.stringify(next))
      return next
    })
  }

  function updateAttributeDetail(attributeId, field, value) {
    setAttributeDetail((current) => {
      if (!current || current.id !== attributeId) {
        return current
      }

      return {
        ...current,
        [field]: value,
      }
    })
  }

  function resetAttributeDetail(attributeId) {
    setAttributeDetail((current) => {
      if (!current || current.id !== attributeId) {
        return current
      }

      return {
        ...current,
        draft_label: current.original_label,
        draft_value: current.original_value,
        draft_validation_status: current.original_validation_status,
      }
    })
  }

  async function saveAttributeDetail(attributeId) {
    const current = attributeDetail
    if (!current || current.id !== attributeId) {
      return
    }

    const response = await fetch(`${apiBaseUrl}/api/v1/attributes/correct`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        document_id: current.document_id,
        corrections: [
          {
            attribute_id: current.id,
            original_prediction: current.original_value,
            corrected_value: current.draft_value,
            corrected_label: current.draft_label,
            validation_status: current.draft_validation_status,
          },
        ],
      }),
    })

    const result = await response.json()
    if (!response.ok) {
      throw new Error(result.detail || `Failed to save attribute ${attributeId}`)
    }

    setAttributeDetail((state) => {
      if (!state || state.id !== attributeId) {
        return state
      }

      return {
        ...state,
        original_label: state.draft_label,
        original_value: state.draft_value,
        original_validation_status: state.draft_validation_status,
      }
    })

    await refreshDocuments()
  }

  async function deleteAttributeDetail(attributeId) {
    const current = attributeDetail
    if (!current || current.id !== attributeId) {
      return
    }

    const confirmed = window.confirm('Delete this attribute? This cannot be undone.')
    if (!confirmed) {
      return
    }

    const response = await fetch(`${apiBaseUrl}/api/v1/attributes/${attributeId}`, {
      method: 'DELETE',
    })

    const result = await response.json()
    if (!response.ok) {
      throw new Error(result.detail || `Failed to delete attribute ${attributeId}`)
    }

    setAttributeDetail(null)
    setCurrentAttribute(null)
    setActiveAttributeId(null)
    setSelectedAttributeId(null)
    await refreshDocuments()
    setPage('viewer')
    if (result.document_id) {
      setActiveDocumentId(result.document_id)
      await loadViewerBundle(result.document_id)
    }
  }

  async function uploadDocument(file) {
    setUploadError('')
    setProcessingError('')
    setViewerError('')
    setViewerBundle(null)
    setUploadPreview({ url: URL.createObjectURL(file), type: file.type })
    setUploadName(file.name)
    setProcessingProgress(12)
    setActivity('Uploading document...')
    setProcessingDocument({ name: file.name, status: 'UPLOADING' })
    setPage('processing')
    processingAbortRef.current = false

    let responseData
    try {
      responseData = await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest()
        uploadXhrRef.current = xhr
        xhr.open('POST', `${apiBaseUrl}/api/v1/ingest`)
        xhr.responseType = 'json'

        xhr.upload.onprogress = (event) => {
          if (!event.lengthComputable) {
            return
          }
          const uploadProgress = Math.min(45, Math.round((event.loaded / event.total) * 45))
          setProcessingProgress(uploadProgress)
        }

        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve(xhr.response || JSON.parse(xhr.responseText || '{}'))
            return
          }

          const detail = xhr.response?.detail || xhr.responseText || `Upload failed with status ${xhr.status}`
          reject(new Error(detail))
        }

        xhr.onerror = () => reject(new Error('Network error while uploading document'))

        const formData = new FormData()
        formData.append('file', file)
        getAccessToken().then((token) => {
          if (token) xhr.setRequestHeader('Authorization', `Bearer ${token}`)
          xhr.send(formData)
        }).catch(reject)
      })
    } catch (error) {
      processingAbortRef.current = true
      setUploadError(error.message || 'Unable to upload document. Please try again.')
      setProcessingError('')
      setProcessingDocument(null)
      setActiveDocumentId(null)
      setProcessingProgress(0)
      setActivity('Preparing upload...')
      setUploadPreview(null)
      setUploadName('')
      setPage('landing')
      return
    } finally {
      uploadXhrRef.current = null
    }

    setProcessingProgress(50)
    setActivity('Queued for extraction...')
    setActiveDocumentId(responseData.document_id)
    setProcessingDocument((current) => ({
      ...(current || {}),
      id: responseData.document_id,
      name: file.name,
      status: responseData.status || 'QUEUED',
    }))
    try {
      await refreshDocuments()
    } catch (error) {
      setProcessingError(error.message)
    }
  }

  async function cancelProcessing() {
    processingAbortRef.current = true
    const documentIdToCancel = activeDocumentId
    if (uploadXhrRef.current) {
      uploadXhrRef.current.abort()
      uploadXhrRef.current = null
    }

    setPage('landing')
    setUploadError('Upload canceled.')
    setProcessingError('')
    setProcessingProgress(0)
    setActivity('Preparing upload...')

    if (documentIdToCancel) {
      try {
        await fetch(`${apiBaseUrl}/api/v1/documents/${documentIdToCancel}/status?status=CANCELED`, { method: 'PATCH' })
        await refreshDocuments()
      } catch (error) {
        setUploadError(error.message || 'Upload canceled, but the server status could not be updated.')
      }
    }
  }

  function backToLibrary() {
    processingAbortRef.current = true
    setPage('landing')
    setUploadError('')
    setViewerError('')
    setCurrentAttribute(null)
    setActiveAttributeId(null)
    setViewerBundle(null)
    setSavedAttributes([])
    setShowSavedPanel(false)
    setActiveDocumentId(null)
    setSelectedAttributeId(null)
    setProcessingDocument(null)
    setProcessingProgress(0)
  }

  function handleNavigateViewer() {
    if (!activeDocumentId) {
      setPage('landing')
      return
    }

    const selected = documents.find((document) => document.id === activeDocumentId)
    if (selected) {
      openDocument(selected)
    }
  }

  function handleSelectSearchResult(item) {
    if (item?.id && item?.document_id && item?.entity_type_label) {
      setActiveAttributeId(item.id)
      setActiveDocumentId(item.document_id)
      setViewerBundle(null)
      setServerPopupOpen(false)
      setPage('attribute')
      return
    }

    if (item.document_id && item.entity_type_label) {
      void openDocumentById(item.document_id)
    } else if (item.id) {
      void openDocumentById(item.id)
    }
    setServerPopupOpen(false)
  }

  function updateAttribute(attributeId, field, value) {
    setViewerBundle((current) => {
      if (!current) {
        return current
      }

      return {
        ...current,
        attributes: current.attributes.map((attribute) =>
          attribute.id === attributeId ? { ...attribute, [field]: value } : attribute,
        ),
      }
    })
  }

  function resetAttribute(attributeId) {
    setViewerBundle((current) => {
      if (!current) {
        return current
      }

      return {
        ...current,
        attributes: current.attributes.map((attribute) =>
          attribute.id === attributeId
            ? {
                ...attribute,
                draft_label: attribute.original_label,
                draft_value: attribute.original_value,
                draft_validation_status: attribute.original_validation_status,
              }
            : attribute,
        ),
      }
    })
  }

  async function saveAttribute(attributeId) {
    const attribute = viewerAttributes.find((item) => item.id === attributeId)
    if (!attribute) {
      return
    }

    const response = await fetch(`${apiBaseUrl}/api/v1/attributes/correct`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        document_id: activeDocumentId,
        corrections: [
          {
            attribute_id: attribute.id,
            original_prediction: attribute.original_value,
            corrected_value: attribute.draft_value,
            corrected_label: attribute.draft_label,
            validation_status: attribute.draft_validation_status,
          },
        ],
      }),
    })

    const result = await response.json()
    if (!response.ok) {
      throw new Error(result.detail || `Failed to save attribute ${attributeId}`)
    }

    setViewerBundle((current) => {
      if (!current) {
        return current
      }

      return {
        ...current,
        attributes: current.attributes.map((item) =>
          item.id === attributeId
            ? {
                ...item,
                original_label: item.draft_label,
                original_value: item.draft_value,
                original_validation_status: item.draft_validation_status,
              }
            : item,
        ),
      }
    })

    await refreshDocuments()
    setViewerError(result.status === 'accepted' ? '' : viewerError)
  }

  async function saveSelectedAttributes(selectedAttributes) {
    const response = await fetch(`${apiBaseUrl}/api/v1/documents/${activeDocumentId}/attributes/save-selected`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        document_id: activeDocumentId,
        attributes: selectedAttributes.map((attribute) => ({
          temp_id: attribute.temp_id || String(attribute.id),
          source: attribute.source || 'model',
          entity_type_label: attribute.draft_label,
          extracted_value: attribute.draft_value,
          page_number: attribute.page_number || 1,
          page_id: attribute.page_id,
          bounding_boxes: attribute.bounding_boxes || [],
          page_spans: attribute.page_spans || [],
          confidence_score: attribute.confidence_score || 0,
          validation_status: attribute.draft_validation_status || attribute.validation_status || 'PENDING',
          kind: attribute.kind,
          metadata: attribute.kind === 'table' ? (attribute.metadata || attribute.table_data) : attribute.metadata,
        })),
      }),
    })

    const result = await response.json()
    if (!response.ok) {
      throw new Error(describeApiError(result.detail, 'Failed to save selected attributes'))
    }

    // Refresh document list
    await refreshDocuments()

    // If backend returned saved attribute ids, fetch their details and show only saved attributes
    try {
      const savedIds = result.saved_attribute_ids || []
      if (savedIds.length) {
        const detailPromises = savedIds.map((id) => fetch(`${apiBaseUrl}/api/v1/attributes/${id}`))
        const detailResponses = await Promise.all(detailPromises)
        const detailJsons = await Promise.all(detailResponses.map((r) => r.json()))
        const savedAttrs = detailJsons
          .filter((j) => j && j.attribute)
          .map((j) => ({
            id: j.attribute.id,
            page_id: j.attribute.page_id,
            document_id: j.attribute.document_id,
            entity_type_label: j.attribute.entity_type_label,
            extracted_value: j.attribute.extracted_value,
            original_prediction: j.attribute.extracted_value,
            bounding_boxes: j.attribute.bounding_boxes || [],
            page_spans: j.attribute.page_spans || [],
            confidence_score: j.attribute.confidence_score || 0,
            validation_status: j.attribute.validation_status || 'PENDING',
            source: j.attribute.source || 'manual',
            saved: true,
            page_number: j.attribute.page_number || 1,
          }))

        // Update viewer bundle to show only saved attributes
        setViewerBundle((current) => {
          if (!current) return current
          return {
            ...current,
            attributes: normalizeDocumentAttributes(savedAttrs),
          }
        })

        // Prepare UI payload for saved attributes modal
        const savedPayload = selectedAttributes.map((attribute) => ({
          label: attribute.draft_label,
          value: attribute.draft_value,
          page_number: attribute.page_number || 1,
        }))
        setSavedAttributes(savedPayload)
        setShowSavedPanel(true)
        // show short toast notification
        try {
          setSavedToast(`${savedPayload.length} attribute${savedPayload.length === 1 ? '' : 's'} saved`)
          window.setTimeout(() => setSavedToast(''), 4000)
        } catch (e) {
          // ignore
        }
        return
      }
    } catch (err) {
      // fallback to reloading viewer bundle
      // eslint-disable-next-line no-console
      console.warn('Failed to fetch saved attribute details', err)
    }

    await loadViewerBundle(activeDocumentId)
  }

  async function refreshCurrentDocument() {
    if (!activeDocumentId) {
      return
    }
    if (page === 'viewer') {
      await loadViewerBundle(activeDocumentId)
    }
    await refreshDocuments()
  }

  const visibleDocuments = useMemo(() => {
    // Visible documents for the library should not be filtered by the
    // global search input — searches are displayed in the popup. Keep
    // library visibility governed only by history filter and favorites.
    let rows = documents

    if (historyFilter === 'favorites') {
      rows = rows.filter((document) => favoriteIds.includes(document.id))
    }

    if (historyFilter === 'recent') {
      rows = rows.slice(0, 12)
    }

    return rows
  }, [documents, favoriteIds, historyFilter, searchQuery])

  useEffect(() => {
    if (page === 'attribute' && activeAttributeId) {
      void loadAttributeDetail(activeAttributeId)
      return
    }

    if (page !== 'attribute') {
      setAttributeDetail(null)
      setAttributeDetailError('')
    }
  }, [activeAttributeId, page])

  useEffect(() => {
    if (page !== 'viewer' || !activeDocumentId || viewerBundle) {
      return undefined
    }
    void loadViewerBundle(activeDocumentId)
  }, [activeDocumentId, page, viewerBundle])

  const processingEta = `${Math.max(1, Math.ceil((100 - processingProgress) / 14))}s remaining`

  if (authRequired && !supabase) {
    return <AuthConfigurationScreen />
  }

  if (authRequired && supabase && sessionLoading) {
    return <main role="status" className="flex min-h-screen items-center justify-center text-sm text-slate-500">Restoring your session...</main>
  }

  if (authRequired && supabase && !session) {
    return (
      <AuthScreen
        mode={authMode}
        onModeChange={setAuthMode}
        onSignedIn={async (signupName) => {
          const { data } = await supabase.auth.getSession()
          setSession(data.session)
          setSessionLoading(false)
          if (signupName && data.session) await saveProfileName(signupName)
        }}
      />
    )
  }

  if (authRequired && supabase && session && profileUserId !== session.user.id) {
    return <main role="status" className="flex min-h-screen items-center justify-center text-sm text-slate-500">Loading your profile...</main>
  }

  if (authRequired && supabase && session && !profile) {
    return (
      <CompleteProfilePage
        initialName={profileName}
        error={profileError}
        onSubmit={saveProfileName}
        onSignOut={handleSignOut}
      />
    )
  }

  if (profileOpen) {
    return (
      <div className="min-h-screen bg-white text-slate-900">
        <TopBar
          pageLabel="Profile"
          documentData={null}
          onNavigateLanding={() => {
            setProfileOpen(false)
            setPage('landing')
          }}
          onNavigateViewer={() => setPage('landing')}
          searchQuery={searchQuery}
          setSearchQuery={handleSetSearchQuery}
          searchMode={searchMode}
          setSearchMode={setSearchMode}
          onRefresh={() => {}}
          serverResults={serverResults}
          serverPopupOpen={serverPopupOpen}
          onCloseSearchResults={() => setServerPopupOpen(false)}
          onSelectSearchResult={handleSelectSearchResult}
          userEmail={currentUserEmail}
          onOpenProfile={() => setProfileOpen(true)}
          onOpenLibrary={() => {
            setProfileOpen(false)
            setPage('library')
          }}
          onSignOut={handleSignOut}
        />
        <ProfilePage
          name={profile?.name || profileName || 'Name not set'}
          userEmail={currentUserEmail}
          userId={session?.user?.id}
          createdAt={createdAt}
          updatedAt={profile?.updated_at}
          lastSignInAt={session?.user?.last_sign_in_at}
          emailConfirmed={Boolean(session?.user?.email_confirmed_at || session?.user?.confirmed_at)}
          documentCount={documents.length}
          processedCount={documents.filter((document) => String(document.status || '').toUpperCase() === 'PROCESSED').length}
          favoriteCount={documents.filter((document) => favoriteIds.includes(document.id)).length}
          onSaveName={saveProfileName}
          onBackToLibrary={() => {
            setProfileOpen(false)
            setPage('library')
          }}
        />
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-white text-slate-900">
      <TopBar
        pageLabel={pageLabel}
        documentData={documentData}
        onNavigateLanding={() => setPage('landing')}
        onNavigateViewer={handleNavigateViewer}
        searchQuery={searchQuery}
        setSearchQuery={handleSetSearchQuery}
        searchMode={searchMode}
        setSearchMode={setSearchMode}
        onRefresh={() => {
          void refreshDocuments().catch(() => {})
        }}
        serverResults={serverResults}
        serverPopupOpen={serverPopupOpen}
        onCloseSearchResults={() => setServerPopupOpen(false)}
        onSelectSearchResult={handleSelectSearchResult}
        userEmail={currentUserEmail}
        onOpenProfile={() => setProfileOpen(true)}
        onOpenLibrary={() => {
          setProfileOpen(false)
          setPage('library')
        }}
        onSignOut={handleSignOut}
      />

      {/* Saved toast */}
      {savedToast ? (
        <div className="fixed right-6 top-6 z-50">
          <div className="rounded-md bg-emerald-600 px-4 py-2 text-white shadow">{savedToast}</div>
        </div>
      ) : null}

      <main className="page-fade-in">
        {page === 'landing' || page === 'library' ? (
          <LandingPage
            documents={visibleDocuments}
            allDocuments={documents}
            libraryOnly={page === 'library'}
            onOpenLibrary={() => setPage('library')}
            onBackToDashboard={() => setPage('landing')}
            documentsLoading={documentsLoading}
            searchQuery={searchQuery}
            setSearchQuery={handleSetSearchQuery}
            searchMode={searchMode}
            setSearchMode={setSearchMode}
            historyFilter={historyFilter}
            setHistoryFilter={setHistoryFilter}
            favoriteIds={favoriteIds}
            toggleFavorite={toggleFavorite}
            onOpenDocument={openDocument}
            onUploadDocument={uploadDocument}
            loadError={documentsError || uploadError}
            apiBaseUrl={apiBaseUrl}
            onServerResults={handleServerResults}
            serverResults={serverResults}
            serverLoading={serverLoading}
            serverError={serverError}
            serverTotal={serverTotal}
          />
        ) : null}

        {page === 'processing' ? (
          <ProcessingPage
            activity={activity}
            progress={processingProgress}
            etaLabel={processingEta}
            documentData={processingDocument || documentData}
            processingError={processingError}
            onRetry={retryProcessing}
            onCancel={cancelProcessing}
            onBackToLibrary={backToLibrary}
          />
        ) : null}

        {page === 'canceled' ? (
          <CanceledDocumentPage
            documentData={documentData}
            onBackToLibrary={backToLibrary}
            onProcessAgain={() => void processDocumentAgain()}
            processingError={processingError}
          />
        ) : null}

        {page === 'viewer' ? (
          <ViewerPage
            apiBaseUrl={apiBaseUrl}
            documentData={documentData}
            attributes={viewerAttributes}
            ocrPages={ocrPages}
            viewerLoading={viewerLoading}
            viewerError={viewerError}
            viewerSearch={viewerSearch}
            setViewerSearch={setViewerSearch}
            zoom={zoom}
            setZoom={setZoom}
            selectedAttributeId={selectedAttributeId}
            setSelectedAttributeId={setSelectedAttributeId}
            onBackToLibrary={backToLibrary}
            onUpdateAttribute={updateAttribute}
            onResetAttribute={resetAttribute}
            onSaveAttribute={saveAttribute}
            onSaveSelectedAttributes={saveSelectedAttributes}
            onRefreshDocument={refreshCurrentDocument}
          />
        ) : null}

        {page === 'attribute' ? (
          <AttributePage
            apiBaseUrl={apiBaseUrl}
            attributeDetail={attributeDetail}
            attributeLoading={attributeDetailLoading}
            attributeError={attributeDetailError}
            onBackToLibrary={backToLibrary}
            onOpenDocument={(documentId) => {
              void openDocumentById(documentId)
            }}
            onUpdateAttribute={updateAttributeDetail}
            onResetAttribute={resetAttributeDetail}
            onSaveAttribute={(attributeId) => {
              void saveAttributeDetail(attributeId).catch((error) => {
                setAttributeDetailError(error.message || 'Failed to save attribute.')
              })
            }}
            onDeleteAttribute={(attributeId) => {
              void deleteAttributeDetail(attributeId).catch((error) => {
                setAttributeDetailError(error.message || 'Failed to delete attribute.')
              })
            }}
            savedAttributes={savedAttributes}
            showSavedPanel={showSavedPanel}
            onCloseSavedPanel={() => setShowSavedPanel(false)}
          />
        ) : null}
      </main>
    </div>
  )
}

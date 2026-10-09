import { createClient } from '@supabase/supabase-js'

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL
const supabasePublishableKey = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY || import.meta.env.VITE_SUPABASE_ANON_KEY
export const authRequired = ['1', 'true', 'yes', 'on'].includes(
  String(import.meta.env.VITE_AUTH_REQUIRED || 'true').trim().toLowerCase(),
)

export const supabase = supabaseUrl && supabasePublishableKey
  ? createClient(supabaseUrl, supabasePublishableKey, {
      auth: {
        persistSession: true,
        autoRefreshToken: true,
        detectSessionInUrl: true,
      },
    })
  : null

let localSignOutPromise = null

async function clearStaleSession() {
  if (!supabase || localSignOutPromise) {
    return localSignOutPromise
  }
  localSignOutPromise = supabase.auth.signOut({ scope: 'local' }).finally(() => {
    localSignOutPromise = null
  })
  return localSignOutPromise
}

export async function getAccessToken() {
  if (!supabase) return null
  const { data } = await supabase.auth.getSession()
  return data.session?.access_token || null
}

export async function authenticatedFetch(input, init = {}) {
  const token = await getAccessToken()
  if (authRequired && supabase && !token) {
    throw new Error('Please sign in to continue')
  }
  const headers = new Headers(init.headers || {})
  if (token) {
    headers.set('Authorization', `Bearer ${token}`)
  }
  const response = await window.fetch(input, { ...init, headers })
  if (response.status === 401 && authRequired && supabase) {
    await clearStaleSession()
  }
  return response
}

export async function signOut() {
  if (!supabase) return { error: null }
  return supabase.auth.signOut()
}

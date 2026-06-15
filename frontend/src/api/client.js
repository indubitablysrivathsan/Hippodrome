/**
 * RWITC API Client
 * All requests are proxied through /api → http://127.0.0.1:8000
 * The CRA proxy (package.json "proxy") handles dev-time forwarding.
 */

const BASE = "";  // CRA proxy handles it; in prod set to your API base URL

async function request(path, params = {}) {
  const url = new URL(BASE + path, window.location.origin);
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, v);
  });

  const res = await fetch(url.toString());
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// ── Venues ────────────────────────────────────────────────────────────────
export const venues = {
  list: (p = {}) => request("/venues/", p),
  get: (id) => request(`/venues/${id}`),
  meetings: (id, p = {}) => request(`/venues/${id}/meetings`, p),
};

// ── Meetings ──────────────────────────────────────────────────────────────
export const meetings = {
  list: (p = {}) => request("/meetings/", p),
  get: (date, venueId) => request(`/meetings/${date}/${venueId}`),
  races: (date, venueId) => request(`/meetings/${date}/${venueId}/races`),
};

// ── Races ─────────────────────────────────────────────────────────────────
export const races = {
  list: (p = {}) => request("/races/", p),
  get: (id) => request(`/races/${id}`),
  runners: (id) => request(`/races/${id}/runners`),
  dividends: (id) => request(`/races/${id}/dividends`),
  exotics: (id) => request(`/races/${id}/exotics`),
  remarks: (id) => request(`/races/${id}/remarks`),
  penalties: (id) => request(`/races/${id}/penalties`),
};

// ── Runners ───────────────────────────────────────────────────────────────
export const runners = {
  get: (raceId, horseId) => request(`/runners/${raceId}/${horseId}`),
  acceptance: (raceId, horseId) => request(`/runners/${raceId}/${horseId}/acceptance`),
  declaration: (raceId, horseId) => request(`/runners/${raceId}/${horseId}/declaration`),
  equipment: (raceId, horseId) => request(`/runners/${raceId}/${horseId}/equipment`),
  equipmentChanges: (raceId, horseId) => request(`/runners/${raceId}/${horseId}/equipment-changes`),
  jockeyChange: (raceId, horseId) => request(`/runners/${raceId}/${horseId}/jockey-change`),
};

// ── Horses ────────────────────────────────────────────────────────────────
export const horses = {
  list: (p = {}) => request("/horses/", p),
  search: (q, limit = 10) => request("/horses/search", { q, limit }),
  get: (id) => request(`/horses/${id}`),
  aliases: (id) => request(`/horses/${id}/aliases`),
  ratings: (id) => request(`/horses/${id}/ratings`),
  currentRating: (id) => request(`/horses/${id}/current-rating`),
  races: (id, p = {}) => request(`/horses/${id}/races`, p),
  medical: (id) => request(`/horses/${id}/medical`),
  treadmill: (id) => request(`/horses/${id}/treadmill`),
};

// ── Jockeys ───────────────────────────────────────────────────────────────
export const jockeys = {
  list: (p = {}) => request("/jockeys/", p),
  get: (id) => request(`/jockeys/${id}`),
  rides: (id, p = {}) => request(`/jockeys/${id}/rides`, p),
  penalties: (id) => request(`/jockeys/${id}/penalties`),
  stats: (id) => request(`/jockeys/${id}/stats`),
};

// ── Trainers ──────────────────────────────────────────────────────────────
export const trainers = {
  list: (p = {}) => request("/trainers/", p),
  get: (id) => request(`/trainers/${id}`),
  runners: (id, p = {}) => request(`/trainers/${id}/runners`, p),
  stats: (id) => request(`/trainers/${id}/stats`),
  penalties: (id) => request(`/trainers/${id}/penalties`),
};

// ── Schema / Stats ────────────────────────────────────────────────────────
export const schema = {
  tables: () => request("/schema/tables"),
  columns: (table) => request(`/schema/tables/${table}`),
  stats: () => request("/schema/stats"),
  recentActivity: (limit = 10) => request("/schema/recent-activity", { limit }),
};
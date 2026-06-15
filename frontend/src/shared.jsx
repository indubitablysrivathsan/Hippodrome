import { useState, useEffect, useCallback } from "react";

// ── Hook ──────────────────────────────────────────────────────────────────
export function useApi(fn, deps = []) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const run = useCallback(async (...args) => {
    setLoading(true);
    setError(null);
    try {
      const r = await fn(...args);
      setData(r);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, deps); // eslint-disable-line

  return { data, loading, error, run };
}

// ── Simple table ──────────────────────────────────────────────────────────
export function Table({ rows }) {
  if (!rows || rows.length === 0) return <p style={{ color: "#888" }}>No data.</p>;
  const cols = Object.keys(rows[0]);
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 13 }}>
        <thead>
          <tr style={{ background: "#f0f0f0" }}>
            {cols.map((c) => (
              <th key={c} style={{ border: "1px solid #ccc", padding: "4px 8px", textAlign: "left" }}>
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} style={{ background: i % 2 ? "#fafafa" : "#fff" }}>
              {cols.map((c) => (
                <td key={c} style={{ border: "1px solid #ccc", padding: "4px 8px", maxWidth: 240, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {typeof row[c] === "object" ? JSON.stringify(row[c]) : String(row[c] ?? "")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Status ─────────────────────────────────────────────────────────────────
export function Status({ loading, error }) {
  if (loading) return <p>Loading…</p>;
  if (error) return <p style={{ color: "red" }}>Error: {error}</p>;
  return null;
}

// ── Section ────────────────────────────────────────────────────────────────
export function Section({ title, children }) {
  const [open, setOpen] = useState(true);
  return (
    <div style={{ marginTop: 16, border: "1px solid #ddd", borderRadius: 4 }}>
      <div
        onClick={() => setOpen(!open)}
        style={{ padding: "6px 12px", background: "#eee", cursor: "pointer", fontWeight: "bold", userSelect: "none" }}
      >
        {open ? "▾" : "▸"} {title}
      </div>
      {open && <div style={{ padding: 12 }}>{children}</div>}
    </div>
  );
}

// ── Input row ──────────────────────────────────────────────────────────────
export function Field({ label, value, onChange, placeholder, type = "text" }) {
  return (
    <label style={{ display: "inline-flex", flexDirection: "column", fontSize: 12, marginRight: 12 }}>
      {label}
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        style={{ marginTop: 2, padding: "3px 6px", border: "1px solid #aaa", borderRadius: 3, width: 160 }}
      />
    </label>
  );
}

export function Btn({ onClick, children, disabled }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{ padding: "4px 14px", cursor: "pointer", borderRadius: 3, border: "1px solid #888" }}
    >
      {children}
    </button>
  );
}

// ── Normalise API response to array ───────────────────────────────────────
export function toRows(data) {
  if (!data) return [];
  if (Array.isArray(data)) return data;
  if (data.items) return data.items;
  if (data.results) return data.results;
  if (data.data) return Array.isArray(data.data) ? data.data : [data.data];
  // single object
  return [data];
}

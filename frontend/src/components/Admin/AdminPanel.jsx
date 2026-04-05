/**
 * Admin panel — user management and usage statistics.
 * Only accessible to users with is_admin = true.
 */

import { useState, useEffect } from "react";
import { adminApi, authApi } from "../../services/api";

export default function AdminPanel({ currentUser, onClose }) {
  const [tab, setTab] = useState("users"); // "users" | "stats" | "profile"

  return (
    <div style={overlayStyle}>
      <div style={panelStyle}>
        {/* Header */}
        <div style={headerStyle}>
          <div>
            <span style={{ fontWeight: 800, fontSize: "1.1rem", color: "#1d4ed8" }}>PIL</span>
            <span style={{ marginLeft: "8px", fontWeight: 600, color: "#111827" }}>Admin Panel</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <span style={{ fontSize: "0.8rem", color: "#6b7280" }}>{currentUser.email}</span>
            <button onClick={onClose} style={closeBtnStyle}>✕ Close</button>
          </div>
        </div>

        {/* Tabs */}
        <div style={tabBarStyle}>
          <button style={tabBtnStyle(tab === "users")}   onClick={() => setTab("users")}>👥 Users</button>
          <button style={tabBtnStyle(tab === "stats")}   onClick={() => setTab("stats")}>📊 Usage Stats</button>
          <button style={tabBtnStyle(tab === "profile")} onClick={() => setTab("profile")}>🔑 My Password</button>
        </div>

        <div style={{ padding: "1.5rem", overflowY: "auto", flex: 1 }}>
          {tab === "users"   && <UsersTab currentUser={currentUser} />}
          {tab === "stats"   && <StatsTab />}
          {tab === "profile" && <ChangePasswordTab />}
        </div>
      </div>
    </div>
  );
}

// ── Users Tab ────────────────────────────────────────────────────────────────

function UsersTab({ currentUser }) {
  const [users,      setUsers]      = useState([]);
  const [loading,    setLoading]    = useState(true);
  const [error,      setError]      = useState(null);
  const [showNew,    setShowNew]    = useState(false);
  const [resetInfo,  setResetInfo]  = useState(null);   // {email, reset_url, expires_at}
  const [accessUser, setAccessUser] = useState(null);   // user whose DB access is being edited

  // All databases the current admin can assign (admins have full access).
  const availableDatabases = currentUser.accessible_databases || [];

  const load = async () => {
    setLoading(true);
    try { setUsers(await adminApi.listUsers()); }
    catch { setError("Failed to load users."); }
    finally { setLoading(false); }
  };

  useEffect(() => { load(); }, []);

  const toggleActive = async (user) => {
    try {
      await adminApi.updateUser(user.id, { is_active: !user.is_active });
      load();
    } catch { alert("Update failed."); }
  };

  const generateReset = async (user) => {
    try {
      const data = await adminApi.generateReset(user.id);
      setResetInfo({ email: user.email, ...data });
    } catch { alert("Failed to generate reset link."); }
  };

  if (loading) return <p style={mutedStyle}>Loading…</p>;
  if (error)   return <p style={{ color: "#dc2626" }}>{error}</p>;

  return (
    <>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
        <h3 style={sectionTitleStyle}>Users ({users.length})</h3>
        <button style={primaryBtnStyle} onClick={() => setShowNew(true)}>+ New User</button>
      </div>

      {/* Password reset link modal */}
      {resetInfo && (
        <div style={modalStyle}>
          <div style={modalCardStyle}>
            <h4 style={{ margin: "0 0 0.75rem" }}>Reset Link for {resetInfo.email}</h4>
            <p style={{ fontSize: "0.8rem", color: "#6b7280", marginBottom: "0.5rem" }}>
              Expires: {new Date(resetInfo.expires_at).toLocaleString()}
            </p>
            <div style={resetUrlStyle}>{resetInfo.reset_url}</div>
            <div style={{ display: "flex", gap: "0.5rem", marginTop: "0.75rem" }}>
              <button style={primaryBtnStyle}
                onClick={() => navigator.clipboard.writeText(resetInfo.reset_url)}>
                Copy Link
              </button>
              <button style={secondaryBtnStyle} onClick={() => setResetInfo(null)}>Close</button>
            </div>
          </div>
        </div>
      )}

      {/* Database access editor modal */}
      {accessUser && (
        <ManageAccessModal
          user={accessUser}
          availableDatabases={availableDatabases}
          onSaved={() => { setAccessUser(null); load(); }}
          onCancel={() => setAccessUser(null)}
        />
      )}

      {/* New user form */}
      {showNew && (
        <NewUserForm
          availableDatabases={availableDatabases}
          onCreated={() => { setShowNew(false); load(); }}
          onCancel={() => setShowNew(false)}
        />
      )}

      {/* Users table */}
      <div style={{ overflowX: "auto" }}>
        <table style={tableStyle}>
          <thead>
            <tr>
              {["Email", "Name", "Role", "Status", "Databases", "Last Login", "Actions"].map((h) => (
                <th key={h} style={thStyle}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {users.map((u) => (
              <tr key={u.id} style={{ background: u.is_active ? "#fff" : "#f9fafb" }}>
                <td style={tdStyle}>{u.email}</td>
                <td style={tdStyle}>{u.full_name || "—"}</td>
                <td style={tdStyle}>
                  <span style={badgeStyle(u.is_admin ? "#dbeafe" : "#f3f4f6", u.is_admin ? "#1d4ed8" : "#374151")}>
                    {u.is_admin ? "Admin" : "User"}
                  </span>
                </td>
                <td style={tdStyle}>
                  <span style={badgeStyle(u.is_active ? "#dcfce7" : "#fee2e2", u.is_active ? "#15803d" : "#b91c1c")}>
                    {u.is_active ? "Active" : "Inactive"}
                  </span>
                </td>
                <td style={tdStyle}>
                  {u.is_admin ? (
                    <span style={badgeStyle("#fef3c7", "#92400e")}>All (admin)</span>
                  ) : (
                    <div style={{ display: "flex", flexWrap: "wrap", gap: "3px" }}>
                      {u.accessible_databases.length === 0
                        ? <span style={{ color: "#9ca3af", fontSize: "0.75rem" }}>None</span>
                        : u.accessible_databases.map((d) => (
                            <span key={d.db_name} style={badgeStyle("#e0f2fe", "#0369a1")}>
                              {d.label}
                            </span>
                          ))
                      }
                    </div>
                  )}
                </td>
                <td style={tdStyle}>
                  {u.last_login_at ? new Date(u.last_login_at).toLocaleDateString() : "Never"}
                </td>
                <td style={tdStyle}>
                  <div style={{ display: "flex", gap: "4px", flexWrap: "wrap" }}>
                    {u.id !== currentUser.user_id && (
                      <button style={smallBtnStyle} onClick={() => toggleActive(u)}>
                        {u.is_active ? "Disable" : "Enable"}
                      </button>
                    )}
                    <button style={smallBtnStyle} onClick={() => generateReset(u)}>Reset PW</button>
                    {!u.is_admin && (
                      <button style={{ ...smallBtnStyle, color: "#2563eb", borderColor: "#bfdbfe" }}
                        onClick={() => setAccessUser(u)}>
                        DB Access
                      </button>
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

// ── Manage DB Access Modal ────────────────────────────────────────────────────

function ManageAccessModal({ user, availableDatabases, onSaved, onCancel }) {
  // Initialize checkboxes from the user's current access.
  const currentDbNames = user.accessible_databases.map((d) => d.db_name);
  const [selected, setSelected] = useState(new Set(currentDbNames));
  const [saving,   setSaving]   = useState(false);
  const [error,    setError]    = useState(null);

  const toggle = (dbName) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(dbName)) next.delete(dbName);
      else next.add(dbName);
      return next;
    });
  };

  const save = async () => {
    setSaving(true); setError(null);
    try {
      await adminApi.updateAccess(user.id, Array.from(selected));
      onSaved();
    } catch (err) {
      setError(err.response?.data?.detail || "Save failed.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div style={modalStyle}>
      <div style={modalCardStyle}>
        <h4 style={{ margin: "0 0 0.25rem" }}>Database Access</h4>
        <p style={{ fontSize: "0.8rem", color: "#6b7280", margin: "0 0 1rem" }}>
          {user.email}
        </p>

        <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", marginBottom: "1rem" }}>
          {availableDatabases.map((d) => (
            <label key={d.db_name}
              style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "0.88rem", cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={selected.has(d.db_name)}
                onChange={() => toggle(d.db_name)}
              />
              <span style={{ fontWeight: 600 }}>{d.label}</span>
              <span style={{ color: "#6b7280", fontSize: "0.78rem" }}>({d.db_name})</span>
            </label>
          ))}
        </div>

        {error && <p style={{ color: "#dc2626", fontSize: "0.8rem", margin: "0 0 0.75rem" }}>{error}</p>}

        <div style={{ display: "flex", gap: "0.5rem" }}>
          <button style={primaryBtnStyle} onClick={save} disabled={saving}>
            {saving ? "Saving…" : "Save"}
          </button>
          <button style={secondaryBtnStyle} onClick={onCancel}>Cancel</button>
        </div>
      </div>
    </div>
  );
}

// ── New User Form ─────────────────────────────────────────────────────────────

function NewUserForm({ availableDatabases, onCreated, onCancel }) {
  const [form,     setForm]     = useState({ email: "", full_name: "", password: "", is_admin: false });
  const [selected, setSelected] = useState(new Set());
  const [error,    setError]    = useState(null);
  const [saving,   setSaving]   = useState(false);

  const toggleDb = (dbName) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(dbName)) next.delete(dbName);
      else next.add(dbName);
      return next;
    });
  };

  const submit = async (e) => {
    e.preventDefault();
    setSaving(true); setError(null);
    try {
      await adminApi.createUser({
        ...form,
        database_access: form.is_admin ? [] : Array.from(selected),
      });
      onCreated();
    } catch (err) {
      setError(err.response?.data?.detail || "Creation failed.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div style={{ background: "#f8fafc", border: "1px solid #e2e8f0", borderRadius: "8px", padding: "1rem", marginBottom: "1rem" }}>
      <h4 style={{ margin: "0 0 0.75rem", fontSize: "0.95rem" }}>New User</h4>
      <form onSubmit={submit}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem" }}>
          <div>
            <label style={labelStyle}>Email *</label>
            <input type="email" required style={inputStyle}
              value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} />
          </div>
          <div>
            <label style={labelStyle}>Full Name</label>
            <input type="text" style={inputStyle}
              value={form.full_name} onChange={(e) => setForm({ ...form, full_name: e.target.value })} />
          </div>
          <div>
            <label style={labelStyle}>Password * (min 8 chars)</label>
            <input type="password" required minLength={8} style={inputStyle}
              value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} />
          </div>
          <div style={{ display: "flex", alignItems: "flex-end", paddingBottom: "2px" }}>
            <label style={{ fontSize: "0.82rem", cursor: "pointer", display: "flex", alignItems: "center", gap: "6px" }}>
              <input type="checkbox" checked={form.is_admin}
                onChange={(e) => setForm({ ...form, is_admin: e.target.checked })} />
              Admin
            </label>
          </div>
        </div>

        {/* Database access section — hidden for admin users */}
        {!form.is_admin && (
          <div style={{ marginTop: "0.75rem" }}>
            <label style={{ ...labelStyle, marginBottom: "0.4rem" }}>Database Access</label>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.6rem" }}>
              {availableDatabases.map((d) => (
                <label key={d.db_name}
                  style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "0.82rem", cursor: "pointer" }}>
                  <input
                    type="checkbox"
                    checked={selected.has(d.db_name)}
                    onChange={() => toggleDb(d.db_name)}
                  />
                  {d.label}
                </label>
              ))}
            </div>
          </div>
        )}
        {form.is_admin && (
          <p style={{ fontSize: "0.78rem", color: "#6b7280", marginTop: "0.5rem" }}>
            Admin users have access to all databases automatically.
          </p>
        )}

        {error && <p style={{ color: "#dc2626", fontSize: "0.8rem", margin: "0.5rem 0 0" }}>{error}</p>}
        <div style={{ display: "flex", gap: "0.5rem", marginTop: "0.75rem" }}>
          <button type="submit" disabled={saving} style={primaryBtnStyle}>
            {saving ? "Creating…" : "Create User"}
          </button>
          <button type="button" style={secondaryBtnStyle} onClick={onCancel}>Cancel</button>
        </div>
      </form>
    </div>
  );
}

// ── Stats Tab ─────────────────────────────────────────────────────────────────

function StatsTab() {
  const [stats,   setStats]   = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    adminApi.getStats()
      .then(setStats)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <p style={mutedStyle}>Loading statistics…</p>;
  if (!stats)  return <p style={{ color: "#dc2626" }}>Failed to load statistics.</p>;

  const { overview, user_activity, endpoint_usage, daily_trend } = stats;
  const maxReqs = Math.max(...user_activity.map((u) => u.total_requests), 1);
  const maxEp   = Math.max(...endpoint_usage.map((e) => e.total), 1);

  return (
    <>
      {/* Overview cards */}
      <h3 style={sectionTitleStyle}>Overview</h3>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "0.75rem", marginBottom: "1.5rem" }}>
        {[
          ["Total Users",        overview.total_users],
          ["Active Users",       overview.active_users],
          ["Requests Today",     overview.requests_today],
          ["Requests This Week", overview.requests_this_week],
          ["Requests Total",     overview.requests_total],
          ["Avg Duration (wk)",  `${overview.avg_duration_ms_week} ms`],
        ].map(([label, value]) => (
          <div key={label} style={statCardStyle}>
            <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "#1d4ed8" }}>{value}</div>
            <div style={{ fontSize: "0.75rem", color: "#6b7280", marginTop: "2px" }}>{label}</div>
          </div>
        ))}
      </div>

      {/* User activity */}
      <h3 style={sectionTitleStyle}>User Activity (last 30 days)</h3>
      <table style={tableStyle}>
        <thead>
          <tr>
            {["User", "Requests", "Activity", "Avg Duration", "Last Active"].map((h) => (
              <th key={h} style={thStyle}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {user_activity.map((u) => (
            <tr key={u.user_id}>
              <td style={tdStyle}>
                <div style={{ fontSize: "0.85rem" }}>{u.email}</div>
                {u.full_name && <div style={{ fontSize: "0.75rem", color: "#6b7280" }}>{u.full_name}</div>}
              </td>
              <td style={tdStyle}>{u.total_requests}</td>
              <td style={{ ...tdStyle, minWidth: "120px" }}>
                <div style={{ background: "#e5e7eb", borderRadius: "4px", height: "10px", width: "100%" }}>
                  <div style={{
                    background: "#2563eb",
                    height: "10px",
                    borderRadius: "4px",
                    width: `${Math.round((u.total_requests / maxReqs) * 100)}%`,
                  }} />
                </div>
              </td>
              <td style={tdStyle}>{u.avg_duration_ms > 0 ? `${u.avg_duration_ms} ms` : "—"}</td>
              <td style={tdStyle}>{u.last_active ? new Date(u.last_active).toLocaleDateString() : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* Endpoint usage */}
      <h3 style={{ ...sectionTitleStyle, marginTop: "1.5rem" }}>Top Endpoints (last 30 days)</h3>
      <table style={tableStyle}>
        <thead>
          <tr>
            {["Method", "Endpoint", "Requests", "Usage", "Avg ms", "Max ms"].map((h) => (
              <th key={h} style={thStyle}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {endpoint_usage.map((e, i) => (
            <tr key={i}>
              <td style={tdStyle}><span style={methodBadge(e.method)}>{e.method}</span></td>
              <td style={{ ...tdStyle, fontFamily: "monospace", fontSize: "0.8rem" }}>{e.endpoint}</td>
              <td style={tdStyle}>{e.total}</td>
              <td style={{ ...tdStyle, minWidth: "100px" }}>
                <div style={{ background: "#e5e7eb", borderRadius: "4px", height: "8px" }}>
                  <div style={{
                    background: "#7c3aed",
                    height: "8px",
                    borderRadius: "4px",
                    width: `${Math.round((e.total / maxEp) * 100)}%`,
                  }} />
                </div>
              </td>
              <td style={tdStyle}>{e.avg_ms}</td>
              <td style={tdStyle}>{e.max_ms}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* Daily trend */}
      <h3 style={{ ...sectionTitleStyle, marginTop: "1.5rem" }}>Daily Requests (last 14 days)</h3>
      <table style={tableStyle}>
        <thead>
          <tr>
            {["Date", "Requests", "Trend", "Unique Users"].map((h) => (
              <th key={h} style={thStyle}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {(() => {
            const maxDay = Math.max(...daily_trend.map((d) => d.requests), 1);
            return daily_trend.map((d) => (
              <tr key={d.day}>
                <td style={tdStyle}>{d.day}</td>
                <td style={tdStyle}>{d.requests}</td>
                <td style={{ ...tdStyle, minWidth: "120px" }}>
                  <div style={{ background: "#e5e7eb", borderRadius: "4px", height: "10px" }}>
                    <div style={{
                      background: "#16a34a",
                      height: "10px",
                      borderRadius: "4px",
                      width: `${Math.round((d.requests / maxDay) * 100)}%`,
                    }} />
                  </div>
                </td>
                <td style={tdStyle}>{d.unique_users}</td>
              </tr>
            ));
          })()}
        </tbody>
      </table>
    </>
  );
}

// ── Change Password Tab ───────────────────────────────────────────────────────

function ChangePasswordTab() {
  const [form,    setForm]    = useState({ current_password: "", new_password: "", confirm: "" });
  const [error,   setError]   = useState(null);
  const [success, setSuccess] = useState(false);
  const [saving,  setSaving]  = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    if (form.new_password !== form.confirm) { setError("Passwords do not match."); return; }
    setSaving(true); setError(null);
    try {
      await authApi.changePassword({
        current_password: form.current_password,
        new_password:     form.new_password,
      });
      setSuccess(true);
      setForm({ current_password: "", new_password: "", confirm: "" });
    } catch (err) {
      setError(err.response?.data?.detail || "Password change failed.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div style={{ maxWidth: "380px" }}>
      <h3 style={sectionTitleStyle}>Change Password</h3>
      {success && <p style={{ color: "#16a34a", marginBottom: "1rem" }}>Password changed successfully.</p>}
      <form onSubmit={submit}>
        {[
          ["Current password",        "current_password", "password"],
          ["New password (min 8 chars)", "new_password",  "password"],
          ["Confirm new password",    "confirm",          "password"],
        ].map(([label, key, type]) => (
          <div key={key} style={{ marginBottom: "0.75rem" }}>
            <label style={labelStyle}>{label}</label>
            <input type={type} required minLength={key !== "current_password" ? 8 : 1}
              style={inputStyle} value={form[key]}
              onChange={(e) => setForm({ ...form, [key]: e.target.value })} />
          </div>
        ))}
        {error && <p style={{ color: "#dc2626", fontSize: "0.82rem" }}>{error}</p>}
        <button type="submit" disabled={saving}
          style={{ ...primaryBtnStyle, width: "auto", marginTop: "0.5rem" }}>
          {saving ? "Saving…" : "Change Password"}
        </button>
      </form>
    </div>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const overlayStyle = {
  position: "fixed", inset: 0, zIndex: 9000,
  background: "rgba(0,0,0,0.45)",
  display: "flex", alignItems: "stretch",
};

const panelStyle = {
  background: "#fff",
  width: "100%", maxWidth: "1100px",
  margin: "0 auto",
  display: "flex", flexDirection: "column",
  overflowY: "auto",
};

const headerStyle = {
  display: "flex", justifyContent: "space-between", alignItems: "center",
  padding: "0.85rem 1.5rem",
  borderBottom: "1px solid #e5e7eb",
  background: "#fff",
  position: "sticky", top: 0, zIndex: 1,
};

const tabBarStyle = {
  display: "flex", gap: "2px",
  padding: "0 1.5rem",
  borderBottom: "1px solid #e5e7eb",
  background: "#f8fafc",
};

const tabBtnStyle = (active) => ({
  padding: "0.6rem 1rem",
  background: "none",
  border: "none",
  borderBottom: active ? "2px solid #2563eb" : "2px solid transparent",
  color: active ? "#2563eb" : "#6b7280",
  fontWeight: active ? 700 : 500,
  fontSize: "0.85rem",
  cursor: "pointer",
  marginBottom: "-1px",
});

const closeBtnStyle = {
  padding: "5px 12px",
  background: "#f3f4f6", border: "1px solid #d1d5db",
  borderRadius: "6px", fontSize: "0.82rem", cursor: "pointer",
};

const primaryBtnStyle = {
  padding: "7px 14px",
  background: "#2563eb", color: "#fff",
  border: "none", borderRadius: "6px",
  fontSize: "0.82rem", fontWeight: 600, cursor: "pointer",
};

const secondaryBtnStyle = {
  padding: "7px 14px",
  background: "#f3f4f6", color: "#374151",
  border: "1px solid #d1d5db", borderRadius: "6px",
  fontSize: "0.82rem", cursor: "pointer",
};

const smallBtnStyle = {
  padding: "3px 8px",
  background: "#f3f4f6", color: "#374151",
  border: "1px solid #d1d5db", borderRadius: "4px",
  fontSize: "0.75rem", cursor: "pointer", whiteSpace: "nowrap",
};

const sectionTitleStyle = {
  fontSize: "0.95rem", fontWeight: 700, color: "#111827", margin: "0 0 0.75rem",
};

const tableStyle = {
  width: "100%", borderCollapse: "collapse",
  fontSize: "0.82rem",
};

const thStyle = {
  textAlign: "left",
  padding: "6px 10px",
  background: "#f8fafc",
  borderBottom: "1px solid #e5e7eb",
  fontWeight: 600, color: "#374151",
  whiteSpace: "nowrap",
};

const tdStyle = {
  padding: "7px 10px",
  borderBottom: "1px solid #f3f4f6",
  color: "#111827",
};

const labelStyle = {
  display: "block", fontSize: "0.8rem", fontWeight: 600,
  color: "#374151", marginBottom: "3px",
};

const inputStyle = {
  width: "100%", padding: "7px 10px",
  border: "1px solid #d1d5db", borderRadius: "6px",
  fontSize: "0.85rem", boxSizing: "border-box",
};

const statCardStyle = {
  background: "#f8fafc", border: "1px solid #e5e7eb",
  borderRadius: "8px", padding: "0.85rem 1rem",
};

const mutedStyle = { color: "#6b7280", fontSize: "0.85rem" };

const badgeStyle = (bg, color) => ({
  display: "inline-block", padding: "2px 7px",
  background: bg, color, borderRadius: "10px",
  fontSize: "0.73rem", fontWeight: 600,
});

const methodBadge = (method) => {
  const colors = {
    GET:    ["#dcfce7", "#15803d"],
    POST:   ["#dbeafe", "#1d4ed8"],
    DELETE: ["#fee2e2", "#b91c1c"],
    PUT:    ["#fef3c7", "#92400e"],
  };
  const [bg, color] = colors[method] || ["#f3f4f6", "#374151"];
  return badgeStyle(bg, color);
};

const modalStyle = {
  position: "fixed", inset: 0, zIndex: 9100,
  background: "rgba(0,0,0,0.4)",
  display: "flex", alignItems: "center", justifyContent: "center",
};

const modalCardStyle = {
  background: "#fff", borderRadius: "10px",
  padding: "1.5rem", maxWidth: "520px", width: "100%",
  boxShadow: "0 8px 32px rgba(0,0,0,0.2)",
};

const resetUrlStyle = {
  background: "#f1f5f9", border: "1px solid #e2e8f0",
  borderRadius: "6px", padding: "8px 10px",
  fontFamily: "monospace", fontSize: "0.78rem",
  wordBreak: "break-all", color: "#374151",
};

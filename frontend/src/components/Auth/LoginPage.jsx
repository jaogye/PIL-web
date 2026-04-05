/**
 * Login page — shown when no valid JWT is stored.
 * Also handles the password-reset flow (?reset_token=...).
 */

import { useState } from "react";
import { authApi, tokenStore } from "../../services/api";

export default function LoginPage({ onLogin, resetToken = null }) {
  const isReset = !!resetToken;

  const [email,    setEmail]    = useState("");
  const [password, setPassword] = useState("");
  const [newPw,    setNewPw]    = useState("");
  const [confirmPw,setConfirmPw]= useState("");
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState(null);
  const [success,  setSuccess]  = useState(null);

  // ── Login ────────────────────────────────────────────────────────────
  const handleLogin = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const data = await authApi.login(email, password);
      tokenStore.set(data.access_token);
      onLogin(data);
    } catch (err) {
      setError(err.response?.data?.detail || "Login failed. Check your credentials.");
    } finally {
      setLoading(false);
    }
  };

  // ── Reset password confirm ────────────────────────────────────────────
  const handleReset = async (e) => {
    e.preventDefault();
    if (newPw !== confirmPw) { setError("Passwords do not match."); return; }
    setError(null);
    setLoading(true);
    try {
      await authApi.confirmReset({ token: resetToken, new_password: newPw });
      setSuccess("Password updated successfully. You can now log in.");
    } catch (err) {
      setError(err.response?.data?.detail || "Reset failed. The link may have expired.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={pageStyle}>
      <div style={cardStyle}>
        {/* Logo / title */}
        <div style={{ textAlign: "center", marginBottom: "2rem" }}>
          <div style={{ fontSize: "2rem", fontWeight: 800, color: "#1d4ed8", letterSpacing: "-0.5px" }}>
            PIL
          </div>
          <div style={{ fontSize: "0.85rem", color: "#6b7280", marginTop: "4px" }}>
            Public Infrastructure Locator
          </div>
        </div>

        {/* ── Reset password form ── */}
        {isReset && !success && (
          <>
            <h2 style={headingStyle}>Set New Password</h2>
            <form onSubmit={handleReset}>
              <label style={labelStyle}>New password</label>
              <input
                type="password"
                value={newPw}
                onChange={(e) => setNewPw(e.target.value)}
                required minLength={8}
                style={inputStyle}
                placeholder="Min. 8 characters"
              />
              <label style={labelStyle}>Confirm password</label>
              <input
                type="password"
                value={confirmPw}
                onChange={(e) => setConfirmPw(e.target.value)}
                required minLength={8}
                style={inputStyle}
              />
              {error   && <p style={errorStyle}>{error}</p>}
              <button type="submit" disabled={loading} style={btnStyle}>
                {loading ? "Saving…" : "Set Password"}
              </button>
            </form>
          </>
        )}

        {/* ── Reset success ── */}
        {isReset && success && (
          <div style={{ textAlign: "center" }}>
            <p style={{ color: "#16a34a", marginBottom: "1.5rem" }}>{success}</p>
            <button onClick={() => window.location.replace("/")} style={btnStyle}>
              Go to Login
            </button>
          </div>
        )}

        {/* ── Login form ── */}
        {!isReset && (
          <>
            <h2 style={headingStyle}>Sign In</h2>
            <form onSubmit={handleLogin}>
              <label style={labelStyle}>Email</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required autoFocus
                style={inputStyle}
                placeholder="you@example.com"
              />
              <label style={labelStyle}>Password</label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                style={inputStyle}
              />
              {error && <p style={errorStyle}>{error}</p>}
              <button type="submit" disabled={loading} style={btnStyle}>
                {loading ? "Signing in…" : "Sign In"}
              </button>
            </form>
            <p style={{ fontSize: "0.78rem", color: "#9ca3af", textAlign: "center", marginTop: "1.25rem" }}>
              Forgot your password? Contact your administrator.
            </p>
          </>
        )}
      </div>
    </div>
  );
}

// ── Styles ──────────────────────────────────────────────────────────────────

const pageStyle = {
  minHeight: "100vh",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  background: "linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%)",
};

const cardStyle = {
  background: "#ffffff",
  borderRadius: "12px",
  boxShadow: "0 8px 32px rgba(0,0,0,0.12)",
  padding: "2.5rem 2rem",
  width: "100%",
  maxWidth: "380px",
};

const headingStyle = {
  fontSize: "1.15rem",
  fontWeight: 700,
  color: "#111827",
  margin: "0 0 1.25rem",
};

const labelStyle = {
  display: "block",
  fontSize: "0.82rem",
  fontWeight: 600,
  color: "#374151",
  marginBottom: "4px",
  marginTop: "0.85rem",
};

const inputStyle = {
  width: "100%",
  padding: "9px 12px",
  border: "1px solid #d1d5db",
  borderRadius: "6px",
  fontSize: "0.9rem",
  outline: "none",
  boxSizing: "border-box",
};

const btnStyle = {
  marginTop: "1.5rem",
  width: "100%",
  padding: "10px",
  background: "#1d4ed8",
  color: "#fff",
  border: "none",
  borderRadius: "6px",
  fontSize: "0.92rem",
  fontWeight: 600,
  cursor: "pointer",
};

const errorStyle = {
  color: "#dc2626",
  fontSize: "0.82rem",
  marginTop: "0.5rem",
};

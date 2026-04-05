/**
 * API client for the LIP2 FastAPI backend.
 * All methods return the parsed JSON response body.
 */

import axios from "axios";

// ── Token helpers ───────────────────────────────────────────────────────────

const TOKEN_KEY = "lip2_access_token";

export const tokenStore = {
  get:    ()      => localStorage.getItem(TOKEN_KEY),
  set:    (token) => localStorage.setItem(TOKEN_KEY, token),
  clear:  ()      => localStorage.removeItem(TOKEN_KEY),
};

// ── Axios client ────────────────────────────────────────────────────────────

const client = axios.create({
  baseURL: "/api/v1",
  headers: {
    "Content-Type": "application/json",
    "X-LIP2-Database": "lip2_ecuador",
  },
});

/** Attach the stored JWT to every request. */
client.interceptors.request.use((config) => {
  const token = tokenStore.get();
  if (token) config.headers["Authorization"] = `Bearer ${token}`;
  return config;
});

/** On 401: clear token and reload (forces login page). */
client.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      tokenStore.clear();
      window.location.reload();
    }
    return Promise.reject(err);
  }
);

/** Switch all subsequent API calls to a different database. */
export function setDatabase(dbName) {
  client.defaults.headers["X-LIP2-Database"] = dbName;
}

// ── Auth ────────────────────────────────────────────────────────────────────

export const authApi = {
  login: (email, password) => {
    // OAuth2PasswordRequestForm expects form-encoded body.
    const form = new URLSearchParams();
    form.append("username", email);
    form.append("password", password);
    return axios.post("/api/v1/auth/login", form, {
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
    }).then((r) => r.data);
  },
  me:             ()      => client.get("/auth/me").then((r) => r.data),
  changePassword: (body)  => client.put("/auth/me/password", body),
  confirmReset:   (body)  => axios.post("/api/v1/auth/reset-password/confirm", body).then((r) => r.data),
};

// ── Admin ───────────────────────────────────────────────────────────────────

export const adminApi = {
  listUsers:      ()         => client.get("/admin/users").then((r) => r.data),
  createUser:     (body)     => client.post("/admin/users", body).then((r) => r.data),
  updateUser:     (id, body) => client.put(`/admin/users/${id}`, body).then((r) => r.data),
  // Update only the database access list for a user.
  updateAccess:   (id, dbs)  => client.put(`/admin/users/${id}`, { database_access: dbs }).then((r) => r.data),
  generateReset:  (id)       => client.post(`/admin/users/${id}/reset`).then((r) => r.data),
  getStats:       ()         => client.get("/admin/stats").then((r) => r.data),
};

// ── Databases ─────────────────────────────────────────────────────────────
// Requires authentication — returns only the databases the current user can access.

export const databasesApi = {
  list: () => client.get("/databases").then((r) => r.data),
};

// ── Optimization ──────────────────────────────────────────────────────────

export const optimizationApi = {
  run:         (payload)       => client.post("/optimization/run", payload).then((r) => r.data),
  list:        ()              => client.get("/optimization/").then((r) => r.data),
  get:         (id)            => client.get(`/optimization/${id}`).then((r) => r.data),
  getWithSignal:(id, signal)   => client.get(`/optimization/${id}`, { signal }).then((r) => r.data),
  delete:      (id)            => client.delete(`/optimization/${id}`),
  rebalance:   (id, params)    => client.post(`/optimization/${id}/rebalance`, params).then((r) => r.data),
};

// ── Infrastructure ────────────────────────────────────────────────────────

export const infrastructureApi = {
  list:   (params)        => client.get("/infrastructure/", { params }).then((r) => r.data),
  create: (payload)       => client.post("/infrastructure/", payload).then((r) => r.data),
  get:    (id)            => client.get(`/infrastructure/${id}`).then((r) => r.data),
  update: (id, payload)   => client.put(`/infrastructure/${id}`, payload).then((r) => r.data),
  delete: (id)            => client.delete(`/infrastructure/${id}`),
};

// ── Impacts ───────────────────────────────────────────────────────────────

export const impactsApi = {
  calculate: (payload) => client.post("/impacts/calculate", payload).then((r) => r.data),
};

// ── Political Divisions ───────────────────────────────────────────────────

export const politicalDivisionsApi = {
  tree:          ()     => client.get("/political-divisions/tree").then((r) => r.data),
  censusSummary: (body) => client.post("/political-divisions/census-summary", body).then((r) => r.data),
};

// ── Target Populations ────────────────────────────────────────────────────

export const targetPopulationsApi = {
  list:          () => client.get("/target-populations/").then((r) => r.data),
  facilityTypes: () => client.get("/target-populations/facility-types").then((r) => r.data),
};

// ── Reports ───────────────────────────────────────────────────────────────

export const reportsApi = {
  /**
   * Download report as a blob using fetch so the Authorization header
   * is included (plain <a href> links cannot send headers).
   */
  downloadExcel: async (scenarioId, db) => {
    const token = tokenStore.get();
    const url   = `/api/v1/reports/scenario/${scenarioId}/excel${db ? `?db=${db}` : ""}`;
    const res   = await fetch(url, {
      headers: {
        Authorization:    token ? `Bearer ${token}` : "",
        "X-LIP2-Database": db || "lip2_ecuador",
      },
    });
    if (!res.ok) throw new Error("Download failed");
    const blob = await res.blob();
    const a    = document.createElement("a");
    a.href     = URL.createObjectURL(blob);
    a.download = `lip2_scenario_${scenarioId}.xlsx`;
    a.click();
    URL.revokeObjectURL(a.href);
  },
  downloadJson: async (scenarioId, db) => {
    const token = tokenStore.get();
    const url   = `/api/v1/reports/scenario/${scenarioId}/json${db ? `?db=${db}` : ""}`;
    const res   = await fetch(url, {
      headers: {
        Authorization:    token ? `Bearer ${token}` : "",
        "X-LIP2-Database": db || "lip2_ecuador",
      },
    });
    if (!res.ok) throw new Error("Download failed");
    const blob = await res.blob();
    const a    = document.createElement("a");
    a.href     = URL.createObjectURL(blob);
    a.download = `lip2_scenario_${scenarioId}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  },
};

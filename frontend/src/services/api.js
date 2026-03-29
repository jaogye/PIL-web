/**
 * API client for the LIP2 FastAPI backend.
 * All methods return the parsed JSON response body.
 */

import axios from "axios";

const client = axios.create({
  baseURL: "/api/v1",
  headers: {
    "Content-Type": "application/json",
    "X-LIP2-Database": "lip2_ecuador",   // default; changed via setDatabase()
  },
});

/** Switch all subsequent API calls to a different database. */
export function setDatabase(dbName) {
  client.defaults.headers["X-LIP2-Database"] = dbName;
}

// ─── Databases ─────────────────────────────────────────────────────────────

export const databasesApi = {
  list: () => axios.get("/api/v1/databases").then((r) => r.data),
};

// ─── Optimization ──────────────────────────────────────────────────────────

export const optimizationApi = {
  /** Submit and run an optimization scenario. */
  run: (payload) => client.post("/optimization/run", payload).then((r) => r.data),

  /** List all scenarios (summary). */
  list: () => client.get("/optimization/").then((r) => r.data),

  /** Get a scenario with full facility locations. */
  get: (id) => client.get(`/optimization/${id}`).then((r) => r.data),

  /** Get a scenario with an AbortSignal for cancellation. */
  getWithSignal: (id, signal) =>
    client.get(`/optimization/${id}`, { signal }).then((r) => r.data),

  /** Delete a scenario. */
  delete: (id) => client.delete(`/optimization/${id}`),

  /** Run the capacity rebalancing algorithm on a completed scenario. */
  rebalance: (id, params) =>
    client.post(`/optimization/${id}/rebalance`, params).then((r) => r.data),
};

// ─── Infrastructure ────────────────────────────────────────────────────────

export const infrastructureApi = {
  list: (params) => client.get("/infrastructure/", { params }).then((r) => r.data),
  create: (payload) => client.post("/infrastructure/", payload).then((r) => r.data),
  get: (id) => client.get(`/infrastructure/${id}`).then((r) => r.data),
  update: (id, payload) => client.put(`/infrastructure/${id}`, payload).then((r) => r.data),
  delete: (id) => client.delete(`/infrastructure/${id}`),
};

// ─── Impacts ───────────────────────────────────────────────────────────────

export const impactsApi = {
  calculate: (payload) => client.post("/impacts/calculate", payload).then((r) => r.data),
};

// ─── Political Divisions ───────────────────────────────────────────────────

export const politicalDivisionsApi = {
  /** Full provincia > canton > parroquia tree. */
  tree: () => client.get("/political-divisions/tree").then((r) => r.data),

  /** Count and population for a set of parroquia IDs and a target population group. */
  censusSummary: ({ parish_ids, target_population_id = null }) =>
    client.post("/political-divisions/census-summary", { parish_ids, target_population_id }).then((r) => r.data),
};

// ─── Target Populations ────────────────────────────────────────────────────

export const targetPopulationsApi = {
  /** List all demographic target population groups. */
  list: () => client.get("/target-populations/").then((r) => r.data),
  /** List facility types with their default target_population_id. */
  facilityTypes: () => client.get("/target-populations/facility-types").then((r) => r.data),
};

// ─── Reports ───────────────────────────────────────────────────────────────

export const reportsApi = {
  /** Returns a URL to trigger the Excel download directly in the browser. */
  excelUrl: (scenarioId) => `/api/v1/reports/scenario/${scenarioId}/excel`,
  jsonUrl: (scenarioId) => `/api/v1/reports/scenario/${scenarioId}/json`,
};

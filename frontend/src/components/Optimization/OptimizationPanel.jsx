/**
 * Optimization control panel.
 *
 * Allows the user to:
 *  1. Choose a model (p-median, p-center, max-coverage).
 *  2. Set parameters (p, radius, scope filters, capacity).
 *  3. Submit the job (async), poll every 30 s, and view results.
 *  4. Load a previous scenario from localStorage or a JSON file.
 */

import { useState, useEffect, useRef } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { optimizationApi, reportsApi, targetPopulationsApi } from "../../services/api";
import PoliticalDivisionTree from "./PoliticalDivisionTree";

// ─── localStorage helpers ────────────────────────────────────────────────────

const LS_INDEX_KEY = "lip2_scenarios";
const MAX_STORED    = 20;

function lsGetIndex() {
  try { return JSON.parse(localStorage.getItem(LS_INDEX_KEY) || "[]"); }
  catch { return []; }
}

function lsSave(id, data, facilityType, mode, payload = null) {
  try {
    localStorage.setItem(`lip2_scenario_${id}`, JSON.stringify({
      ...data,
      _facilityType: facilityType,
      _mode: mode,
      _payload: payload,
    }));
    const index = lsGetIndex();
    const entry = {
      id,
      name: data.name,
      model_type: data.model_type,
      p_facilities: data.p_facilities,
      stored_at: new Date().toISOString(),
    };
    const existing = index.findIndex((s) => s.id === id);
    if (existing >= 0) index[existing] = entry;
    else index.unshift(entry);
    if (index.length > MAX_STORED) {
      index.splice(MAX_STORED).forEach((s) =>
        localStorage.removeItem(`lip2_scenario_${s.id}`)
      );
    }
    localStorage.setItem(LS_INDEX_KEY, JSON.stringify(index));
  } catch (_) { /* quota exceeded – ignore */ }
}

function lsLoad(id) {
  try { return JSON.parse(localStorage.getItem(`lip2_scenario_${id}`)); }
  catch { return null; }
}

// ─── Default name builder ────────────────────────────────────────────────────

const MODEL_SHORT    = { p_median: "PMedian", p_center: "PCenter", max_coverage: "MaxCov" };
const FACILITY_SHORT = {
  school: "School", high_school: "HighSchool",
  health_center: "HealthCenter", hospital: "Hospital", other: "Other",
};

function buildDefaultName(model_type, facility_type, min_capacity, max_capacity, scopeLabel) {
  return [
    MODEL_SHORT[model_type] || model_type,
    FACILITY_SHORT[facility_type] || facility_type,
    min_capacity !== "" ? String(min_capacity) : "0",
    max_capacity !== "" ? String(max_capacity) : "inf",
    scopeLabel || "National",
  ].join("_");
}

function formatTime(sec) {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

// ─── Constants ────────────────────────────────────────────────────────────────

const MODEL_OPTIONS = [
  { value: "p_median",     label: "P-Median",          description: "Minimise total weighted travel time (efficiency)." },
  { value: "p_center",     label: "P-Center",          description: "Minimise the maximum travel time (equity)." },
  { value: "max_coverage", label: "Maximum Coverage",  description: "Maximise population covered within a service radius." },
  { value: "bump_hunter",  label: "Bump Hunter",       description: "Find census areas that are local demand-density peaks — candidate facility sites." },
];

const FACILITY_TYPE_OPTIONS = [
  { value: "school",        label: "School" },
  { value: "high_school",   label: "High School" },
  { value: "health_center", label: "Health Center" },
  { value: "hospital",      label: "Hospital" },
  { value: "nursery",       label: "Nursery (Guardería)" },
  { value: "other",         label: "Other" },
];

// ─── Component ───────────────────────────────────────────────────────────────

export default function OptimizationPanel({ onResultsReady, onRebalancingResult, selectedDb = "lip2_ecuador" }) {
  const queryClient = useQueryClient();

  const [form, setForm] = useState({
    name:                 "",
    model_type:           "p_median",
    p_facilities:         5,
    service_radius:       30,
    mode:                 "from_scratch",
    facility_type:        "high_school",
    target_population_id: null,
    min_capacity:         "",
    max_capacity:         "",
    k_neighbors:          "10",   // bump_hunter neighbourhood size for local-maxima detection
    k_vec:                "100",  // bump_hunter neighbours used in gravity score
  });

  const [scopeFilters, setScopeFilters]     = useState(null);
  const [scopeLabel,   setScopeLabel]       = useState(null);
  const [nameIsManual, setNameIsManual]     = useState(false);
  const [activeScenarioId, setActiveScenarioId] = useState(null);

  // ── Run state ──
  const [runStatus, setRunStatus]   = useState("idle"); // idle | running | completed | failed | cancelled
  const [elapsed,   setElapsed]     = useState(0);
  const [runError,  setRunError]    = useState(null);
  const [runData,   setRunData]     = useState(null);   // last completed response

  // Active scenario data — set from runs AND from loading existing scenarios.
  const [activeScenarioData, setActiveScenarioData] = useState(null);

  // ── Rebalancing state ──
  const [rebalForm,   setRebalForm]   = useState({ capacity_per_facility: "", min_capacity: 0, max_transfers: 20 });
  const [rebalStatus, setRebalStatus] = useState("idle"); // idle | running | completed | failed
  const [rebalError,  setRebalError]  = useState(null);
  const [rebalResult, setRebalResult] = useState(null);
  const [showRebal,   setShowRebal]   = useState(false);

  const elapsedRef      = useRef(null);   // setInterval id
  const pollRef         = useRef(null);   // setTimeout id
  const runningId       = useRef(null);   // scenario_id being polled
  const currentPayload  = useRef(null);   // last submitted payload (for reoptimization)

  // File picker
  const fileInputRef = useRef(null);

  // ── Auto-name ──
  const { model_type, facility_type, min_capacity, max_capacity } = form;
  useEffect(() => {
    if (nameIsManual) return;
    setForm((f) => ({
      ...f,
      name: buildDefaultName(model_type, facility_type, min_capacity, max_capacity, scopeLabel),
    }));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model_type, facility_type, min_capacity, max_capacity, scopeLabel]);

  // ── Elapsed timer (runs every second while status === 'running') ──
  useEffect(() => {
    if (runStatus !== "running") {
      clearInterval(elapsedRef.current);
      return;
    }
    const start = Date.now();
    setElapsed(0);
    elapsedRef.current = setInterval(() => {
      setElapsed(Math.floor((Date.now() - start) / 1000));
    }, 1000);
    return () => clearInterval(elapsedRef.current);
  }, [runStatus]);

  // ── Cleanup on unmount ──
  useEffect(() => () => {
    clearInterval(elapsedRef.current);
    clearTimeout(pollRef.current);
  }, []);

  // ── Polling logic ──
  const handleSuccess = (data) => {
    const meta = data.stats?._meta || {};
    const ft   = meta.facility_type || form.facility_type;
    const mode = meta.mode          || form.mode;
    lsSave(data.scenario_id, data, ft, mode, currentPayload.current);
    queryClient.invalidateQueries({ queryKey: ["scenarios"] });
    setActiveScenarioId(data.scenario_id);
    setRunData(data);
    setRunStatus("completed");
    setActiveScenarioData(data);
    setRebalResult(null);
    setRebalStatus("idle");
    onResultsReady?.({ ...data, _facilityType: ft, _mode: mode, _payload: currentPayload.current });
  };

  const schedulePoll = (scenarioId) => {
    pollRef.current = setTimeout(async () => {
      if (runningId.current !== scenarioId) return; // cancelled
      try {
        const data = await optimizationApi.get(scenarioId);
        if (data.status === "completed") {
          handleSuccess(data);
        } else if (data.status === "failed") {
          setRunStatus("failed");
          setRunError(data.stats?.error_message || "Optimization failed");
        } else {
          schedulePoll(scenarioId); // still running → next poll in 30 s
        }
      } catch {
        schedulePoll(scenarioId); // network error → retry
      }
    }, 30_000);
  };

  // ── Scope ──
  const handleScopeChange = (val) => {
    if (!val) { setScopeFilters(null); setScopeLabel(null); }
    else { const { scope_label, ...filters } = val; setScopeFilters(filters); setScopeLabel(scope_label || null); }
  };

  // Notify parent of initial facility_type so the map loads existing markers immediately.
  useEffect(() => {
    onResultsReady?.({ _facilityType: form.facility_type });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Target populations & facility type defaults ──
  const { data: targetPopulations = [] } = useQuery({
    queryKey: ["target-populations"],
    queryFn: targetPopulationsApi.list,
    staleTime: Infinity,
  });
  const { data: facilityTypesData = [] } = useQuery({
    queryKey: ["facility-types"],
    queryFn: targetPopulationsApi.facilityTypes,
    staleTime: Infinity,
  });

  // When facility type changes, auto-set the default target population.
  useEffect(() => {
    const ft = facilityTypesData.find((f) => f.code === form.facility_type);
    if (ft) {
      setForm((prev) => ({ ...prev, target_population_id: ft.default_target_population_id }));
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [form.facility_type, facilityTypesData]);

  // ── Scenarios list ──
  const { data: scenarios = [] } = useQuery({
    queryKey: ["scenarios"],
    queryFn: optimizationApi.list,
    refetchInterval: runStatus === "running" ? 30_000 : false,
  });

  // ── Submit ──
  const handleSubmit = async (e) => {
    e.preventDefault();
    const isBumpHunter = form.model_type === "bump_hunter";
    const payload = { ...form };
    // These fields are not top-level API fields — remove them before sending.
    delete payload.k_neighbors;
    delete payload.k_vec;

    if (isBumpHunter) {
      // Bump hunter has no p_facilities.
      delete payload.p_facilities;
      delete payload.min_capacity;
      delete payload.max_capacity;
      // service_radius is required for bump hunter.
      payload.service_radius = Number(form.service_radius);
      // Pack bump-hunter-specific params.
      payload.parameters = {
        k_neighbors: form.k_neighbors !== "" ? Number(form.k_neighbors) : 10,
        k_vec:       form.k_vec       !== "" ? Number(form.k_vec)       : 100,
      };
    } else {
      if (form.model_type === "max_coverage") {
        delete payload.p_facilities;
        payload.service_radius = Number(form.service_radius);
      } else {
        payload.p_facilities = Number(form.p_facilities);
        delete payload.service_radius;
      }
      if (form.min_capacity !== "") payload.min_capacity = Number(form.min_capacity);
      else delete payload.min_capacity;
      if (form.max_capacity !== "") payload.max_capacity = Number(form.max_capacity);
      else delete payload.max_capacity;
    }

    if (scopeFilters) payload.scope_filters = scopeFilters;
    // Always keep facility_type (used for target population resolution even in from_scratch mode).
    if (form.target_population_id != null) payload.target_population_id = form.target_population_id;
    else delete payload.target_population_id;

    setRunStatus("running");
    setRunError(null);
    setRunData(null);
    currentPayload.current = payload;

    try {
      const { scenario_id } = await optimizationApi.run(payload);
      runningId.current = scenario_id;
      schedulePoll(scenario_id);
    } catch (err) {
      setRunStatus("failed");
      setRunError(err?.response?.data?.detail || "Failed to start optimization");
    }
  };

  // ── Cancel ──
  const handleCancel = () => {
    runningId.current = null;
    clearTimeout(pollRef.current);
    setRunStatus("cancelled");
  };

  // ── Load recent scenario ──
  const handleScenarioClick = async (s) => {
    setRebalResult(null);
    setRebalStatus("idle");
    const stored = lsLoad(s.id);
    if (stored) {
      setActiveScenarioId(s.id);
      setActiveScenarioData(stored);
      onResultsReady?.({
        ...stored,
        _facilityType: stored._facilityType || "high_school",
        _mode:         stored._mode         || "from_scratch",
      });
      return;
    }
    // Not in localStorage → fetch from API
    try {
      const data = await optimizationApi.get(s.id);
      if (data.status === "completed") {
        const meta = data.stats?._meta || {};
        lsSave(s.id, data, meta.facility_type || "high_school", meta.mode || "from_scratch");
        setActiveScenarioId(s.id);
        setActiveScenarioData(data);
        onResultsReady?.({
          ...data,
          _facilityType: meta.facility_type || "high_school",
          _mode:         meta.mode          || "from_scratch",
        });
      }
    } catch (_) {}
  };

  // ── File picker ──
  const handleFileOpen = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (evt) => {
      try {
        const data = JSON.parse(evt.target.result);
        if (!data.facility_locations) throw new Error("invalid");
        const meta = data.stats?._meta || {};
        const ft   = data._facilityType || meta.facility_type || "high_school";
        const mode = data._mode         || meta.mode          || "from_scratch";
        if (data.scenario_id) lsSave(data.scenario_id, data, ft, mode);
        setRunData(data);
        onResultsReady?.({ ...data, _facilityType: ft, _mode: mode });
      } catch {
        alert("Archivo JSON inválido.");
      }
    };
    reader.readAsText(file);
    e.target.value = "";
  };

  // ── Rebalancing ──
  const handleRebalance = async () => {
    if (!activeScenarioData?.scenario_id) return;

    const locs = activeScenarioData.facility_locations || [];
    const totalCovered = locs.reduce((s, f) => s + (f.covered_demand || 0), 0);
    const defaultCap = locs.length > 0 ? Math.round(totalCovered / locs.length) : 1000;

    const params = {
      capacity_per_facility: rebalForm.capacity_per_facility !== ""
        ? Number(rebalForm.capacity_per_facility)
        : defaultCap,
      min_capacity:   Number(rebalForm.min_capacity)   || 0,
      max_transfers:  Number(rebalForm.max_transfers)  || 20,
    };

    setRebalStatus("running");
    setRebalError(null);
    try {
      const result = await optimizationApi.rebalance(activeScenarioData.scenario_id, params);
      setRebalResult(result);
      setRebalStatus("completed");
      onRebalancingResult?.(result.transfers);
    } catch (err) {
      setRebalStatus("failed");
      setRebalError(err?.response?.data?.detail || "Rebalancing failed.");
      onRebalancingResult?.(null);
    }
  };

  const handleClearRebal = () => {
    setRebalResult(null);
    setRebalStatus("idle");
    setRebalError(null);
    onRebalancingResult?.(null);
  };

  // Default capacity hint for the form placeholder.
  const rebalDefaultCap = (() => {
    const locs = activeScenarioData?.facility_locations || [];
    const total = locs.reduce((s, f) => s + (f.covered_demand || 0), 0);
    return locs.length > 0 ? Math.round(total / locs.length) : null;
  })();

  const selectedModel = MODEL_OPTIONS.find((m) => m.value === form.model_type);

  return (
    <div style={{ padding: "1rem", overflowY: "auto", height: "100%" }}>
      <h2 style={{ marginBottom: "1rem", fontSize: "1.1rem", fontWeight: 700 }}>
        Optimization
      </h2>

      {/* ── Form ── */}
      <form onSubmit={handleSubmit}>
        <label style={labelStyle}>Model</label>
        <select
          style={inputStyle}
          value={form.model_type}
          onChange={(e) => setForm({ ...form, model_type: e.target.value })}
        >
          {MODEL_OPTIONS.map((m) => (
            <option key={m.value} value={m.value}>{m.label}</option>
          ))}
        </select>
        {selectedModel && (
          <p style={{ fontSize: "0.78rem", color: "#6b7280", marginBottom: "0.75rem" }}>
            {selectedModel.description}
          </p>
        )}

        <label style={labelStyle}>Facility Type</label>
        <select
          style={inputStyle}
          value={form.facility_type}
          onChange={(e) => {
            setForm({ ...form, facility_type: e.target.value });
            onResultsReady?.({ _facilityType: e.target.value });
          }}
        >
          {FACILITY_TYPE_OPTIONS.map((t) => (
            <option key={t.value} value={t.value}>{t.label}</option>
          ))}
        </select>

        {targetPopulations.length > 0 && (
          <>
            <label style={labelStyle}>Target Population</label>
            <select
              style={inputStyle}
              value={form.target_population_id ?? ""}
              onChange={(e) =>
                setForm({ ...form, target_population_id: e.target.value ? Number(e.target.value) : null })
              }
            >
              {targetPopulations.map((tp) => (
                <option key={tp.id} value={tp.id}>{tp.label}</option>
              ))}
            </select>
          </>
        )}

        {form.model_type !== "bump_hunter" && (
          <>
            <label style={labelStyle}>Planning Mode</label>
            <select
              style={inputStyle}
              value={form.mode}
              onChange={(e) => setForm({ ...form, mode: e.target.value })}
            >
              <option value="from_scratch">From scratch</option>
              <option value="complete_existing">Complete existing</option>
            </select>
            {form.mode === "complete_existing" && (
              <p style={{ fontSize: "0.78rem", color: "#6b7280", marginBottom: "0.75rem" }}>
                Existing facilities of the selected type will be fixed.
              </p>
            )}
          </>
        )}

        {/* p (p-median / p-center — not bump_hunter or max_coverage) */}
        {form.model_type !== "bump_hunter" && form.model_type !== "max_coverage" && (
          <>
            <label style={labelStyle}>Number of Facilities (p)</label>
            <input
              style={inputStyle}
              type="number" min={1} max={500}
              value={form.p_facilities}
              onChange={(e) => setForm({ ...form, p_facilities: e.target.value })}
            />
          </>
        )}

        {/* service radius (max_coverage and bump_hunter) */}
        {(form.model_type === "max_coverage" || form.model_type === "bump_hunter") && (
          <>
            <label style={labelStyle}>Service Radius (minutes)</label>
            <input
              style={inputStyle}
              type="number" min={1} max={600}
              value={form.service_radius}
              onChange={(e) => setForm({ ...form, service_radius: e.target.value })}
            />
          </>
        )}

        {/* bump_hunter parameters */}
        {form.model_type === "bump_hunter" && (
          <>
            <label style={labelStyle}>Neighbourhood Size (k)</label>
            <input
              style={inputStyle}
              type="number" min={1} max={500}
              value={form.k_neighbors}
              placeholder="10"
              onChange={(e) => setForm({ ...form, k_neighbors: e.target.value })}
            />
            <label style={labelStyle}>Momentum Neighbours (k_vec)</label>
            <input
              style={inputStyle}
              type="number" min={1} max={5000}
              value={form.k_vec}
              placeholder="100"
              onChange={(e) => setForm({ ...form, k_vec: e.target.value })}
            />
          </>
        )}

        {/* Capacity controls – hidden for bump_hunter */}
        {form.model_type !== "bump_hunter" && (
          <div style={{ display: "flex", gap: "0.5rem" }}>
            <div style={{ flex: 1 }}>
              <label style={labelStyle}>Min Capacity</label>
              <input
                style={inputStyle}
                type="number" min={0}
                value={form.min_capacity}
                placeholder="None"
                onChange={(e) => setForm({ ...form, min_capacity: e.target.value })}
              />
            </div>
            <div style={{ flex: 1 }}>
              <label style={labelStyle}>Max Capacity</label>
              <input
                style={inputStyle}
                type="number" min={0}
                value={form.max_capacity}
                placeholder="∞"
                onChange={(e) => setForm({ ...form, max_capacity: e.target.value })}
              />
            </div>
          </div>
        )}

        {/* Geographic Scope */}
        <label style={labelStyle}>Geographic Scope</label>
        <p style={{ fontSize: "0.75rem", color: "#6b7280", marginBottom: "0.4rem" }}>
          No selection uses all census areas.
        </p>
        <PoliticalDivisionTree
          onChange={handleScopeChange}
          selectedDb={selectedDb}
          targetPopulationId={form.target_population_id}
          targetPopulationLabel={
            targetPopulations.find((tp) => tp.id === form.target_population_id)?.label
            ?? "Target population"
          }
        />

        {/* Scenario Name */}
        <label style={{ ...labelStyle, marginTop: "0.75rem", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span>Scenario Name</span>
          {nameIsManual && (
            <button type="button" onClick={() => setNameIsManual(false)} style={resetNameBtnStyle}>
              Reset
            </button>
          )}
        </label>
        <input
          style={inputStyle}
          required
          value={form.name}
          onChange={(e) => {
            setNameIsManual(e.target.value !== "");
            setForm({ ...form, name: e.target.value });
          }}
          placeholder="Auto-generated…"
        />

        {/* Action buttons */}
        <div style={{ display: "flex", gap: "0.5rem", marginTop: "0.25rem" }}>
          <button
            type="submit"
            disabled={runStatus === "running"}
            style={btnStyle(runStatus === "running", "#2563eb")}
          >
            {runStatus === "running" ? "Running…" : "Run Optimization"}
          </button>
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            style={btnStyle(false, "#374151")}
            title="Open a previously saved scenario JSON file"
          >
            Open
          </button>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept=".json"
          style={{ display: "none" }}
          onChange={handleFileOpen}
        />
      </form>

      {/* ── Progress bar ── */}
      {runStatus === "running" && (
        <div style={progressWrapStyle}>
          <div style={{ flex: 1 }}>
            <div style={progressTrackStyle}>
              <div style={progressFillStyle(Math.min(elapsed / 180, 0.99))} />
            </div>
            <span style={{ fontSize: "0.72rem", color: "#6b7280" }}>
              Running… {formatTime(elapsed)} elapsed — next status check in ~30 s
            </span>
          </div>
          <button onClick={handleCancel} style={cancelBtnStyle} title="Cancel">✕</button>
        </div>
      )}

      {runStatus === "cancelled" && (
        <p style={{ fontSize: "0.82rem", color: "#6b7280", marginTop: "0.5rem" }}>
          Optimization cancelled.
        </p>
      )}

      {/* ── Error ── */}
      {runStatus === "failed" && (
        <p style={{ color: "red", marginTop: "0.5rem", fontSize: "0.85rem" }}>
          {runError || "An error occurred."}
        </p>
      )}

      {/* ── Results summary ── */}
      {runStatus === "completed" && runData && (
        <div style={resultBoxStyle}>
          <strong>Completed</strong>
          <ul style={{ marginTop: "0.5rem", paddingLeft: "1rem" }}>
            {Object.entries(runData.stats || {})
              .filter(([k]) => !k.startsWith("_"))
              .map(([k, v]) => (
                <li key={k} style={{ fontSize: "0.82rem" }}>
                  {k.replace(/_/g, " ")}: <strong>{v}</strong>
                </li>
              ))}
          </ul>
          {runData.scenario_id && (
            <div style={{ marginTop: "0.75rem", display: "flex", gap: "0.5rem" }}>
              <button onClick={() => reportsApi.downloadExcel(runData.scenario_id, selectedDb)} style={linkBtnStyle("#16a34a")}>
                Download Excel
              </button>
              <button onClick={() => reportsApi.downloadJson(runData.scenario_id, selectedDb)} style={linkBtnStyle("#374151")}>
                Download JSON
              </button>
            </div>
          )}
        </div>
      )}

      {/* ── Rebalancing section ── */}
      {activeScenarioData?.scenario_id && activeScenarioData?.model_type !== "bump_hunter" && (
        <div style={rebalSectionStyle}>
          <button
            type="button"
            onClick={() => setShowRebal((v) => !v)}
            style={rebalToggleBtnStyle}
          >
            {showRebal ? "▾" : "▸"} Capacity Rebalancing
          </button>
          {showRebal && (
            <div style={{ marginTop: "0.6rem" }}>
              <p style={{ fontSize: "0.75rem", color: "#6b7280", marginBottom: "0.6rem" }}>
                Identifies over- and under-served facilities and proposes capacity
                transfers to reduce unmet demand.
              </p>
              <label style={labelStyle}>Capacity per Facility</label>
              <input
                style={inputStyle}
                type="number"
                min={1}
                value={rebalForm.capacity_per_facility}
                placeholder={rebalDefaultCap != null ? `Default: ${rebalDefaultCap.toLocaleString()}` : "Auto"}
                onChange={(e) => setRebalForm({ ...rebalForm, capacity_per_facility: e.target.value })}
              />
              <div style={{ display: "flex", gap: "0.5rem" }}>
                <div style={{ flex: 1 }}>
                  <label style={labelStyle}>Min. Operational Floor</label>
                  <input
                    style={inputStyle}
                    type="number"
                    min={0}
                    value={rebalForm.min_capacity}
                    onChange={(e) => setRebalForm({ ...rebalForm, min_capacity: e.target.value })}
                  />
                </div>
                <div style={{ flex: 1 }}>
                  <label style={labelStyle}>Max Transfers</label>
                  <input
                    style={inputStyle}
                    type="number"
                    min={1}
                    max={100}
                    value={rebalForm.max_transfers}
                    onChange={(e) => setRebalForm({ ...rebalForm, max_transfers: e.target.value })}
                  />
                </div>
              </div>
              <div style={{ display: "flex", gap: "0.5rem" }}>
                <button
                  type="button"
                  onClick={handleRebalance}
                  disabled={rebalStatus === "running"}
                  style={btnStyle(rebalStatus === "running", "#7c3aed")}
                >
                  {rebalStatus === "running" ? "Running…" : "Run Rebalancing"}
                </button>
                {rebalResult && (
                  <button type="button" onClick={handleClearRebal} style={btnStyle(false, "#6b7280")}>
                    Clear
                  </button>
                )}
              </div>

              {rebalStatus === "failed" && (
                <p style={{ color: "red", fontSize: "0.8rem", marginTop: "0.4rem" }}>{rebalError}</p>
              )}

              {rebalStatus === "completed" && rebalResult && (
                <div style={rebalResultBoxStyle}>
                  <div style={{ display: "flex", gap: "1.5rem", marginBottom: "0.5rem" }}>
                    <div>
                      <div style={{ fontSize: "0.7rem", color: "#6b7280", textTransform: "uppercase" }}>Unmet before</div>
                      <div style={{ fontWeight: 700, fontSize: "0.9rem" }}>
                        {rebalResult.unmet_demand_before.toLocaleString()}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: "0.7rem", color: "#6b7280", textTransform: "uppercase" }}>Unmet after</div>
                      <div style={{ fontWeight: 700, fontSize: "0.9rem", color: "#16a34a" }}>
                        {rebalResult.unmet_demand_after.toLocaleString()}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: "0.7rem", color: "#6b7280", textTransform: "uppercase" }}>Improvement</div>
                      <div style={{ fontWeight: 700, fontSize: "0.9rem", color: "#2563eb" }}>
                        {rebalResult.improvement_pct}%
                      </div>
                    </div>
                  </div>

                  {rebalResult.transfers.length === 0 ? (
                    <p style={{ fontSize: "0.8rem", color: "#6b7280" }}>
                      No transfers needed — all facilities are balanced.
                    </p>
                  ) : (
                    <>
                      <div style={{ fontSize: "0.78rem", fontWeight: 600, marginBottom: "0.35rem" }}>
                        {rebalResult.transfers.length} Recommended Transfer{rebalResult.transfers.length !== 1 ? "s" : ""}
                        <span style={{ fontSize: "0.72rem", color: "#6b7280", fontWeight: 400, marginLeft: "0.5rem" }}>
                          (shown on map as orange lines)
                        </span>
                      </div>
                      {rebalResult.transfers.map((t, i) => (
                        <div key={i} style={transferRowStyle}>
                          <span style={{ fontWeight: 600, color: "#ef4444" }}>{t.from_area_code || "F" + i}</span>
                          <span style={{ color: "#6b7280", fontSize: "0.75rem" }}>→</span>
                          <span style={{ fontWeight: 600, color: "#16a34a" }}>{t.to_area_code || "T" + i}</span>
                          <span style={{ marginLeft: "auto", fontSize: "0.75rem", color: "#374151" }}>
                            {t.amount.toLocaleString(undefined, { maximumFractionDigits: 0 })} units
                          </span>
                        </div>
                      ))}
                    </>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ── Recent Scenarios ── */}
      {scenarios.length > 0 && (
        <div style={{ marginTop: "1.5rem" }}>
          <h3 style={{ fontSize: "0.9rem", fontWeight: 600, marginBottom: "0.5rem" }}>
            Recent Scenarios
          </h3>
          {scenarios.slice(0, 8).map((s) => (
            <div
              key={s.id}
              style={scenarioRowStyle(s.id === activeScenarioId)}
              onClick={() => handleScenarioClick(s)}
            >
              <span style={{ fontWeight: 600, fontSize: "0.82rem" }}>{s.name}</span>
              <span style={{ fontSize: "0.75rem", color: "#6b7280" }}>
                {s.model_type} · p={s.p_facilities} · {s.status}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Inline styles ────────────────────────────────────────────────────────────

const labelStyle = {
  display: "block", fontSize: "0.82rem", fontWeight: 600,
  color: "#374151", marginBottom: "0.25rem",
};

const inputStyle = {
  display: "block", width: "100%", padding: "0.45rem 0.65rem",
  fontSize: "0.9rem", border: "1px solid #d1d5db", borderRadius: "6px",
  marginBottom: "0.75rem", background: "#fff", boxSizing: "border-box",
};

const btnStyle = (disabled, bg) => ({
  flex: 1, padding: "0.6rem", background: disabled ? "#93c5fd" : bg,
  color: "#fff", border: "none", borderRadius: "6px", fontWeight: 700,
  fontSize: "0.9rem", cursor: disabled ? "default" : "pointer",
});

const resetNameBtnStyle = {
  fontSize: "0.72rem", fontWeight: 500, color: "#2563eb",
  background: "none", border: "none", cursor: "pointer",
  padding: 0, textDecoration: "underline",
};

const progressWrapStyle = {
  marginTop: "0.75rem", display: "flex", alignItems: "center", gap: "0.5rem",
};

const progressTrackStyle = {
  height: "8px", background: "#e5e7eb", borderRadius: "4px",
  overflow: "hidden", marginBottom: "4px",
};

const progressFillStyle = (pct) => ({
  height: "100%", width: `${Math.round(pct * 100)}%`,
  background: "linear-gradient(90deg, #2563eb, #60a5fa)",
  borderRadius: "4px",
  transition: "width 1s linear",
});

const cancelBtnStyle = {
  padding: "0.3rem 0.6rem", background: "#ef4444", color: "#fff",
  border: "none", borderRadius: "4px", cursor: "pointer", fontWeight: 700,
  fontSize: "0.85rem", flexShrink: 0,
};

const resultBoxStyle = {
  marginTop: "1rem", padding: "0.75rem",
  background: "#f0fdf4", border: "1px solid #86efac", borderRadius: "8px",
};

const linkBtnStyle = (bg) => ({
  display: "inline-block", padding: "0.35rem 0.75rem", background: bg,
  color: "#fff", borderRadius: "6px", fontSize: "0.8rem",
  textDecoration: "none", fontWeight: 600,
});

const rebalSectionStyle = {
  marginTop: "1rem",
  border: "1px solid #e9d5ff",
  borderRadius: "8px",
  padding: "0.6rem 0.75rem",
  background: "#faf5ff",
};

const rebalToggleBtnStyle = {
  background: "none",
  border: "none",
  cursor: "pointer",
  fontSize: "0.85rem",
  fontWeight: 700,
  color: "#7c3aed",
  padding: 0,
  width: "100%",
  textAlign: "left",
};

const rebalResultBoxStyle = {
  marginTop: "0.75rem",
  padding: "0.65rem",
  background: "#ffffff",
  border: "1px solid #e9d5ff",
  borderRadius: "6px",
};

const transferRowStyle = {
  display: "flex",
  alignItems: "center",
  gap: "0.4rem",
  padding: "0.3rem 0",
  fontSize: "0.82rem",
  borderBottom: "1px solid #f3f4f6",
};

const scenarioRowStyle = (active) => ({
  display: "flex", flexDirection: "column",
  padding: "0.5rem 0.65rem", borderRadius: "6px",
  marginBottom: "0.35rem",
  background: active ? "#eff6ff" : "#f9fafb",
  border: `1px solid ${active ? "#93c5fd" : "#e5e7eb"}`,
  cursor: "pointer",
});

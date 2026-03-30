/**
 * Root application component.
 * Sets up the main two-panel layout: sidebar + map.
 */

import { useState, useRef, useEffect } from "react";
import { QueryClient, QueryClientProvider, useQuery, useQueryClient } from "@tanstack/react-query";
import MapView from "./components/Map/MapView";
import OptimizationPanel from "./components/Optimization/OptimizationPanel";
import { infrastructureApi, databasesApi, optimizationApi, setDatabase } from "./services/api";

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, staleTime: 30_000 } },
});

// ── localStorage helper (for reoptimization saves) ──────────────────────────

function lsSaveReopt(id, data, facilityType, mode, payload) {
  try {
    localStorage.setItem(`lip2_scenario_${id}`, JSON.stringify({
      ...data,
      _facilityType: facilityType,
      _mode: mode,
      _payload: payload,
    }));
    const index = JSON.parse(localStorage.getItem("lip2_scenarios") || "[]");
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
    localStorage.setItem("lip2_scenarios", JSON.stringify(index.slice(0, 20)));
  } catch (_) { /* quota exceeded */ }
}

function AppInner() {
  const qc = useQueryClient();

  const [optimizationResult, setOptimizationResult] = useState(null);
  const [facilityType, setFacilityType]             = useState(null);
  const [planningMode, setPlanningMode]             = useState(null);
  const [selectedDb,   setSelectedDb]               = useState("lip2_ecuador");
  const [panelKey,     setPanelKey]                 = useState(0); // forces OptimizationPanel remount on DB change

  // ── Rebalancing state ──
  const [rebalancingTransfers, setRebalancingTransfers] = useState(null);

  // ── Reoptimization state ──
  const [userEdits, setUserEdits]               = useState({ removed: new Set(), added: new Set() });
  const [previousResult, setPreviousResult]     = useState(null); // scenario before reoptimization
  const [reoptimizeStatus, setReoptimizeStatus] = useState("idle"); // idle|running|completed|failed
  const [reoptimizeError,  setReoptimizeError]  = useState(null);
  const reoptimizeRunningId = useRef(null);
  const reoptimizePollRef   = useRef(null);

  // Reset edits when a brand-new scenario (different id) is loaded.
  const prevScenarioId = useRef(null);
  useEffect(() => {
    if (optimizationResult?.scenario_id !== prevScenarioId.current) {
      prevScenarioId.current = optimizationResult?.scenario_id ?? null;
      setUserEdits({ removed: new Set(), added: new Set() });
      setPreviousResult(null);
      setReoptimizeStatus("idle");
      setReoptimizeError(null);
      setRebalancingTransfers(null);
    }
  }, [optimizationResult?.scenario_id]);

  // Cleanup polling on unmount.
  useEffect(() => () => clearTimeout(reoptimizePollRef.current), []);

  // ── Database switcher ──
  const { data: databases = [] } = useQuery({
    queryKey: ["databases"],
    queryFn: databasesApi.list,
    staleTime: Infinity,
  });

  const handleDbChange = (dbName) => {
    setDatabase(dbName);
    setSelectedDb(dbName);
    setOptimizationResult(null);
    setFacilityType(null);
    setPlanningMode(null);
    setUserEdits({ removed: new Set(), added: new Set() });
    setPreviousResult(null);
    setReoptimizeStatus("idle");
    setPanelKey((k) => k + 1);
  };

  // ── Existing facilities query ──
  const { data: existingFacilities = [] } = useQuery({
    queryKey: ["existing-facilities", facilityType],
    queryFn: () =>
      infrastructureApi.list({ facility_type: facilityType, status: "existing" }),
    enabled: !!facilityType,
  });

  const handleResultsReady = (data) => {
    if (data._facilityType) setFacilityType(data._facilityType);
    if (data._mode) setPlanningMode(data._mode);
    if (data.scenario_id) setOptimizationResult(data);
  };

  const visibleExistingFacilities =
    optimizationResult && planningMode === "complete_existing"
      ? existingFacilities
      : [];

  // ── Right-click handlers ──
  const handleFacilityContextMenu = (censusAreaId) => {
    setUserEdits((prev) => {
      const newRemoved = new Set(prev.removed);
      if (newRemoved.has(censusAreaId)) newRemoved.delete(censusAreaId);
      else newRemoved.add(censusAreaId);
      return { ...prev, removed: newRemoved };
    });
  };

  const handleAreaContextMenu = (censusAreaId) => {
    setUserEdits((prev) => {
      const newAdded = new Set(prev.added);
      if (newAdded.has(censusAreaId)) newAdded.delete(censusAreaId);
      else newAdded.add(censusAreaId);
      return { ...prev, added: newAdded };
    });
  };

  // ── Reoptimization polling ──
  const scheduleReoptimizePoll = (scenarioId, preservedPayload) => {
    reoptimizePollRef.current = setTimeout(async () => {
      if (reoptimizeRunningId.current !== scenarioId) return;
      try {
        const data = await optimizationApi.get(scenarioId);
        if (data.status === "completed") {
          const meta = data.stats?._meta || {};
          const ft   = meta.facility_type || "high_school";
          const mode = meta.mode          || "from_scratch";
          lsSaveReopt(data.scenario_id, data, ft, mode, preservedPayload);
          setFacilityType(ft);
          setPlanningMode(mode);
          setOptimizationResult({ ...data, _facilityType: ft, _mode: mode, _payload: preservedPayload });
          setUserEdits({ removed: new Set(), added: new Set() });
          setReoptimizeStatus("completed");
          qc.invalidateQueries({ queryKey: ["scenarios"] });
        } else if (data.status === "failed") {
          setReoptimizeStatus("failed");
          setReoptimizeError(data.stats?.error_message || "Reoptimization failed.");
          setPreviousResult(null);
        } else {
          scheduleReoptimizePoll(scenarioId, preservedPayload);
        }
      } catch {
        scheduleReoptimizePoll(scenarioId, preservedPayload);
      }
    }, 30_000);
  };

  // ── Reoptimize ──
  const handleReoptimize = async () => {
    if (!optimizationResult?._payload) return;

    // Only fix facilities that were already existing (user-created) or manually
    // added by the user.  Algorithm-suggested planned facilities are NOT fixed
    // so the solver can reassign them freely.
    const isBumpHunter = optimizationResult.model_type === "bump_hunter";
    const baseName = optimizationResult.name || "Scenario";
    const reoptName = `Reopt_${baseName}`.slice(0, 255);

    let payload;
    if (isBumpHunter) {
      // Fix all current bumps + manually added areas; re-run only the assignment step.
      const allBumpIds = (optimizationResult.facility_locations || [])
        .filter((f) => !userEdits.removed.has(f.census_area_id))
        .map((f) => f.census_area_id);
      const fixedIds = Array.from(new Set([...allBumpIds, ...userEdits.added]));
      payload = {
        ...optimizationResult._payload,
        name: reoptName,
        mode: "from_scratch",
        fixed_census_area_ids: fixedIds,
        parameters: undefined,  // skip bump-hunter algorithm; just reassign
      };
    } else {
      // All facilities the user has kept (not explicitly removed) become fixed.
      // The solver is skipped; a constrained nearest-assignment runs instead.
      const keptIds = (optimizationResult.facility_locations || [])
        .filter((f) => !userEdits.removed.has(f.census_area_id))
        .map((f) => f.census_area_id);
      const newFacilityIds = Array.from(userEdits.added);
      const fixedIds = Array.from(new Set([...keptIds, ...newFacilityIds]));
      const baseP = optimizationResult.p_facilities ?? fixedIds.length;
      const newP = Math.min(200, Math.max(baseP, fixedIds.length));
      payload = {
        ...optimizationResult._payload,
        name: reoptName,
        p_facilities: newP,
        mode: "from_scratch",
        fixed_census_area_ids: fixedIds,
        // User-created facilities need different assignment rules (radius for
        // max_coverage; no cap_min floor since they start empty).
        reopt_new_facility_ids: newFacilityIds.length > 0 ? newFacilityIds : undefined,
      };
    }

    setReoptimizeStatus("running");
    setReoptimizeError(null);

    try {
      const { scenario_id } = await optimizationApi.run(payload);
      reoptimizeRunningId.current = scenario_id;
      setPreviousResult(optimizationResult);
      scheduleReoptimizePoll(scenario_id, optimizationResult._payload);
    } catch (err) {
      setReoptimizeStatus("failed");
      setReoptimizeError(err?.response?.data?.detail || "Failed to start reoptimization.");
    }
  };

  const hasEdits = userEdits.removed.size > 0 || userEdits.added.size > 0;
  const canReoptimize = !!optimizationResult?._payload;

  return (
    <div style={{ display: "flex", height: "100vh", overflow: "hidden" }}>

      {/* ── Sidebar ── */}
      <aside style={sidebarStyle}>
        {/* Header */}
        <div style={headerStyle}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontWeight: 800, fontSize: "1rem", color: "#1e40af" }}>PIL</span>
            {databases.length > 0 && (
              <select
                value={selectedDb}
                onChange={(e) => handleDbChange(e.target.value)}
                style={dbSelectStyle}
                title="Select country database"
              >
                {databases.map((d) => (
                  <option key={d.db_name} value={d.db_name}>{d.label}</option>
                ))}
              </select>
            )}
          </div>
          <span style={{ fontSize: "0.75rem", color: "#6b7280", marginTop: "2px" }}>
            Public Infrastructure Locator
          </span>
        </div>

        {/* Panel content */}
        <div style={{ flex: 1, overflow: "hidden" }}>
          <OptimizationPanel
          key={panelKey}
          onResultsReady={handleResultsReady}
          onRebalancingResult={setRebalancingTransfers}
          selectedDb={selectedDb}
        />
        </div>
      </aside>

      {/* ── Map ── */}
      <main style={{ flex: 1, position: "relative" }}>
        <MapView
          facilities={optimizationResult?.facility_locations ?? []}
          existingFacilities={visibleExistingFacilities}
          unassignedAreas={optimizationResult?.unassigned_areas ?? []}
          serviceRadius={optimizationResult?.service_radius}
          selectedDb={selectedDb}
          removedFacilityIds={userEdits.removed}
          addedFacilityAreaIds={userEdits.added}
          onFacilityContextMenu={canReoptimize ? handleFacilityContextMenu : null}
          onAreaContextMenu={canReoptimize ? handleAreaContextMenu : null}
          rebalancingTransfers={rebalancingTransfers}
        />

        {/* Stats overlay — top left */}
        {optimizationResult?.stats && (
          <div style={statsOverlayStyle}>
            <strong style={{ fontSize: "0.8rem" }}>
              {optimizationResult.name}
            </strong>
            {Object.entries(optimizationResult.stats)
              .filter(([k]) => !k.startsWith("_"))
              .slice(0, 4)
              .map(([k, v]) => (
                <div key={k} style={{ fontSize: "0.75rem", color: "#374151" }}>
                  {k.replace(/_/g, " ")}: <strong>{v}</strong>
                </div>
              ))}
          </div>
        )}

        {/* Edit overlay — bottom right, shown when there are pending edits */}
        {(hasEdits || reoptimizeStatus === "running" || reoptimizeStatus === "failed") && (
          <div style={editOverlayStyle}>
            <div style={{ fontWeight: 700, fontSize: "0.82rem", marginBottom: "0.35rem" }}>
              Manual Edits
            </div>
            {userEdits.removed.size > 0 && (
              <div style={{ fontSize: "0.78rem", color: "#374151" }}>
                {userEdits.removed.size} facilit{userEdits.removed.size === 1 ? "y" : "ies"} marked for removal
              </div>
            )}
            {userEdits.added.size > 0 && (
              <div style={{ fontSize: "0.78rem", color: "#374151" }}>
                {userEdits.added.size} area{userEdits.added.size === 1 ? "" : "s"} added as facility
              </div>
            )}
            {reoptimizeStatus === "running" && (
              <div style={{ fontSize: "0.78rem", color: "#2563eb", marginTop: "0.25rem" }}>
                Reoptimization running…
              </div>
            )}
            {reoptimizeStatus === "failed" && (
              <div style={{ fontSize: "0.75rem", color: "#ef4444", marginTop: "0.25rem" }}>
                {reoptimizeError}
              </div>
            )}
            <div style={{ display: "flex", gap: "0.4rem", marginTop: "0.5rem" }}>
              {hasEdits && canReoptimize && (
                <button
                  onClick={handleReoptimize}
                  disabled={reoptimizeStatus === "running"}
                  style={reoptimizeBtnStyle(reoptimizeStatus === "running")}
                >
                  {reoptimizeStatus === "running" ? "Running…" : "Reoptimize"}
                </button>
              )}
              {hasEdits && (
                <button
                  onClick={() => setUserEdits({ removed: new Set(), added: new Set() })}
                  style={clearEditsBtnStyle}
                >
                  Clear
                </button>
              )}
            </div>
            {!canReoptimize && hasEdits && (
              <div style={{ fontSize: "0.72rem", color: "#9ca3af", marginTop: "0.25rem" }}>
                Run an optimization first to enable reoptimization.
              </div>
            )}
          </div>
        )}

        {/* Comparison overlay — top right, shown after reoptimization completes */}
        {previousResult && reoptimizeStatus === "completed" && optimizationResult && (
          <div style={comparisonOverlayStyle}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
              <span style={{ fontWeight: 700, fontSize: "0.82rem" }}>Comparison</span>
              <button onClick={() => setPreviousResult(null)} style={closeCompBtnStyle}>✕</button>
            </div>
            <div style={{ display: "flex", gap: "1rem" }}>
              {/* Previous scenario */}
              <div style={{ flex: 1, paddingRight: "0.75rem", borderRight: "1px solid #e5e7eb" }}>
                <div style={{ fontSize: "0.7rem", fontWeight: 700, color: "#6b7280", textTransform: "uppercase", marginBottom: "0.25rem" }}>
                  Previous
                </div>
                <div style={{ fontWeight: 600, fontSize: "0.78rem", marginBottom: "0.35rem", color: "#374151" }}>
                  {previousResult.name}
                </div>
                {Object.entries(previousResult.stats || {})
                  .filter(([k]) => !k.startsWith("_"))
                  .slice(0, 4)
                  .map(([k, v]) => {
                    const newV = optimizationResult.stats?.[k];
                    const delta = (typeof v === "number" && typeof newV === "number")
                      ? newV - v : null;
                    return (
                      <div key={k} style={{ fontSize: "0.73rem", color: "#6b7280", marginBottom: "1px" }}>
                        {k.replace(/_/g, " ")}: <strong style={{ color: "#111827" }}>{v}</strong>
                        {delta !== null && (
                          <span style={{ marginLeft: "4px", color: delta < 0 ? "#16a34a" : delta > 0 ? "#dc2626" : "#9ca3af", fontSize: "0.7rem" }}>
                            ({delta > 0 ? "+" : ""}{delta.toFixed(2)})
                          </span>
                        )}
                      </div>
                    );
                  })}
              </div>
              {/* Reoptimized scenario */}
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: "0.7rem", fontWeight: 700, color: "#2563eb", textTransform: "uppercase", marginBottom: "0.25rem" }}>
                  Reoptimized
                </div>
                <div style={{ fontWeight: 600, fontSize: "0.78rem", marginBottom: "0.35rem", color: "#374151" }}>
                  {optimizationResult.name}
                </div>
                {Object.entries(optimizationResult.stats || {})
                  .filter(([k]) => !k.startsWith("_"))
                  .slice(0, 4)
                  .map(([k, v]) => (
                    <div key={k} style={{ fontSize: "0.73rem", color: "#6b7280", marginBottom: "1px" }}>
                      {k.replace(/_/g, " ")}: <strong style={{ color: "#2563eb" }}>{v}</strong>
                    </div>
                  ))}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppInner />
    </QueryClientProvider>
  );
}

// ─── Inline styles ─────────────────────────────────────────────────────────

const sidebarStyle = {
  width: 320,
  minWidth: 300,
  height: "100vh",
  display: "flex",
  flexDirection: "column",
  background: "#ffffff",
  borderRight: "1px solid #e5e7eb",
  boxShadow: "2px 0 8px rgba(0,0,0,0.06)",
  overflow: "hidden",
};

const headerStyle = {
  display: "flex",
  flexDirection: "column",
  padding: "1rem",
  borderBottom: "1px solid #f3f4f6",
  background: "#f8fafc",
};

const dbSelectStyle = {
  fontSize: "0.78rem",
  fontWeight: 600,
  color: "#1e40af",
  background: "#eff6ff",
  border: "1px solid #bfdbfe",
  borderRadius: "6px",
  padding: "0.2rem 0.5rem",
  cursor: "pointer",
  outline: "none",
};

const statsOverlayStyle = {
  position: "absolute",
  top: 12,
  left: 12,
  background: "rgba(255,255,255,0.93)",
  borderRadius: 8,
  padding: "0.6rem 0.9rem",
  boxShadow: "0 2px 8px rgba(0,0,0,0.15)",
  display: "flex",
  flexDirection: "column",
  gap: "0.2rem",
  maxWidth: 220,
};

const editOverlayStyle = {
  position: "absolute",
  bottom: 32,
  right: 16,
  background: "rgba(255,255,255,0.96)",
  borderRadius: 10,
  padding: "0.7rem 1rem",
  boxShadow: "0 2px 12px rgba(0,0,0,0.18)",
  display: "flex",
  flexDirection: "column",
  minWidth: 200,
  maxWidth: 260,
  border: "1px solid #e5e7eb",
};

const reoptimizeBtnStyle = (disabled) => ({
  flex: 1,
  padding: "0.45rem 0.75rem",
  background: disabled ? "#93c5fd" : "#2563eb",
  color: "#fff",
  border: "none",
  borderRadius: "6px",
  fontWeight: 700,
  fontSize: "0.82rem",
  cursor: disabled ? "default" : "pointer",
});

const clearEditsBtnStyle = {
  padding: "0.45rem 0.75rem",
  background: "#f3f4f6",
  color: "#374151",
  border: "1px solid #e5e7eb",
  borderRadius: "6px",
  fontWeight: 600,
  fontSize: "0.82rem",
  cursor: "pointer",
};

const comparisonOverlayStyle = {
  position: "absolute",
  top: 12,
  right: 16,
  background: "rgba(255,255,255,0.97)",
  borderRadius: 10,
  padding: "0.75rem 1rem",
  boxShadow: "0 2px 12px rgba(0,0,0,0.18)",
  border: "1px solid #e5e7eb",
  maxWidth: 420,
  minWidth: 300,
};

const closeCompBtnStyle = {
  background: "none",
  border: "none",
  cursor: "pointer",
  fontSize: "0.85rem",
  color: "#9ca3af",
  padding: "0 2px",
  lineHeight: 1,
};

/**
 * PoliticalDivisionTree
 *
 * Hierarchical checkbox tree for selecting provincia > canton > parroquia.
 * Calls onChange({ parish_ids, label }) whenever the selection changes.
 * Pass onChange(null) when nothing is selected (= no filter, all areas).
 */

import { useState, useCallback, useEffect, useRef } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { politicalDivisionsApi } from "../../services/api";

// ─── Scope label builder ─────────────────────────────────────────────────────

/**
 * Builds a geographic scope string from the selected parroquias.
 * Format: Provincia_Canton_Parroquia (levels separated by _).
 * Multiple siblings at the same level are joined with -.
 * Example: "Pichincha_Quito_Calderon-Pomasqui"
 */
function buildScopeLabel(tree, selectedParroquias, getCheckState) {
  if (selectedParroquias.size === 0) return "National";

  const san = (name) =>
    name
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .replace(/[^a-zA-Z0-9]/g, "");

  const provinceParts = [];
  for (const province of tree) {
    const pState = getCheckState(province);
    if (pState === "unchecked") continue;
    if (pState === "checked") {
      provinceParts.push(san(province.name));
      continue;
    }
    // indeterminate → drill into cantons
    const cantonParts = [];
    for (const canton of province.children || []) {
      const cState = getCheckState(canton);
      if (cState === "unchecked") continue;
      if (cState === "checked") {
        cantonParts.push(san(canton.name));
        continue;
      }
      // indeterminate → drill into parroquias
      const parroquiaParts = (canton.children || [])
        .filter((p) => selectedParroquias.has(p.id))
        .map((p) => san(p.name));
      cantonParts.push(san(canton.name) + "_" + parroquiaParts.join("-"));
    }
    provinceParts.push(san(province.name) + "_" + cantonParts.join("-"));
  }

  return provinceParts.join("-") || "National";
}

// ─── Main component ─────────────────────────────────────────────────────────

export default function PoliticalDivisionTree({
  onChange,
  selectedDb = "lip2_ecuador",
  targetPopulationId = null,
  targetPopulationLabel = "Target population",
}) {
  const [expanded, setExpanded] = useState(new Set());
  // selectedParroquias: Set of parroquia node IDs (leaf selection)
  const [selectedParroquias, setSelectedParroquias] = useState(new Set());

  const { data: tree = [], isLoading, isError } = useQuery({
    queryKey: ["political-divisions-tree", selectedDb],
    queryFn: politicalDivisionsApi.tree,
    staleTime: Infinity,
  });

  const summaryMutation = useMutation({
    mutationFn: politicalDivisionsApi.censusSummary,
  });

  // Collect all leaf nodes under a subtree (works for any country hierarchy)
  const getAllParroquias = useCallback((node) => {
    if (!node.children?.length) return [node];
    return node.children.flatMap(getAllParroquias);
  }, []);

  // Derived check state for a node
  const getCheckState = useCallback(
    (node) => {
      const parroquias = getAllParroquias(node);
      if (parroquias.length === 0) return "unchecked";
      const n = parroquias.filter((p) => selectedParroquias.has(p.id)).length;
      if (n === 0) return "unchecked";
      if (n === parroquias.length) return "checked";
      return "indeterminate";
    },
    [selectedParroquias, getAllParroquias]
  );

  const toggleExpand = (id) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });

  const handleCheck = useCallback(
    (node) => {
      const parroquias = getAllParroquias(node);
      const ids = parroquias.map((p) => p.id);
      const allSelected = ids.every((id) => selectedParroquias.has(id));

      const next = new Set(selectedParroquias);
      if (allSelected) {
        ids.forEach((id) => next.delete(id));
      } else {
        ids.forEach((id) => next.add(id));
      }
      setSelectedParroquias(next);
    },
    [selectedParroquias, getAllParroquias]
  );

  // Notify parent and fetch census summary whenever selection or target population changes
  useEffect(() => {
    const ids = [...selectedParroquias];
    if (ids.length === 0) {
      onChange(null);
      return;
    }
    const scope_label = buildScopeLabel(tree, selectedParroquias, getCheckState);
    onChange({ parish_ids: ids, scope_label });
    summaryMutation.mutate({ parish_ids: ids, target_population_id: targetPopulationId });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedParroquias, targetPopulationId]);

  if (isLoading) return <p style={hintStyle}>Loading political divisions…</p>;
  if (isError)   return <p style={{ ...hintStyle, color: "#ef4444" }}>Error loading tree.</p>;

  const summary = summaryMutation.data;

  return (
    <div>
      {/* Tree */}
      <div style={treeContainerStyle}>
        {tree.map((provincia) => (
          <TreeNode
            key={provincia.id}
            node={provincia}
            depth={0}
            expanded={expanded}
            onToggleExpand={toggleExpand}
            onCheck={handleCheck}
            getCheckState={getCheckState}
          />
        ))}
      </div>

      {/* Summary */}
      {selectedParroquias.size > 0 && (
        <div style={summaryStyle}>
          <div style={summaryRowStyle}>
            <span>Selected parishes:</span>
            <strong>{selectedParroquias.size}</strong>
          </div>
          {summary && (
            <>
              <div style={summaryRowStyle}>
                <span>Census areas:</span>
                <strong>{summary.census_area_count.toLocaleString()}</strong>
              </div>
              <div style={summaryRowStyle}>
                <span>Total population:</span>
                <strong>{summary.total_demand.toLocaleString(undefined, { maximumFractionDigits: 0 })}</strong>
              </div>
              {summary.target_population !== undefined && summary.target_population !== summary.total_demand && (
                <div style={{ ...summaryRowStyle, color: "#2563eb" }}>
                  <span>{targetPopulationLabel}:</span>
                  <strong>{summary.target_population.toLocaleString(undefined, { maximumFractionDigits: 0 })}</strong>
                </div>
              )}
            </>
          )}
          <button
            onClick={() => setSelectedParroquias(new Set())}
            style={clearButtonStyle}
          >
            Clear selection
          </button>
        </div>
      )}
    </div>
  );
}

// ─── Tree node ───────────────────────────────────────────────────────────────

function TreeNode({ node, depth, expanded, onToggleExpand, onCheck, getCheckState }) {
  const isExpanded = expanded.has(node.id);
  const checkState = getCheckState(node);
  const hasChildren = !!node.children?.length;
  const checkboxRef = useRef(null);

  useEffect(() => {
    if (checkboxRef.current) {
      checkboxRef.current.indeterminate = checkState === "indeterminate";
    }
  }, [checkState]);

  return (
    <div>
      <div style={rowStyle(depth)}>
        {/* Expand toggle */}
        <span
          onClick={() => hasChildren && onToggleExpand(node.id)}
          style={arrowStyle(hasChildren)}
        >
          {hasChildren ? (isExpanded ? "▼" : "▶") : ""}
        </span>

        {/* Checkbox */}
        <input
          ref={checkboxRef}
          type="checkbox"
          checked={checkState === "checked"}
          onChange={() => onCheck(node)}
          style={checkboxStyle}
        />

        {/* Label */}
        <span
          onClick={() => hasChildren && onToggleExpand(node.id)}
          style={labelStyle(hasChildren, depth)}
          title={node.name}
        >
          {node.name}
        </span>
      </div>

      {/* Children */}
      {isExpanded && hasChildren && (
        <div>
          {node.children.map((child) => (
            <TreeNode
              key={child.id}
              node={child}
              depth={depth + 1}
              expanded={expanded}
              onToggleExpand={onToggleExpand}
              onCheck={onCheck}
              getCheckState={getCheckState}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Styles ──────────────────────────────────────────────────────────────────

const treeContainerStyle = {
  maxHeight: "260px",
  overflowY: "auto",
  border: "1px solid #d1d5db",
  borderRadius: "6px",
  padding: "0.25rem 0",
  background: "#fff",
  fontSize: "0.8rem",
};

const rowStyle = (depth) => ({
  display: "flex",
  alignItems: "center",
  padding: "0.18rem 0.4rem",
  paddingLeft: `${0.4 + depth * 1.1}rem`,
  cursor: "default",
  userSelect: "none",
  ":hover": { background: "#f3f4f6" },
});

const arrowStyle = (hasChildren) => ({
  width: "1rem",
  flexShrink: 0,
  fontSize: "0.6rem",
  color: "#6b7280",
  cursor: hasChildren ? "pointer" : "default",
  textAlign: "center",
});

const checkboxStyle = {
  marginRight: "0.35rem",
  flexShrink: 0,
  cursor: "pointer",
  width: "13px",
  height: "13px",
};

const labelStyle = (hasChildren, depth) => ({
  flex: 1,
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
  cursor: hasChildren ? "pointer" : "default",
  fontWeight: depth === 0 ? 600 : depth === 1 ? 500 : 400,
  color: depth === 0 ? "#111827" : depth === 1 ? "#374151" : "#4b5563",
});

const summaryStyle = {
  marginTop: "0.5rem",
  padding: "0.5rem 0.65rem",
  background: "#eff6ff",
  border: "1px solid #bfdbfe",
  borderRadius: "6px",
  fontSize: "0.78rem",
};

const summaryRowStyle = {
  display: "flex",
  justifyContent: "space-between",
  marginBottom: "0.2rem",
};

const clearButtonStyle = {
  marginTop: "0.35rem",
  width: "100%",
  padding: "0.25rem",
  fontSize: "0.75rem",
  background: "none",
  border: "1px solid #93c5fd",
  borderRadius: "4px",
  color: "#2563eb",
  cursor: "pointer",
};

const hintStyle = {
  fontSize: "0.78rem",
  color: "#6b7280",
  marginBottom: "0.5rem",
};

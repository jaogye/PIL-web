/**
 * Interactive map using MapLibre GL.
 *
 * Layer selector covers all 124 layers of the OpenFreeMap liberty style,
 * grouped into 15 semantic categories.
 *
 * Supports right-click context menus on facility markers and census area
 * markers for the reoptimization feature.
 */

import { useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";

const DB_VIEWS = {
  lip2_ecuador: { center: [-78.4678, -1.8312], zoom: 6 },
  lip2_belgium: { center: [4.4699, 50.5039],   zoom: 7 },
};

const SERVICE_LINES_SOURCE    = "service-lines";
const SERVICE_LINES_LAYER     = "service-lines-layer";
const REBAL_SOURCE            = "rebalancing-transfers";
const REBAL_LAYER             = "rebalancing-transfers-layer";
const REBAL_ARROWS_LAYER      = "rebalancing-arrows-layer";

// ── All liberty style layers grouped by category ───────────────────────────
const LAYER_GROUPS = [
  {
    id: "fondo",
    label: "Background",
    layers: ["background", "natural_earth"],
  },
  {
    id: "agua",
    label: "Water",
    layers: [
      "water",
      "waterway_river", "waterway_other", "waterway_tunnel",
      "water_name_point_label", "water_name_line_label", "waterway_line_label",
    ],
  },
  {
    id: "cobertura",
    label: "Land Cover",
    layers: [
      "landcover_grass", "landcover_ice", "landcover_sand",
      "landcover_wetland", "landcover_wood",
    ],
  },
  {
    id: "uso_suelo",
    label: "Land Use",
    layers: [
      "landuse_cemetery", "landuse_hospital", "landuse_pitch",
      "landuse_residential", "landuse_school", "landuse_track",
    ],
  },
  {
    id: "parques",
    label: "Parks",
    layers: ["park", "park_outline"],
  },
  {
    id: "calles_principales",
    label: "Main Streets",
    layers: [
      "road_motorway", "road_motorway_casing",
      "road_trunk_primary", "road_trunk_primary_casing",
      "road_secondary_tertiary", "road_secondary_tertiary_casing",
      "bridge_motorway", "bridge_motorway_casing",
      "bridge_trunk_primary", "bridge_trunk_primary_casing",
      "bridge_secondary_tertiary", "bridge_secondary_tertiary_casing",
      "tunnel_motorway", "tunnel_motorway_casing",
      "tunnel_trunk_primary", "tunnel_trunk_primary_casing",
      "tunnel_secondary_tertiary", "tunnel_secondary_tertiary_casing",
    ],
  },
  {
    id: "calles_menores",
    label: "Minor Streets",
    layers: [
      "road_minor", "road_minor_casing",
      "road_link", "road_link_casing",
      "road_motorway_link", "road_motorway_link_casing",
      "road_service_track", "road_service_track_casing",
      "road_area_pattern",
      "road_one_way_arrow", "road_one_way_arrow_opposite",
      "bridge_link", "bridge_link_casing",
      "bridge_motorway_link", "bridge_motorway_link_casing",
      "bridge_service_track", "bridge_service_track_casing",
      "bridge_street", "bridge_street_casing",
      "tunnel_link", "tunnel_link_casing",
      "tunnel_minor",
      "tunnel_motorway_link", "tunnel_motorway_link_casing",
      "tunnel_service_track", "tunnel_service_track_casing",
      "tunnel_street_casing",
    ],
  },
  {
    id: "caminos",
    label: "Paths & Pedestrians",
    layers: [
      "road_path_pedestrian",
      "bridge_path_pedestrian", "bridge_path_pedestrian_casing",
      "tunnel_path_pedestrian",
    ],
  },
  {
    id: "ferrocarril",
    label: "Railways",
    layers: [
      "road_major_rail", "road_major_rail_hatching",
      "road_transit_rail", "road_transit_rail_hatching",
      "bridge_major_rail", "bridge_major_rail_hatching",
      "bridge_transit_rail", "bridge_transit_rail_hatching",
      "tunnel_major_rail", "tunnel_major_rail_hatching",
      "tunnel_transit_rail", "tunnel_transit_rail_hatching",
    ],
  },
  {
    id: "aeropuertos",
    label: "Airports",
    layers: ["airport", "aeroway_fill", "aeroway_runway", "aeroway_taxiway"],
  },
  {
    id: "edificios",
    label: "Buildings",
    layers: ["building", "building-3d"],
  },
  {
    id: "limites",
    label: "Boundaries",
    layers: ["boundary_2", "boundary_3", "boundary_disputed"],
  },
  {
    id: "poi",
    label: "Points of Interest",
    layers: ["poi_r1", "poi_r7", "poi_r20", "poi_transit"],
  },
  {
    id: "nombres_calles",
    label: "Street Names",
    layers: [
      "highway-name-major", "highway-name-minor", "highway-name-path",
      "highway-shield-non-us", "highway-shield-us-interstate", "road_shield_us",
    ],
  },
  {
    id: "etiquetas",
    label: "Place Labels",
    layers: [
      "label_country_1", "label_country_2", "label_country_3",
      "label_state", "label_city_capital", "label_city",
      "label_town", "label_village", "label_other",
    ],
  },
];

const ALL_VISIBLE = Object.fromEntries(LAYER_GROUPS.map((g) => [g.id, true]));

export default function MapView({
  facilities = [],
  existingFacilities = [],
  unassignedAreas = [],
  serviceRadius = null,
  selectedDb = "lip2_ecuador",
  removedFacilityIds = null,
  addedFacilityAreaIds = null,
  onFacilityContextMenu = null,
  onAreaContextMenu = null,
  rebalancingTransfers = null,
}) {
  const mapContainer      = useRef(null);
  const mapRef            = useRef(null);
  const markersRef        = useRef([]);
  const linesReadyRef     = useRef(false);
  const prevFacilitiesRef = useRef(null);

  // Use refs so event listeners always call the latest version without re-creating markers.
  const onFacilityContextMenuRef = useRef(onFacilityContextMenu);
  const onAreaContextMenuRef     = useRef(onAreaContextMenu);
  useEffect(() => { onFacilityContextMenuRef.current = onFacilityContextMenu; });
  useEffect(() => { onAreaContextMenuRef.current     = onAreaContextMenu; });

  const [showLayerPanel, setShowLayerPanel] = useState(false);
  const [layerVisibility, setLayerVisibility] = useState(ALL_VISIBLE);

  // Context menu state: null = hidden, or { x, y, type, censusAreaId, isEdited }
  const [contextMenu, setContextMenu] = useState(null);

  // ── Initialise map once ────────────────────────────────────────────
  useEffect(() => {
    if (mapRef.current) return;

    const { center, zoom } = DB_VIEWS[selectedDb] ?? DB_VIEWS.lip2_ecuador;
    const map = new maplibregl.Map({
      container: mapContainer.current,
      style: "https://tiles.openfreemap.org/styles/liberty",
      center,
      zoom,
    });

    map.addControl(new maplibregl.NavigationControl(), "top-right");
    map.addControl(new maplibregl.ScaleControl(), "bottom-left");

    map.on("load", () => {
      // Service lines (facility → census area)
      map.addSource(SERVICE_LINES_SOURCE, {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
      map.addLayer({
        id: SERVICE_LINES_LAYER,
        type: "line",
        source: SERVICE_LINES_SOURCE,
        paint: {
          "line-color": "#6b7280",
          "line-width": 1,
          "line-opacity": 0.5,
          "line-dasharray": [3, 2],
        },
      });

      // Rebalancing transfer lines (facility → facility)
      map.addSource(REBAL_SOURCE, {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
      map.addLayer({
        id: REBAL_LAYER,
        type: "line",
        source: REBAL_SOURCE,
        layout: { "line-cap": "round", "line-join": "round" },
        paint: {
          "line-color": "#f97316",
          "line-width": ["interpolate", ["linear"], ["get", "amount"], 0, 2, 10000, 8],
          "line-opacity": 0.85,
        },
      });
      // Arrow symbols at the destination end
      map.addLayer({
        id: REBAL_ARROWS_LAYER,
        type: "symbol",
        source: REBAL_SOURCE,
        layout: {
          "symbol-placement": "line-center",
          "text-field": "▶",
          "text-size": 14,
          "text-rotate": 0,
          "text-allow-overlap": true,
          "text-ignore-placement": true,
        },
        paint: {
          "text-color": "#f97316",
          "text-halo-color": "#fff",
          "text-halo-width": 1,
        },
      });

      // Click on rebalancing line → popup with transfer details
      map.on("click", REBAL_LAYER, (e) => {
        if (!e.features?.length) return;
        const f = e.features[0].properties;
        new maplibregl.Popup()
          .setLngLat(e.lngLat)
          .setHTML(`
            <strong>Capacity Transfer</strong><br/>
            From: <strong>${f.from_code || "—"}</strong><br/>
            To: &nbsp;&nbsp;<strong>${f.to_code || "—"}</strong><br/>
            Amount: <strong>${Number(f.amount).toLocaleString(undefined, { maximumFractionDigits: 0 })}</strong><br/>
            Impact: <strong>${Number(f.impact).toLocaleString(undefined, { maximumFractionDigits: 0 })}</strong>
          `)
          .addTo(map);
      });
      map.on("mouseenter", REBAL_LAYER, () => { map.getCanvas().style.cursor = "pointer"; });
      map.on("mouseleave", REBAL_LAYER, () => { map.getCanvas().style.cursor = ""; });

      linesReadyRef.current = true;
    });

    mapRef.current = map;

    return () => {
      mapRef.current?.remove();
      mapRef.current = null;
      linesReadyRef.current = false;
    };
  }, []);

  // ── Apply layer visibility when toggles change ─────────────────────
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const apply = () => {
      LAYER_GROUPS.forEach((group) => {
        const visibility = layerVisibility[group.id] ? "visible" : "none";
        group.layers.forEach((layerId) => {
          if (map.getLayer(layerId)) {
            map.setLayoutProperty(layerId, "visibility", visibility);
          }
        });
      });
    };

    if (map.isStyleLoaded()) {
      apply();
    } else {
      map.once("load", apply);
    }
  }, [layerVisibility]);

  // ── Update rebalancing transfer lines ─────────────────────────────
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const update = () => {
      const source = map.getSource(REBAL_SOURCE);
      if (!source) return;

      if (!rebalancingTransfers || rebalancingTransfers.length === 0) {
        source.setData({ type: "FeatureCollection", features: [] });
        return;
      }

      const features = rebalancingTransfers
        .filter((t) => t.from_x && t.from_y && t.to_x && t.to_y)
        .map((t) => ({
          type: "Feature",
          properties: {
            from_code: t.from_area_code,
            to_code:   t.to_area_code,
            amount:    t.amount,
            impact:    t.impact,
          },
          geometry: {
            type: "LineString",
            coordinates: [[t.from_x, t.from_y], [t.to_x, t.to_y]],
          },
        }));

      source.setData({ type: "FeatureCollection", features });
    };

    if (map.isStyleLoaded()) update();
    else map.once("load", update);
  }, [rebalancingTransfers]);

  // ── Fly to country when database changes ──────────────────────────
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const { center, zoom } = DB_VIEWS[selectedDb] ?? DB_VIEWS.lip2_ecuador;
    map.flyTo({ center, zoom, speed: 1.2 });
  }, [selectedDb]);

  // ── Close context menu on click-outside or Escape ─────────────────
  useEffect(() => {
    if (!contextMenu) return;
    const handleClick = () => setContextMenu(null);
    const handleKey   = (e) => { if (e.key === "Escape") setContextMenu(null); };
    document.addEventListener("click", handleClick);
    document.addEventListener("keydown", handleKey);
    return () => {
      document.removeEventListener("click", handleClick);
      document.removeEventListener("keydown", handleKey);
    };
  }, [contextMenu]);

  // ── Re-render markers and service lines ───────────────────────────
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const removed = removedFacilityIds  ?? new Set();
    const added   = addedFacilityAreaIds ?? new Set();

    // Helper: open right-click context menu for a facility marker.
    const openFacilityMenu = (e, censusAreaId) => {
      e.preventDefault();
      e.stopPropagation();
      setContextMenu({
        x: e.clientX,
        y: e.clientY,
        type: "facility",
        censusAreaId,
        isEdited: removed.has(censusAreaId),
      });
    };

    // Helper: open right-click context menu for a census area marker.
    const openAreaMenu = (e, censusAreaId) => {
      e.preventDefault();
      e.stopPropagation();
      setContextMenu({
        x: e.clientX,
        y: e.clientY,
        type: "area",
        censusAreaId,
        isEdited: added.has(censusAreaId),
      });
    };

    const render = () => {
      const wasEmpty = !prevFacilitiesRef.current || prevFacilitiesRef.current.length === 0;
      prevFacilitiesRef.current = facilities;

      markersRef.current.forEach((m) => m.remove());
      markersRef.current = [];

      const facilityCensusAreaIds = new Set(facilities.map((f) => f.census_area_id));

      // 1. Service lines + square (census area) data.
      const lineFeatures = [];
      // squareMap: census_area_id → { censusAreaId, x, y, demand, travel_time, area_code, facility_area_code, facility_x, facility_y }
      const squareMap = new Map();

      facilities.forEach((facility) => {
        if (!facility.x || !facility.y) return;
        (facility.served_areas || []).forEach((sa) => {
          if (!sa.x || !sa.y) return;
          lineFeatures.push({
            type: "Feature",
            geometry: {
              type: "LineString",
              coordinates: [[sa.x, sa.y], [facility.x, facility.y]],
            },
          });
          if (!facilityCensusAreaIds.has(sa.census_area_id)) {
            squareMap.set(sa.census_area_id, {
              censusAreaId: sa.census_area_id,
              x: sa.x,
              y: sa.y,
              demand: sa.assigned_demand,
              travel_time: sa.travel_time,
              area_code: sa.area_code,
              facility_area_code: facility.area_code,
              facility_x: facility.x,
              facility_y: facility.y,
            });
          }
        });
      });

      if (linesReadyRef.current) {
        map.getSource(SERVICE_LINES_SOURCE)?.setData({
          type: "FeatureCollection",
          features: lineFeatures,
        });
      }

      // 2. Yellow squares (census areas) — clickable popup + right-click to add as facility.
      squareMap.forEach(({ censusAreaId, x, y, demand, travel_time, area_code, facility_area_code, facility_x, facility_y }) => {
        const isAdded = added.has(censusAreaId);
        const el = document.createElement("div");

        if (isAdded) {
          // Proposed facility (user added) — purple circle
          el.style.cssText = `
            width: 18px; height: 18px;
            background: #7c3aed; border: 2.5px solid #fff; border-radius: 50%;
            box-shadow: 0 2px 5px rgba(0,0,0,0.35); cursor: pointer;
            display: flex; align-items: center; justify-content: center;
            font-size: 10px; color: #fff; font-weight: 700;
          `;
          el.textContent = "+";
        } else {
          // Normal census area — yellow square
          el.style.cssText = `
            width: 10px; height: 10px;
            background: #fbbf24; border: 1.5px solid #78350f; cursor: pointer;
          `;
        }

        const ttStr = travel_time != null ? `${Number(travel_time).toFixed(1)} min` : "—";

        // Flat-earth Euclidean distance between census area and its facility.
        let distKmStr = "—";
        let speedStr  = "—";
        if (facility_x != null && facility_y != null) {
          const latMid = (y + facility_y) / 2 * Math.PI / 180;
          const dxKm   = (facility_x - x) * 111.0 * Math.cos(latMid);
          const dyKm   = (facility_y - y) * 111.0;
          const distKm = Math.sqrt(dxKm * dxKm + dyKm * dyKm);
          distKmStr = `${distKm.toFixed(2)} km`;
          if (travel_time != null && travel_time > 0) {
            speedStr = `${(distKm / (travel_time / 60)).toFixed(1)} km/h`;
          }
        }

        const popupHtml = isAdded
          ? `<strong>Proposed Facility</strong><br/>
             Code: <strong>${area_code || "—"}</strong><br/>
             Right-click to cancel addition`
          : `<strong>Census Area</strong><br/>
             Code: <strong>${area_code || "—"}</strong><br/>
             Demand: <strong>${(demand || 0).toLocaleString()}</strong><br/>
             Access time: <strong>${ttStr}</strong><br/>
             Distance: <strong>${distKmStr}</strong><br/>
             Travel speed: <strong>${speedStr}</strong><br/>
             Nearest facility: <strong>${facility_area_code || "—"}</strong><br/>
             <em style="color:#6b7280;font-size:0.75em">Right-click to add facility here</em>`;

        const popup = new maplibregl.Popup({ offset: 8 }).setHTML(popupHtml);

        el.addEventListener("contextmenu", (e) => openAreaMenu(e, censusAreaId));

        markersRef.current.push(
          new maplibregl.Marker({ element: el, anchor: "center" })
            .setLngLat([x, y]).setPopup(popup).addTo(map)
        );
      });

      // 3. Unassigned areas — pink squares (areas not served by any facility).
      unassignedAreas.forEach(({ census_area_id, area_code, name, x, y }) => {
        if (!x || !y) return;
        const isAdded = added.has(census_area_id);
        const el = document.createElement("div");

        if (isAdded) {
          el.style.cssText = `
            width: 18px; height: 18px;
            background: #7c3aed; border: 2.5px solid #fff; border-radius: 50%;
            box-shadow: 0 2px 5px rgba(0,0,0,0.35); cursor: pointer;
            display: flex; align-items: center; justify-content: center;
            font-size: 10px; color: #fff; font-weight: 700;
          `;
          el.textContent = "+";
        } else {
          el.style.cssText = `
            width: 10px; height: 10px;
            background: #f472b6; border: 1.5px solid #9d174d; cursor: pointer;
          `;
        }

        const displayName = name || area_code || "—";
        const popupHtml = isAdded
          ? `<strong>Proposed Facility</strong><br/>
             ${displayName}<br/>
             <em style="color:#6b7280;font-size:0.75em">Right-click to cancel addition</em>`
          : `<strong>Unassigned Area</strong><br/>
             ${displayName}<br/>
             <em style="color:#6b7280;font-size:0.75em">Not covered — right-click to add facility here</em>`;

        const popup = new maplibregl.Popup({ offset: 8 }).setHTML(popupHtml);
        el.addEventListener("contextmenu", (e) => openAreaMenu(e, census_area_id));

        markersRef.current.push(
          new maplibregl.Marker({ element: el, anchor: "center" })
            .setLngLat([x, y]).setPopup(popup).addTo(map)
        );
      });

      // 4. Existing facilities from infrastructure DB (orange).
      existingFacilities.forEach((fac) => {
        if (!fac.x || !fac.y) return;
        const el = document.createElement("div");
        el.style.cssText = `
          width: 16px; height: 16px; background: #f97316;
          border: 2px solid #fff; border-radius: 50%;
          box-shadow: 0 2px 5px rgba(0,0,0,0.35); cursor: pointer;
        `;
        const popup = new maplibregl.Popup({ offset: 12 }).setHTML(
          `<strong>Existing</strong><br/>${fac.name || fac.facility_type}`
        );
        markersRef.current.push(
          new maplibregl.Marker({ element: el, anchor: "center" })
            .setLngLat([fac.x, fac.y]).setPopup(popup).addTo(map)
        );
      });

      // 4. Optimized facilities (on top) — left-click popup, right-click context menu.
      if (facilities.length === 0) return;
      const bounds = new maplibregl.LngLatBounds();

      facilities.forEach((facility, idx) => {
        if (!facility.x || !facility.y) return;
        const isExisting = facility.is_existing;
        const isRemoved  = removed.has(facility.census_area_id);

        // Compute population-weighted average access time from served areas.
        const servedAreas = facility.served_areas || [];
        const totalDemand = servedAreas.reduce((s, a) => s + (a.assigned_demand || 0), 0);
        const weightedSum = servedAreas.reduce(
          (s, a) => s + (a.assigned_demand || 0) * (a.travel_time != null ? a.travel_time : 0),
          0
        );
        const avgTime = totalDemand > 0
          ? `${(weightedSum / totalDemand).toFixed(1)} min`
          : "—";
        const maxTime = facility.max_travel_time != null
          ? `${Number(facility.max_travel_time).toFixed(1)} min`
          : "—";

        const el = document.createElement("div");

        if (isRemoved) {
          // Removed facility — grayed out with X
          el.style.cssText = `
            width: 20px; height: 20px;
            background: #9ca3af; opacity: 0.7;
            border: 3px solid #ef4444; border-radius: 50%;
            box-shadow: 0 2px 6px rgba(0,0,0,0.4); cursor: pointer;
            display: flex; align-items: center; justify-content: center;
            font-size: 11px; color: #ef4444; font-weight: 900;
          `;
          el.textContent = "✕";
        } else {
          el.style.cssText = `
            width: 20px; height: 20px;
            background: ${isExisting ? "#16a34a" : "#2563eb"};
            border: 3px solid #fff; border-radius: 50%;
            box-shadow: 0 2px 6px rgba(0,0,0,0.4); cursor: pointer;
          `;
        }

        const popupHtml = isRemoved
          ? `<strong>Facility ${idx + 1} — Marked for removal</strong><br/>
             ${facility.name || facility.area_code}<br/>
             <em style="color:#6b7280;font-size:0.75em">Right-click to restore</em>`
          : `<strong>Facility ${idx + 1}${isExisting ? " (existing)" : ""}</strong><br/>
             ${facility.name || facility.area_code}<br/>
             Demand: <strong>${(facility.covered_demand || 0).toLocaleString()}</strong><br/>
             Max access: <strong>${maxTime}</strong><br/>
             Avg access: <strong>${avgTime}</strong><br/>
             <em style="color:#6b7280;font-size:0.75em">Right-click to remove</em>`;

        const popup = new maplibregl.Popup({ offset: 14 }).setHTML(popupHtml);

        el.addEventListener("contextmenu", (e) => openFacilityMenu(e, facility.census_area_id));

        markersRef.current.push(
          new maplibregl.Marker({ element: el, anchor: "center" })
            .setLngLat([facility.x, facility.y]).setPopup(popup).addTo(map)
        );
        if (!isRemoved) bounds.extend([facility.x, facility.y]);
      });

      if (!bounds.isEmpty() && wasEmpty) {
        map.fitBounds(bounds, { padding: 60, maxZoom: 12 });
      }
    };

    if (map.isStyleLoaded()) render();
    else map.once("load", render);
  }, [facilities, existingFacilities, unassignedAreas, removedFacilityIds, addedFacilityAreaIds]);

  // ── Helpers ────────────────────────────────────────────────────────
  const toggleLayer = (id) =>
    setLayerVisibility((prev) => ({ ...prev, [id]: !prev[id] }));

  const allChecked = Object.values(layerVisibility).every(Boolean);
  const toggleAll  = () =>
    setLayerVisibility(Object.fromEntries(LAYER_GROUPS.map((g) => [g.id, !allChecked])));

  const handleContextMenuAction = (e) => {
    e.stopPropagation(); // prevent document click from closing the menu immediately
    if (!contextMenu) return;
    if (contextMenu.type === "facility") {
      onFacilityContextMenuRef.current?.(contextMenu.censusAreaId);
    } else {
      onAreaContextMenuRef.current?.(contextMenu.censusAreaId);
    }
    setContextMenu(null);
  };

  // ── Render ─────────────────────────────────────────────────────────
  return (
    <div style={{ width: "100%", height: "100%", minHeight: 400, position: "relative" }}>
      <div ref={mapContainer} style={{ width: "100%", height: "100%" }} />

      {/* Layer selector */}
      <div style={{ position: "absolute", top: 10, left: 10, zIndex: 10 }}>
        <button
          onClick={() => setShowLayerPanel((v) => !v)}
          title="Select layers"
          style={layerBtnStyle(showLayerPanel)}
        >
          ⊞ Layers
        </button>

        {showLayerPanel && (
          <div style={layerPanelStyle}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
              <span style={panelTitleStyle}>Map Layers</span>
              <button onClick={toggleAll} style={toggleAllBtnStyle}>
                {allChecked ? "Hide all" : "Show all"}
              </button>
            </div>
            {LAYER_GROUPS.map((group) => (
              <label key={group.id} style={layerRowStyle}>
                <input
                  type="checkbox"
                  checked={layerVisibility[group.id]}
                  onChange={() => toggleLayer(group.id)}
                  style={{ marginRight: "8px", cursor: "pointer" }}
                />
                {group.label}
              </label>
            ))}
          </div>
        )}
      </div>

      {/* Right-click context menu */}
      {contextMenu && (
        <div
          style={{
            position: "fixed",
            left: contextMenu.x,
            top: contextMenu.y,
            background: "#ffffff",
            border: "1px solid #e5e7eb",
            borderRadius: "8px",
            boxShadow: "0 4px 16px rgba(0,0,0,0.18)",
            padding: "4px 0",
            zIndex: 9999,
            minWidth: "180px",
          }}
          onClick={(e) => e.stopPropagation()}
        >
          <div style={ctxMenuItemStyle} onClick={handleContextMenuAction}>
            {contextMenu.type === "facility"
              ? (contextMenu.isEdited ? "↩ Restore Facility" : "✕ Remove Facility")
              : (contextMenu.isEdited ? "↩ Cancel Addition"  : "+ Add Facility Here")}
          </div>
          <div style={{ borderTop: "1px solid #f3f4f6", margin: "4px 0" }} />
          <div style={{ ...ctxMenuItemStyle, color: "#9ca3af" }} onClick={() => setContextMenu(null)}>
            Cancel
          </div>
        </div>
      )}
    </div>
  );
}

// ── Styles ──────────────────────────────────────────────────────────────────

const layerBtnStyle = (active) => ({
  padding: "6px 12px",
  background: active ? "#1d4ed8" : "#ffffff",
  color: active ? "#ffffff" : "#374151",
  border: "1px solid #d1d5db",
  borderRadius: "6px",
  fontWeight: 600,
  fontSize: "0.82rem",
  cursor: "pointer",
  boxShadow: "0 2px 6px rgba(0,0,0,0.18)",
  whiteSpace: "nowrap",
});

const layerPanelStyle = {
  marginTop: "6px",
  background: "#ffffff",
  border: "1px solid #e5e7eb",
  borderRadius: "8px",
  padding: "10px 14px",
  boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
  minWidth: "200px",
  maxHeight: "70vh",
  overflowY: "auto",
};

const panelTitleStyle = {
  fontSize: "0.75rem",
  fontWeight: 700,
  color: "#6b7280",
  textTransform: "uppercase",
  letterSpacing: "0.05em",
};

const toggleAllBtnStyle = {
  fontSize: "0.72rem",
  color: "#2563eb",
  background: "none",
  border: "none",
  cursor: "pointer",
  padding: 0,
  fontWeight: 600,
};

const layerRowStyle = {
  display: "flex",
  alignItems: "center",
  fontSize: "0.85rem",
  color: "#111827",
  padding: "4px 0",
  cursor: "pointer",
  borderBottom: "1px solid #f3f4f6",
};

const ctxMenuItemStyle = {
  padding: "8px 14px",
  fontSize: "0.85rem",
  color: "#111827",
  cursor: "pointer",
  fontWeight: 500,
  whiteSpace: "nowrap",
  userSelect: "none",
};

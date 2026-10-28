# %% -------------------- Power-balance checks by zone (P and Q) --------------------
# HYBRID storage approach:
#   1) Wide DataFrames (easy global scanning):
#        - p_fail_df[time, zone]      : True where |P| <= tol_p  (your "failure" definition)
#        - q_fail_any_df[time, zone]  : True where ANY capacitor-combination makes |Q_adj| <= tol_q
#        - pq_fail_df[time, zone]     : True where (P_fail AND Q_fail_any) are both True
#   2) Dict-of-DataFrames by zone (easy drill-down):
#        - q_adj_by_zone[zone]        : adjusted Q time series for each capacitor combination
#        - q_fail_by_zone[zone]       : boolean mask per combination (|Q_adj| <= tol_q)
#        - q_combo_meta_by_zone[zone] : list of metadata dicts describing each combo column
#
# Sign convention:
#   - Inductive load is +kVAR in AMI
#   - Capacitors provide reactive support, so we SUBTRACT capacitor kVAR:
#       Q_adj(t) = Q(t) - Qcap_on
#
# Requirements:
#   - zone_p_df: DataFrame of total zone P (kW). columns = zone ids
#   - zone_q_df: DataFrame of total zone Q (kVAR). columns = zone ids
#   - zone_stat_df: DataFrame from dg.get_zone_stat_df(graph, zones) that includes a 'cap_details' column
#       where each cell is a list of dicts like:
#           {"cap_name": "...", "bus": "...", "kvar_steps": [..], "num_steps": N}
#
# Notes:
#   - "Fail" is defined exactly as: value is close to 0 (within +/- tolerance).
#   - If a zone has no capacitors, Q has only one "scenario": Q_adj = Q (no subtraction).

import re
import pandas as pd
import numpy as np
from itertools import product


def _unique_subset_sums(step_values):
    """
    Given a list of capacitor step kvar values (e.g., [50, 150, 400]),
    return the sorted list of UNIQUE achievable kvar outputs, INCLUDING 0 (all off).

    This computes all subset sums, treating each list element as a distinct step
    (so repeated values like [200,200,200] are allowed and produce 0,200,400,600).
    """
    sums = {0.0}
    for v in step_values:
        # For each existing sum, you can either add this step or not
        sums |= {s + float(v) for s in sums}
    return sorted(sums)

def _zone_capacitor_combos(cap_details):
    """
    Build all unique zone-level capacitor combinations.

    Parameters
    ----------
    cap_details : list[dict]
        Each dict should contain:
          - 'cap_name'
          - 'kvar_steps' : list[float]

    Returns
    -------
    combos : list[dict]
        Each combo dict contains:
          - 'combo_label' : a human-readable label describing capacitor settings
          - 'qcap_on'      : total capacitor kvar on for this combo (float)
          - 'per_cap'      : list of per-cap settings (cap_name, cap_kvar_on)
    """
    # If no capacitors, only one scenario: no capacitor kvar on
    if not cap_details:
        return [{
            "combo_label": "no_caps",
            "qcap_on": 0.0,
            "per_cap": []
        }]

    # For each capacitor, compute its unique achievable outputs (including 0)
    per_cap_options = []
    for cap in cap_details:
        cap_name = cap.get("cap_name", "Capacitor.unknown")
        step_vals = cap.get("kvar_steps", [])
        achievable = _unique_subset_sums(step_vals)  # includes 0
        per_cap_options.append((cap_name, achievable))

    # Cartesian product across capacitors to get zone-level combos
    # Each element of 'choice_tuple' is a selected kvar output for the corresponding capacitor.
    combos_raw = []
    for choice_tuple in product(*[opts for _, opts in per_cap_options]):
        per_cap = []
        for (cap_name, _), cap_kvar_on in zip(per_cap_options, choice_tuple):
            per_cap.append({"cap_name": cap_name, "cap_kvar_on": float(cap_kvar_on)})

        qcap_on = sum(x["cap_kvar_on"] for x in per_cap)

        # Create a readable label like: "CapA=0|CapB=200|CapB=400" (one entry per cap)
        # Keep it stable so you can quickly inspect which combination caused failure.
        label_parts = [f"{x['cap_name']}={x['cap_kvar_on']:g}" for x in per_cap]
        combo_label = " | ".join(label_parts)

        combos_raw.append({"combo_label": combo_label, "qcap_on": float(qcap_on), "per_cap": per_cap})

    # Deduplicate combos that end up with identical per-cap settings labels (should already be unique),
    # but also deduplicate by total qcap if you only care about total.
    # We want to inspect which capacitor-state combination caused failure, so we keep per-cap labels.
    # Still, there can be rare duplicates if cap_details has duplicates; remove exact label duplicates.
    seen = set()
    combos = []
    for c in combos_raw:
        if c["combo_label"] in seen:
            continue
        seen.add(c["combo_label"])
        combos.append(c)

    return combos


def _zero_cross_or_touch_mask(x):
    """
    Boolean mask for zero cross/touch events, flagging BOTH endpoints.
    Avoids pandas FutureWarning by not using fillna on object dtype.
    """
    if isinstance(x, pd.Series):
        # event is True at t_k when (t_{k-1}, t_k) crosses/touches
        event = (x.shift(1) * x) <= 0

        # Comparisons with NaN yield False, but event may be boolean dtype with <NA> depending on pandas;
        # force missing to False using a boolean comparison rather than fillna.
        event = (event == True)

        # Flag both endpoints: event at k implies mark k, and mark k-1 via shifting
        mask = event | event.shift(-1)
        mask = (mask == True)
        return mask

    if isinstance(x, pd.DataFrame):
        event = (x.shift(1) * x) <= 0
        event = (event == True)

        mask = event | event.shift(-1)
        mask = (mask == True)
        return mask

    raise TypeError(f"_zero_cross_or_touch_mask expects a pandas Series or DataFrame, got {type(x)}")


def zero_cross_or_touch_mask(x):
    """
    Public API wrapper for the cross/touch mask.

    Use this in notebooks/plots instead of calling the underscore-prefixed helper.
    """
    return _zero_cross_or_touch_mask(x)


def build_zone_power_balance_masks(
    zone_p_df,
    zone_q_df,
    zone_stat_df,
    tol_p=50.0,
    tol_q=50.0,
):
    """
    Build P/Q pass/fail masks and Q scenario drill-down data.

    Failure definition:
      - Tolerance failure:
          P_fail_tol(t, zone)      = |P| <= tol_p
          Q_fail_tol(t, zone, scn) = |Q_adj| <= tol_q

      - Zero cross/touch failure (flags BOTH endpoints of an event):
          cross_or_touch(x[k-1], x[k]) if x[k-1] * x[k] <= 0
          (includes 0->0 segments)

      - Combined failures (logical OR):
          P_fail = P_fail_tol OR P_fail_cross
          Q_fail = Q_fail_tol OR Q_fail_cross

      - Zone-level Q failure:
          Q_fail_any(t, zone) = any scenario fails at time t

      - Combined P&Q failure (logical AND):
          PQ_fail(t, zone) = P_fail(t, zone) AND Q_fail_any(t, zone)

    Returns
    ------------------------
    p_fail_df : pd.DataFrame (bool)
    q_fail_any_df : pd.DataFrame (bool)
    pq_fail_df : pd.DataFrame (bool)

    q_adj_by_zone : dict[int, pd.DataFrame]
    q_fail_by_zone : dict[int, pd.DataFrame]
    q_combo_meta_by_zone : dict[int, list[dict]]
    """
    # -------------------- P failure (wide) --------------------
    # Tolerance-based "failure"
    p_fail_tol_df = zone_p_df.abs().le(tol_p)

    # Crossing/touch "failure" (flags both endpoints)
    p_fail_cross_df = _zero_cross_or_touch_mask(zone_p_df)

    # Combined P failure
    p_fail_df = p_fail_tol_df | p_fail_cross_df

    # -------------------- Q failure (hybrid) --------------------
    # Prepare wide Q "any scenario" fail df with same shape/index/columns as zone_q_df
    q_fail_any_df = pd.DataFrame(False, index=zone_q_df.index, columns=zone_q_df.columns)

    # Per-zone drill-down containers
    q_adj_by_zone = {}
    q_fail_by_zone = {}
    q_combo_meta_by_zone = {}

    for zone_id in zone_q_df.columns:
        # Baseline Q(t) for this zone
        q_series = zone_q_df[zone_id]

        # Pull capacitor details from zone_stat_df if present
        if ("cap_details" in zone_stat_df.columns) and (zone_id in zone_stat_df.index):
            cap_details = zone_stat_df.loc[zone_id, "cap_details"] or []
        else:
            cap_details = []

        # Build all capacitor-state combos for this zone
        combos = _zone_capacitor_combos(cap_details)
        q_combo_meta_by_zone[zone_id] = combos

        # Build adjusted-Q DataFrame for this zone: one column per combo label
        # Q_adj(t) = Q(t) - Qcap_on   (capacitors subtract from inductive +kVAR)
        q_adj_cols = {}
        for combo in combos:
            label = combo["combo_label"]
            qcap_on = combo["qcap_on"]
            q_adj_cols[label] = q_series - qcap_on

        q_adj_df = pd.DataFrame(q_adj_cols, index=zone_q_df.index)

        # Tolerance-based Q failure, per scenario
        q_fail_tol_df = q_adj_df.abs().le(tol_q)

        # Crossing/touch Q failure, per scenario (includes baseline scenario too)
        q_fail_cross_df = _zero_cross_or_touch_mask(q_adj_df)

        # Combined Q failure, per scenario
        q_fail_df = q_fail_tol_df | q_fail_cross_df

        # "Any scenario fails" mask for this zone/time
        q_fail_any_df[zone_id] = q_fail_df.any(axis=1)

        # Store drill-down details
        q_adj_by_zone[zone_id] = q_adj_df
        q_fail_by_zone[zone_id] = q_fail_df

    # -------------------- Combined P & Q failure --------------------
    pq_fail_df = p_fail_df & q_fail_any_df

    return p_fail_df, q_fail_any_df, pq_fail_df, q_adj_by_zone, q_fail_by_zone, q_combo_meta_by_zone


def summarize_pq_fail_df(pq_fail_df):
    """
    Summarize combined P&Q failure mask (pq_fail_df) by zone.

    Parameters
    ----------
    pq_fail_df : pd.DataFrame (bool)
        Index = time (e.g., unix_time), columns = zone_id strings like "lvl 1 - 2".
        True indicates a failure at that timestamp for that zone.

    Returns
    -------
    summary_df : pd.DataFrame
        One row per zone with:
          - level: "lvl 1" or "lvl 2" (parsed from zone_id)
          - zone_num: integer parsed from zone_id (for sorting)
          - n_total, n_true, pct_true
          - dt: timestep estimated from mode of index diffs (same units as index)
          - true_time: n_true * dt
          - true_time_hours: true_time converted to hours (assumes dt is seconds)
    """
    if not isinstance(pq_fail_df, pd.DataFrame):
        raise TypeError("pq_fail_df must be a pandas DataFrame")

    if pq_fail_df.empty:
        return pd.DataFrame()

    # --- Compute dt using mode of index differences (assumes mostly regular) ---
    # Convert index to a numeric Series and compute diffs
    idx = pd.Series(pq_fail_df.index)

    # If index is not numeric, try converting to numeric
    try:
        idx_num = pd.to_numeric(idx)
    except Exception as exc:
        raise ValueError("pq_fail_df.index must be numeric-like to compute dt") from exc

    diffs = idx_num.diff().dropna()

    if diffs.empty:
        dt = np.nan
    else:
        # mode() can return multiple values; take the first
        # TODO: figure out why dt is off by 1000
        dt = diffs.mode().iloc[0]

    rows = []
    for zone_id in pq_fail_df.columns:
        s = pq_fail_df[zone_id]

        n_total = int(s.shape[0])
        n_true = int(s.sum())  # True counts as 1
        pct_true = (100.0 * n_true / n_total) if n_total else np.nan

        # Parse level and zone number from strings
        zone_str = str(zone_id)

        # # level_num: 1 or 2 (numeric)
        # m_level = re.search(r"lvl\s*([12])", zone_str, flags=re.IGNORECASE)
        # level = int(m_level.group(1)) if m_level else np.nan

        # # zone_num: take trailing integer after hyphen if present
        # # e.g. "lvl 1 - 2" -> 2
        # m_num = re.search(r"-\s*(\d+)\s*$", zone_str)
        # zone_num = int(m_num.group(1)) if m_num else np.nan

        # --- Parse level + zone token from strings like:
        #     "lvl 1.2"
        #     "lvl 2.1-2"
        zone_str = str(zone_id)

        # level: 1 or 2
        m_level = re.search(r"lvl\s*([12])", zone_str, flags=re.IGNORECASE)
        level = int(m_level.group(1)) if m_level else np.nan

        # zone_token: everything after the first dot, e.g. "2" or "1-2"
        m_token = re.search(r"lvl\s*[12]\.(.+)$", zone_str, flags=re.IGNORECASE)
        zone_token = m_token.group(1).strip() if m_token else ""

        # zone_sort_key: tuple of all integers appearing in zone_token
        #   "2"   -> (2,)
        #   "1-2" -> (1, 2)
        token_ints = [int(x) for x in re.findall(r"\d+", zone_token)]
        zone_sort_key = tuple(token_ints)

        # Convenience numeric zone number for level 1 zones only (NaN otherwise)
        # For "lvl 1.2", zone_num = 2
        zone_num = token_ints[0] if (level == 1 and len(token_ints) >= 1) else np.nan

        true_time = (n_true * dt) if pd.notna(dt) else np.nan

        # Interpret as seconds -> hours (matches your unix_time-looking index)
        true_time_hours = (true_time / 3600.0) if pd.notna(true_time) else np.nan

        rows.append({
            "zone_id": zone_id,
            "level": level,
            "zone_num": zone_num,
            "zone_token": zone_token,
            "zone_sort_key": zone_sort_key,
            "n_total": n_total,
            "n_true": n_true,
            "pct_true": pct_true,
            "dt": dt,
            "true_time": true_time,
            "true_time_hours": true_time_hours,
        })

    summary_df = pd.DataFrame(rows).set_index("zone_id")

    # Sort by level then by zone_num (and fallback to zone_id string)
    summary_df = summary_df.sort_values(by=["level", "zone_sort_key"], kind="stable")

    return summary_df


def summarize_power_headroom(zone_p_df, q_adj_by_zone, tol_p=50.0, tol_q=50.0):
    """
    Summarize signed upper-bound headroom for P and Q by zone.

    Headroom definitions (signed, upper-bound reference):
      - P headroom at time t:
            h_P(t) = P(t) - tol_p
        but forced to 0 when P(t) <= tol_p

      - Q headroom at time t, scenario s:
            h_Q(t,s) = Q_adj(t,s) - tol_q
        but forced to 0 when Q_adj(t,s) <= tol_q

    Outputs (per zone):
      - p_min_headroom (kW)          : min over time of forced headroom values
      - p_worst_time                 : timestamp where p_min_headroom occurs
      - q_min_headroom (kVAR)        : min over all time/scenarios of forced headroom values
      - q_worst_time                 : timestamp where q_min_headroom occurs
      - q_worst_scenario             : scenario label where q_min_headroom occurs

    Notes:
      - With this definition, any timepoint at/below tolerance forces headroom to 0,
        so p_min_headroom or q_min_headroom will be 0 if any such point exists.
      - This function assumes q_adj_by_zone keys match zone_p_df columns.
    """
    rows = []

    for zone_id in zone_p_df.columns:
        # -------------------- P headroom --------------------
        p_series = zone_p_df[zone_id]

        # Compute headroom and force failures (P <= tol_p) to 0
        p_headroom = p_series - tol_p
        p_headroom = p_headroom.where(p_series > tol_p, 0.0)

        # Find minimum headroom and when it occurs
        # (skipna in case of missing data)
        p_min_headroom = float(p_headroom.min(skipna=True))
        p_worst_time = p_headroom.idxmin(skipna=True) if p_headroom.notna().any() else np.nan

        # -------------------- Q headroom (worst over scenarios) --------------------
        q_min_headroom = np.nan
        q_worst_time = np.nan
        q_worst_scenario = np.nan

        q_adj_df = q_adj_by_zone.get(zone_id, None)

        if isinstance(q_adj_df, pd.DataFrame) and (not q_adj_df.empty):
            # Compute headroom and force failures (Q_adj <= tol_q) to 0
            q_headroom_df = q_adj_df - tol_q
            q_headroom_df = q_headroom_df.where(q_adj_df > tol_q, 0.0)

            # Find global minimum across all time/scenarios
            # Stack -> Series with MultiIndex (time, scenario)
            stacked = q_headroom_df.stack().dropna()

            if not stacked.empty:
                q_min_headroom = float(stacked.min())
                worst_idx = stacked.idxmin()  # (time, scenario)
                q_worst_time = worst_idx[0]
                q_worst_scenario = worst_idx[1]

        rows.append({
            "zone_id": zone_id,
            "p_min_headroom": p_min_headroom,
            "p_worst_time": p_worst_time,
            "q_min_headroom": q_min_headroom,
            "q_worst_time": q_worst_time,
            "q_worst_scenario": q_worst_scenario,
        })

    headroom_df = pd.DataFrame(rows).set_index("zone_id")
    return headroom_df


def summarize_bus_headroom(zones, zone_headroom_df):
    """
    Build bus-level headroom summary from zone-level headroom results.

    For each bus, find all zones that include it and compute:
      - p_min_headroom_kw   = min(p_min_headroom_kw over those zones)
      - q_min_headroom_kvar = min(q_min_headroom_kvar over those zones)

    Also returns the "limiting" zone (argmin) for P and Q:
      - p_limiting_zone
      - q_limiting_zone

    Parameters
    ----------
    zones : dict[str, set[str] or list[str]]
        Mapping of zone_id -> iterable of bus names (from dg.get_zones()).

    zone_headroom_df : pd.DataFrame
        Output of summarize_power_headroom(...), indexed by zone_id and containing:
          - 'p_min_headroom'  (kW)
          - 'q_min_headroom'  (kVAR)

    Returns
    -------
    bus_headroom_df : pd.DataFrame
        Index: bus name
        Columns:
          - p_min_headroom_kw
          - p_limiting_zone
          - q_min_headroom_kvar
          - q_limiting_zone
          - n_zones (how many zones contained this bus)
    """
    if not isinstance(zones, dict):
        raise TypeError("zones must be a dict of zone_id -> bus iterable")

    if not isinstance(zone_headroom_df, pd.DataFrame):
        raise TypeError("zone_headroom_df must be a pandas DataFrame")

    # Note: zone_headroom_df currently uses non-unit-suffixed names.
    required_cols = {"p_min_headroom", "q_min_headroom"}
    missing = required_cols - set(zone_headroom_df.columns)
    if missing:
        raise ValueError(f"zone_headroom_df is missing required columns: {sorted(missing)}")

    bus_to_zones = {}
    for zone_id, zone_info in zones.items():
        # zones are dicts like {"buses": set(...), "type": "..."}
        bus_iter = zone_info["buses"]
        for bus in bus_iter:
            bus_to_zones.setdefault(bus, []).append(zone_id)

    rows = []
    for bus, bus_zones in bus_to_zones.items():
        # Keep only zones that exist in the headroom DF (else NaN for debugging)
        valid_zones = [z for z in bus_zones if z in zone_headroom_df.index]

        if len(valid_zones) == 0:
            rows.append({
                "bus": bus,
                "p_min_headroom_kw": np.nan,
                "p_limiting_zone": np.nan,
                "q_min_headroom_kvar": np.nan,
                "q_limiting_zone": np.nan,
                "n_zones": len(bus_zones),
            })
            continue

        sub = zone_headroom_df.loc[valid_zones, ["p_min_headroom", "q_min_headroom"]]

        # Min headroom values
        p_min = sub["p_min_headroom"].min(skipna=True)
        q_min = sub["q_min_headroom"].min(skipna=True)

        # Limiting zones (idxmin); handle all-NaN columns safely
        p_lim = sub["p_min_headroom"].idxmin(skipna=True) if sub["p_min_headroom"].notna().any() else np.nan
        q_lim = sub["q_min_headroom"].idxmin(skipna=True) if sub["q_min_headroom"].notna().any() else np.nan

        rows.append({
            "bus": bus,
            "p_min_headroom_kw": float(p_min) if pd.notna(p_min) else np.nan,
            "p_limiting_zone": p_lim,
            "q_min_headroom_kvar": float(q_min) if pd.notna(q_min) else np.nan,
            "q_limiting_zone": q_lim,
            "n_zones": len(bus_zones),
        })

    bus_headroom_df = pd.DataFrame(rows).set_index("bus").sort_index()
    return bus_headroom_df


def summarize_rf_zone_p_headroom(zone_p_df_rf, reverse_flow_threshold_kw=0.0):
    """
    Compute real-power headroom-to-reverse-flow for each RF zone, using zone-level P time series.

    Inputs
    ------
    zone_p_df_rf : pd.DataFrame
        Index = time, columns = RF zone_id, values = total zone real power (kW).
        You will typically build this by summing AMI kW across *load buses* per RF zone.

    reverse_flow_threshold_kw : float
        Reverse-flow threshold in kW (default 0.0). Headroom is forced to 0 when P <= threshold.

    Headroom definition
    -------------------
      headroom(t) = P_zone(t) if P_zone(t) > threshold else 0
      p_min_headroom_kw = min_t headroom(t)

    Returns
    -------
    rf_zone_headroom_df : pd.DataFrame
        Index = zone_id
        Columns:
          - p_min_headroom_kw
          - p_limiting_time
          - p_zone_at_limiting_time_kw
          - reverse_flow_threshold_kw
    """
    if not isinstance(zone_p_df_rf, pd.DataFrame):
        raise TypeError("zone_p_df_rf must be a pandas DataFrame")

    rows = []
    for zone_id in zone_p_df_rf.columns:
        p_zone = zone_p_df_rf[zone_id]

        # Headroom-to-threshold series
        headroom_series = p_zone.where(p_zone > reverse_flow_threshold_kw, 0.0)

        # Scalar minimum headroom
        p_min_headroom_kw = float(headroom_series.min(skipna=True)) if headroom_series.notna().any() else np.nan

        # Limiting time (where headroom is smallest)
        if headroom_series.notna().any():
            p_limiting_time = headroom_series.idxmin(skipna=True)
            p_zone_at_limiting_time_kw = float(p_zone.loc[p_limiting_time]) if pd.notna(p_zone.loc[p_limiting_time]) else np.nan
        else:
            p_limiting_time = np.nan
            p_zone_at_limiting_time_kw = np.nan

        rows.append({
            "zone_id": zone_id,
            "p_min_headroom_kw": p_min_headroom_kw,
            "p_limiting_time": p_limiting_time,
            "p_zone_at_limiting_time_kw": p_zone_at_limiting_time_kw,
            "reverse_flow_threshold_kw": float(reverse_flow_threshold_kw),
        })
    
    rf_zone_headroom_df = pd.DataFrame(rows).set_index("zone_id")
    return rf_zone_headroom_df



def summarize_rf_bus_p_headroom(zones_rf, rf_zone_headroom_df):
    """
    Aggregate RF zone headroom to the bus level.

    For each bus:
      - p_min_headroom_kw = min over all RF zones that contain the bus
      - p_limiting_zone   = which RF zone produced that minimum

    Parameters
    ----------
    zones_rf : dict
        dg.get_zones(...) output for RF zones:
          zones_rf[zone_id]["buses"] = set of bus names

    rf_zone_headroom_df : pd.DataFrame
        Output of summarize_rf_zone_p_headroom(...), indexed by zone_id and containing:
          - p_min_headroom_kw

    Returns
    -------
    rf_bus_headroom_df : pd.DataFrame
        Index = bus
        Columns:
          - p_min_headroom_kw
          - p_limiting_zone
          - n_zones
    """
    if not isinstance(zones_rf, dict):
        raise TypeError("zones_rf must be a dict")

    if not isinstance(rf_zone_headroom_df, pd.DataFrame):
        raise TypeError("rf_zone_headroom_df must be a pandas DataFrame")

    if "p_min_headroom_kw" not in rf_zone_headroom_df.columns:
        raise ValueError("rf_zone_headroom_df must contain 'p_min_headroom_kw'")

    # Build bus -> zones mapping
    bus_to_zones = {}
    for zone_id, zone_info in zones_rf.items():
        for bus in zone_info["buses"]:
            bus_to_zones.setdefault(bus, []).append(zone_id)

    rows = []
    for bus, zlist in bus_to_zones.items():
        valid = [z for z in zlist if z in rf_zone_headroom_df.index]

        if len(valid) == 0:
            rows.append({
                "bus": bus,
                "p_min_headroom_kw": np.nan,
                "p_limiting_zone": np.nan,
                "n_zones": len(zlist),
            })
            continue

        s = rf_zone_headroom_df.loc[valid, "p_min_headroom_kw"]
        p_min = s.min(skipna=True)
        p_lim = s.idxmin(skipna=True) if s.notna().any() else np.nan

        rows.append({
            "bus": bus,
            "p_min_headroom_kw": float(p_min) if pd.notna(p_min) else np.nan,
            "p_limiting_zone": p_lim,
            "n_zones": len(zlist),
        })

    rf_bus_headroom_df = pd.DataFrame(rows).set_index("bus").sort_index()
    return rf_bus_headroom_df


def combine_ui_rf_bus_headroom(
    ui_bus_headroom_df,
    rf_bus_headroom_df,
    include_ui_q=True,
):
    """
    Create a simplified, combined bus-level output for UI and RF results.

    Output columns (default):
      - rf_p_min_headroom_kw
      - ui_p_min_headroom_kw
      - ui_q_min_headroom_kvar  (optional, include_ui_q=True)

    Parameters
    ----------
    ui_bus_headroom_df : pd.DataFrame
        Output of pb.summarize_bus_headroom(...). Expected to contain:
          - p_min_headroom_kw
          - q_min_headroom_kvar (if include_ui_q=True)

    rf_bus_headroom_df : pd.DataFrame
        Output of pb.summarize_rf_bus_p_headroom(...). Expected to contain:
          - p_min_headroom_kw

    include_ui_q : bool
        If True, include UI reactive headroom column in combined output.

    Returns
    -------
    combined_df : pd.DataFrame
        Index = bus
        Columns = simplified numeric results only.
    """
    # Start with empty DataFrame indexed by the union of buses
    buses = ui_bus_headroom_df.index.union(rf_bus_headroom_df.index)
    combined_df = pd.DataFrame(index=buses)

    # RF: rename to distinguish from UI
    if "p_min_headroom_kw" in rf_bus_headroom_df.columns:
        combined_df["rf_p_min_headroom_kw"] = rf_bus_headroom_df["p_min_headroom_kw"]
    else:
        combined_df["rf_p_min_headroom_kw"] = pd.NA

    # UI: rename to distinguish from RF
    if "p_min_headroom_kw" in ui_bus_headroom_df.columns:
        combined_df["ui_p_min_headroom_kw"] = ui_bus_headroom_df["p_min_headroom_kw"]
    else:
        combined_df["ui_p_min_headroom_kw"] = pd.NA

    # Optional UI Q
    if include_ui_q:
        if "q_min_headroom_kvar" in ui_bus_headroom_df.columns:
            combined_df["ui_q_min_headroom_kvar"] = ui_bus_headroom_df["q_min_headroom_kvar"]
        else:
            combined_df["ui_q_min_headroom_kvar"] = pd.NA

    ordered_cols = ["rf_p_min_headroom_kw", "ui_p_min_headroom_kw"]
    if include_ui_q:
        ordered_cols.append("ui_q_min_headroom_kvar")

    combined_df = combined_df[ordered_cols].sort_index()
    return combined_df
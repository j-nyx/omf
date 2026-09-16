"""KKT-screened, response-based and catalog-aware upgrade planning.

KKT information comes from the current maximum-HC LP. It is used only to
screen physical actions. Every accepted action is selected from realized HC
gain obtained by applying the action and re-solving that LP.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

import numpy as np

from .data import load_runtime_data, site_index
from .lp import initial_network_state, solve_hc_lp
from .sensitivity import response_for_taps, weighted_constraint_relief


MAX_ITERATIONS = 24
SCREEN_LIMIT = 6
KKT_SCORE_TOL = 1.0e-9
HC_NUMERICAL_TOL_MW = 1.0e-9


def _catalog(data: dict[str, Any], catalog_id: str) -> dict[str, Any]:
    ids = data["catalogs"]["id"].astype(str).tolist()
    if catalog_id not in ids:
        raise ValueError(f"Unknown catalog_id '{catalog_id}'. Available catalogs: {', '.join(ids)}")
    return data["catalogs"]


def _catalog_arrays(catalog: dict[str, Any], asset_type: str) -> tuple[np.ndarray, np.ndarray]:
    if asset_type == "primary_line":
        return catalog["line_rating_a"].astype(float), catalog["line_cost_usd_per_km"].astype(float)
    return catalog["transformer_rating_kva"].astype(float), catalog["transformer_cost_usd"].astype(float)


def _catalog_action_cost(catalog: dict[str, Any], asset_type: str, rating: float, length_km: float) -> float:
    ratings, costs = _catalog_arrays(catalog, asset_type)
    match = np.flatnonzero(np.isclose(ratings, rating))
    if not len(match):
        raise ValueError(f"Rating {rating:g} is not in the {asset_type} catalog")
    cost = float(costs[match[0]])
    return cost * float(length_km) if asset_type == "primary_line" else cost


def _incremental_cost(
    catalog: dict[str, Any], asset_type: str, new_rating: float, length_km: float,
    previously_paid_catalog_cost: float,
) -> tuple[float, float]:
    """Return additional project cost and cumulative catalog cost.

    The first replacement pays its catalog project cost. A later replacement
    of the same physical asset pays only the positive difference, so prior
    accepted expenditure is never charged again.
    """
    cumulative = _catalog_action_cost(catalog, asset_type, new_rating, length_km)
    return max(0.0, cumulative - float(previously_paid_catalog_cost)), cumulative


def _physical_asset_groups(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Group phase constraint rows into one physical line/transformer asset."""
    groups: OrderedDict[tuple[str, str], list[int]] = OrderedDict()
    ids = data["thermal"]["asset_id"].astype(str)
    types = data["thermal"]["asset_type"].astype(str)
    for row_index, key in enumerate(zip(types, ids)):
        groups.setdefault(key, []).append(row_index)
    return [
        {"asset_type": asset_type, "asset_id": asset_id, "row_indices": np.asarray(indices, dtype=int)}
        for (asset_type, asset_id), indices in groups.items()
    ]


def _pareto_efficient(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove rating/cost actions dominated within one physical asset."""
    efficient = []
    for candidate in candidates:
        dominated = any(
            other is not candidate
            and other["new_rating"] >= candidate["new_rating"]
            and other["incremental_cost_usd"] <= candidate["incremental_cost_usd"]
            and (
                other["new_rating"] > candidate["new_rating"]
                or other["incremental_cost_usd"] < candidate["incremental_cost_usd"]
            )
            for other in candidates
        )
        if not dominated:
            efficient.append(candidate)
    return efficient


def _maximum_hc_guidance(solution) -> dict[str, Any]:
    if solution.inequality_duals is None or solution.residuals is None:
        raise RuntimeError("Maximum-HC LP did not return KKT diagnostics")
    return {
        "duals": solution.inequality_duals,
        "residuals": solution.residuals,
        "row_ranges": solution.row_ranges,
        "source": "current maximum-HC LP KKT",
    }


def _thermal_candidates(
    data: dict[str, Any], state, guidance: dict[str, Any], catalog: dict[str, Any],
    selected: dict[tuple[str, str], dict[str, float]],
) -> list[dict[str, Any]]:
    t0, t1 = guidance["row_ranges"]["thermal"]
    n_hours = len(data["critical"]["hours"])
    n_rows = len(data["thermal"]["asset_id"])
    dual = guidance["duals"][t0:t1].reshape(n_hours, n_rows)
    original = data["thermal"]["original_rating"].astype(float)
    lengths = data["thermal"]["length_km"].astype(float)
    phases = data["thermal"]["asset_phase"].astype(str)
    candidates_by_asset: list[list[dict[str, Any]]] = []

    for group in _physical_asset_groups(data):
        rows = group["row_indices"]
        key = (group["asset_type"], group["asset_id"])
        prior = selected.get(key)
        current_rating = float(prior["rating"] if prior else np.max(original[rows]))
        previously_paid = float(prior["paid_catalog_cost"] if prior else 0.0)
        ratings, _ = _catalog_arrays(catalog, group["asset_type"])
        length_km = float(np.max(lengths[rows]))
        asset_candidates = []
        for new_rating in ratings[ratings > current_rating * (1.0 + 1.0e-10)]:
            incremental_cost, cumulative_cost = _incremental_cost(
                catalog, group["asset_type"], float(new_rating), length_km, previously_paid
            )
            delta_multiplier = float(new_rating) / original[rows] - state.thermal_rating_multipliers[rows]
            kkt_score = float(np.sum(dual[:, rows] * delta_multiplier.reshape(1, -1)))
            if kkt_score <= KKT_SCORE_TOL:
                continue
            asset_candidates.append({
                "kind": "thermal", "asset_id": group["asset_id"], "asset_type": group["asset_type"],
                "physical_row_indices": rows, "new_rating": float(new_rating),
                "rating_unit": str(data["thermal"]["rating_unit"][rows[0]]),
                "incremental_cost_usd": incremental_cost,
                "cumulative_catalog_cost_usd": cumulative_cost,
                "kkt_weighted_score": kkt_score,
                "screening_score": kkt_score / max(incremental_cost, 1.0),
                "affected_constraint_rows": int(n_hours * len(rows)),
                "affected_phases": sorted(set(phases[rows].tolist())),
                "guidance_source": guidance["source"],
            })
        efficient = _pareto_efficient(asset_candidates)
        if efficient:
            candidates_by_asset.append(efficient)

    # Screen physical assets after phase aggregation, then preserve every
    # Pareto-efficient rating option for each retained asset. This prevents a
    # large-rating option from crowding a cheaper option for the same asset out
    # of the realized-gain LP evaluation set.
    candidates_by_asset.sort(
        key=lambda items: max(item["screening_score"] for item in items), reverse=True
    )
    retained = [candidate for items in candidates_by_asset[:SCREEN_LIMIT] for candidate in items]
    retained.sort(key=lambda item: (item["screening_score"], item["kkt_weighted_score"]), reverse=True)
    return retained


def _voltage_candidates(data: dict[str, Any], state, guidance: dict[str, Any], catalog: dict[str, Any]) -> list[dict[str, Any]]:
    v0, v1 = guidance["row_ranges"]["voltage"]
    n_hours = len(data["critical"]["hours"])
    n_outputs = len(data["outputs"]["name"])
    dual = guidance["duals"][v0:v1].reshape(n_hours, n_outputs)
    current_response = response_for_taps(data, state.regulator_taps)
    allowed = data["regulator"]["allowed_taps"].astype(int)
    anchors = data["regulator"]["tap_anchors"].astype(int)
    per_tap_cost = float(catalog["regulator_setting_cost_usd_per_tap"].reshape(-1)[0])
    candidates = []
    for location_index, location in enumerate(data["regulator"]["locations"].astype(str)):
        current = int(state.regulator_taps[location_index])
        for tap in allowed[allowed < current]:
            trial_taps = state.regulator_taps.copy()
            trial_taps[location_index] = int(tap)
            delta = response_for_taps(data, trial_taps) - current_response
            kkt_score = weighted_constraint_relief(dual, delta)
            if kkt_score <= KKT_SCORE_TOL:
                continue
            cost = abs(int(tap) - current) * per_tap_cost
            candidates.append({
                "kind": "regulator", "asset_id": str(location), "asset_type": "regulator",
                "location_index": location_index, "location": str(location), "new_tap": int(tap),
                "incremental_cost_usd": cost, "kkt_weighted_score": kkt_score,
                "screening_score": kkt_score / max(cost, 1.0),
                "response_source": str(data["regulator"]["response_source"].reshape(-1)[0]),
                "response_anchors": anchors.tolist(), "guidance_source": guidance["source"],
            })
    candidates.sort(key=lambda item: (item["screening_score"], item["kkt_weighted_score"]), reverse=True)
    return candidates[:SCREEN_LIMIT]


def _apply_candidate(data: dict[str, Any], state, candidate: dict[str, Any]):
    trial = state.copy()
    if candidate["kind"] == "thermal":
        rows = candidate["physical_row_indices"]
        originals = data["thermal"]["original_rating"].astype(float)[rows]
        trial.thermal_rating_multipliers[rows] = candidate["new_rating"] / originals
    else:
        trial.regulator_taps[candidate["location_index"]] = candidate["new_tap"]
    return trial


def _action_record(
    data: dict[str, Any], iteration: int, candidate: dict[str, Any], old_state, new_state,
    old_solution, new_solution, selected: dict[tuple[str, str], dict[str, float]],
) -> dict[str, Any]:
    common = {
        "iteration": int(iteration), "asset_id": str(candidate["asset_id"]),
        "asset_type": str(candidate["asset_type"]),
        "incremental_cost_usd": float(candidate["incremental_cost_usd"]),
        "kkt_weighted_score": float(candidate["kkt_weighted_score"]),
        "hc_before_mw": float(old_solution.hosting_capacity_mw),
        "hc_after_mw": float(new_solution.hosting_capacity_mw),
        "realized_hc_gain_mw": float(new_solution.hosting_capacity_mw - old_solution.hosting_capacity_mw),
        "binding_constraints_before": old_solution.binding_constraints,
        "guidance_source": str(candidate["guidance_source"]),
    }
    if candidate["kind"] == "thermal":
        key = (candidate["asset_type"], candidate["asset_id"])
        rows = candidate["physical_row_indices"]
        old_rating = float(selected[key]["rating"] if key in selected else np.max(data["thermal"]["original_rating"].astype(float)[rows]))
        return common | {
            "old_state": {"rating": old_rating, "unit": candidate["rating_unit"]},
            "new_state": {"rating": float(candidate["new_rating"]), "unit": candidate["rating_unit"]},
            "old_rating": old_rating, "new_rating": float(candidate["new_rating"]),
            "rating_unit": candidate["rating_unit"], "affected_phases": candidate["affected_phases"],
            "affected_constraint_rows": int(candidate["affected_constraint_rows"]),
            "reason": "physical-asset KKT screening; Pareto catalog action selected by realized HC gain per incremental cost after LP re-solve",
            "action_type": "catalog_replacement",
        }
    i = candidate["location_index"]
    old_tap, new_tap = int(old_state.regulator_taps[i]), int(new_state.regulator_taps[i])
    anchors = candidate["response_anchors"]
    return common | {
        "location": str(candidate["location"]), "regulator_location": str(candidate["location"]),
        "old_state": {"tap": old_tap}, "new_state": {"tap": new_tap},
        "old_tap": old_tap, "new_tap": new_tap,
        "selected_response_source": str(candidate["response_source"]),
        "response_anchor_information": {
            "anchors": anchors,
            "old_tap_mode": "exact_anchor" if old_tap in anchors else "linear_interpolation",
            "new_tap_mode": "exact_anchor" if new_tap in anchors else "linear_interpolation",
        },
        "reason": "maximum-HC KKT screening with frozen Iowa240 response; selected by realized HC gain per incremental cost after LP re-solve",
        "action_type": "regulator_tap_setting",
    }


def plan_upgrade_impl(site_id: str, target_hc_mw: float, catalog_id: str) -> dict:
    if not np.isfinite(target_hc_mw) or float(target_hc_mw) <= 0.0:
        raise ValueError("target_hc_mw must be a finite positive absolute MW target")
    started = time.perf_counter()
    data = load_runtime_data()
    index = site_index(data, site_id)
    catalog = _catalog(data, catalog_id)
    state = initial_network_state(data)
    baseline = solve_hc_lp(site_id, data, state)
    if not baseline.success:
        raise RuntimeError(f"Baseline LP failed for {site_id}: {baseline.message}")
    if baseline.hosting_capacity_mw + HC_NUMERICAL_TOL_MW >= float(target_hc_mw):
        return {
            "site_id": str(data["sites"]["id"][index]), "baseline_hc_mw": baseline.hosting_capacity_mw,
            "target_hc_mw": float(target_hc_mw), "upgrade_required": False, "target_achieved": True,
            "post_upgrade_hc_mw": baseline.hosting_capacity_mw, "released_hc_mw": 0.0,
            "total_cost_usd": 0.0, "upgrades": [], "runtime_sec": time.perf_counter() - started,
            "success": True, "failure_reason": None,
        }

    current = baseline
    actions: list[dict[str, Any]] = []
    selected: dict[tuple[str, str], dict[str, float]] = {}
    failure = None
    for iteration in range(1, MAX_ITERATIONS + 1):
        guidance = _maximum_hc_guidance(current)
        candidates = _thermal_candidates(data, state, guidance, catalog, selected)
        candidates += _voltage_candidates(data, state, guidance, catalog)
        candidates.sort(key=lambda item: (item["screening_score"], item["kkt_weighted_score"]), reverse=True)
        if not candidates:
            failure = "catalog exhausted or no maximum-HC KKT/response-supported physical action"
            break

        evaluated = []
        for candidate in candidates:
            trial_state = _apply_candidate(data, state, candidate)
            trial_result = solve_hc_lp(site_id, data, trial_state)
            if trial_result.success and trial_result.hosting_capacity_mw > current.hosting_capacity_mw + HC_NUMERICAL_TOL_MW:
                realized_gain = trial_result.hosting_capacity_mw - current.hosting_capacity_mw
                decision_metric = realized_gain / max(candidate["incremental_cost_usd"], 1.0)
                evaluated.append((decision_metric, realized_gain, candidate, trial_state, trial_result))
        if not evaluated:
            failure = "screened physical actions did not increase HC after maximum-HC LP re-solve"
            break

        _, _, chosen, next_state, next_result = max(evaluated, key=lambda item: (item[0], item[1]))
        action = _action_record(data, iteration, chosen, state, next_state, current, next_result, selected)
        actions.append(action)
        if chosen["kind"] == "thermal":
            selected[(chosen["asset_type"], chosen["asset_id"])] = {
                "rating": float(chosen["new_rating"]),
                "paid_catalog_cost": float(chosen["cumulative_catalog_cost_usd"]),
            }
        state, current = next_state, next_result
        if current.hosting_capacity_mw + HC_NUMERICAL_TOL_MW >= float(target_hc_mw):
            break

    achieved = bool(current.hosting_capacity_mw + HC_NUMERICAL_TOL_MW >= float(target_hc_mw))
    if not achieved and failure is None:
        failure = f"iteration limit {MAX_ITERATIONS} reached"
    return {
        "site_id": str(data["sites"]["id"][index]), "baseline_hc_mw": baseline.hosting_capacity_mw,
        "target_hc_mw": float(target_hc_mw), "upgrade_required": True, "target_achieved": achieved,
        "post_upgrade_hc_mw": current.hosting_capacity_mw,
        "released_hc_mw": current.hosting_capacity_mw - baseline.hosting_capacity_mw,
        "total_cost_usd": float(sum(action["incremental_cost_usd"] for action in actions)),
        "upgrades": actions, "runtime_sec": time.perf_counter() - started,
        "success": achieved, "failure_reason": None if achieved else failure,
    }

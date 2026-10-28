"""
top level function for sandia protection analysis.

may make more sense as a class...
"""

import time
from pathlib import Path
import pandas as pd
import os

from . import decaf_db as ddb
from . import decaf_graph as dg
from . import decaf_power_balance as pb

PARENT_DIR = Path(__file__).parent.resolve()
TEMP_DIR = Path(PARENT_DIR, "temp_delme")  # location for db to be created

def protection_analysis(
    ckt_dss_fp=None,  # should probably be non-null - required for graph
    ckt_name='Test Circuit',
    ami_fp=None,
    scada_fp=None,
    der_pipeline_csv_fp=None,  # systems approved, but not yet connected
    new_interconnection_csv_fp=None,  # systems not yet approved, may required mitigation
    database_fp=None,
    init_db=True,
    is_single_file_model=1,  # ckt_dss_fp points to a single file model
        ):
    """
    NOTE: expects inputs as file path

    NOTE: unsure what the difference between `der_pipeline_csv_fp` and 
    `new_interconnection_csv_fp` is.  Code accounts for interconnections.

    Original Description:
    Sandia: do a model-based HC analysis to calculate short circuit current 
    contributions at all circuit buses, stepping through each interconnection 
    and showing the changes over the pipeline.

    Description: Calculates the bus-level protection-constrained hosting capacity based on 
    limits for reverse power flow and unintentional islanding, stepping through each 
    interconnection and showing the changes over the pipeline.'

    """
    print()
    print('Running protection_analysis...')

    # init unique database using time
    if database_fp is None:
        init_time = str(time.time()).replace('.','_')
        db_fp = f'unnamed_database_{init_time}.db'
    else:
        db_fp = database_fp

    if not isinstance(db_fp, Path):
        db_fp = Path(db_fp)

    if init_db:
        # ensure db doesn't exist.
        if os.path.exists(db_fp):
            print(f'Deleting existing database at {db_fp}')
            os.remove(db_fp)

        print(f'Creating database from inputs at {db_fp}')

        ddb.init_db(db_fp)

        # add inputs to database
        if ami_fp is not None:
            ddb.infuse_ami_to_db(db_fp, ami_fp)
            print('Added AMI to db...')
        if scada_fp is not None:
            ddb.infuse_scada_to_db(db_fp, scada_fp)
            print('Added SCADA to db...')
        if new_interconnection_csv_fp is not None:
            ddb.infuse_interconnect(db_fp, new_interconnection_csv_fp)
            print('Added interconnection data to db...')
        if ckt_dss_fp is not None:
            print('Added circuit data to db...')
            ddb.insert_circuit_model(db_fp,
                ckt_dss_fp,
                model_name=ckt_name,
                is_single_file_model=is_single_file_model,
                )
        print(f'Database creation complete: {db_fp}')
    else:

        print(f'ASSERT database exists at {db_fp}')
        # TODO: handle circuit dss... depending on if single file model
        ckt_dss_fp = ddb.retrieve_circuit_model(db_fp)

    print(f'ASSERT openDSS main dss file exists at {ckt_dss_fp}')

    # execute algorithms on database at db_fp and dss model at ckt_dss_fp

    # Required parameters (can later expose these as function arguments)
    tol_p = 50.0 # tolerance for active power balance for unintentional islanding, in kW
    tol_q = 50.0 # tolerance for reactive power balance for unintentional islanding, in kVAR
    reverse_flow_threshold_kw = 0.0 # threshold for reverse power flow, in kW

    # Get AMI from database
    ami = ddb.get_ami(db_fp)

    # Build graph from DSS model
    inc_matrix = dg.get_incidence_matrix(ckt_dss_fp)
    graph = dg.get_directed_graph(inc_matrix)

    # Identify zone boundary buses and split by type (ui vs rf)
    zone_bus = dg.get_zone_buses()  # returns dict: {bus: "ui" or "rf"}

    zone_bus_ui = {bus: zt for bus, zt in zone_bus.items() if zt == "ui"}
    zone_bus_rf = {bus: zt for bus, zt in zone_bus.items() if zt == "rf"}

    # Build zones separately (so numbering is independent)
    zones_ui = dg.get_zones(graph, zone_bus_ui)
    zones_rf = dg.get_zones(graph, zone_bus_rf)

    # -------------------- UI pipeline (P/Q balance + capacitor scenarios) --------------
    # Zone stats for UI zones
    zone_stat_df_ui = dg.get_zone_stat_df(graph, zones_ui)

    # Identify load buses per UI zone, then build zone P/Q dataframes from AMI
    zone_load_buses_ui = dg.get_zone_load_buses(zones_ui)

    total_zone_p_ui = {}
    total_zone_q_ui = {}

    for zone_id, zone_loads in zone_load_buses_ui.items():
        total_zone_p_ui[zone_id] = ami["kW"][zone_loads].sum(axis=1)
        total_zone_q_ui[zone_id] = ami["kVAR"][zone_loads].sum(axis=1)

    zone_p_df_ui = pd.DataFrame(total_zone_p_ui)
    zone_q_df_ui = pd.DataFrame(total_zone_q_ui)

    # Run zone power balance masks + Q scenario generation
    p_fail_df, q_fail_any_df, pq_fail_df, q_adj_by_zone, q_fail_by_zone, q_combo_meta_by_zone = pb.build_zone_power_balance_masks(
        zone_p_df_ui,
        zone_q_df_ui,
        zone_stat_df_ui,
        tol_p=tol_p,
        tol_q=tol_q,
    )

    # Compute UI headroom (zone-level), then bus-level aggregation
    headroom_df_ui = pb.summarize_power_headroom(
        zone_p_df_ui,
        q_adj_by_zone,
        tol_p=tol_p,
        tol_q=tol_q,
    )

    ui_bus_headroom_df = pb.summarize_bus_headroom(zones_ui, headroom_df_ui)

    # -------------------- RF pipeline (P only, reverse-flow threshold) --------------------
    zone_load_buses_rf = dg.get_zone_load_buses(zones_rf)

    total_zone_p_rf = {}
    for zone_id, zone_loads in zone_load_buses_rf.items():
        total_zone_p_rf[zone_id] = ami["kW"][zone_loads].sum(axis=1)

    zone_p_df_rf = pd.DataFrame(total_zone_p_rf)

    rf_zone_headroom_df = pb.summarize_rf_zone_p_headroom(
        zone_p_df_rf,
        reverse_flow_threshold_kw=reverse_flow_threshold_kw,
    )

    rf_bus_headroom_df = pb.summarize_rf_bus_p_headroom(zones_rf, rf_zone_headroom_df)

    # Combine UI and RF headroom dataframes
    combined_bus_headroom_df = pb.combine_ui_rf_bus_headroom(
        ui_bus_headroom_df=ui_bus_headroom_df,
        rf_bus_headroom_df=rf_bus_headroom_df,
        include_ui_q=True,
    )

    return combined_bus_headroom_df

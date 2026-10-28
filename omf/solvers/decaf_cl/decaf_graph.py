"""
Code used to creat directed graph of a given opendss feeder
"""

import os
import random
import json
import opendssdirect as dss
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import re

# NOTE: bokeh plotting may not be compatible with OMF packages
from bokeh.plotting import figure # type: ignore
from bokeh.models import ColumnDataSource, Legend, LegendItem # type: ignore
from bokeh.palettes import Category20, Category10 # type: ignore

def get_incidence_matrix(model_fp):
    """
    Clears any openDSS direct feeder,
    Solves system at model_fp,
    and returns incidence matrix where the `value` column indicates flow
    entering bus via pde.
    """
    dss.run_command('clear')
    dss.run_command(f"""compile "{model_fp}" """)  # quotes intentional
    dss.run_command('solve')

    dss.run_command("calcincmatrix_o")

    paths_to_remove = []

    inc_matrix_fp = dss.run_command("export incmatrix incidence_matrix.csv")
    inc_matrix = pd.read_csv(inc_matrix_fp)
    paths_to_remove.append(inc_matrix_fp)

    row_fp = dss.run_command("export incmatrixrow incidence_matrix_row.csv")
    row_names = pd.read_csv(row_fp)
    paths_to_remove.append(row_fp)

    col_fp = dss.run_command("export incmatrixcols incidence_matrix_cols.csv")
    col_names = pd.read_csv(col_fp)
    paths_to_remove.append(col_fp)

    # remove temporary opendss exports
    for temp_path in paths_to_remove:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    inc_bus_names = col_names.iloc[inc_matrix['Col']]
    inc_matrix['bus_name'] = inc_bus_names.reset_index(drop=True)
    inc_matrix['bus_name'] = inc_matrix['bus_name'].astype(str)
    inc_pde_name = row_names.iloc[inc_matrix['Row']]
    inc_matrix['pde_name'] = inc_pde_name.reset_index(drop=True)

    # addition to help with per phase graph
    inc_matrix['bus_nodes'] = ''
    inc_matrix['n_bus_nodes'] = 0
    inc_matrix['kv_base'] = 0.0

    for idx, row in inc_matrix.iterrows():
        bus_name = row['bus_name']
        dss.Circuit.SetActiveBus(bus_name)
        inc_matrix.at[idx, 'n_bus_nodes'] = int(dss.Bus.NumNodes())
        inc_matrix.at[idx, 'bus_nodes'] = dss.Bus.Nodes()
        inc_matrix.at[idx, 'kv_base'] = float(dss.Bus.kVBase())
        inc_matrix.at[idx, 'x_coord'] = float(dss.Bus.X())
        inc_matrix.at[idx, 'y_coord'] = float(dss.Bus.Y())

    return inc_matrix


def get_directed_graph(incidence_matrix):
    """
    Create networkx directed graph from incidence matrix

    ASSERT: feeder used to create incidence matrix is active in dss direct
    This will be the case if exqecuted immediately after `get_incidence_matrix`
    """
    # init graph
    graph = nx.MultiDiGraph()

    # create nodes
    bus_names = dss.Circuit.AllBusNames()
    bus_dict = {}
    for bus_name in bus_names:
        dss.Circuit.SetActiveBus(bus_name)
        bus_dict[bus_name] = {}
        bus_dict[bus_name]['kv_base'] = float(dss.Bus.kVBase())
        bus_dict[bus_name]['x_coord'] = float(dss.Bus.X())
        bus_dict[bus_name]['y_coord'] = float(dss.Bus.Y())

    # add nodes to graph
    for bus_name, bus_attributes in bus_dict.items():
        graph.add_node(bus_name, attr=bus_attributes)

    # add edges (if possible) to graph
    for pde_name, pde_df in incidence_matrix.groupby('pde_name'):
        n_connects = len(pde_df)

        if n_connects == 1:
            # print(f"Skipping edge {pde_name} - only one connection")
            continue

        if n_connects > 2:
            # print(f"Skipping edge {pde_name} - too many connections")
            # should probably not happen...
            continue

        sorted_pde = pde_df.sort_values('Value', ascending=False)

        bus_1 = sorted_pde.iloc[0]['bus_name']  # value of 1
        bus_2 = sorted_pde.iloc[1]['bus_name']  # value of -1

        edge_dict = {
            'kind': pde_name.split('.')[0],
            'pde_name': pde_name,
        }

        graph.add_edge(bus_1, bus_2, attr=edge_dict)

    # check for loops
    graph_copy = graph.copy()
    cycles = []

    while True:
        try:
            cycle = nx.find_cycle(graph_copy)
            cycles.append(cycle)
            # Remove the edges forming the cycle to find the next one
            graph_copy.remove_edge(*cycle[-1])
        except nx.NetworkXNoCycle:
            break
    if len(cycles) > 0:
        print(f"WARNING: directed graph has {len(cycles)} loops!")

    return graph


def get_directed_nodal_graph(incidence_matrix):
    """
    Create networkx directed graph from incidence matrix

    Uses a 'per phase' / nodal approach instead of just bus to bus.

    ASSERT: feeder used to create incidence matrix is active in dss direct
    This will be the case if exqecuted immediately after `get_incidence_matrix`
    """

    # collect pde 'long bus' connection data (for phase to phase connections)
    full_pde_df = None
    collected_classes = []

    multi_bus_xfmr = 0  # used to count xfmr with more than 2 bus (possibly split phase)
    # collect pde data
    for _, row, in incidence_matrix.iterrows():
        dss.Circuit.SetActiveElement(row['pde_name'])
        class_name = dss.ActiveClass.ActiveClassName()
        if class_name in collected_classes:
            continue
        else:
            active_class_json = json.loads(dss.ActiveClass.ToJSON())
            current_ac_df = pd.DataFrame(active_class_json)

            if full_pde_df is None:
                full_pde_df = current_ac_df
            else:
                full_pde_df = pd.concat([full_pde_df, current_ac_df])

            collected_classes.append(class_name)

    # creation of graph
    # which is essentially 3 graphs in one object....

    graph = nx.MultiDiGraph()

    bus_dict = {}

    # for maybe nicer plots later...
    general_coord_offset = 0.0001

    # collect all information for nodes
    for _, row, in incidence_matrix.drop_duplicates('bus_name').iterrows():
        short_bus = row['bus_name']
        node_list = row['bus_nodes']

        for node in node_list:
            if node == 0:
                # skip grounds...
                continue

            node_name = f"{short_bus}.{node}"
            node_dict = row[['kv_base', 'x_coord', 'y_coord']].to_dict()

            # add corrdinate offset per phase
            node_offset = general_coord_offset * node
            node_dict['x_coord'] += node_offset
            node_dict['y_coord'] += node_offset
            node_dict['src_phase'] = np.nan

            bus_dict[node_name] = node_dict

    # add nodes to graph
    for bus_name, bus_attributes in bus_dict.items():
        graph.add_node(bus_name, attr=bus_attributes)

    # handle edges
    for pde_long_name, pde_df in incidence_matrix.groupby('pde_name'):
        n_connects = len(pde_df)

        if n_connects == 1:
            # print(f"Skipping edge {pde_name} - only one connection")
            continue

        if n_connects > 2:
            # print(f"Skipping edge {pde_name} - too many connections")
            # should probably not happen...
            continue

        sorted_pde = pde_df.sort_values('Value', ascending=False)

        bus_1 = ''
        bus_2 = ''

        for _, pde_row in sorted_pde.iterrows():
            if pde_row['Value'] == 1:
                # bus 1 i.e. from bus
                bus_1 = pde_row['bus_name']
            elif pde_row['Value'] == -1:
                # bus 2 i.e. to bus
                bus_2 = pde_row['bus_name']
            else:
                print('error in value...')  # shouldn't happen.

        # idenfity pde from pde_df
        pde_kind, pde_name = pde_long_name.split('.')

        mask_0 = full_pde_df['DSSClass'] == pde_kind
        mask_1 = full_pde_df['name'] == pde_name
        mask = mask_0 & mask_1
        current_pde = full_pde_df[mask]

        # different pde have different locations of bus information
        is_switch = False

        if pde_kind == 'Line':
            # for lines
            if 'Switch' in current_pde:
                is_switch = current_pde['Switch'].values[0]

            # get long bus
            pde_long_bus1 = current_pde['bus1'].values[0]
            pde_long_bus2 = current_pde['bus2'].values[0]

        elif pde_kind == 'Transformer':
            buses_list = current_pde['bus'].values[0]

            if len(buses_list) > 2:
                # NOTE: likely split phase situations...
                # print(f'ignoring more than 2 buses: {pde_long_name}')
                multi_bus_xfmr += 1

            pde_long_bus1 = buses_list[0]
            pde_long_bus2 = buses_list[1]
        else:
            print(f"ERROR pde not handeled : {pde_kind}")
            print(f"Skipped : {pde_long_name}")

            continue

        pde_n_phases = current_pde['phases'].values[0]
        # get the phase splits
        pde_bus1_splits = pde_long_bus1.split('.')
        pde_bus2_splits = pde_long_bus2.split('.')

        # get the short bus name
        pde_bus1 = pde_bus1_splits.pop(0)
        pde_bus2 = pde_bus2_splits.pop(0)

        # check for short bus match (else edge defined reverse of flow)
        # NOTE: bus_1 -> bus_2 from incidence matrix ie. trusted as true
        if pde_bus1 != bus_1:
            print(f'mis match of bus 1 : {pde_long_name} - {bus_1} != {pde_bus1}')
        if pde_bus2 != bus_2:
            print(f'mis match of bus 2 : {pde_long_name} - {bus_2} != {pde_bus2}')

        current_phase = 0
        # ASSERT - same number of phase connections...
        # NOTE: if pde defined in reverse, swap bus order here, else, directed graph will break
        for b1_phase, b2_phase in zip(pde_bus1_splits, pde_bus2_splits):
            # define phase
            if b1_phase == b2_phase:
                edge_phase = b1_phase
            else:
                edge_phase = f"{b1_phase}-{b2_phase}"

            # skip common ground definitions
            if str(edge_phase) == '0':
                continue

            edge_name = f"{pde_name}.{edge_phase}"
            b1_name = f"{pde_bus1}.{b1_phase}"
            b2_name = f"{pde_bus2}.{b2_phase}"

            edge_dict = {
                    'edge_name': edge_name,
                    'pde_long_name': pde_long_name,
                    'pde_name': pde_name,
                    'kind': pde_kind,
                    'is_switch': is_switch,
                }

            graph.add_edge(b1_name, b2_name, attr=edge_dict)
            current_phase += 1

        if current_phase != pde_n_phases:
            print(f'maybe a phase is dropped? : {edge_name}')

    # nodal graph created - is effectively 3 graphs in one object
    # identify source bus and total phases
    dss.Circuit.SetActiveClass('Vsource')
    src_json = dss.ActiveClass.ToJSON()
    vsrc_df = pd.DataFrame(json.loads(src_json))
    long_bus_1 = vsrc_df['bus1'].values[0]
    short_bus = long_bus_1.split('.')[0]

    if 'phases' in vsrc_df:
        n_phases = vsrc_df['phases'].values[0]
    else:
        n_phases = 3

    # from source bus phases, cast all descendants primary phase
    for phase in list(range(1, n_phases+1)):
        src_node = f"{short_bus}.{phase}"
        descendents = nx.descendants(graph, src_node)
        for node in descendents:
            graph.nodes[node]['attr']['src_phase'] = phase

    return graph


def _assign_src_phase(nodal_graph, df, bus1='bus1'):
    """
    assign source phase to dataframe based on nodal graph.
    """
    df['short_bus'] = [x[0] for x in df[bus1].str.split('.')]
    df['on_primary_1'] = False
    df['on_primary_2'] = False
    df['on_primary_3'] = False

    possible_phases = [1, 2, 3]
    for ndx, row in df.iterrows():
        short_bus = row['short_bus']
        for phase in possible_phases:
            node_name = f"{short_bus}.{phase}"
            if node_name in nodal_graph.nodes:
                src_node = nodal_graph.nodes[node_name]['attr']['src_phase']
                if src_node is np.nan:
                    continue
                df.at[ndx, f'on_primary_{src_node}'] = True
    df['n_primary_connections'] = df[['on_primary_1', 'on_primary_2', 'on_primary_3']].astype(int).sum(axis=1)

    return df


def get_element_src_phase_df(nodal_graph):
    """
    return dataframe of dss elements with source phase identified
    """
    # transformers
    dss.Circuit.SetActiveClass('Transformer')
    xfmr_df = pd.DataFrame(json.loads(dss.ActiveClass.ToJSON()))

    # xfmr buses are in a list, bus1 isolated
    xfmr_df['bus1'] = ''
    for ndx, xfmr_row in xfmr_df.iterrows():
        xfmr_df.at[ndx, 'bus1'] = xfmr_row['bus'][0]

    xfmr_df = _assign_src_phase(nodal_graph, xfmr_df)

    # loads
    dss.Circuit.SetActiveClass('Load')
    load_df = pd.DataFrame(json.loads(dss.ActiveClass.ToJSON()))
    load_df = _assign_src_phase(nodal_graph, load_df)

    # pv systems
    dss.Circuit.SetActiveClass('PVSystem')
    pv_df = pd.DataFrame(json.loads(dss.ActiveClass.ToJSON()))
    pv_df = _assign_src_phase(nodal_graph, pv_df)

    # capacitors
    dss.Circuit.SetActiveClass('Capacitor')
    cap_df = pd.DataFrame(json.loads(dss.ActiveClass.ToJSON()))
    cap_df = _assign_src_phase(nodal_graph, cap_df)

    # combine all into one df
    combined_df = pd.concat([
        xfmr_df,
        load_df,
        pv_df,
        cap_df
    ]
    )
    # slim down df to 'important' columns
    columns_to_keep = [
        'DSSClass', 'name', 'phases', 'conn', 'bus1', 'bus', 'kV', 'kv', 'kVA', 'kW', 'kvar',
        'short_bus', 'model', 'enabled', 'on_primary_1', 'on_primary_2', 'on_primary_3',
        'n_primary_connections'
    ]
    return combined_df[columns_to_keep]


def get_p_per_phase():
    """
    get per phase powers to check per phase groupings...

    ASSERT model is loaded in openDSSdirect
    """
    export_path = dss.run_command('export p_byphase')

    # non-default import due to export file structure oddities
    per_phase_p = pd.read_csv(export_path, header=None, usecols=range(10))

    # replacing of header columns
    header_col = per_phase_p.iloc[0]
    header_col = [x.strip() for x in header_col]
    per_phase_p.columns = header_col
    per_phase_p.drop(index=0, inplace=True)
    per_phase_p.reset_index(inplace=True, drop=True)
    # cleanup of types
    numeric_cols = [
        'NumTerminals', 'NumConductors', 
        'NumPhases', 'kW1', 'kvar1',
        'kW2', 'kvar2', 'kW3', 'kvar3']
    for col in numeric_cols:
        per_phase_p[col] = pd.to_numeric(per_phase_p[col])

    return per_phase_p


def get_zone_buses(feeder_fp=None):
    """
    return list of:
    bus2 from current dss feeder switches,
    transformer monitored bus from voltage regulators, 
    and voltage source bus 1.

    if feeder_fp is given, will load that feeder instead of using existing

    TODO: return should be a dictionary, with bus:zone_type, where
    zone_type is either ui (unintentional island) or rf (reverse flow)

    """
    if feeder_fp is not None:
        dss.run_command('clear')
        dss.run_command(f"""compile "{feeder_fp}" """)  # quotes intentional
        dss.run_command('solve')

    line_df = dss.utils.lines_to_dataframe()
    switch_mask = line_df['IsSwitch']
    switches = line_df[switch_mask]

    switch_bus = []
    for _, switch_row in switches.iterrows():
        bus_name = switch_row['Bus2']  # ASSERT: bus2 is 'downstream' of switch
        short_bus = bus_name.split('.')[0]  # to remove phase numbers
        switch_bus.append(short_bus)

    # TODO: collect bus 2 of vsrc, any vreg
    # id unique vreg buses
    reg_bus = []
    for reg_name, reg_row in dss.utils.regcontrols_to_dataframe().iterrows():
        bus = dss.run_command(f"? transformer.{reg_row['Transformer']}.bus")
        short_bus = bus.split('.')[0]
        reg_bus.append(short_bus)

    reg_bus = list(set(reg_bus))

    # vsrc
    vsrc_df = dss.utils.vsources_to_dataframe()
    v_name = vsrc_df.index[0]

    v_bus = dss.run_command(f"? vsource.{v_name}.bus1")
    short_v_bus = [v_bus.split('.')[0]]

    all_bus = []
    all_bus.extend(switch_bus)
    all_bus.extend(reg_bus)
    all_bus.extend(short_v_bus)

    all_bus_types = {}
    for bus in switch_bus:
        all_bus_types[bus] = 'ui'
    for bus in reg_bus:
        all_bus_types[bus] = 'rf'
    for bus in short_v_bus:
        all_bus_types[bus] = 'rf'

    return all_bus_types


def get_zones(graph, bus_list, min_bus_n=1, lvl2=False):
    """
    Based on input graph, find all zones downstream of givel bus_list.
    Then, find all new zones from subtraction and intersection.
    Finally, reduce all results to only unique zones

    TODO: accomodate for bus_list now describing the zone type of ui or rf

    """
    zone_types = {}
    # find all downstream zones
    # level 1
    level_1_zones = {}
    zone_n = 1
    for bus_name, zone_type in bus_list.items():
        descendents = list(nx.descendants(graph, bus_name))
        # ensure bus in set
        descendents.append(bus_name)  # added 20260304
        level_1_zones[zone_n] = set(descendents)

        # note: used internal to function only
        zone_types[zone_n] = {}
        zone_types[zone_n]['bus_name'] = bus_name
        zone_types[zone_n]['zone_type'] = zone_type

        zone_n += 1

    # level 2
    if lvl2:
        level_2_zones = {}
        # for each zone
        for zone_n, zone_set in level_1_zones.items():
            # find difference of all other zones.
            for next_zone_n, next_zone_set in level_1_zones.items():
                # NOTE: zone types not included in level 2 zones

                if zone_n == next_zone_n:
                    continue
                set_dif = zone_set - next_zone_set

                # check if new zone
                if len(set_dif) == 0:
                    # empty set - i.e self-self
                    continue

                if len(set_dif) == len(zone_set):
                    # no overlap
                    continue

                # length of set dif must be different than original and non zero,
                # so sub zone exists...
                zone_name = f"{zone_n}-{next_zone_n}"
                level_2_zones[zone_name] = set_dif

                # if set dif exists, set intersection must also exist.
                set_intersection = zone_set & next_zone_set
                zone_name = f"{zone_n}&{next_zone_n}"
                level_2_zones[zone_name] = set_intersection

        # NOTE: the above could likely be carried out again for lvl 3 zones
        zones = {'lvl 1': level_1_zones, 'lvl 2': level_2_zones}

    else:
        zones = {'lvl 1': level_1_zones}

    unique_sets = set()
    u_zone_dict = {}
    for zone_lvl, zone_dict in zones.items():
        for zone_name, zone_data in zone_dict.items():
            # get zone type...
            if zone_lvl == 'lvl 1':
                zone_type = zone_types[zone_name]['zone_type']
            else:
                zone_type = ''

            # skip too small
            if len(zone_data) < min_bus_n:
                continue
            # freeze for hash check
            fset = frozenset(zone_data)

            if fset not in unique_sets:
                # add to unique sets
                unique_sets.add(fset)
                # modify zone name to lvl x.zone_name
                lvl_zone_name = f"{zone_lvl}.{zone_name}"
                # add to u_zone_dict
                u_zone_dict[lvl_zone_name] = {}
                u_zone_dict[lvl_zone_name]['buses'] = set(zone_data)
                u_zone_dict[lvl_zone_name]['type'] = zone_type

    return u_zone_dict


def _parse_dss_kvar_steps(kvar_text):
    """
    Parse OpenDSS capacitor kvar output into a list of floats.

    Handles:
      - "300"                  -> [300.0]
      - "[100, 200, 50]"       -> [100.0, 200.0, 50.0]
      - "[200 200 200]"        -> [200.0, 200.0, 200.0]  (space-separated)
      - "[200, 200 200]"       -> [200.0, 200.0, 200.0]  (mixed separators)
    """
    if kvar_text is None:
        return []

    s = str(kvar_text).strip()

    # Scalar case
    if not (s.startswith("[") and s.endswith("]")):
        try:
            return [float(s)]
        except ValueError:
            return []

    # Remove brackets and split into tokens by comma and/or whitespace
    inner = s[1:-1].strip()
    if inner == "":
        return []

    # Split on commas OR any whitespace (one or more)
    tokens = re.split(r"[,\s]+", inner)

    # Convert tokens to floats, skipping any empty tokens
    out = []
    for tok in tokens:
        if tok == "":
            continue
        out.append(float(tok))

    return out


def get_zone_stat_df(graph, zones):
    """
    tally static zone results.

    ASSERT: feeder associated with zones is still active in dss direct
    """
    zone_dict = {}
    for zone_id, new_zone_dict in zones.items():

        descendents = new_zone_dict['buses']

        zone_load_n = 0
        zone_pv_n = 0
        zone_cap_n = 0
        zone_cap_steps_n = 0

        zone_load_kw = 0
        zone_load_kvar = 0

        zone_cap_kvar_total = 0
        zone_pv_kva = 0

        # Per-capacitor details for this zone
        cap_details = []  # list[dict]

        set_coords = []

        is_source = False
        for bus in descendents:
            node_dict = graph.nodes[bus]

            node_coord = (
                node_dict['attr']['x_coord'],
                node_dict['attr']['y_coord'])
            set_coords.append(node_coord)

            dss.Circuit.SetActiveBus(bus)
            all_pce = dss.Bus.AllPCEatBus()

            n_pce = len(all_pce)

            if n_pce == 0:
                continue
            # check for load and pvsystem
            for pce in all_pce:
                if 'Load' in pce:
                    # handle load
                    load_kw = float(dss.run_command(f"? {pce}.kw"))
                    load_kvar = float(dss.run_command(f"? {pce}.kvar"))

                    zone_load_kw += load_kw
                    zone_load_kvar += load_kvar

                    zone_load_n += 1

                elif 'PVSystem' in pce:
                    # handle pv system
                    pv_kva = float(dss.run_command(f"? {pce}.kva"))
                    zone_pv_kva += pv_kva
                    zone_pv_n += 1

                elif "Capacitor" in pce:
                    # handle capacitors
                    cap_phases = dss.run_command(f"? {pce}.phases")
                    cap_kvar = dss.run_command(f"? {pce}.kvar")

                    # Parse to list of step sizes (always a list)
                    kvar_steps = _parse_dss_kvar_steps(cap_kvar)
                    num_steps = len(kvar_steps)
                    zone_cap_steps_n += num_steps

                    # Aggregate totals: sum all step kvar values
                    zone_cap_kvar_total += sum(kvar_steps)
                    zone_cap_n += 1

                    # Strip leading "Capacitor." (case-insensitive) so we store only the object name
                    # Examples:
                    #   "capacitor.CapA"    -> "CapA"
                    cap_name = re.sub(r"^capacitor\.", "", pce, flags=re.IGNORECASE)                  

                    # Record per-capacitor details for later inspection/export
                    cap_details.append({
                        "cap_name": cap_name,
                        # "bus": bus,
                        "kvar_steps": kvar_steps,
                        "phases": cap_phases,
                    })
                elif "Vsource" in pce:
                    is_source = True
                else:
                    print(f"{pce} - not accounted for")

        zone_dict[zone_id] = {}
        zone_dict[zone_id]['type'] = new_zone_dict['type'] 
        zone_dict[zone_id]['is_source'] = is_source
        zone_dict[zone_id]['n_bus'] = len(descendents)

        zone_dict[zone_id]['n_load'] = zone_load_n
        zone_dict[zone_id]['total_load_kw'] = zone_load_kw
        zone_dict[zone_id]['total_load_kvar'] = zone_load_kvar

        zone_dict[zone_id]['n_capacitor'] = zone_cap_n
        zone_dict[zone_id]['total_cap_kvar'] = zone_cap_kvar_total
        zone_dict[zone_id]['total_cap_steps'] = zone_cap_steps_n
        zone_dict[zone_id]['cap_details'] = cap_details

        zone_dict[zone_id]['n_pv'] = zone_pv_n
        zone_dict[zone_id]['total_pv_kva'] = zone_pv_kva

    return pd.DataFrame(zone_dict).T


def get_per_phase_zone_stat_df(zones, element_src_df):
    phases = [1,2,3]
    zone_sums = {}

    for zone_name, zone_dict in zones.items():
        zone_buses = zone_dict['buses']

        zone_n_xfmrs = 0
        zone_xfmr_kva = 0

        zone_n_loads = 0

        zone_n_pv = 0
        zone_pv_kva = 0

        zone_n_caps = 0
        zone_cap_kvar = 0

        zone_phase_load = {
            'kw_phase_1': 0,
            'kw_phase_2': 0,
            'kw_phase_3': 0,
            'kvar_phase_1': 0,
            'kvar_phase_2': 0,
            'kvar_phase_3': 0,
        }
        zone_phase_pv = {
            'pv_kva_phase_1': 0,
            'pv_kva_phase_2': 0,
            'pv_kva_phase_3': 0,
        }
        zone_phase_cap = {
            'cap_kvar_phase_1': 0,
            'cap_kvar_phase_2': 0,
            'cap_kvar_phase_3': 0,
        }

        for bus in zone_buses:
            mask = element_src_df['short_bus'] == bus
            if sum(mask) == 0:
                continue
            connected_df = element_src_df[mask]

            # NOTE: will return multiple elements
            for _, element_row in connected_df.iterrows():
                # handling here will be based on DSSClass
                dss_class = element_row['DSSClass']

                if element_row['n_primary_connections'] == 0:
                    # why is this now... zero?
                    print(element_row)
                    continue

                if dss_class == 'Load':
                    zone_n_loads += 1
                    # get kw and kvar
                    load_kw = element_row['kW']
                    load_kvar = element_row['kvar']
                    # figure out scaling (for delta and 3 phase)
                    scale = 1 / element_row['n_primary_connections']
                    # figure out phase assignment
                    for phase in phases:
                        if element_row[f'on_primary_{phase}']:
                            zone_phase_load[f'kw_phase_{phase}'] += load_kw * scale
                            zone_phase_load[f'kvar_phase_{phase}'] += load_kvar * scale

                elif dss_class == 'Transformer':
                    zone_n_xfmrs += 1
                    zone_xfmr_kva += element_row['kVA'][0]

                elif dss_class == 'PVSystem':
                    zone_n_pv += 1
                    pv_kva = element_row['kVA']
                    zone_pv_kva += pv_kva

                    scale = 1 / element_row['n_primary_connections']
                    # figure out phase assignment
                    for phase in phases:
                        if element_row[f'on_primary_{phase}']:
                            zone_phase_pv[f'pv_kva_phase_{phase}'] += pv_kva * scale

                elif dss_class == 'Capacitor':
                    zone_n_caps += 1

                    cap_array = np.array(element_row['kvar'])
                    total_cap = cap_array.sum()
                    zone_cap_kvar += total_cap

                    scale = 1 / element_row['n_primary_connections']
                    # figure out phase assignment
                    for phase in phases:
                        if element_row[f'on_primary_{phase}']:
                            zone_phase_cap[f'cap_kvar_phase_{phase}'] += total_cap * scale

                else:
                    print(f'WARNING "{dss_class}" dss class not handled...')

        zone_sums[zone_name] = zone_phase_load
        zone_sums[zone_name]['n_loads'] = zone_n_loads
        zone_sums[zone_name]['n_xfmrs'] = zone_n_xfmrs
        zone_sums[zone_name]['total_xfmr_kva'] = zone_xfmr_kva
        zone_sums[zone_name]['n_pv'] = zone_n_pv
        zone_sums[zone_name]['total_pv_kva'] = zone_pv_kva
        zone_sums[zone_name].update(zone_phase_pv)
        zone_sums[zone_name]['n_caps'] = zone_n_caps
        zone_sums[zone_name]['total_cap_kvar'] = zone_cap_kvar
        zone_sums[zone_name].update(zone_phase_cap)
        zone_sums[zone_name]['type'] = zone_dict['type']

    per_phase_zone_stats = pd.DataFrame(zone_sums).T
    return per_phase_zone_stats


def get_zone_firsts(graph, zones, get_zone_flow=True):
    """
    Based on bus graph and zones, identify first bus and switched edge of zones
    """
    # identify first bus in zone
    # NOTE: may not work if cycles (loops) exist
    zone_firsts = {}

    # for names to index
    topo_index = {node: i for i, node in enumerate(nx.topological_sort(graph))}

    for zone_name, zone_dict in zones.items():
        zone_set = zone_dict['buses']

        # initialize first bus with unreal value
        first_bus = {}
        first_bus['bus_name'] = ''
        first_bus['ndx'] = 1e6

        # check all buses for smallest topo index
        for bus in zone_set:
            if bus in topo_index:
                ndx = topo_index[bus]
                if ndx < first_bus['ndx']:
                    first_bus['bus_name'] = bus
                    first_bus['ndx'] = ndx

        # save first bus (typically switch bus2)
        zone_firsts[zone_name] = {}
        zone_firsts[zone_name]['type'] = zone_dict['type']
        zone_firsts[zone_name]['first_bus'] = first_bus['bus_name']

        # identify immediately upstream bus (likely switch bus1)
        try:
            zone_firsts[zone_name]['zeroth_bus'] = list(graph.predecessors(first_bus['bus_name']))[0]
            # edge estimation
            possible_edge = (zone_firsts[zone_name]['zeroth_bus'], zone_firsts[zone_name]['first_bus'], 0)
        except IndexError:
            # due to voltage source/ no predecessor - use first edge
            possible_edge = list(graph.edges.items())[0][0]

        if possible_edge in graph.edges:
            zone_firsts[zone_name]['switched_edge_graph_name'] = possible_edge
            zone_firsts[zone_name]['switched_edge_pde_name'] = graph.edges[possible_edge]['attr']['pde_name']

            zone_firsts[zone_name]['switched_edge_kind'] = graph.edges[possible_edge]['attr']['kind']
        else:
            print(f"ERROR - no switch edge found {possible_edge}")

    if get_zone_flow:
        get_dss_zone_flow(zone_firsts)

    return zone_firsts

def get_dss_zone_flow(zone_firsts):
    p_per_phase = get_p_per_phase()

    for zone_id, zone_dict in zone_firsts.items():
        edge_name = zone_dict['switched_edge_pde_name'].lower()
        if edge_name in p_per_phase['Element'].str.lower().values:
            mask = p_per_phase['Element'].str.lower().values == edge_name
            zone_dict['dss_flow'] = p_per_phase[mask]
        else:
            print(f"WARNING dss edge flow for {zone_id} not found!")

    return zone_firsts


def get_zone_load_buses(zones):
    """
    return dictionary of buses with load in zone.
    This will help select AMI from database.

    ASSERT: feeder associated with zones is still active in dss direct

    """
    zone_dict = {}
    for zone_id, zone_dict_iter in zones.items():

        descendents = zone_dict_iter['buses']
        zone_loads = []

        for bus in descendents:
            dss.Circuit.SetActiveBus(bus)
            all_pce = dss.Bus.AllPCEatBus()

            n_pce = len(all_pce)

            if n_pce == 0:
                continue
            # check for load
            for pce in all_pce:
                if 'Load' in pce:
                    zone_loads.append(bus)

        # ensure unique
        zone_loads = list(set(zone_loads))
        zone_dict[zone_id] = zone_loads

    return zone_dict


def get_zone_load_dict(zones):
    """
    Return dictionary of zone load bus : load name
    Useful for identifying AMI phase

    Assuming ami named after bus name,
    and load name will connect to phase infromation
    """
    zone_dict = {}
    for zone_id, descendents in zones.items():

        zone_loads = {}

        for bus in descendents:
            dss.Circuit.SetActiveBus(bus)
            all_pce = dss.Bus.AllPCEatBus()

            n_pce = len(all_pce)

            if n_pce == 0:
                continue

            # check for load
            for pce in all_pce:
                if 'Load' in pce:
                    # NOTE: will cause issues if bus has more than 1 load..
                    if bus in zone_loads:
                        print(f'WARNING: duplicate load on bus {bus}')
                    zone_loads[bus] = pce

        zone_dict[zone_id] = zone_loads

    return zone_dict


#  plot functions
def plot_basic_graph(graph):
    """
    return basic plot of graph using bus coordinates
    """
    pos = {}
    for node in graph.nodes():
        x = graph.nodes[str(node)]['attr']['x_coord']
        y = graph.nodes[str(node)]['attr']['y_coord']
        pos[node] = (x, y)

    fig, ax = plt.subplots()
    nx.draw(
        graph,
        pos,
        with_labels=False,
        node_color='skyblue',
        node_size=1,
        edge_color='k',
        arrows=True,
        ax=ax)

    plt.axis('off')
    plt.show()

    return fig, ax


def plot_zone_graph(graph, zones):
    """
    plot zone convex hulls of graph
    """
    # collect spatial information of zone
    u_zone_descendents = {}
    for zone_id, zone_dict in zones.items():
        descendents = zone_dict['buses']

        set_coords = []
        for bus in descendents:
            node_dict = graph.nodes[bus]

            node_coord = (
                node_dict['attr']['x_coord'],
                node_dict['attr']['y_coord'])
            set_coords.append(node_coord)

        u_zone_descendents[zone_id] = {}
        u_zone_descendents[zone_id]['descendents'] = descendents
        u_zone_descendents[zone_id]['coords'] = set_coords

        # create convex hull of zone
        points = set_coords
        if len(points) < 3:
            # not enough points for polygon
            #  print(f"skipping {zone_id}")
            continue

        hull = ConvexHull(points)

        x = []
        y = []
        for pt_index in hull.vertices:
            coords = points[pt_index]
            x.append(coords[0])
            y.append(coords[1])

        # for enclosed hull
        x.append(x[0])
        y.append(y[0])

        u_zone_descendents[zone_id]['shape_x'] = x
        u_zone_descendents[zone_id]['shape_y'] = y

    # plot setup
    n_zones = len(zones)

    colors = generate_colors(n_zones, cmap_name='gist_ncar')
    # shuffle colors
    random.seed(0)
    random.shuffle(colors)

    symbols = get_symbols(n_zones)

    pos = {}
    for node in graph.nodes():
        x = graph.nodes[str(node)]['attr']['x_coord']
        y = graph.nodes[str(node)]['attr']['y_coord']
        pos[node] = (x, y)

    fig, ax = plt.subplots(figsize=(10, 10))
    marker_index = 0

    # for each identified zone
    zone_n = -1
    for zone_id, zone_dict in u_zone_descendents.items():
        zone_n += 1
        if len(zone_dict['descendents']) < 3:
            # point plotting
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=zone_dict['descendents'],
                node_color=colors[zone_n],
                node_shape=symbols[marker_index],
                label=zone_id,
                ax=ax,
                )
            marker_index += 1

        else:
            # draw fill...
            x = zone_dict['shape_x']
            y = zone_dict['shape_y']
            ax.fill(
                x, y,
                facecolor=colors[zone_n],
                edgecolor=colors[zone_n],
                linestyle='dashed',
                linewidth=2,
                label=zone_id,
                alpha=0.4,
                zorder=0
                )

            # zone id plotting... may clutter
            id_pt = (zone_n+1) % len(x)
            ax.scatter(
                x[id_pt],
                y[id_pt],
                marker=f'${zone_id}$',
                s=1010,
                color='white',
                zorder=3)
            ax.scatter(
                x[id_pt],
                y[id_pt],
                marker=f'${zone_id}$',
                s=1000,
                color=colors[zone_n],
                zorder=4)

    # graph nodes and edges
    nx.draw(
        graph,
        pos,
        with_labels=False,
        node_color='skyblue',
        node_size=1,
        edge_color='k',
        arrows=True,
        ax=ax)
    nx.draw_networkx_edges(graph, pos, arrows=True, ax=ax)
    ax.legend(loc='upper right')

    return fig, ax


#  AI functions
def generate_colors(num_colors, cmap_name='viridis'):
    # Function to generate colors from a colormap
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i / num_colors) for i in range(num_colors)]


def get_symbols(num_symbols):
    # Function to get a list of symbols
    # List of available symbols
    symbols = ['o', 's', '^', 'D', 'P', '*', 'X', 'H', 'v', '<', '>', '1', '2', '3', '4', '8']
    symbols = symbols * 10
    return symbols[:num_symbols]  # Return only the required number of symbols


def _pick_palette(n):
    # fallback palette picker
    if n <= 10:
        return list(Category10[10])[:n]
    if n <= 20:
        return list(Category20[20])[:n]
    # repeat if more than 20
    pal = list(Category20[20])
    return [pal[i % len(pal)] for i in range(n)]


def plot_zone_graph_bokeh(
        graph,
        zones,
        width=900,
        height=900,
        show_zone_ids=True,
        ):
    """
    Bokeh version of plot_zone_graph with legend click-to-hide for each zone.
    Returns a Bokeh Figure.

    Created using Chat GPT 5.2 and original plot code as input
    """
    # --- collect zone spatial info + hulls
    u_zone_descendents = {}
    for zone_id, zone_dict in zones.items():
        descendents = zone_dict['buses']

        coords = []
        for bus in descendents:
            nd = graph.nodes[bus]
            coords.append((nd["attr"]["x_coord"], nd["attr"]["y_coord"]))

        u_zone_descendents[zone_id] = {
            "descendents": descendents,
            "coords": coords,
        }

        if len(coords) >= 3:
            hull = ConvexHull(coords)
            xs = [coords[i][0] for i in hull.vertices]
            ys = [coords[i][1] for i in hull.vertices]
            # close polygon
            xs.append(xs[0])
            ys.append(ys[0])
            u_zone_descendents[zone_id]["shape_x"] = xs
            u_zone_descendents[zone_id]["shape_y"] = ys

    # --- colors (you used generate_colors + shuffle; mimic deterministically)
    zone_ids = list(u_zone_descendents.keys())
    n_zones = len(zone_ids)
    colors = _pick_palette(max(n_zones, 1))
    random.seed(0)
    random.shuffle(colors)
    color_map = {zid: colors[i] for i, zid in enumerate(zone_ids)}

    # --- build plot
    p = figure(
        width=width,
        height=height,
        title="Identified Zones",
        match_aspect=True,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )

    # --- draw graph edges
    xs_e, ys_e = [], []
    for u, v in graph.edges():
        x0 = graph.nodes[str(u)]["attr"]["x_coord"]
        y0 = graph.nodes[str(u)]["attr"]["y_coord"]
        x1 = graph.nodes[str(v)]["attr"]["x_coord"]
        y1 = graph.nodes[str(v)]["attr"]["y_coord"]
        xs_e.append([x0, x1])
        ys_e.append([y0, y1])

    edge_src = ColumnDataSource({"xs": xs_e, "ys": ys_e})
    p.multi_line(xs="xs", ys="ys", source=edge_src, line_color="black", line_width=1, alpha=0.8)

    # --- draw graph nodes (tiny like your matplotlib version)
    node_x = [graph.nodes[str(n)]["attr"]["x_coord"] for n in graph.nodes()]
    node_y = [graph.nodes[str(n)]["attr"]["y_coord"] for n in graph.nodes()]
    node_src = ColumnDataSource({"x": node_x, "y": node_y})
    p.scatter("x", "y", source=node_src, size=3, color="skyblue", alpha=0.9)

    # --- per-zone renderers for legend toggling
    legend_items = []

    zone_n = 0
    for zone_id in zone_ids:
        zd = u_zone_descendents[zone_id]
        color = color_map[zone_id]
        label = str(zone_id)

        if len(zd["descendents"]) < 3:
            # points for small zones
            zx = [graph.nodes[n]["attr"]["x_coord"] for n in zd["descendents"]]
            zy = [graph.nodes[n]["attr"]["y_coord"] for n in zd["descendents"]]
            src = ColumnDataSource({"x": zx, "y": zy})

            r = p.scatter(
                "x", "y",
                source=src,
                size=10,
                color=color,
                alpha=0.9,
            )
            legend_items.append(LegendItem(label=label, renderers=[r]))

        else:
            # polygon fill
            xs = [zd["shape_x"]]
            ys = [zd["shape_y"]]
            src = ColumnDataSource({"xs": xs, "ys": ys})

            r = p.patches(
                xs="xs", ys="ys",
                source=src,
                fill_color=color,
                fill_alpha=0.4,
                line_color=color,
                line_dash="dashed",
                line_width=2,
            )
            legend_items.append(LegendItem(label=label, renderers=[r]))

            # optional zone id text near an arbitrary hull vertex (like your id_pt)
            if show_zone_ids:
                x = zd["shape_x"]
                y = zd["shape_y"]
                id_pt = (zone_n + 1) % len(x)
                # Bokeh text marker: use text glyph (simpler than matplotlib's '$...$')

                # optional background dot to mimic your outlined marker
                txt_src = ColumnDataSource({"x": [x[id_pt]], "y": [y[id_pt]], "t": [str(zone_id)]})

                p.scatter([x[id_pt]], [y[id_pt]], size=25, color=color, alpha=0.9)

                p.text(
                    x="x", y="y", text="t",
                    source=txt_src,
                    text_align="center", text_baseline="middle",
                    text_color="white",
                    text_font_size="12pt",
                )

    # --- legend with click-to-hide
    legend = Legend(items=legend_items, location="top_right", click_policy="hide")
    p.add_layout(legend, "right")
    
    return p

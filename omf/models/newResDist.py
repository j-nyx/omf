"""
The Restructure of Resilient Dist
"""

import json, os, shutil, subprocess, datetime, random, copy, base64, platform
import os.path
from pathlib import Path
import numpy as np
import networkx as nx
import omf
from omf.models import __neoMetaModel__
from omf.models.__neoMetaModel__ import *
from omf.solvers import opendss

# Model metadata:
modelName, template = __neoMetaModel__.metadata(__file__)
tooltip = "Outage and Mitigation Modeling"
hidden = False

# Causes that may be included but not currently listed
# temporary_fault, permanent_break, overload, scheduled_maintenance, unknown

WEATHER_CATEGORIES = {
	"underground weather": {"flooding", "earthquakes"},
	"above-ground weather": {"wind", "ice", "lightning", "hail", "tornadoes", "hurricanes"}
}

MITIGATIONS = [
	{
		"id": "undergrounding",
		"component": "line",
		"causes": {"vegetation_contact", "wildlife", "above-ground weather", "equipment_failure"},
		"excluded_causes": {"underground weather"},
	},
	{
		"id": "vegetation_clearance",
		"component": "line",
		"causes": {"vegetation_contact", "equipment_failure"},
		"excluded_causes": set(),
	},
	{
		"id": "recloser",
		"component": {"line", "transformer"},
		"causes": {"above-ground weather", "wildlife", "vegetation_contact"},
		"excluded_causes": {"scheduled_maintenance"},
	},
	{
		"id": "backup_generation",
		"component": "load",
		"causes": {"equipment_failure", "underground weather", "wildlife", "vegetation_contact", "above-ground weather"},
		"excluded_causes": None,
	},
]


def mitigationAnalysis(mitigationOptions, dssCircuitPath, inputDict, outage, prices_per_ft):
	"""Calculate mitigation costs based on line lengths in a DSS file.

	Parameters:
		mitigationOptions (list): list of mitigation.
		dssCircuitPath (str or Path): path to the mitigation circuit.

	Returns:
	"""
	tree = opendss.dssConvert.dssToTree(dssCircuitPath)
	results = []
	total = 0.0

	line_objs = { ob.get('object','').split('.')[1]: ob for ob in tree if isinstance(ob, dict) and ob.get('object','').startswith('line.') }

	target_obj = line_objs.get(outage.get('component_name'))
	if target_obj == None:
		raise Exception(f"Could not find line object for component_name '{outage.get('component_name')}' in mitigation circuit.")

	# for mitigation in mitigationOptions:
	# 	mit_id = mitigation.get('id')

	# 	length_val = target_obj.get('length')
	# 	units_val = target_obj.get('units', 'ft')

	# 	# Jenny Stopped here

	# 	# results.append({'mitigation_id': mit_id, 'component': outage.get('component_name'), 'length_ft': length_ft, 'unit': units_val, 'price_per_ft': price, 'cost': cost})
	# 	# total += cost

	return {'items': results, 'total_cost': total}

def validateData(previousOutageData, criticalLoads):
	if previousOutageData is None or not Path(previousOutageData).exists():
		raise Exception("Outage data file does not exist.")
	if criticalLoads is None:
		raise Exception("Critical loads list is not provided.")
	
	return True

def insertOutage(modelDir, outage, dssCircuit):
	''' Insert the outage data into the model. '''
	tree = opendss.dssConvert.dssToTree(dssCircuit)
	comp_name = outage['component_name']
	inserted = False
	for i, ob in enumerate(tree):
		obj = ob.get('object','') if isinstance(ob, dict) else ''
		# object strings look like 'line.NAME'
		if obj.startswith('line.') and obj.split('.')[1] == comp_name:
			edit_cmd = {'!CMD': 'edit', 'object': f'line.{comp_name}', 'enabled': 'false'}
			tree.insert(i+1, edit_cmd)
			inserted = True
			break
	if inserted:
		opendss.dssConvert.treeToDss(tree, Path(modelDir, 'outage_circuit.dss'))
		return True
	# If no matching line found, return False
	return False

def CLPR(modelDir, dssCircuit):
	''' Run Powerflow Analysis on Model and identify loads that did not receive power.'''
	opendss.runDSS(dssCircuit)
	opendss.runDssCommand(f"export voltages '{modelDir}/voltages.csv'")
	voltages = pd.read_csv(Path(modelDir, "voltages.csv"))
	return True


def identifyMitigationOptions(outage):
	"""Return mitigation IDs that match the cause/component in the given outage row.

	The function expects `outage` to be a dict-like or pandas Series with keys
	"cause_of_outage" and "component_type" must be present. TODO: Add checks for this kind of stuff.
	"""
	cause = str(outage['cause_of_outage']).lower() if 'cause_of_outage' in outage else None
	component_type = str(outage['component_type'].lower()) if 'component_type' in outage else None
	
	# name = str(outage['component_name'])
	# if '_' in name:
	# comp = name.split('_', 1)[0]
	# In case component_type doesn't exist ^ save this. We can maybe get it from the name.

	if cause == None or component_type == None:
		raise ValueError("Outage must have 'cause_of_outage' and 'component_type' fields.")

	matches = []
	for m in MITIGATIONS:
		# determine if component matches
		mitigation_components = m.get('component')
		comp_match = False
		if isinstance(mitigation_components, (set, list, tuple)):
			if component_type in mitigation_components or 'any' in mitigation_components:
				comp_match = True
		else:
			mitigation_components = str(mitigation_components).lower()
			if mitigation_components == 'any' or mitigation_components == component_type:
				comp_match = True
		excluded = m.get('excluded_causes')

		# determine if cause is allowed
		allowed = m.get('causes')
		allowed_match = cause in allowed

		# only append if component matches, cause is allowed, and not excluded
		if comp_match and allowed_match and (not excluded):
			matches.append(m.get('id'))

	return matches

def insertMitigation():
	''' Insert the mitigation options into the model. '''
	return True

def work(modelDir, inputDict):
	''' Run the model in its directory. '''
	outData = {}

	validateData(inputDict['outageDataFileName'], inputDict['criticalLoads'])
	previousOutages = pd.read_csv(Path(modelDir, inputDict['outageDataFileName']))

	feederName = [x for x in os.listdir(modelDir) if x.endswith('.omd')][0][:-4]
	inputDict["feederName1"] = feederName
	pathToOmd = Path(modelDir, feederName)
	tree = opendss.dssConvert.omdToTree(pathToOmd)
	opendss.dssConvert.treeToDss(tree, Path(modelDir, 'circuit.dss'))

	# For now, testing purposes, we are just going to take the first outage.

	insertOutage(modelDir, previousOutages.iloc[0], Path(modelDir, 'circuit.dss'))

	CLPR(modelDir, Path(modelDir, 'outage_circuit.dss'))
	mitigationOptions = identifyMitigationOptions(previousOutages.iloc[0])
	# Insert mitigation into the model (placeholder)
	insertMitigation()
	CLPR(modelDir, Path(modelDir, 'mitigation_circuit.dss'))
	# Build per-foot prices from inputs (defaults to 0 if missing)
	prices_per_ft = {
		'undergrounding': float(inputDict.get('undergroundLineCost', 0)),
		'vegetation_clearance': float(inputDict.get('vegetationClearanceCost', 0)),
		'recloser': float(inputDict.get('recloserCost', 0)),
		'backup_generation': float(inputDict.get('dgUnitCost', 0)),
	}
	costs = mitigationAnalysis(mitigationOptions, Path(modelDir, 'mitigation_circuit.dss'), prices_per_ft)
	outData['mitigation_costs'] = costs


	outData['hostingCapacityMap'] = open(Path(modelDir, "geoJson_offline.html"), 'r').read()

	# And we're done.
	return outData

def new(modelDir):
	''' Create a new instance of this model. Returns true on success, false on failure. '''
	defaultCircuit = "iowa240.clean.dss"
	previousOutageData = "input_previousOutageData.csv"
	previousOutageDataPath = Path(omf.omfDir, 'static', 'testFiles', 'newResDist', previousOutageData)
	
	defaultInputs = {
		"modelType": modelName,
		"modelName": modelDir,
		"user": "admin",
		"created": str(datetime.datetime.now()),
		"feederName1": defaultCircuit,
		"outageDataFileName": previousOutageData,
		"outageUIDisplay": previousOutageData,
		"undergroundLineCost": "15",
		"recloserCost": "10000.0",
		"vegetationClearanceCost": "5",
		"dgUnitCost": "1000000.0",
		"maxDGPerGenerator": "1.0",
		"criticalLoads": "load_001,load_004",
	}

	creationCode = __neoMetaModel__.new(modelDir, defaultInputs)
	try:
		# Copy the feeder from one place to another
		shutil.copyfile(pJoin(__neoMetaModel__._omfDir, "static", "publicFeeders", defaultInputs["feederName1"]+'.omd'), pJoin(modelDir, defaultInputs["feederName1"]+'.omd'))
		shutil.copyfile( previousOutageDataPath, Path(modelDir, previousOutageData) )
	except:
		return False
	return creationCode

def _runModel():
	# Testing the hazard class.
	"""
	Internal helper for resilient dist run model processing.
	"""
	_testHazards()
	# Location
	modelLoc = pJoin(__neoMetaModel__._omfDir,"data","Model","admin","Automated Testing of " + modelName)
	# Blow away old test results if necessary.
	try:
		shutil.rmtree(modelLoc)
	except:
		# No previous test results.
		pass 
	# Create New.
	new(modelLoc)
	# Pre-run.
	# renderAndShow(modelLoc)
	# Run the model.
	__neoMetaModel__.runForeground(modelLoc)
	# Show the output.
	__neoMetaModel__.renderAndShow(modelLoc)

if __name__ == '__main__':
	_runModel()

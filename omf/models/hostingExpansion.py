"""
Estimate distribution hosting-capacity expansion options
& visualize upgrade scenarios for additional DER adoption.
"""

# Python Imports
import shutil
import datetime
import json
import os
import pandas as pd
from pathlib import Path
import logging
import plotly.utils as pu
import plotly.graph_objects as go

# OMF imports
from omf.models import __neoMetaModel__
from omf.models.__neoMetaModel__ import *
from omf import weather
from omf.solvers import opendss
from omf.solvers import pysam

# Model metadata
modelName, template = __neoMetaModel__.metadata(__file__)
hidden = False


def checkCircuitSolar(modelDir, inputDict: dict):
	'''
	Reviews any pvsystems or batteries in the circuit and sums up their kW values
	'''
	returningKW = 0
	feederName = [x for x in os.listdir(modelDir) if x.endswith('.omd')][0]
	inputDict['feederName1'] = feederName[:-4]
	pathToOmd = Path(modelDir, feederName)
	tree = opendss.dssConvert.omdToTree(pathToOmd)
	pvsystems = [x for x in tree if x.get('object', 'N/A').startswith('pvsystem.')]
	batteries = [x for x in tree if x.get('object', 'N/A').startswith('battery.')]
	if len(pvsystems) == 0 and len(batteries) == 0:
		return returningKW
	if len(pvsystems) != 0:
		kwFromPV = [x['kw'] for x in pvsystems if 'kw' in x]
		for item in kwFromPV:
			returningKW += float(item)
	if len(batteries) != 0:
		kwFromBattery = [x['kw'] for x in batteries if 'kw' in x]
		for item in kwFromBattery:
			returningKW += float(item)
	return returningKW


def processOptimalUpgrades(results: dict) -> dict:
	'''
	Turns the decaf.optimal_upgrades result dict into tables and a figure for the HTML template.
	'''
	outData = {}
	upgrades = results.get("upgrades", []) or []
	baseline = results.get("baseline_hc_mw")
	target = results.get("target_hc_mw")
	postUpgrade = results.get("post_upgrade_hc_mw")
	released = results.get("released_hc_mw")
	totalCost = results.get("total_cost_usd")
	runtime = results.get("runtime_sec")

	def fmtMW(val):
		return "N/A" if val is None else f"{val:,.3f}"

	def fmtUSD(val):
		return "N/A" if val is None else f"${val:,.2f}"

	def fmtBool(val):
		return "N/A" if val is None else ("Yes" if val else "No")

	# Status flags for the template
	outData["optUpg_success"] = bool(results.get("success", False))
	outData["optUpg_failureReason"] = results.get("failure_reason") or ""
	outData["optUpg_upgradeRequired"] = bool(results.get("upgrade_required", False))
	outData["optUpg_targetAchieved"] = bool(results.get("target_achieved", False))

	outData["optUpg_summaryHeadings"] = [
		"Site (Bus)",
		"Baseline HC (MW)",
		"Target HC (MW)",
		"Post Upgrade HC (MW)",
		"HC Released (MW)",
		"Upgrade Required",
		"Target Achieved",
		"Number of Upgrades",
		"Total Upgrade Cost (USD)",
		"Runtime (sec)",
	]
	outData["optUpg_summaryValues"] = [[
		results.get("site_id", "N/A"),
		fmtMW(baseline),
		fmtMW(target),
		fmtMW(postUpgrade),
		fmtMW(released),
		fmtBool(results.get("upgrade_required")),
		fmtBool(results.get("target_achieved")),
		len(upgrades),
		fmtUSD(totalCost),
		"N/A" if runtime is None else f"{runtime:,.1f}",
	]]

	# Ranked upgrade table, ordered by iteration
	upgrades = sorted(upgrades, key=lambda u: u.get("iteration", 0))
	outData["optUpg_upgradeHeadings"] = [
		"Rank", "Asset ID", "Asset Type", "Action", "Old Setting", "New Setting",
		"Cost (USD)", "HC Before (MW)", "HC After (MW)", "HC Gain (MW)", "kW Gained per $1k"
	]
	upgradeRows = []
	constraintRows = []
	for u in upgrades:
		cost = u.get("incremental_cost_usd")
		gain = u.get("realized_hc_gain_mw")
		if cost and gain is not None:
			gainPerK = f"{(gain * 1000) / (cost / 1000):,.2f}"
		else:
			gainPerK = "N/A"
		oldState = u.get("old_state") or {}
		newState = u.get("new_state") or {}
		upgradeRows.append([
			u.get("iteration", "N/A"),
			u.get("asset_id", "N/A"),
			str(u.get("asset_type", "N/A")).replace("_", " ").title(),
			str(u.get("action_type", "N/A")).replace("_", " ").title(),
			", ".join(f"{k}: {v}" for k, v in oldState.items()) or "N/A",
			", ".join(f"{k}: {v}" for k, v in newState.items()) or "N/A",
			fmtUSD(cost),
			fmtMW(u.get("hc_before_mw")),
			fmtMW(u.get("hc_after_mw")),
			fmtMW(gain),
			gainPerK,
		])
		for c in u.get("binding_constraints_before", []) or []:
			slack = c.get("slack")
			dual = c.get("dual_kw_per_pu")
			constraintRows.append([
				u.get("iteration", "N/A"),
				u.get("asset_id", "N/A"),
				str(c.get("family", "N/A")).title(),
				c.get("hour", "N/A"),
				c.get("constraint", "N/A"),
				"N/A" if slack is None else f"{slack:,.4f}",
				"N/A" if dual is None else f"{dual:,.2f}",
			])
	outData["optUpg_upgradeValues"] = upgradeRows

	# Binding constraints that the upgrades were chosen to relieve
	outData["optUpg_constraintHeadings"] = [
		"Rank", "Asset ID", "Constraint Family", "Hour", "Constraint", "Slack", "Dual (kW/pu)"
	]
	outData["optUpg_constraintValues"] = constraintRows

	# Hosting capacity by upgrade step, with the target as a dashed line
	stepLabels = ["Baseline"]
	stepValues = [baseline if baseline is not None else 0]
	for u in upgrades:
		stepLabels.append(f"After #{u.get('iteration', '')} {u.get('asset_id', '')}")
		stepValues.append(u.get("hc_after_mw", 0) or 0)
	hcFigure = go.Figure()
	hcFigure.add_trace(go.Bar(
		x=stepLabels,
		y=stepValues,
		name="Hosting Capacity (MW)",
		marker=dict(color="steelblue"),
		text=[f"{v:.3f}" for v in stepValues],
		textposition="outside"
	))
	if target is not None:
		hcFigure.add_trace(go.Scatter(
			x=stepLabels,
			y=[target] * len(stepLabels),
			name="Target (MW)",
			mode="lines+markers" if len(stepLabels) == 1 else "lines",
			line=dict(color="red", width=2, dash="dash")
		))
	yMax = max(stepValues + ([target] if target is not None else []))
	hcFigure.update_layout(
		xaxis_title=None,
		yaxis_title="Hosting Capacity (MW)",
		yaxis=dict(range=[0, yMax * 1.15 if yMax > 0 else 1]),
		legend={
			"orientation": "h",
			"yanchor": "bottom",
			"y": 1.02,
			"xanchor": "right",
			"x": 1
		}
	)
	outData["optUpg_hcFigure"] = json.dumps(hcFigure, cls=pu.PlotlyJSONEncoder)
	return outData


def work(modelDir, inputDict: dict) -> dict:
	''' Run the model in its directory. '''
	# Delete output file every run if it exists
	outData = {}
	# Model operations goes here.
	lat = float( inputDict['latitude'] )
	long = float( inputDict['longitude'] )
	year = int( inputDict['year'] )
	sys_design = pysam._pysam_sysDesignSetup(inputDict, lat, long)
	attributes = ['dni,dhi,ghi,wind_speed,air_temperature']
	nrlAPIResponse = weather.nlr_get_nsrdb_data(data_set="goes_aggregated", longitude=long, latitude=lat, year=year, api_key="rnvNJxNENljf60SBKGxkGVwkXls4IAKs1M8uZl56", attributes=attributes, filename=Path(modelDir,"output_aggregated_data.csv"))
	requestSuccess = True if nrlAPIResponse.status_code == 200 else False
	if requestSuccess:
		pvwatts_model, pvwatts_data = pysam.run_pvwatts(
			modelDir=modelDir,
			sys_design=sys_design,
			dataFile="output_aggregated_data.csv"
		)
	else:
		raise Exception("hostingExpansion.py: API request 1 Failed")
	# For Max Solar - Set tilt = latitude
	inputDict['tilt'] = lat
	sys_design_max = pysam._pysam_sysDesignSetup(inputDict, lat, long)
	attributes_clearsky = ['clearsky_dhi', 'clearsky_dni', 'clearsky_ghi']
	nrlAPIResponse_clearsky = weather.nlr_get_nsrdb_data(data_set="goes_aggregated", longitude=long, latitude=lat, year=year, api_key="rnvNJxNENljf60SBKGxkGVwkXls4IAKs1M8uZl56", attributes=attributes_clearsky, filename=Path(modelDir,"output_aggregated_clearsky_data.csv"))
	requestSuccess = True if nrlAPIResponse_clearsky.status_code == 200 else False
	if requestSuccess:
		maxSolar_model, maxSolar_data = pysam.run_pvwatts_historical_max(modelDir=modelDir, sys_design=sys_design_max, dataFile="output_aggregated_clearsky_data.csv")
	else:
		raise Exception("hostingExpansion.py: API request 2 Failed")
	amiData = pd.read_csv( Path(modelDir, inputDict["AmiDataFileName"]) )
	# Determine the length of available data (use load data length, should be max 1 year)
	data_length = len(amiData)
	# Slice solar data to match load data length
	pvwatts_data_sliced = pvwatts_data.iloc[:data_length]
	maxSolar_data_sliced = maxSolar_data.iloc[:data_length]
	# Get existing storage capacity on circuit
	storage_output = checkCircuitSolar(modelDir, inputDict)
	full_df = pd.DataFrame({
			'hour': pvwatts_data_sliced.index,
			'total_load': amiData.iloc[:, 1:].sum(axis=1)*1000,  # Convert kW to W
			'pysam_ac_watts': pvwatts_data_sliced['ac'].values,
			'storage_output_w': storage_output * 1000,  # Convert kW to W
			'dc_nameplate_w': float(inputDict["systemCapacity"])*1000,  # kW to W
			'max_solar_ac_watts': maxSolar_data_sliced['ac'].values
	})
	full_df.to_csv(Path(modelDir, "output_LoadvsPySAM.csv"), index=False)
	scatterFigure = go.Figure()
	scatterFigure.add_trace(go.Scatter(
		x=full_df['hour'],
		y=full_df['storage_output_w'],
		name='Storage Output (W)',
		fill='tozeroy',
		fillcolor='rgba(0, 200, 0, 0.3)',
		line=dict(color='darkgreen', width=2),
		mode='lines'
	))
	scatterFigure.add_trace(go.Scatter(
		x=full_df['hour'],
		y=full_df['pysam_ac_watts'] + full_df['storage_output_w'],
		name='Solar Output (W)',
		line=dict(color='darkgreen', width=2),
		mode='lines'
	))
	scatterFigure.add_trace(go.Scatter(
		x=full_df['hour'],
		y=full_df['total_load'],
		name='Total Load (W)',
		line=dict(color='blue', width=2),
		mode='lines'
	))
	scatterFigure.add_trace(go.Scatter(
		x=full_df['hour'],
		y=full_df['dc_nameplate_w'],
		name='DC Nameplate Capacity (W)',
		line=dict(color='red', width=2, dash='dash'),
		mode='lines'
	))
	# Add max solar output
	scatterFigure.add_trace(go.Scatter(
		x=full_df['hour'],
		y=full_df['max_solar_ac_watts'] + full_df['storage_output_w'],
		name='Max Solar Output (W)',
		line=dict(color='darkgreen', width=2, dash='dash'),
		mode='lines'
	))
	scatterFigure.update_layout(
	title=None,
		xaxis_title=None,
		yaxis_title=None,
		hovermode='x unified',
		legend={
			"orientation": "h",
			"yanchor": "bottom",
			"y": 1.02,
			"xanchor": "right",
			"x": 1
		}
	)
	outData['scatterFigure'] = json.dumps( scatterFigure, cls=pu.PlotlyJSONEncoder )
	feederName = [x for x in os.listdir(modelDir) if x.endswith('.omd')][0]
	pathToOmd = Path(modelDir, feederName)
	tree = opendss.dssConvert.omdToTree(pathToOmd)
	opendss.dssConvert.treeToDss(tree, Path(modelDir, 'circuit.dss'))

	# Can't get decaf_cl solver working.
	# Temporarily copy the output of optimal_upgrades file for testing

	shutil.copyfile( Path(__neoMetaModel__._omfDir, "static", "testFiles", "hostingExpansion", "output_optiUpgrResults.json"),
								  Path(modelDir, "output_optiUpgrResults.json") )

	shutil.copyfile( Path(__neoMetaModel__._omfDir, "static", "testFiles", "hostingExpansion", "output_optiUpgrResults 2.json"),
								  Path(modelDir, "output_optiUpgrResults 2.json") )

	outData.update(processOptimalUpgrades( json.load(open(Path(modelDir, "output_optiUpgrResults.json"))) ))
	# outData.update(processOptimalUpgrades( json.load(open(Path(modelDir, "output_optiUpgrResults 2.json"))) ))

	# Stdout/stderr.
	outData["stdout"] = "Success"
	outData["stderr"] = ""
	return outData


def new(modelDir):
	''' Create a new instance of this model. Returns true on success, false on failure. '''
	amiFileName = "input_mackelroy.csv"
	amiFilePath = Path(omf.omfDir,'static','testFiles', 'hostingExpansion', amiFileName)
	ScadaFileName = "input_ScadaData.csv"
	ScadaFilePath = Path(omf.omfDir,'static','testFiles', 'hostingExpansion', ScadaFileName)
	derPipelineFileName = "input_derPipelineData.csv"
	derPipelineFilePath = Path(omf.omfDir,'static','testFiles', 'hostingExpansion', derPipelineFileName)
	newInterconnFileName = "input_newInterconnData.csv"
	newInterconnFilePath = Path(omf.omfDir,'static','testFiles', 'hostingExpansion', newInterconnFileName)
	
	defaultInputs = {
		"user": "admin",
		"modelType": modelName,
		"created": str(datetime.datetime.now()),
		"feederName1": 'iowa240.clean.dss',
		"AmiUIDisplay": amiFileName,
		"AmiDataFileName": amiFileName,
		"ScadaUIDisplay": ScadaFileName,
		"ScadaDataFileName": ScadaFileName,
		"derPipelineUIDisplay": derPipelineFileName,
		"derPipelineDataFileName": derPipelineFileName,
		"newInterconnUIDisplay": newInterconnFileName,
		"newInterconnDataFileName": newInterconnFileName,
		"longitude": "-94.67",
		"latitude": "39.10",
		"year": "2024",
		"azimuth": "180.0",
		"systemCapacity": 800,
		"tilt": 45,
		"losses": 15.5,
	}
	creationCode = __neoMetaModel__.new(modelDir, defaultInputs)
	# Copy files from the test directory ( or respective places ) and put them in the model for use
	try:
		shutil.copyfile(
			Path(__neoMetaModel__._omfDir, "static", "publicFeeders", defaultInputs["feederName1"]+'.omd'),
			Path(modelDir, defaultInputs["feederName1"]+'.omd'))
		shutil.copyfile( amiFilePath, Path(modelDir, amiFileName) )
		shutil.copyfile( ScadaFilePath, Path(modelDir, ScadaFileName) )
		shutil.copyfile( derPipelineFilePath, Path(modelDir, derPipelineFileName))
		shutil.copyfile( newInterconnFilePath, Path(modelDir, newInterconnFileName))
	except:
		return False
	return creationCode

@neoMetaModel_test_setup
def _tests():
	# Location
	"""
	Run this module's local smoke tests or debugging workflow.
	"""
	modelLoc = Path(__neoMetaModel__._omfDir, "data", "Model", "admin", "Automated Testing of " + modelName)
	# Blow away old test results if necessary.
	try:
		shutil.rmtree(modelLoc)
	except:
		# No previous test results.
		pass
	# Create New.
	new(modelLoc)
	# Pre-run.
	__neoMetaModel__.renderAndShow(modelLoc)
	# Run the model.
	__neoMetaModel__.runForeground(modelLoc)
	# Show the output.
	__neoMetaModel__.renderAndShow(modelLoc)

if __name__ == '__main__':
	_tests()

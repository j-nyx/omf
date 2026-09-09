import pandas as pd
from pathlib import Path
import random
from datetime import timedelta

data = pd.read_csv(Path(__file__).parent / "outages.csv")
# Index,Annual Event #,Metric Interval Event #,Start,Finish,Component Type,Object Name,Inducing Object,Protective Device,Desired Fault type,Fault Type,Number customers affected,Secondary number of customers affected,Location,Meters Affected,Cause
data = data.drop(columns=["Metric Interval Event #", "Inducing Object", "Protective Device", "Desired Fault type", "Fault Type", 'Number customers affected', "Secondary number of customers affected"])

data.rename(columns={'Annual Event #': 'outage_id', 'Start': 'time_of_outage_start', 'Finish': 'time_of_outage_end', 'Component Type': 'component_type', 'Object Name': 'component_name', 'Location': 'component_location', 'Meters Affected': 'meters_affected', 'Cause': 'cause_of_outage'}, inplace=True)
data.to_csv(Path(__file__).parent / "input_previousOutageData.csv", index=False)
data["last_vegetation_trim"] = None
data["component_age"] = None

# Keep only one row for each desired cause (one squirrel, earthquake, wind, tree)
desired_causes = ["squirrel", "earthquake", "wind", "tree"]
# ensure outage start is datetime
data["time_of_outage_start"] = pd.to_datetime(data["time_of_outage_start"], errors="coerce")

rows = []
for cause in desired_causes:
	mask = data["cause_of_outage"].astype(str).str.contains(cause, case=False, na=False)
	subset = data[mask]
	if subset.empty:
		# fallback: copy first row and set cause
		if len(data) > 0:
			row = data.iloc[0].copy()
			row["cause_of_outage"] = cause
		else:
			continue
	else:
		row = subset.iloc[0].copy()
	rows.append(row)

trimmed = pd.DataFrame(rows)

# Fill `last_vegetation_trim` with a random date prior to the outage start
def random_prior_date(start_ts):
	if pd.isna(start_ts):
		# if no outage start, pick a date up to 10 years ago
		start_ts = pd.Timestamp.now()
	days_back = random.randint(30, 3650)  # between ~1 month and 10 years
	prior = start_ts - pd.Timedelta(days=days_back)
	return prior.date().isoformat()

trimmed["last_vegetation_trim"] = trimmed["time_of_outage_start"].apply(random_prior_date)

trimmed["component_age"] = [random.randint(51, 120) for _ in range(len(trimmed))]

'''
 outage id
component_name
component_type
component_location
component_age
last_vegetation_
trim
consumers_affected
time of outage start
time of outage end
cause_of_outage
'''

trimmed = trimmed[["outage_id", "component_name", "component_type", "component_location", "component_age", "last_vegetation_trim", "meters_affected", "time_of_outage_start", "time_of_outage_end", "cause_of_outage"]]

trimmed.to_csv(Path(__file__).parent / "input_previousOutageData.csv", index=False)
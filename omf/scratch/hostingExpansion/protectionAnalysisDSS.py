import omf.solvers.opendss.dssConvert as dssConvert
from pathlib import Path
from omf import omfDir

from omf.solvers.decaf_cl import sandia_toplevel

# dssConvert.dss_to_clean_via_save('./dssfiles/main_no_profiles.dss', './protectionAnalysis.dss')

file = Path(omfDir, "static", "testFiles", "hostingExpansion", 'protectionAnalysis.dss')
amidata = Path(omfDir, "static", "testFiles", "hostingExpansion", '20260129-iowa_240_synthetic_ami_short_04.csv')

iowa240 = Path("/home/jenny/nreca/omf/omf/solvers/opendss/iowa240.clean.dss")

result = sandia_toplevel.protection_analysis(ami_fp=amidata, ckt_dss_fp=file, is_single_file_model=0)
# result = sandia_toplevel.protection_analysis(ami_fp=amidata, ckt_dss_fp=iowa240, is_single_file_model=0)
print(result)
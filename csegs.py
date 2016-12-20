import sys
import pandas as pd
import pickle
import time
import os.path
from collections import defaultdict
from datetime import datetime
import numpy as np
import re

data_file = sys.argv[1]

col_names = "custID days_ahead perf_time trans_date paid_tkt all_tkt paid_aud venue_state venue_name mtyp_p mtyp_s event_name show_desc dist_km  cults sex age_gr pcode cust_state mosaic".split()
# note: perf_time and trans_date are of the form 2014-10-05 19:15:00.000
df = pd.read_csv(data_file, names = col_names, parse_dates = [2,3])

#
# fix this data frame
#
df["perf_time"] = df["perf_time"].dt.date
df["trans_date"] = df["trans_date"].dt.date
df["dist_km"] = df["dist_km"].apply(lambda _: round(_, 1))
print("total rows in data frame:",len(df.index))
unique_customers = set(df["custID"].unique())
number_unique_customers = len(unique_customers)
print("total unique customers:",number_unique_customers)
# create the set of possible culture serments
labels = set(df["cults"].unique())
print("total culture segments:",len(labels))

# missing values; count returns a Series with number of non-NA/null observations over requested axis;
# Series is 1-d ndarray with axis labels 

missing_values_series = df.count() - len(df.index)

print("missing values: ", end="")
for i, v in enumerate(missing_values_series):
	if v < 0:
		print("{}:{} ({}%) ".format(missing_values_series.index[i], -missing_values_series.data[i], 
										round(100*(-missing_values_series.data[i]/len(df.index)),1)), end="")
print()

# 
def _decompose_mosaic(mostype):
	
	mosaic_mask = re.compile('(^[A-M]{1})(\d{2}$)')
	
	match_res = mosaic_mask.match(mostype)  # match objects always have a boolean value of True
	assert match_res, "error! this is not a Mosaic group name.." 
	
	mos_letter = match_res.group(1)  # mosaic letter
	mosn = int(match_res.group(2))  # mosaic number
	assert (mosn < 50), "error! the Mosaic class number should be under 50..."	

	return (mos_letter, mosn)

for cus in unique_customers:

	# select only the part of data frame describing transactions by this customer
	this_df = df[df["customerID"] == cus]
	
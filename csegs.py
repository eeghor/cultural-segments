import sys
import pandas as pd
import pickle
import time
import os.path
from datetime import datetime
import numpy as np
import re
from collections import defaultdict, Counter

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

cust_feature_dict = defaultdict(lambda: defaultdict(int))  # {"customerID1": {"feature1": 1, "feature2": 0, ..}, ..}
cust_mtype_counts = defaultdict(lambda: defaultdict(int))
cust_pmtype_counts = defaultdict(lambda: defaultdict(int))	
# missing values; count returns a Series with number of non-NA/null observations over requested axis;
# Series is 1-d ndarray with axis labels 

missing_values_series = df.count() - len(df.index)

print("missing values: ", end="")
for i, v in enumerate(missing_values_series):
	if v < 0:
		print("{}:{} ({}%) ".format(missing_values_series.index[i], -missing_values_series.data[i], 
										round(100*(-missing_values_series.data[i]/len(df.index)),1)), end="")
print()

print("removing the customers with missing mosaic groups...")
df = df[pd.notnull(df["mosaic"])]
print("done. now there are {} rows left...".format(len(df.index)))

print("removing the customers with missing customer states...")
df = df[pd.notnull(df["cust_state"])]
print("done. now there are {} rows left...".format(len(df.index)))

#print("secondary mtypes in this dataset", Counter(df["mtyp_s"]))

popular_sec_mtypes = Counter(df["mtyp_s"]).most_common(25)
print("most popular secondary mtypes", popular_sec_mtypes)
unique_customers_w_mosaic = set(df["custID"].unique())
print("unique customers with mosaic:", len(unique_customers_w_mosaic))
# 
def _decompose_mosaic(mostype):
	
	mosaic_mask = re.compile('(^[A-M]{1})(\d{2}$)')
	
	match_res = mosaic_mask.match(mostype)  # match objects always have a boolean value of True
	assert match_res, "error! this is not a Mosaic group name.." 
	
	mos_letter = match_res.group(1)  # mosaic letter
	mosn = int(match_res.group(2))  # mosaic number
	assert (mosn < 50), "error! the Mosaic class number should be under 50..."	

	return (mos_letter, mosn)

for cus in unique_customers_w_mosaic:

	# select only the part of data frame describing transactions by this customer
	this_df = df[df["custID"] == cus]
	# print("customer with ID=",cus)
	# print("df for this customer:", this_df)
	# Mosaic group features

	# secondary m type features
	for sec_f in this_df["mtyp_s"].unique():
		cust_feature_dict[cus]["purchased_"+sec_f] = 1

	# primary m type features
	for sec_f in this_df["mtyp_p"].unique():
		cust_feature_dict[cus]["purchased_"+sec_f] = 1
	#cust_pmtype_counts = Counter(this_df["mtyp_p"])

	if str(this_df["mosaic"].unique()[0]).isalnum():
		cus_mosaic = this_df["mosaic"].unique()[0]
		print("mosaic:",cus_mosaic)
		m_letter, m_number = _decompose_mosaic(cus_mosaic)
		cust_feature_dict[cus]["belongs_to_mosaic_letter_" + m_letter]
		cust_feature_dict[cus]["mosaic_number_" + str(m_number)]
		cust_feature_dict[cus]["mosaic_group_" + cus_mosaic]

	# income level features:		
	if (m_letter in ["A","D"] or 					    # all A and D are rich 
		(m_letter == "B" and m_number in range(5,9)) or   # B05 to B08 are rich but B09 aren't ("simple needs")
		(m_letter == "C" and m_number in [10, 12, 13]) or # C11 and C14 are likely to have average income
		(m_letter == "E" and m_number in [17,18]) or  	# E18 and E19 are probably not that rich
		(m_letter == "F" and m_number in [21])):  		# F22 to F24 may have average income
		cust_feature_dict[cus]["high_income"] = 1

	elif ((m_letter in ["B"] and m_number in [9]) or      # these are "the good life" older couples
		(m_letter in ["G","H"]) or
		(m_letter == "C" and m_number in [11]) or   		# educated singles and couples in early career "inner city aspirations" 
		(m_letter == "E" and m_number in [19,20])):
		cust_feature_dict[cus]["average_income"] = 1

	else:
		cust_feature_dict[cus]["low_income"] = 1

	# education features:
	if ((m_letter in ["A","B","C", "I"]) or 
		(m_letter == "H" and m_number in [30])):
		cust_feature_dict[cus]["good_education"] = 1

	elif ((m_letter in ["D","F"]) or
		(m_letter == "H" and m_number in [31,32])):
		cust_feature_dict[cus]["average_education"] = 1

	else:
		cust_feature_dict[cus]["poor_education"] = 1

	# gender feature
	for s in this_df["sex"].tolist():

		if s in "M F".split():
			cust_feature_dict[cus][s] = 1
		else:
			cust_feature_dict[cus]["unknown_gender"] = 1

	# customer state feature

	cstates = this_df["cust_state"].unique()

	if len(cstates) > 1:
			cust_feature_dict[cus]["moved_interstate"] = 1
	else:
		for s in cstates:
			cust_feature_dict[cus]["lives_in_" + s] = 1

	
	# average tickets purchased
	cust_feature_dict[cus]["average_tickets_per_purchase"] = round(this_df.loc[this_df["all_tkt"] > 0, "all_tkt"].mean(),1)

	# even or odd
	#print(this_df.loc[this_df["all_tkt"] > 0, "all_tkt"])
	tk_evens = this_df.loc[this_df["all_tkt"] > 0, "all_tkt"].apply(lambda _: _%2 == 0)
	tot_evens = sum(tk_evens)
	tot_odds = len(this_df.loc[this_df["all_tkt"] > 0, "all_tkt"]) - tot_evens
	if tot_evens > tot_odds:
		cust_feature_dict[cus]["usually_buys_even"] = 1
	else:
		cust_feature_dict[cus]["usually_buys_odd"] = 1

	print(cust_feature_dict[cus])
	#print("mtypes for this customer", cust_mtype_counts[cus])
	#cust_pmtype_counts[cus] = Counter(this_df["mtyp_p"])
	#print("primary mtypes for this customer", cust_pmtype_counts)
	#mos_letter, mosn = _decompose_mosaic(this_df.mosaic[0])
	#print("mosaic letter: {}, mosaic number {}".format(mos_letter, mosn))


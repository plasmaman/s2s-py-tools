import argparse
from ecmwfapi import ECMWFService
from datetime import datetime, timedelta
import os, sys
from settings import *

# Set up argument parser
parser = argparse.ArgumentParser(description="Download ECMWF forecast and hindcast data")
parser.add_argument('--refdate', type=str, help="Reference date in 'YYYY-MM-DD' format, default is today")
parser.add_argument('--days', type=str, help="Number of days to download, default is five days before reference date")
parser.add_argument('--subset', type=str, help="Subset, default is taken from settings file")

args = parser.parse_args()

# Set default values for refdate and first_dt as date objects (not datetime)
if args.refdate:
	refdate = datetime.strptime(args.refdate, "%Y-%m-%d")  # Convert to date object
else:
	#refdate = (datetime.today() - timedelta(days=2)).date()  # Default to 2 days before today
	refdate = datetime.today()

if args.days:
	days = int(args.days)
else:
	days = 5  # Default to 5 days before today

if args.subset:
	subset = args.subset
else:
	subset = SUBSET

first_dt = refdate - timedelta(days=days)
print('Running download_hindcasts.py at', datetime.now())
print(f'refdate = {refdate}, first_dt = {first_dt}, subset={subset}')

# Define other parameters
variable = 'tp'
steps = [24*d for d in LEAD_TIME_DAYS[subset]]

server = None
def retrieve(request, target):
	global server
	if server is None:
		server = ECMWFService('mars')
	server.execute(request, target)

meta = {
	'param': '228.128',
	'levtype': 'sfc',
	'area': AREA[subset],
	'grid': GRID[subset],
	'stream': 'eefh',
	'class': 'od',
	'expver': '1',
	'time': '00:00:00',
	'step': steps
}

current_dt = first_dt
while current_dt <= refdate:
	get = True
	if current_dt.month == 2 and current_dt.day == 29:
		get = False
	if current_dt < datetime(2024,11,11): 
		if current_dt.weekday() not in [0,3]:
			get = False
	else:
		if current_dt.day%2 == 0:
			get = False
	if get:
		for datatype in ('pf','cf',):
			elems = [variable, current_dt.strftime('%Y-%m-%d')]
			#elems.append('%i-%i'%(steps[0], steps[-1]))
			elems.extend([datatype, subset])
			target = '%s/%s.grb'%(HINDCAST_DIR,'_'.join(elems))
			#print(target)
			if os.path.isfile(target) or os.path.isfile(target.replace('grb','nc')):
				print('File exists:', target)
			else:
				request = meta.copy()
				request['type'] = datatype
				request['date'] = current_dt.strftime('%Y-%m-%d')
				print('Fetching:', target)
				hdate = []
				dt = current_dt
				while current_dt.year-dt.year<20:
					day = (28 if dt.month == 2 and current_dt.day == 29 else dt.day)
					dt = datetime(year=dt.year-1,month=dt.month,day=day)
					hdate.append(dt.strftime('%Y-%m-%d'))
				request['hdate'] = hdate
				if datatype in ('pf',):
					request['number'] = '1/2/3/4/5/6/7/8/9/10'
				print('*'*10)
				# for k,v in request.items():
				# 	print(k,v)
				print(request)
				retrieve(request, target)
	current_dt += timedelta(hours=24)


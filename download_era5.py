import cdsapi, os
from calendar import monthrange
from settings import *
from datetime import datetime, date

client = None
months = [3]
years = [2021]
dataset = "reanalysis-era5-single-levels"
subset = SUBSET

if 1:
	d = {
		"download_format": "unarchived",
		'data_format': 'netcdf',
		'product_type': ['reanalysis'],
		'variable': ['total_precipitation'],
		'time': ['%02d:00'%h for h in range(0,24)],
		'grid': GRID[subset],
		'area': AREA[subset]
	}
	for i,month in enumerate(months):
		for year in years:
			dt = datetime(year=year, month=month, day=1).date()
			filename = dt.strftime(ERA5_FILE_WILDCARD)
			print(filename)
			if not os.path.exists(filename):
				d['day'] = ['%02d'%i for i in range(1,monthrange(year,month)[1]+1)]
				d['month'] = ['%i'%(month)]
				d['year'] = ['%i'%(year)] 
				print(d,filename)
				if client is None:
					client = cdsapi.Client()
				client.retrieve(dataset, d, filename)

if 1:
	d = {
		"download_format": "unarchived",
		'data_format': 'netcdf',
		'product_type': ['reanalysis'],
		'variable': ['land_sea_mask'],
		'time': ['00:00'],
		'grid': GRID[subset],
		'area': AREA[subset]
	}
	filename = ERA5_LSM_FILE
	if not os.path.exists(filename):
		d['day'] = ['01']
		d['month'] = ['01']
		d['year'] = ['2000'] 
		print(d,filename)
		if client is None:
			client = cdsapi.Client()
		client.retrieve(dataset, d, filename)


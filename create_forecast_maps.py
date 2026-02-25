#from pyexpat import model
from settings import *
import os, sys
import numpy as np
import xarray as xr
import pickle
from datetime import date, timedelta, datetime
from glob import glob
import pandas as pd
from pathlib import Path
import netCDF4 as nc
import re
import calendar
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import cartopy.feature as cf
from shapely.geometry import Point
from scipy.stats import pearsonr
import cmcrameri.cm as cmc
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator, FuncFormatter
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
import itertools
from typing import Dict, Any, Callable

# ---- your constants ----

# Preferences:
DEBUG_LEVEL = 0
BIAS_CORRECTION_WINDOW_DAYS = 11 # Same as nbr of ensemble members in hindcast
# Increase sample size for computing extreme thresholds: 11 times as many for clim = nbr. of ensemble members
PERCENTILE_WINDOW_DAYS = {'model': 3, 'clim': 33}
COASTLINEWIDTH = 0.6
COASTLINECOLOR = 'k'
BORDERCOLOR = 'y'
fs = FONTSIZE = 9

# Areas and country-level zooming:
# Define these are variables instead of strings, so we can change the definition w/o changing the code
MALAWI = 'MALAWI_3'
EAST_AFRICA = 'EAST_AFRICA'
EAST_AFRICA_EXT = 'EAST_AFRICA_EXT'
EAST_AFRICA_FCST = 'EAST_AFRICA_FCST'
EAST_AFRICA_PALMER = 'EAST_AFRICA_PALMER'
ETHIOPIA = 'ETHIOPIA_2'
MADA = 'MADA'

# Area definitions (lat0, lat1, lon0, lon1):
CBOXES = {
	'MALAWI_1': [-17.5,-9,32.5,36],
	'MALAWI_2': [-17.5,-9,32,36.5],
	'MALAWI_3': [-19,-8,31,37.5],
	'EAST_AFRICA': [-9,7.5,33.5,50],
	'EAST_AFRICA_EXT': [-12,15,28,52],
	'EAST_AFRICA_PALMER': [-5,10,30,50],
	'EAST_AFRICA_FCST': [-26,15,28,52],
	'ETHIOPIA_1': [3,16,32,52],
	'ETHIOPIA_2': [2,16,32,50],
	'MADA': [-26.5,-11,42,51.5],
}

# Some nice colours:
C0 = '#1f77b4'
C1 = '#ff7f0e'
C2 = '#2ca02c'
C3 = '#d62728'
C4 = '#9467bd'
C5 = '#8c564b'
C8 = '#bcbd22'
C9 = '#17becf'

# Caching:
USE_CACHE = True
_GRID_CACHE = None
_DATA_CACHE = {}
_ZOOM_DIC = {} # Sub-region definitions
_BORDERS = None

# Define forecast type specifications
FORECAST_SPECS = {
	'dry_spells': {
		'threshold_param': 'no_rain_thresholds',
		'auxiliary_params': ['consecutive_days', 'threshold_type'],
		'hits_kwargs': lambda threshold, aux: {
			'forecast_type': 'dry_spells',
			'consecutive_days': aux['consecutive_days'],
			'threshold_type': aux['threshold_type']
		}
	},
	# Not implemented yet:
	# 'accumulated_rainfall': {
	# 	'threshold_param': 'accumulated_rainfall_thresholds',
	# 	'auxiliary_params': ['accumulation_days'],
	# 	'hits_kwargs': lambda threshold, aux: {
	# 		'forecast_type': 'accumulated_rainfall',
	# 		'accumulation_days': aux['accumulation_days']
	# 	}
	# },
	'heavy_rainfall_days': {
		'threshold_param': 'daily_rainfall_thresholds',
		'auxiliary_params': ['consecutive_days', 'threshold_type'],
		'hits_kwargs': lambda threshold, aux: {
			'forecast_type': 'heavy_rainfall_days',
			'consecutive_days': aux['consecutive_days'],
			'threshold_type': aux['threshold_type']
		}
	}
}

def savefig(filename,dpi=300,filetype='png',base_path=None):
	if base_path is None:
		base_path = '.'
	filename = f'{base_path}/{filename}.{filetype}'
	debug('Saving figure:',filename, priority=1)
	plt.savefig(filename,dpi=dpi)

def country_code_to_region(cc):
	try:
		return {
			'MG': MADA,
			'ET': ETHIOPIA,
			'MW': MALAWI,
			'EA': EAST_AFRICA_FCST
		}[cc]
	except:
		pass
	return cc

def region_to_country_code(region):
	try:
		return {
			MADA: 'MG',
			ETHIOPIA: 'ET',
			MALAWI: 'MW',
			EAST_AFRICA_FCST: 'EA'
		}[region]
	except:
		pass
	return region

def get_prop(key, default=None, **kw):
	"""
	Used to check the keyword dictionary kw whether a named variable is defined, 
	but also define a default value in case it's not in the kw
	"""
	try:
		val = kw[key]
	except:
		val = default
	return val

def debug(*args, priority=0, **kwargs):
	"""
	Print a message only if the priority (default=0) is higher than or equal to the debug_level (default=1)
	"""
	if priority >= DEBUG_LEVEL:
		print(*args, **kwargs)

def loadpickle(filename):
	"""
	Load a pickle file from the cache
	"""
	a = pickle.load(open(filename,'rb'))
	debug('pickle loaded:',filename)
	# if not filename in self.loaded_files:
	# 	self.loaded_files.append(filename)
	return a

def savepickle(filename, a):
	"""
	Save anything to a pickle file for caching, but only if we use caching
	"""
	if USE_CACHE:
		f = open(filename,'wb')
		pickle.dump(a,f)
		debug('pickle saved:',filename)
		f.close()

def get_data_cache_key(obs_source):
	#return f"{obs_source['model_name']}_{obs_source['resolution']}"
	return f"{obs_source['model_name']}_{obs_source['subset']}"

def era5_new(*, 
	base_dir=ERA5_DIR, 
	variable_name="tp", 
	cache_dir=CACHEDIR, 
	subset = SUBSET,
	#resolution = RESOLUTION,
	precip_mult = 1000.
):
	dic = {
		"model_name": "era5",
		"base_dir": base_dir,       
		"variable_name": variable_name,
		"cache_dir": cache_dir,
		"precip_mult": precip_mult,
		"subset": subset
		#"resolution": resolution
		#"data_cache": {},
	}
	#global_data_cache[dic] = {}
	return dic

def create_model(model_name, **kw):
	if model_name in ('ecmwf', 'ecmwf',):
		return ecmf_native_new(**kw)
	raise RuntimeError('Invalid model name:', model_name)

def create_obs_source(obs_source_name, **kw):
	if obs_source_name in ('era5',):
		return era5_new(**kw)
	raise RuntimeError('Invalid obs source name:', obs_source_name)

def ecmf_native_new(*, 
	#fcst_dir=ECMF_NATIVE_ROOT, 
	hcst_dir=HINDCAST_DIR, 
	variable_name="tp", 
	cache_dir=CACHEDIR, 
	first_lead_time=24, 
	last_lead_time=720, 
	subset = SUBSET,
	#resolution = RESOLUTION,
	precip_mult = 1000.
):
	dic = {
		"model_name": "ecmf",
		#"fcst_dir": fcst_dir,       
		"hcst_dir": hcst_dir,       
		"variable_name": variable_name,
		"first_lead_time": int(first_lead_time),
		"last_lead_time": int(last_lead_time),
		"cache_dir": cache_dir,
		"precip_mult": precip_mult,
		"subset": subset
		#"resolution": resolution
	}
	#global_data_cache[dic] = {}
	return dic

def ensure_valid_time(ds):
	"""
	Ensure ds has a 1D 'valid_time' coordinate that can be used as concat dimension.
	Typical ECMWF GRIB has time + step -> valid_time.
	"""
	if ds is None:
		return None

	# If already present and usable, keep it
	if "valid_time" in ds.coords:
		vt = ds["valid_time"]
		# Sometimes valid_time is 2D (time, step). We want 1D along step for a single time.
		if vt.ndim == 2 and "time" in vt.dims and "step" in vt.dims:
			# assume single time
			if ds.sizes.get("time", 1) == 1:
					ds = ds.isel(time=0)
					ds = ds.assign_coords(valid_time=ds["valid_time"])
		return ds

	# Build from time + step
	if "time" in ds.coords and "step" in ds.coords:
		# If multiple times exist, pick first (your files are typically one init time per message group)
		if ds.sizes.get("time", 1) == 1:
			base = ds["time"].isel(time=0)
			steps = ds["step"]
			vt = base + steps
			ds = ds.isel(time=0)
			ds = ds.assign_coords(valid_time=vt)
			return ds

		# If time has >1, fall back to vectorized valid_time (will be 2D) then flatten later if needed
		vt = ds["time"] + ds["step"]
		ds = ds.assign_coords(valid_time=vt)
		return ds

	raise ValueError("Cannot construct valid_time: missing time/step coordinates.")

def load_cf_pf(pattern):
	"""
	Auto-detect file structure and load cf/pf data accordingly.
	"""

	files = sorted(glob(pattern))
	if not files:
		raise RuntimeError(f"No files found matching pattern: {pattern}")
		#return None, []

	keys = ['cf','pf']
	
	# Check if any file contains 'cf' or 'pf' in the name
	has_cf_pf_markers = any('_cf_' in f or '_pf_' in f for f in files)
	
	if has_cf_pf_markers:
		file_type = 'separated'
	else:
		file_type = 'legacy'
	#debug(f"Detected file structure: {file_type}")

	if file_type == 'legacy':
		files = sorted(glob(pattern))
		parts = {key: [] for key in keys}
		for f in files:
			if '.idx' not in f:
				for key in keys:
					ds = xr.open_dataset(f, engine="cfgrib", filter_by_keys = {"dataType": key})
					ds = ensure_valid_time(ds)
					parts[key].append(ds)

	elif file_type == 'separated':
		files = {key: [] for key in keys}
		for f in glob(pattern):
			for key in keys:
				# Ignore index files
				if key in f and '.idx' not in f:
					files[key].append(f)
		parts = {}
		for key in keys:
			files[key] = sorted(files[key])
			if not files[key]:
				raise RuntimeError(f"No {key} files found....")
			parts[key] = []
			for f in files[key]:
				ds = xr.open_dataset(f, engine="cfgrib")
				ds = ensure_valid_time(ds)
				parts[key].append(ds)

	else:
		raise ValueError(f"Unknown file type: {file_type}")
	
	for key in keys:
		if not parts[key]:
			raise RuntimeError(f"No {key} groups found in files.")
		
	ds_combined = {
		key: xr.concat(
			parts[key],
			dim="valid_time",
			data_vars="minimal",
			coords="minimal",
			compat="override",
			join="override",
			combine_attrs="override",
		).sortby("valid_time")
		for key in keys
	}

	common_vt = np.intersect1d(ds_combined['cf']["valid_time"].values, ds_combined['pf']["valid_time"].values)
	if common_vt.size == 0:
		raise ValueError("No overlapping valid_time between cf and pf.")
	ds_combined = {key: ds_combined[key].sel(valid_time=common_vt) for key in ds_combined.keys()}
	# Concat along the number dimension:
	ds_cf = ds_combined['cf'].expand_dims(number=[0])
	ds_pf = ds_combined['pf'].assign_coords(number=(ds_combined['pf']["number"]))
	ds_merged = xr.concat(
		[ds_cf, ds_pf],
		dim="number",
		data_vars="minimal",
		coords="minimal",
		compat="equals",
		join="override",
		combine_attrs="override",
	)
	return ds_merged

def load_hindcast_cf_pf(pattern, variable, precip_mult=1.0):
	"""
	Load hindcast cf/pf grib files matching pattern.
	Each file = one forecast step, containing 20 hindcast years.
	Returns dict keyed by hdate (hindcast init date), each containing:
		- 'model_data': np.array (number, step, lat, lon)
		- 'valid_times': list of pd.Timestamp
	"""
	files = sorted([f for f in glob(pattern) if '.idx' not in f])
	if not files:
		raise RuntimeError(f"No files found matching: {pattern}")

	parts_cf = []
	parts_pf = []

	for f in files:
		ds_cf = xr.open_dataset(f, engine="cfgrib", filter_by_keys={"dataType": "cf"})
		ds_pf = xr.open_dataset(f, engine="cfgrib", filter_by_keys={"dataType": "pf"})
		parts_cf.append(ds_cf)
		parts_pf.append(ds_pf)

	# Concatenate along step dimension
	ds_cf_all = xr.concat(parts_cf, dim="step", coords="minimal", data_vars="minimal", compat="override", join="override", combine_attrs="override")
	ds_pf_all = xr.concat(parts_pf, dim="step", coords="minimal", data_vars="minimal", compat="override", join="override", combine_attrs="override")
	#ds_cf_all = xr.concat(parts_cf, dim="step")
	#ds_pf_all = xr.concat(parts_pf, dim="step")

	# Combine cf and pf along number dimension (cf becomes number=0)
	ds_cf_exp = ds_cf_all.expand_dims(number=[0])
	ds_merged = xr.concat(
		[ds_cf_exp, ds_pf_all],
		dim="number",
		data_vars="minimal",
		coords="minimal",
		compat="override",
		join="override",
		combine_attrs="override",
	)
	# ds_merged dims: (number=11, time=20, step=N, lat, lon)

	# Build result dict keyed by hindcast init date, matching old format
	result = {}
	hdates = ds_merged["time"].values  # the 20 hindcast years
	for j, hdate in enumerate(hdates):
		da = ds_merged[variable].isel(time=j)
		data = da.values  # shape: (number, step, lat, lon)

		# Drop step=0 (accumulation starts at step 1)
		data = data[:, 1:, :, :]

		# De-accumulate
		mdata = np.empty_like(data)
		mdata[:, 0, :, :] = data[:, 0, :, :]
		for step in range(1, data.shape[1]):
			mdata[:, step, :, :] = data[:, step, :, :] - data[:, step - 1, :, :]

		base = pd.Timestamp(hdate)
		steps = ds_merged["step"].values[1:]  # also drop step=0 from valid times
		valid_times = [base + pd.Timedelta(s) for s in steps]

		result[base] = {
			'model_data': precip_mult * mdata,
			'valid_times': valid_times,
		}

	return result

def collect_hindcast_data_for_refdate(**kw):
	model = kw['model']
	refdate = kw['refdate']

	# os.makedirs(model["cache_dir"], exist_ok=True)
	# cachefile = f'{model["cache_dir"]}/hcst_data_for_refdate_{refdate.strftime("%Y%m%d")}_native'
	# try:
	# 	return loadpickle(cachefile)
	# except Exception:
	# 	pass

	# --- New-style: single files per step containing both cf and pf ---
	result = None
	for delta in [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5]:
		dt = refdate + timedelta(days=delta)
		pattern = dt.strftime(ECMF_HINDCAST_FILE_WILDCARD)
		file_list = [f for f in glob(pattern) if '.idx' not in f]
		if not file_list:
				continue
		result = load_hindcast_cf_pf(
				pattern=pattern,
				variable=model["variable_name"],
				precip_mult=model["precip_mult"],
		)
		return result

	# --- Old-style: separate _cf_ / _pf_ files ---
	try:
		cf_file = read_hindcast_file(refdate=refdate, suffix='cf', raise_on_missing=False)
	except:
		cf_file = None
	try:
		pf_file = read_hindcast_file(refdate=refdate, suffix='pf', raise_on_missing=False)
	except:
		pf_file = None

	if cf_file is not None and pf_file is not None:
		ensemble_ds = read_hindcast_file(refdate=refdate, suffix='pf')
		control_ds  = read_hindcast_file(refdate=refdate, suffix='cf')
		ensemble_precip = ensemble_ds[model["variable_name"]]
		control_precip  = control_ds[model["variable_name"]]
		control_precip_expanded = control_precip.expand_dims(dim='number', axis=0)
		combined_precip = xr.concat([control_precip_expanded, ensemble_precip], dim='number')
		init_times = control_ds['time'][:].values
		result = {}
		for j, init_time in enumerate(init_times):
			first = combined_precip.isel(time=j)
			mdata = np.empty_like(first)
			mdata[:, 0, :, :] = first.isel(step=0)
			for step in range(1, first.shape[1]):
					mdata[:, step, :, :] = first.isel(step=step) - first.isel(step=step-1)
			valid_times = [pd.Timestamp(t) for t in init_time + control_ds['step'][:].values]
			result[pd.Timestamp(init_time)] = {
					'model_data': model["precip_mult"] * mdata,
					'valid_times': valid_times,
			}
		return result

	raise RuntimeError(f"No hindcast files found for refdate {refdate} within ±5 days")

	#savepickle(cachefile, result)

# def collect_h_data_for_refdate(**kw):
# 	model = kw['model']
# 	refdate = kw['refdate']
# 	os.makedirs(model["cache_dir"], exist_ok=True)
# 	cachefile = f'{model["cache_dir"]}/hcst_data_for_refdate_{refdate.strftime("%Y%m%d")}_native'
	
# 	# disk cache (optional)
# 	try:
# 		raise
# 		result = loadpickle(cachefile)
# 		return result
# 	except Exception:
# 		pass
	
# 	for delta in [0,-1,1,-2,2,-3,3,-4,4,-5,5]:
# 		dt = refdate + timedelta(hours=delta*24)
# 		init_tag = dt.strftime("%m%d")
# 		#pattern = f'{model["fcst_dir"]}/{ECMF_NATIVE_FILE_PREFIX}*{init_tag}*'
# 		pattern = dt.strftime(ECMF_HINDCAST_FILE_WILDCARD)
# 		file_list = sorted([f for f in glob(pattern) if '.idx' not in f])
# 		if not file_list:
# 			continue

# 		result = load_hindcast_cf_pf(
# 			pattern=pattern,
# 			variable=model["variable_name"],
# 			precip_mult=model["precip_mult"],
# 		)
# 		#savepickle(cachefile, result)
# 		return result

# 	raise RuntimeError(f"No hindcast files found for refdate {refdate} within ±5 days")
		
def collect_forecast_data_for_refdate(**kw):
	model = kw['model']
	refdate = kw['refdate']
	os.makedirs(model["cache_dir"], exist_ok=True)
	cachefile = f'{model["cache_dir"]}/fcst_data_for_refdate_{refdate.strftime("%Y%m%d")}_native'
	
	# disk cache (optional)
	try:
		result = loadpickle(cachefile)
		return result
	except Exception:
		pass
	
	init_tag = refdate.strftime("%m%d")
	#pattern = f'{model["fcst_dir"]}/{ECMF_NATIVE_FILE_PREFIX}*{init_tag}*'
	pattern = refdate.strftime(ECMF_FORECAST_FILE_WILDCARD)
	#print(pattern)
	#sys.exit()

	# This does a lot of things with the files
	ds_merged = load_cf_pf(pattern)

	# Check that we have the variable
	var = model["variable_name"]
	if var not in ds_merged:
		raise ValueError(f"'{var}' not found in merged dataset.")
	n_vt = ds_merged.sizes["valid_time"]
	# Compute accumulated daily rainfall by subtracting the day before
	daily = ds_merged[var].sortby("valid_time").diff("valid_time")
	ds_merged = ds_merged.isel(valid_time=slice(1, None))
	ds_merged = ds_merged.assign(tp_daily=daily)
	# Check that the number of valid times is consistent
	n_daily = ds_merged["tp_daily"].sizes["valid_time"]
	if n_daily != n_vt - 1:
		raise AssertionError(f"Expected tp_daily times = valid_time-1, got {n_daily} vs {n_vt}")
	#print(ds_merged)
	#sys.exit()
	
	# arr.shape = (nbr of members, nbr of days, lat, lon), e.g.: (101, 30, 125, 78)
	arr = ds_merged["tp_daily"].values 
	result = {
		"model_data": model['precip_mult'] * arr,
		"valid_times": [
			dt.astype("datetime64[ms]").astype(datetime).date()
			for dt in ds_merged["valid_time"].values
		],
	}
	
	# optional: savepickle(cachefile, result)
	return result

def get_grid(cache_dir=CACHEDIR):
	"""
	Procedural replacement for Context().get_grid()

	Returns dict with:
	lon, lat: 2D meshgrids
	lsm: 2D land-sea mask (same shape as lon/lat)
	inside: dict placeholder (fill later when you port that logic)
	"""
	global _GRID_CACHE
	if _GRID_CACHE is not None:
		return _GRID_CACHE
	#cachefile = f"{cache_dir}/grid_{RESOLUTION}"
	cachefile = f"{cache_dir}/grid_{SUBSET}"
	try:
		g = loadpickle(cachefile)
		_GRID_CACHE = g
		return g
	except Exception:
		pass
	# Build grid from the ERA5 LSM file
	file_path = f"{ERA5_LSM_FILE}"
	ds = nc.Dataset(file_path)
	lon_1d = ds["longitude"][:]
	lat_1d = ds["latitude"][:]
	try:
		lsm = ds["lsm"][0, :, :]
	except:
		lsm = None
	ds.close()
	lon, lat = np.meshgrid(lon_1d, lat_1d)
	g = {
		"lon": lon,
		"lat": lat,
		"lsm": lsm
		#"inside": {},  # fill later
	}
	savepickle(cachefile, g)
	_GRID_CACHE = g
	return g

def get_all_hindcast_refdates(first_refdate=None, last_refdate=None, **kw):
	"""
	Returns the reference dates all the hindcasts we have available
	"""
	model = kw['model']
	wildcard = f"{HINDCAST_DIR}/tp_*_pf"
	wildcard += f"_{model['subset']}.grb"
	#wildcard += f"_{model['resolution']}.grb"
	#print(wildcard)
	files = glob(wildcard)
	dates = []
	for f in files:
		match = re.search(r"\d{4}-\d{2}-\d{2}", f)
		if match:
			dates.append(datetime.strptime(match.group(), "%Y-%m-%d").date())
	dates = sorted(dates)
	if not dates:
		return []
	if first_refdate is None:
		first_refdate = dates[0]
	if last_refdate is None:
		last_refdate = dates[-1]
	return [d for d in dates if first_refdate <= d <= last_refdate]

def find_closest_hindcast_refdate(max_dist_days=7, max_dist_years=2, **kw):
	"""
	Code for finding the closest hindcast
	We need that for calibration
	Normally one would download hindcasts from ECMWF which are produced on odd days
	Otherwise one could keep a library of older hindcasts for the same time of the year
	In any case the code will fail unless it finds a hindcast with an initial date within 10 days of the refdate

	:param refdate: The reference initial date
	:param max_dist_days: Maximum distance in days
	:param max_dist_years: Maximum distance in years
	"""
	all_refdates = get_all_hindcast_refdates(period="hcst", **kw)
	#debug(all_refdates)
	refdate = kw['refdate']
	refdate_start_of_year = refdate.replace(month=1, day=1)
	refdate_day_of_year = (refdate - refdate_start_of_year).days
	test_year = refdate.year
	closest_refdate = None
	# Start with current year, then decrease by one at a time:
	while closest_refdate is None and refdate.year - test_year <= max_dist_years:
		candidates = [d for d in all_refdates if d.year == test_year]
		if candidates:
			close = [
					d for d in candidates
					if abs((d - d.replace(month=1, day=1)).days - refdate_day_of_year) <= max_dist_days
			]
			if close:
					closest_refdate = min(
						close,
						key=lambda d: abs((d - d.replace(month=1, day=1)).days - refdate_day_of_year)
					)
		test_year -= 1
	return closest_refdate

def get_hindcast_filename(**kw):
	key = kw['refdate'].strftime('%Y-%m-%d')
	filename = f'{HINDCAST_DIR}/tp_{key}_{kw['suffix']}'
	#filename = f'{filename}_{RESOLUTION}'
	filename = f'{filename}_{SUBSET}'
	filename += '.grb'
	#debug(filename, kw)
	return filename

def read_hindcast_file(**kw):
	filename = get_hindcast_filename(**kw)
	# Open the GRIB file using xarray and the cfgrib engine
	# This could fail, but then the exception has to be handled where read_file is called from
	if os.path.exists(filename):
		return xr.open_dataset(filename, engine='cfgrib')
	raise FileNotFoundError('File does not exist:', filename)

def get_init_times_for_refdate(**kw):
	control_ds = read_hindcast_file(suffix='cf', **kw)
	init_times = control_ds['time'][:].values
	return [pd.Timestamp(init_time) for init_time in sorted(init_times)]

# def collect_hindcast_data_for_refdate(**kw):
# 	model = kw['model']
# 	refdate = kw['refdate']
# 	e = [
# 		'hindcast_data_for_reftime',
# 		refdate.strftime('%Y%m%d'),
# 		SUBSET
# 		#RESOLUTION
# 	]
# 	cachefile = f'{model["cache_dir"]}/{"_".join(e)}'
# 	# try:
# 	#     return data_cache[cachefile]
# 	# except:
# 	#     pass
# 	try:
# 		return loadpickle(cachefile)
# 	except:
# 		pass
# 	# We have to read two files, control (cf) and perturbed (pf):
# 	ensemble_ds = read_hindcast_file(refdate=refdate, suffix='pf')
# 	control_ds = read_hindcast_file(refdate=refdate, suffix='cf')
# 	ensemble_precip = ensemble_ds[model["variable_name"]]
# 	control_precip = control_ds[model["variable_name"]]
# 	control_precip_expanded = control_precip.expand_dims(dim='number', axis=0)
# 	combined_precip = xr.concat([control_precip_expanded, ensemble_precip], dim='number')
# 	init_times = control_ds['time'][:].values
# 	result = {}
# 	for j,init_time in enumerate(init_times):
# 		first = combined_precip.isel(time=j)
# 		mdata = np.empty_like(first)
# 		#Context().debug(j,init_time)
# 		mdata[:, 0, :, :] = first.isel(step=0)
# 		for step in range(1, first.shape[1]):
# 			mdata[:, step, :, :] = first.isel(step=step) - first.isel(step=step-1)
# 		# The new array mdata now has 24-hour accumulated values for the first initial time
# 		# with the shape (number=10, step=30, latitude=100, longitude=61)
# 		# eradata = []
# 		valid_times = []
# 		for i, t in enumerate(init_time + control_ds['step'][:].values):
# 			dt = pd.Timestamp(t)
# 			valid_times.append(dt)
# 		result[pd.Timestamp(init_time)] = {
# 			# 'eradata': eradata,
# 			'model_data': model["precip_mult"]*mdata,
# 			'valid_times': valid_times
# 		}
# 	return result

def read_nc_file(filepath, variable = 'tp'):
	"""
	Read netCDF file and return the precipitation data and dimensions.
	"""
	ds = nc.Dataset(filepath)
	a = ds.variables[variable][:]
	ds.close()
	return a


def compute_daily_precip(tp_current, tp_next = None):
	"""
	Compute daily precipitation sums.
	For the last day of the month, use data from the next month's file.
	"""
	daily_precip = []
	for day in range(31):  # Assuming up to 31 days in a month
		start = day * 24 + 1
		end = (day + 1) * 24 + 1
		if end > tp_current.shape[0]:  # Handle the case for the last day
			if tp_next is not None:
				last_day_sum = np.sum(tp_current[start:,:,:], axis=0) + tp_next[0,:,:]
				daily_precip.append(last_day_sum)
			break
		else:
			daily_sum = np.sum(tp_current[start:end,:,:],axis=0)
			daily_precip.append(daily_sum)
	return daily_precip

def process_monthly_files_era5(year, month, **kw):
	"""
	Process a single month file and the subsequent month file if needed.
	"""
	obs_source = kw['obs_source']
	#resolution = obs_source['resolution']
	subset = obs_source['subset']
	dt = datetime(year=year, month=month, day=1).date()
	file_current = dt.strftime(ERA5_FILE_WILDCARD)
	#file_prefix = f"{obs_source['base_dir']}/{obs_source['model_name']}_{EXPNAME}_{obs_source['variable_name']}"
	#file_current = f"{file_prefix}_{year}_{month:02d}_{resolution}.nc"
	next_month = (1 if month==12 else month+1)
	next_year = (year+1 if month==12 else year)
	dt = datetime(year=next_year, month=next_month, day=1).date()
	file_next = dt.strftime(ERA5_FILE_WILDCARD)
	#file_next = f"{file_prefix}_{next_year}_{next_month:02d}_{resolution}.nc"
	# Read the current month's data
	tp_current = read_nc_file(file_current)
	# Read next month's data for last day computation
	tp_next = read_nc_file(file_next) if os.path.exists(file_next) else None
	# Compute daily precipitation
	daily_precip = compute_daily_precip(tp_current, tp_next)
	return daily_precip
	 
def get_daily_precip_era5(year, month, day, **kw):
	global _DATA_CACHE
	obs_source = kw['obs_source']
	cache_key = get_data_cache_key(obs_source)
	try:
		cache = _DATA_CACHE[cache_key]
	except:
		_DATA_CACHE[cache_key] = cache = {}
	#key = f'{year}{month:02d}{obs_source['resolution']}'
	key = f'{year}{month:02d}{obs_source['subset']}'
	if not key in cache:
		cache[key] = process_monthly_files_era5(year, month, **kw)
	if day > len(cache[key]):
		#Context().debug(f'!!! Daily data not found for {year}, {month}...')
		t = f'Daily data not found for {year}, {month}'
		if len(cache[key]) < calendar.monthrange(year,month)[1]: 
			t += f' -- next month\'s file is probably missing...'
		raise IndexError(t)
	return obs_source['precip_mult']*cache[key][day-1]

def get_daily_precip(year, month, day, **kw):
	obs_source = kw['obs_source']
	if obs_source['model_name'] == 'era5':
		return get_daily_precip_era5(year, month, day, **kw)
	raise ValueError("Unknown obs_source:", obs_source)


def compute_thresholds(refdate, all_data, **kw):
	mask = get_prop('mask', False, **kw)
	thr_window = BIAS_CORRECTION_WINDOW_DAYS
	# This must be defined, cannot use a default
	precip_mm = kw['precip_mm']
	init_times = sorted(all_data.keys())
	if True:
	# e = [
	# 	f'{Context().get_cachedir()}/{self.modelname}',
	# 	'thresholds',
	# 	'window',
	# 	f'{Context().get_attr('obs_source_name')}',
	# 	f'{precip_mm}',
	# 	f'{thr_window}',
	# 	refdate.strftime('%Y-%m-%d'),
	# 	Context().get_resolution()
	# ]
	# if not mask:
	# 	e.append('nomask')
	# cachefile = '_'.join(e)
	# try:
	# 	#raise
	# 	thr = loadpickle(cachefile)
	# except:
		# # Now we compute the thresholds for all the validtimes
		debug('Computing thresholds:', refdate)
		# Make a new data array by combining all initial times for this refdate:
		data = {'model': np.array([all_data[it]['model_data'] for it in init_times])}
		# We now collect the data from the obs source:
		# Pad the validtimes vector first:
		obsdata_padded = None
		n = int((thr_window-1)/2)
		for j, it in enumerate(init_times):
			vtvec = all_data[it]['valid_times']
			vtvec_padded = [vtvec[0] + timedelta(hours = 24*i) for i in range(-n,0)]
			a = [get_daily_precip(vt.year, vt.month, vt.day, **kw) for vt in vtvec_padded]
			a += all_data[it]['obs_data']
			vtvec_padded = [vtvec[-1] + timedelta(hours = 24*i) for i in range(1,n+1)]
			a += [get_daily_precip(vt.year, vt.month, vt.day, **kw) for vt in vtvec_padded]
			if obsdata_padded is None:
				obsdata_padded = np.empty([len(init_times), len(a)] + list(a[0].shape))
			obsdata_padded[j] = np.array(a)
		# We now similarly combine the observational data
		# Use sliding_window_view to create overlapping windows along axis=1
		a = sliding_window_view(obsdata_padded, window_shape=(thr_window,), axis=1)
		data['obs'] = np.moveaxis(a,-1,1)
		#Context().debug(obsdata_padded.shape, a.shape)
		#Context().debug(data['obs'].shape, data['model'].shape)
		if mask:
			thr = {}
			for idx,it in enumerate(init_times):
				keep = np.arange(len(init_times)) != idx
				sliced = {}
				for k,a in data.items():
					a = a[keep,:,:,:,:]
					sliced[k] = a.reshape([np.prod(a.shape[:2])] + list(a.shape[2:]))
				# These don't have to be integers now:
				obs_percentiles = np.sum((sliced['obs'] <= precip_mm), axis=0) / sliced['obs'].shape[0] * 100
				# Vectorizing is much much faster!!
				model_data = np.sort(sliced['model'], axis=0)  # Sort along the "time" axis
				num_samples = model_data.shape[0]
				# Add 0.5 for rounding and multply by (num_samples-1):
				percentile_indices = (obs_percentiles / 100 * (num_samples - 1) + 0.5).astype(int)
				a = np.take_along_axis(model_data, percentile_indices[None, :, :, :], axis=0).squeeze(axis=0)
				debug(it, a.shape, precip_mm, np.mean(a.ravel()))
				thr[it] = a
			#sys.exit()
		else:
			sliced = {}
			for k,a in data.items():
				sliced[k] = a.reshape([np.prod(a.shape[:2])] + list(a.shape[2:]))
			obs_percentiles = np.sum((sliced['obs'] <= precip_mm), axis=0) / sliced['obs'].shape[0] * 100
			model_data = np.sort(sliced['model'], axis=0)  # Sort along the "time" axis
			num_samples = model_data.shape[0]
			percentile_indices = (obs_percentiles / 100 * (num_samples - 1) + 0.5).astype(int)
			thr = np.take_along_axis(model_data, percentile_indices[None, :, :, :], axis=0).squeeze(axis=0)
			debug(thr.shape, precip_mm, np.mean(thr.ravel()))
			
		#savepickle(cachefile, thr)
	return thr

def compute_percentile_thresholds(refdate, all_data, key, percentile=None, **kw):
	window_days = PERCENTILE_WINDOW_DAYS[key]
	init_times = sorted(all_data.keys())
	debug(f'Computing percentile-based thresholds for {key}:', refdate)
	# Make a new data array by combining all initial times for this refdate:
	key2 = ('obs_data' if key=='clim' else 'model_data')
	# Insert empty dimension to make this compatible
	if key=='clim':
		obsdata_padded = None
		n = int((window_days-1)/2)
		for j, it in enumerate(init_times):
			vtvec = all_data[it]['valid_times']
			vtvec_padded = [vtvec[0] + timedelta(hours = 24*i) for i in range(-n,0)]
			a = [get_daily_precip(vt.year, vt.month, vt.day, **kw) for vt in vtvec_padded]
			a += all_data[it][key2]
			vtvec_padded = [vtvec[-1] + timedelta(hours = 24*i) for i in range(1,n+1)]
			a += [get_daily_precip(vt.year, vt.month, vt.day, **kw) for vt in vtvec_padded]
			if obsdata_padded is None:
				obsdata_padded = np.empty([len(init_times), len(a)] + list(a[0].shape))
			obsdata_padded[j] = np.array(a)
		# We now similarly combine the observational data
		# Use sliding_window_view to create overlapping windows along axis=1
		data = sliding_window_view(obsdata_padded, window_shape=(window_days,), axis=1)
		#print(data.shape)
		data = np.moveaxis(data,-1,1)
		#print(data.shape)
		data = data.reshape([np.prod(data.shape[:2])] + list(data.shape[2:]))
		thr = np.percentile(data, percentile, axis=0)
	else:
		data = np.array([all_data[it][key2] for it in init_times])
		#debug(data['model'].shape)
		n_years, n_members, n_lead_times, n_lat, n_lon = data.shape
		windows = sliding_window_view(data, window_shape=window_days, axis=2)
		#n_years, n_members, n_lead_times, n_lat, n_lon, window_days
		#print('windows.shape:',windows.shape)
		data = np.moveaxis(windows,-1,0)
		#print('data.shape:',data.shape)
		windows_reshaped = data.reshape([np.prod(data.shape[:3])] + list(data.shape[3:]))
		#print('windows_reshaped.shape:',windows_reshaped.shape)
		thr_windows = np.percentile(windows_reshaped, percentile, axis=0)
		print('thr_windows.shape:',thr_windows.shape)
		# Pad to get full lead_time dimension
		# For early indices, use first window; for late indices, use last window
		pad_before = window_days // 2
		pad_after = n_lead_times - thr_windows.shape[0] - pad_before
		thr = np.pad(thr_windows, ((pad_before, pad_after), (0, 0), (0, 0)), 
						mode='edge')
	debug(thr.shape, np.mean(thr.ravel()))
	#sys.exit()			
	return thr

def zoom(region = None, **kw):
	"""Create a cutout from the full grid"""
	if not region in _ZOOM_DIC:
		g = get_grid(**kw)
		if region is None:
			bbox = (
				np.min(g['lat'].ravel()),
				np.max(g['lat'].ravel()),
				np.min(g['lon'].ravel()),
				np.max(g['lon'].ravel()),
			)
		else:
			bbox = CBOXES[region]
		J4 = np.nonzero((g['lon'][0,:]>=bbox[2])&(g['lon'][0,:]<=bbox[3]))[0]
		I4 = np.nonzero((g['lat'][:,0]>=bbox[0])&(g['lat'][:,0]<=bbox[1]))[0]
		K4 = np.nonzero(
			(g['lon'].ravel()>=bbox[2])&
			(g['lon'].ravel()<=bbox[3])&
			(g['lat'].ravel()>=bbox[0])&
			(g['lat'].ravel()<=bbox[1])
		)[0]
		ix4 = np.ix_(I4,J4)
		lon4 = g['lon'][ix4]
		lat4 = g['lat'][ix4]
		try:
			lsm4 = g['lsm'][ix4]
		except:
			lsm4 = None
		asp4 = (np.max(lon4.ravel())-np.min(lon4.ravel())) / (np.max(lat4.ravel())-np.min(lat4.ravel()))
		d = {
			'lon': lon4,
			'lat': lat4,
			'lsm': lsm4,
			'I': I4,
			'J': J4,
			'K': K4,
			'ix': ix4,
			'sz': np.prod(lon4.shape),
			'aspect_ratio': asp4
		}
		d['proj'] = ccrs.PlateCarree()
		d['extent'] = [bbox[2],bbox[3],bbox[0],bbox[1]]
		_ZOOM_DIC[region] = d
	return _ZOOM_DIC[region]

def get_borders():
	global _BORDERS
	if _BORDERS is None:
		_BORDERS = cartopy.feature.NaturalEarthFeature(category='cultural',name='admin_0_boundary_lines_land',scale='10m',facecolor='none')
	return _BORDERS

def drawborders(ax):
	b = get_borders()
	#b = cf.BORDERS
	ax.add_feature(b,edgecolor=BORDERCOLOR,linewidth=COASTLINEWIDTH*2)
	#ax.add_feature(b,edgecolor='yellow',linewidth=COASTLINEWIDTH)

def drawfeatures(ax):
	ax.add_feature(cf.COASTLINE,edgecolor='w',linewidth=COASTLINEWIDTH*1.6)
	ax.add_feature(cf.COASTLINE,edgecolor=COASTLINECOLOR,linewidth=COASTLINEWIDTH)
	#ax.add_feature(cf.COASTLINE,edgecolor=BORDERCOLOR,linewidth=COASTLINEWIDTH*0.75)
	ax.add_feature(cf.LAKES,edgecolor=COASTLINECOLOR,linewidth=COASTLINEWIDTH*.75,facecolor='None')
	drawborders(ax)

def define_hits_prob(forecast_period_days, percentile, forecast_type=None, **kw):
	if forecast_type in ['dry_spells','heavy_rainfall_days']:
		consecutive_days = kw['consecutive_days']
		if forecast_type in ['dry_spells']:
			p_daily = percentile / 100.
		elif forecast_type in ['heavy_rainfall_days']:
			p_daily = (100-percentile) / 100.
		# P(all days wet in window) for one window
		p_window = p_daily ** consecutive_days
		# Number of possible windows
		n_windows = forecast_period_days - consecutive_days + 1
		# P(at least one window has wet spell)
		prob = 1 - (1 - p_window) ** n_windows
		print(f"percentile: {percentile}")
		print(f"consecutive_days: {consecutive_days}")
		print(f"p_daily: {p_daily}")
		print(f"p_window: {p_window}")
		print(f"n_windows: {n_windows}")
		return prob
	raise ValueError(f"Unknown forecast type: {forecast_type}")

def define_hits(precip_data, thr, ax=1, forecast_type=None, **kw):
	"""
	This function is key, as it computes whether a criterion is met.
	precip_data is either model data or obs data
	thr is the threshold, either a scalar or an array with dimensions (day, lat, lon)
		-> thr is computed for each day
	For the forecast type "dry_spells", it checks precip_data for whether there are any dry_spells of length "consecutive_days"
	"""
	if forecast_type == 'dry_spells':
		consecutive_days = kw['consecutive_days']
		rain_below_threshold = precip_data < thr
		windows = sliding_window_view(rain_below_threshold, window_shape=consecutive_days, axis=ax)
		dry_spells = np.all(windows, axis=-1)
		hits = (np.any(dry_spells, axis=ax)).astype(int)
	elif forecast_type == 'heavy_rainfall_days':
		consecutive_days = kw['consecutive_days']
		rain_above_threshold = precip_data > thr
		windows = sliding_window_view(rain_above_threshold, window_shape=consecutive_days, axis=ax)
		wet_spells = np.all(windows, axis=-1)
		hits = (np.any(wet_spells, axis=ax)).astype(int)
	elif forecast_type == 'accumulated_rainfall':
		# # Accumulated rainfall forecast
		# forecast_options_rainfall = {
		# 	'forecast_type': 'accumulated_rainfall',
		# 	'rainfall_thresholds': [50],
		# 	'accumulation_days': [2],
		# 	'lead_time_day0': 2,
		# 	'forecast_periods_days': [14],
		# 	'regions': ['EA']
		# }
		# Accumulated rainfall forecast
		# Check if any N-day accumulation exceeds threshold
		accumulation_days = kw['accumulation_days']
		# Create sliding windows of N consecutive days
		windows = sliding_window_view(precip_data, window_shape=accumulation_days, axis=ax)
		# Sum rainfall over each N-day window
		accumulated = np.sum(windows, axis=-1)
		# Check if any window exceeds threshold
		# thr can be a scalar or array - if array, need to handle differently
		if np.isscalar(thr):
			exceeds_threshold = accumulated > thr
		else:
			# If thr varies by day, we need to compare against appropriate threshold
			# For accumulated rainfall, use the threshold from the first day of each window
			# This assumes thr has shape matching the sliding windows
			if thr.ndim > 1:
				# Take threshold from first day of accumulation period
				thr_for_windows = thr[:accumulated.shape[ax]] if ax == 0 else thr
			exceeds_threshold = accumulated > thr
		# Check if ANY accumulation period exceeds threshold
		hits = (np.any(exceeds_threshold, axis=ax)).astype(int)
	else:
		raise ValueError(f"Unknown forecast type: {forecast_type}")
	return hits

def create_map_axes(z,
			figw_inches=3.5, 
			margin_left_inches=0.05, 
			margin_right_inches=0.05, 
			margin_top_inches=0.7, 
			margin_bottom_inches=0.8,
			dpi=150):
	"""
	Create a matplotlib figure and axes with specified dimensions and margins.
	
	Parameters
	----------
	z = the zoom dictionary
	figw_inches : float
		Figure width in inches
	margin_left_inches : float
		Left margin in inches
	margin_right_inches : float
		Right margin in inches
	margin_top_inches : float
		Top margin in inches
	margin_bottom_inches : float
		Bottom margin in inches
	dpi : int
		Dots per inch for the figure
	Returns
	-------
	fig : matplotlib.figure.Figure
		The created figure
	ax : matplotlib.axes.Axes
		The created axes
	"""
	aspect_ratio = z['aspect_ratio']
	# Compute axes width
	axes_width_inches = figw_inches - margin_left_inches - margin_right_inches	
	# Compute axes height from aspect ratio
	axes_height_inches = axes_width_inches / aspect_ratio	
	# Compute figure height
	figh_inches = axes_height_inches + margin_top_inches + margin_bottom_inches	
	# Create figure
	fig = plt.figure(figsize=(figw_inches, figh_inches), dpi=dpi)
	# Add axes with the specified margins (in normalized coordinates)
	ax = fig.add_axes([
		margin_left_inches / figw_inches,      # left
		margin_bottom_inches / figh_inches,    # bottom
		axes_width_inches / figw_inches,       # width
		axes_height_inches / figh_inches],       # height
		projection = z['proj']
	)
	return fig, ax


def make_forecast_for_refdate(forecast_options, **kw):
	"""
	Generic forecast processor that works with any forecast type.
	
	Users only need to:
	1. Set forecast_type in their options
	2. Provide the relevant parameters (thresholds and auxiliary params)
	"""
    
	forecast_type = forecast_options['forecast_type']
	spec = FORECAST_SPECS[forecast_type]
	
	# Get the main threshold parameter name and values
	threshold_param = spec['threshold_param']
	thresholds = forecast_options[threshold_param]
	
	# Get auxiliary parameters (like consecutive_days, accumulation_days, etc.)
	aux_param_names = spec['auxiliary_params']
	aux_param_values = []
	for param in aux_param_names:
		if param == 'threshold_type':
			# Default to 'absolute' if not specified
			aux_param_values.append(forecast_options.get('threshold_type', ['absolute']))
		else:
			aux_param_values.append(forecast_options[param])

	aux_param_values = [forecast_options[param] for param in aux_param_names]
	
	lead_time_day0 = forecast_options['lead_time_day0']

	# Allow strings here
	try:
		model = kw['model']
	except:
		model = create_model(kw['model_name'])
		kw['model'] = model
	try:
		obs_source = kw['obs_source']
	except:
		obs_source = create_obs_source(kw['obs_source_name'])
		kw['obs_source'] = obs_source

	refdate = kw['refdate']

	try:
		fig_dir = kw['fig_dir']
	except:
		fig_dir = FIGDIR

	# See if we're testing  
	testing = get_prop('testing', False, **kw)
	force = get_prop('force', False, **kw)

	# Set some constants:
	ms = 2.5
	mew = 0.3
	lw = 0.75
	figw_inches = 3.5
	dpi = 150
	margin_left_inches = 0.15
	margin_right_inches = 0.15
	margin_top_inches = 0.6
	margin_bottom_inches = 0.5
	# Compute axes width
	axes_width_inches = figw_inches - margin_left_inches - margin_right_inches	
	cbar_padding_inches = 0.05  # padding between axes and colorbar
	cbar_height_inches = 0.2  # height of colorbar
	cbar_bottom_inches = margin_bottom_inches - cbar_height_inches - cbar_padding_inches # space from bottom of figure
	
	# Make a list of regions for which we show maps:
	display_regions = []
	for region in forecast_options['regions']:
		display_regions.append(country_code_to_region(region))

	# Get the forecast data for the refdate (initial date):
	f = collect_forecast_data_for_refdate(model=model, refdate=refdate)
	# debug("model_data shape:", f["model_data"].shape)
	# debug("first valid time:", f["valid_times"][0], "last:", f["valid_times"][-1])
	
	g = get_grid()
	# debug("grid shape:", g['lon'].shape)

	# Get the closest hindcast date for calibration
	closest_refdate = find_closest_hindcast_refdate(model=model, refdate=refdate)
	if closest_refdate is None:
		raise RuntimeError(f"No hindcast refdate found within search window for refdate={refdate}")
	debug('refdate, closest_refdate:', refdate, closest_refdate)
	
	# Now get all the initial times for that closest refdate
	init_times = get_init_times_for_refdate(model=model, refdate=closest_refdate)
	#debug('Initial times for closest refdate:', init_times)

	# For each of these initial times, collect hindcast data
	# This returns a dictionary with key init_time, and for each init_time another
	# dictionary with keys "model_data" and "valid_times"
	all_data = collect_hindcast_data_for_refdate(model=model, refdate=closest_refdate)
	# Print these if you want:
	if False:
		for key,val in h.items():
			debug(key, val['model_data'].shape, val['valid_times'])

	# Now we collect "observational data" for these init_times
	# In our case this is ERA5 data
	# Just add another key to the all_data dictionary for each init_time
	for j,it in enumerate(init_times):
		obs_data = []
		for vt in all_data[it]['valid_times']:
			obs_data.append(get_daily_precip(vt.year, vt.month, vt.day, obs_source=obs_source))
			#debug(vt, obs_data[-1].shape)
			#sys.exit()
		all_data[it]['obs_data'] = obs_data

	# Main processing loop - completely generic
	for threshold in thresholds:

		for forecast_period in forecast_options['forecast_periods_days']:
			lead_time_day1 = lead_time_day0 + forecast_period - 1
			# Find the day index for this subselection:
			it = init_times[0]
			valid_times = all_data[it]['valid_times']
			idx = []
			for i,vt in enumerate(valid_times):
				td = vt - it
				if td.days > lead_time_day1:
					break
				if td.days < lead_time_day0:
					continue
				idx.append(i)
			#debug(lead_time_day0, lead_time_day1, idx, len(idx))
			fcst_data = f['model_data'][:,idx,:,:]
			all_obsdata = np.array([
				[all_data[it]['obs_data'][i] for i in idx] 
				for it in init_times
			])
			bad = (all_obsdata[0,0,:,:]>999)
            
			# Loop over all combinations of auxiliary parameters
			for aux_combo in itertools.product(*aux_param_values):
					
				# Create dict of auxiliary params for this iteration
				aux_dict = dict(zip(aux_param_names, aux_combo))

				# Compute thresholds based on threshold type
				threshold_type = aux_dict.get('threshold_type')
				#print(aux_dict, threshold_type)
				#sys.exit()

				# Get the kwargs for define_hits using the spec's function
				hits_kwargs = spec['hits_kwargs'](threshold, aux_dict)
		
				if threshold_type == 'absolute':
					# This is for bias-correction using quantile mapping
					# We must map the threshold in the obs_data to a corresponding threshold in the model_data
					# thr contains the tresholds per day and grid point : (day, lat, lon)
					thr = compute_thresholds(
						closest_refdate, 
						all_data,
						precip_mm=threshold,
						obs_source=kw.get('obs_source')
					)
					# Compute hits
					hits = {
						'model': define_hits(fcst_data, thr[idx, :, :], **hits_kwargs),
						'clim': define_hits(all_obsdata, threshold, **hits_kwargs)
					}
				elif threshold_type == 'percentile':
					# No bias-correction needed, as we just compute the model-internal thresholds
					# thr contains the tresholds per day and grid point : (day, lat, lon)
					#h = define_hits_prob(forecast_period, threshold, **hits_kwargs)
					#print(h)
					thr = compute_percentile_thresholds(
						closest_refdate, 
						all_data,
						'model',
						percentile=threshold
					)
					# Compute hits
					hits = {
						'model': define_hits(fcst_data, thr[idx, :, :], **hits_kwargs),
					}
					thr = compute_percentile_thresholds(
						closest_refdate, 
						all_data,
						'clim',
						percentile=threshold,
						obs_source=kw.get('obs_source')
					)
					hits['clim'] = define_hits(all_obsdata, thr[idx, :, :], **hits_kwargs)
					# print(hits['model'].shape)
					# print(np.mean(hits['model'], axis=0).ravel().mean())
					# below_threshold = fcst_data < thr[idx, :, :]
					# print(f"Fraction of forecast days below threshold: {below_threshold.mean()}")
					# sys.exit()

				for key in ['model','clim','diff']:
					for dregion in display_regions:
						z = zoom(dregion)
						if key in ['model','clim']:
							a = np.mean(hits[key], axis=0)[np.ix_(z['I'],z['J'])]
							cv = np.arange(0,1.1,.1)  
							cmap = plt.get_cmap('Reds',len(cv)-1)
							extend = 'neither'
							#ticks = (cv[::2] if not vbar else cv)
						else: 
							a = np.mean(hits['model'], axis=0)[np.ix_(z['I'],z['J'])] - np.mean(hits['clim'], axis=0)[np.ix_(z['I'],z['J'])]
							cv = np.arange(-.5,0.51,.1)
							cmap = plt.get_cmap('RdBu_r',len(cv)+1)
							cmap = ListedColormap(cmap(np.linspace(0, 1, len(cv)+1))[1:-1])
							extend = 'both'
							#ticks = (cv[1:-1:2] if not vbar else cv[1:-1])
						cmap.set_bad(color='.7')
						# Compute axes height from aspect ratio
						axes_height_inches = axes_width_inches / z['aspect_ratio']	
						# Compute figure height
						figh_inches = axes_height_inches + margin_top_inches + margin_bottom_inches	
						# Create figure
						fig = plt.figure(figsize=(figw_inches, figh_inches), dpi=dpi)
						# Add axes with the specified margins (in normalized coordinates)
						ax = fig.add_axes([
							margin_left_inches / figw_inches,      # left
							margin_bottom_inches / figh_inches,    # bottom
							axes_width_inches / figw_inches,       # width
							axes_height_inches / figh_inches],       # height
							projection = z['proj']
						)
						ax.set_aspect('auto')
						ax.add_feature(cf.COASTLINE,edgecolor='k',linewidth=lw)
						ax.add_feature(cf.LAKES,edgecolor='k',linewidth=lw,facecolor='None')
						borders = get_borders()
						a[bad[np.ix_(z['I'],z['J'])]] = np.nan
						data = ax.pcolormesh(z['lon'],z['lat'],a,cmap=cmap,vmin=cv[0],vmax=cv[-1],shading='auto',transform=ccrs.PlateCarree())
						srcdesc = {'model': 'ECMWF', 'clim': 'Climatology', 'diff': 'ECMWF$-$Climatology'}[key]
						ax.add_feature(borders,edgecolor='y',linewidth=lw)
						t = f'Dry Spell Forecast ({srcdesc})'
						t += '\nInitial Date: %s'%(refdate.strftime('%d %B, %Y'))
						t += '\nForecast Period: %s to %s'%(
							(refdate+timedelta(days=lead_time_day0-1)).strftime('%d %b'),
							(refdate+timedelta(days=lead_time_day1-1)).strftime('%d %b')
						)
						t += '\nNo Rain Threshold: %s mm per day'%(threshold)
						plt.title(t, fontsize=fs-1)
						cbar_ax = fig.add_axes([
							margin_left_inches / figw_inches,                    # same left as main axes
							cbar_bottom_inches / figh_inches,                    # bottom position
							axes_width_inches / figw_inches,                     # same width as main axes
							cbar_height_inches / figh_inches                     # colorbar height
						])
						# Create colorbar
						cbar = ColorbarBase(cbar_ax, 
							cmap=cmap, 
							norm=data.norm,
							orientation='horizontal',
							extend=extend
						)
						cbar_ax.tick_params(labelsize=fs-1)

						if forecast_type == 'dry_spells':
							e = [
								'fcst',
								refdate.strftime('%Y-%m-%d'),
								f"{lead_time_day0}-{lead_time_day1}",
								f"{threshold}",
								f"{hits_kwargs['consecutive_days']}",
								key,
								region_to_country_code(dregion)
							]
							filename = f'_'.join(e)
						else:
							filename = f"{key}_{threshold}_{i}_{dregion}"
						savefig(filename, base_path = fig_dir)

def test():

	# Example configurations - users just need to edit these dictionaries:

	# Heavy rainfall days forecast in millimeters
	forecast_options_heavy_absolute = {
		'forecast_type': 'heavy_rainfall_days',
		'daily_rainfall_thresholds': [50],
		'consecutive_days': [1],
		'threshold_type': ['absolute'],
		'lead_time_day0': 2,
		'forecast_periods_days': [14],
		'regions': ['EA']
	}

	# Heavy rainfall days forecast with percentile thresholds
	forecast_options_heavy_percentile = {
		'forecast_type': 'heavy_rainfall_days',
		'daily_rainfall_thresholds': [95],
		'consecutive_days': [1],
		'threshold_type': ['percentile'],
		'lead_time_day0': 2,
		'forecast_periods_days': [14],
		'regions': ['EA']
	}
		
	# Dry spell forecast
	forecast_options_dry = {
		'forecast_type': 'dry_spells',
		'no_rain_thresholds': [1,2,5],
		'consecutive_days': [5,7,9],
		'threshold_type': ['absolute'],
		'lead_time_day0': 2,
		'forecast_periods_days': [14,21],
		'regions': ['EA','ET','MW','MG']
	}
	forecast_options_dry_single = {
		'forecast_type': 'dry_spells',
		'no_rain_thresholds': [1],
		'consecutive_days': [5],
		'threshold_type': ['absolute'],
		'lead_time_day0': 2,
		'forecast_periods_days': [14],
		'regions': ['MW']
	}
		
	# Dry spell forecast
	forecast_options_dry_percentile = {
		'forecast_type': 'dry_spells',
		'no_rain_thresholds': [50],
		'consecutive_days': [7],
		'threshold_type': ['percentile'],
		'lead_time_day0': 2,
		'forecast_periods_days': [14],
		'regions': ['EA']
	}

	# # Accumulated rainfall forecast
	# Not implemented yet
	# forecast_options_rainfall = {
	# 	'forecast_type': 'accumulated_rainfall',
	# 	'accumulated_rainfall_thresholds': [50],
	# 	'accumulation_days': [2],
	# 	'lead_time_day0': 2,
	# 	'forecast_periods_days': [14],
	# 	'regions': ['EA']
	# }

	#forecast_options = forecast_options_dry_percentile
	forecast_options = forecast_options_dry_single

	make_forecast_for_refdate(
		forecast_options,
		model_name = 'ecmwf',
		refdate = (datetime.today() - timedelta(days=1)).date(),
		#refdate = datetime(2026,2,17).date(),
		obs_source_name = 'era5'
	)

def test_hindcasts():
	# res = collect_hindcast_data_for_refdate(
	# 	model = ecmf_native_new(),
	# 	refdate = datetime(2026,2,15).date()
	# )
	# print('-------- OLD')
	# for k, v in res.items():
	# 	print(v['valid_times'])
	# 	print(v['model_data'].shape)
	# 	break
	# collect_forecast_data_for_refdate(
	# 	model = ecmf_native_new(),
	# 	refdate = datetime(2026,2,17).date()
	# )
	res = collect_hindcast_data_for_refdate(
		model = ecmf_native_new(),
		refdate = datetime(2026,2,23).date()
	)
	print('-------- NEW')
	for k, v in res.items():
		print(v['valid_times'])
		print(v['model_data'].shape)
		break

if __name__ == "__main__":
	#test_hindcasts()
	#test_hindcasts()
	test()
	#plt.show()

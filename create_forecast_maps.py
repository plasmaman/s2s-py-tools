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


# ---- your constants ----
EXPNAME = "africa"
HOMEDIR = "/nird/home/kolstad"
DATADIR = "/nird/datapeak/NS9873K/kolstad"
DATALAKEDIR = "/nird/datalake/NS9853K/kolstad"
ECMF_NATIVE_ROOT = f"{DATALAKEDIR}/ecwmf"
CACHEROOT = f"{DATADIR}/cache/python"
FIGDIR = f"{DATADIR}/../www/{EXPNAME}"
HINDCAST_DIR = f"{DATADIR}/{EXPNAME}"
ERA5_DIR = f"{DATADIR}/{EXPNAME}"
RESOLUTION = 'native'

# Preferences:
DEBUG_LEVEL = 0
BIAS_CORRECTION_WINDOW_DAYS = 11 # Same as nbr of ensemble members in hindcast

# Caching:
CACHEDIR = f"{CACHEROOT}/{EXPNAME}/native"
USE_CACHE = True
_GRID_CACHE = None
_DATA_CACHE = {}

# optionally: savepickle(path, obj)

# Global variables:
#global_data_cache = {}

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
	return f"{obs_source['model_name']}_{obs_source['resolution']}"

def era5_new(*, 
	base_dir=ERA5_DIR, 
	variable_name="tp", 
	cache_dir=CACHEDIR, 
	resolution = RESOLUTION,
	precip_mult = 1000.
):
	dic = {
		"model_name": "era5",
		"base_dir": base_dir,       
		"variable_name": variable_name,
		"cache_dir": cache_dir,
		"precip_mult": precip_mult,
		"resolution": resolution
		#"data_cache": {},
	}
	#global_data_cache[dic] = {}
	return dic

def ecmf_native_new(*, 
	fcst_dir=ECMF_NATIVE_ROOT, 
	hcst_dir=HINDCAST_DIR, 
	variable_name="tp", 
	cache_dir=CACHEDIR, 
	first_lead_time=24, 
	last_lead_time=720, 
	resolution = RESOLUTION,
	precip_mult = 1000.
):
	dic = {
		"model_name": "ecmf",
		"fcst_dir": fcst_dir,       
		"hcst_dir": hcst_dir,       
		"variable_name": variable_name,
		"first_lead_time": int(first_lead_time),
		"last_lead_time": int(last_lead_time),
		"cache_dir": cache_dir,
		"precip_mult": precip_mult,
		"resolution": resolution
	}
	#global_data_cache[dic] = {}
	return dic

def open_group_once(grib_path, group, idx_path=None):
	"""
	Open one cfgrib 'filter_by_keys' group from a file.
	group: 'cf' or 'pf' (as in your original code)
	idx_path: path to cfgrib index file (optional)
	"""
	backend_kwargs = {
		"filter_by_keys": {"dataType": group},
	}
	if idx_path is not None:
		backend_kwargs["indexpath"] = str(idx_path)

	return xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs=backend_kwargs)


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
	# Return CF and PF datasets with cumulative data, stacked over valid_time.
	files = sorted(glob(pattern))
	cf_parts, pf_parts = [], []

	for f in files:
		idx = f"{Path(f).with_suffix('').as_posix()}.idx"
		try:
			ds_cf = open_group_once(f, "cf", idx)
			ds_cf = ensure_valid_time(ds_cf)
			cf_parts.append(ds_cf)
		except Exception:
			pass

		try:
			ds_pf = open_group_once(f, "pf", idx)
			ds_pf = ensure_valid_time(ds_pf)
			pf_parts.append(ds_pf)
		except Exception:
			pass

	if not cf_parts and not pf_parts:
		raise RuntimeError("No cf/pf groups found in files.")
	
	ds_cf_all = (
		xr.concat(
			cf_parts,
			dim="valid_time",
			data_vars="minimal",
			coords="minimal",
			compat="override",
			join="override",
			combine_attrs="override",
		).sortby("valid_time")
		if cf_parts
		else None
	)

	ds_pf_all = (
		xr.concat(
			pf_parts,
			dim="valid_time",
			data_vars="minimal",
			coords="minimal",
			compat="override",
			join="override",
			combine_attrs="override",
		).sortby("valid_time")
		if pf_parts
		else None
	)

	return ds_cf_all, ds_pf_all

def merge_cf_pf_along_number(model, ds_cf, ds_pf):
	if ds_cf is not None and ds_pf is not None:
		common_vt = np.intersect1d(ds_cf["valid_time"].values, ds_pf["valid_time"].values)
		if common_vt.size == 0:
			raise ValueError("No overlapping valid_time between cf and pf.")
		ds_cf = ds_cf.sel(valid_time=common_vt)
		ds_pf = ds_pf.sel(valid_time=common_vt)

	if ds_cf is not None:
		ds_cf = ds_cf.expand_dims(number=[0])

	if ds_pf is not None and "number" in ds_pf.dims:
		ds_pf = ds_pf.assign_coords(number=(ds_pf["number"] + 1))

	if ds_cf is not None and ds_pf is not None:
		ds_all = xr.concat(
			[ds_cf, ds_pf],
			dim="number",
			data_vars="minimal",
			coords="minimal",
			compat="equals",
			join="override",
			combine_attrs="override",
		)
	else:
		ds_all = ds_cf if ds_cf is not None else ds_pf

	if ds_all is None:
		raise ValueError("Both ds_cf and ds_pf are None.")

	var = model["variable_name"]
	if var not in ds_all:
		raise ValueError(f"'{var}' not found in merged dataset.")

	n_vt = ds_all.sizes["valid_time"]

	daily = ds_all[var].sortby("valid_time").diff("valid_time")
	ds_all = ds_all.isel(valid_time=slice(1, None))
	ds_all = ds_all.assign(tp_daily=daily)

	n_daily = ds_all["tp_daily"].sizes["valid_time"]
	if n_daily != n_vt - 1:
		raise AssertionError(f"Expected tp_daily times = valid_time-1, got {n_daily} vs {n_vt}")

	return ds_all


def collect_forecast_data_for_refdate(**kw):
	model = kw['model']
	refdate = kw['refdate']
	os.makedirs(model["cache_dir"], exist_ok=True)
	cachefile = f'{model["cache_dir"]}/fcst_data_for_refdate_{refdate.strftime("%Y%m%d")}_native'
	# memory cache
#  if cachefile in model["data_cache"]:
#      return model["data_cache"][cachefile]
	# disk cache (optional)
	try:
		result = loadpickle(cachefile)
		#model["data_cache"][cachefile] = result
		return result
	except Exception:
		pass
	init_tag = refdate.strftime("%m%d")
	pattern = f'{model["fcst_dir"]}/A1F{init_tag}*1'
	ds_cf, ds_pf = load_cf_pf(pattern)
	ds_all = merge_cf_pf_along_number(model, ds_cf, ds_pf)
	# arr.shape = (nbr of members, nbr of days, lat, lon), e.g.: (101, 30, 125, 78)
	arr = ds_all["tp_daily"].values 
	result = {
		"model_data": arr,  # keep same for now
		"valid_times": [
			dt.astype("datetime64[ms]").astype(datetime).date()
			for dt in ds_all["valid_time"].values
		],
	}
	# optional: savepickle(cachefile, result)
	#model["data_cache"][cachefile] = result
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

	cachefile = f"{cache_dir}/grid_{RESOLUTION}"
	try:
		g = loadpickle(cachefile)
		_GRID_CACHE = g
		return g
	except Exception:
		pass

	# Build grid from the ERA5 LSM file
	file_path = f"{DATADIR}/{EXPNAME}/era5_africa_lsm_{RESOLUTION}"
	file_path += ".nc"

	ds = nc.Dataset(file_path)
	lon_1d = ds["longitude"][:]
	lat_1d = ds["latitude"][:]
	lsm = ds["lsm"][0, :, :]
	ds.close()

	lon, lat = np.meshgrid(lon_1d, lat_1d)

	g = {
		"lon": lon,
		"lat": lat,
		"lsm": lsm,
		"inside": {},  # fill later
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
	wildcard += f"_{model['resolution']}.grb"
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
	debug(all_refdates)
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
	filename = f'{filename}_{RESOLUTION}'
	filename += '.grb'
	debug(filename, kw)
	return filename

def read_hindcast_file(**kw):
	filename = get_hindcast_filename(**kw)
	# Open the GRIB file using xarray and the cfgrib engine
	# This could fail, but then the exception has to be handled where read_file is called from
	ds = xr.open_dataset(filename, engine='cfgrib')
	return ds

def get_init_times_for_refdate(**kw):
	control_ds = read_hindcast_file(suffix='cf', **kw)
	init_times = control_ds['time'][:].values
	return [pd.Timestamp(init_time) for init_time in sorted(init_times)]

def collect_hindcast_data_for_refdate(**kw):
	model = kw['model']
	refdate = kw['refdate']
	e = [
		'hindcast_data_for_reftime',
		refdate.strftime('%Y%m%d'),
		RESOLUTION
	]
	cachefile = f'{model["cache_dir"]}/{"_".join(e)}'
	# try:
	#     return data_cache[cachefile]
	# except:
	#     pass
	try:
		return loadpickle(cachefile)
	except:
		pass
	# We have to read two files, control (cf) and perturbed (pf):
	ensemble_ds = read_hindcast_file(refdate=refdate, suffix='pf')
	control_ds = read_hindcast_file(refdate=refdate, suffix='cf')
	ensemble_precip = ensemble_ds[model["variable_name"]]
	control_precip = control_ds[model["variable_name"]]
	control_precip_expanded = control_precip.expand_dims(dim='number', axis=0)
	combined_precip = xr.concat([control_precip_expanded, ensemble_precip], dim='number')
	init_times = control_ds['time'][:].values
	result = {}
	for j,init_time in enumerate(init_times):
		first = combined_precip.isel(time=j)
		mdata = np.empty_like(first)
		#Context().debug(j,init_time)
		mdata[:, 0, :, :] = first.isel(step=0)
		for step in range(1, first.shape[1]):
			mdata[:, step, :, :] = first.isel(step=step) - first.isel(step=step-1)
		# The new array mdata now has 24-hour accumulated values for the first initial time
		# with the shape (number=10, step=30, latitude=100, longitude=61)
		# eradata = []
		valid_times = []
		for i, t in enumerate(init_time + control_ds['step'][:].values):
			dt = pd.Timestamp(t)
			valid_times.append(dt)
		result[pd.Timestamp(init_time)] = {
			# 'eradata': eradata,
			'model_data': model["precip_mult"]*mdata,
			'valid_times': valid_times
		}
	return result

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
	resolution = obs_source['resolution']
	file_prefix = f"{obs_source['base_dir']}/{obs_source['model_name']}_{EXPNAME}_{obs_source['variable_name']}"
	file_current = f"{file_prefix}_{year}_{month:02d}_{resolution}.nc"
	next_month = (1 if month==12 else month+1)
	next_year = (year+1 if month==12 else year)
	file_next = f"{file_prefix}_{next_year}_{next_month:02d}_{resolution}.nc"
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
	key = f'{year}{month:02d}{obs_source['resolution']}'
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


def define_hits(precip_data, thr, ax=1, forecast_type=None, **kw):
	"""
	This function is key, as it computes whether a criterion is met.
	precip_data is either model data or obs data
	thr is the threshold, either a scalar or an array with dimensions (day, lat, lon)
		-> thr is computed for each day
	For the forecast type "dry_spells", it checks precip_data for whether there are any dry_spells of length "dry_spell_max_length"
	"""
	if forecast_type == 'dry_spells':
		dry_spell_max_length = kw['dry_spell_max_length']
		rain_below_threshold = precip_data < thr
		windows = sliding_window_view(rain_below_threshold, window_shape=dry_spell_max_length, axis=ax)
		dry_spells = np.all(windows, axis=-1)
		hits = (np.any(dry_spells, axis=ax)).astype(int)
	else:
		raise ValueError(f"Unknown forecast type: {forecast_type}")
	return hits

def make_forecast_for_refdate(force = False, testing = False, **kw):
	model = kw['model']
	refdate = kw['refdate']
	obs_source = kw['obs_source']
	forecast_options = kw['forecast_options']
	lead_time_day0 = forecast_options['lead_time_day0']
	# Create model dictionary

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
	debug('Initial times for closest refdate:', init_times)

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

	forecast_type = forecast_options['forecast_type']
	for precip_mm in forecast_options['no_rain_thresholds']:
				
		# This is for bias-correction using quantile mapping
		# We must map the threshold in the obs_data to a corresponding threshold in the model_data
		# thr contains the tresholds per day and grid point : (day, lat, lon)
		thr = compute_thresholds(
			closest_refdate, 
			all_data,
			precip_mm = precip_mm,
			obs_source = kw['obs_source']
		)

		for i in forecast_options['forecast_periods_days']:
			lead_time_day1 = lead_time_day0 + i - 1
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
			all_obsdata = np.array([[all_data[it]['obs_data'][i] for i in idx] for it in init_times])
			bad = (all_obsdata[0,0,:,:]>999)
			for dry_spell_max_length in forecast_options['dry_spell_lengths']:
				# Compute a boolean for each day
				# Is there a dry spell of the specified length within the specified period?
				# hits['model'] has dimensions (member, lat, lon), shape e.g. (101, lat, lon)
				# hits['clim'] has dimensions (year, lat, lon), shape e.g. (20, lat, lon)
				hits = {
					'model': define_hits(fcst_data, thr[idx,:,:], forecast_type=forecast_type, dry_spell_max_length=dry_spell_max_length),
					'clim': define_hits(all_obsdata, precip_mm, forecast_type=forecast_type, dry_spell_max_length=dry_spell_max_length)
				}

def test():
		
	# Define which forecasts to make
	forecast_options = {
		'forecast_type': 'dry_spells',
		'no_rain_thresholds': [1],
		'dry_spell_lengths': [7],
		'lead_time_day0': 2,
		'forecast_periods_days': [14],
		'regions': ['EA']
	}
	make_forecast_for_refdate(
		model = ecmf_native_new(),
		refdate = (datetime.today() - timedelta(days=2)).date(),
		obs_source = era5_new(),
		forecast_options = forecast_options
	)

if __name__ == "__main__":
	 test()

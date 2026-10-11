
#SUBSET = 'malawi'
#SUBSET = 'native'
DEFAULT_SUBSET = 'native'
HOMEDIR = "/nird/home/kolstad"
#DATADIR = "/nird/datapeak/NS9873K/kolstad"
DATADIR = "/nird/datalake/NS9853K/kolstad"
ECMF_FORECAST_FILE_WILDCARD_OLD = f"{DATADIR}/ecwmf/A1F%m%d*1"
ECMF_HINDCAST_FILE_WILDCARD_OLD = f"{DATADIR}/ecwmf/A1H%m%d*1"
#ECMF_FORECAST_FILE_WILDCARD = f"/nird/datalake/NS9853K/users/rondro/DATA/IFS_example/ECMWFupload/mal_a1_ifs-subs_od_eefo_*_%Y%m%d*"
CACHEROOT = f"{DATADIR}/cache/python"
CACHEDIR = f"{CACHEROOT}/africa/native"
FIGDIR = '/nird/home/kolstad/python/s2s-py-tools/figs'
HINDCAST_DIR = f"{DATADIR}/africa"
ERA5_DIR = f"{DATADIR}/africa"
#ERA5_FILE_WILDCARD = f'{ERA5_DIR}/era5_tp_%Y_%m_{SUBSET}.nc'
#ERA5_LSM_FILE = f"{ERA5_DIR}/era5_lsm_{SUBSET}.nc"
ECMF_FORECAST_FILE_WILDCARD = {
	'malawi': f'{DATADIR}/ecwmf/A1F%m%d*1',
	'native': f'{DATADIR}/ecwmf/A1F%m%d*1',
	'new': f'{DATADIR}/ecwmf/noe_t1_ifs-subs_od_eefo_*_%Y%m%d*'
}
GRID = {
	'malawi': ['0.4/0.4'],
	'new': ['0.4/0.4'],
	'native': ['0.4/0.4']
}
AREA = {
	'malawi': [-5.2,25.2,-24,50],
	'native': [22.8,21.2,-26.8,52],
	'new': [72,-20.8,-38.4,54.8]
}
LEAD_TIME_DAYS = {
	'malawi': range(1,47),
	'native': range(1,31),
	'new': range(1,31)
}

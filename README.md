# s2s-py-tools (working title)

Procedural Python code for reading ECMWF S2S forecast data and generating forecast products. The first version included code to create maps representing subseasonal dry-spell forecasts.

## Dry-spell forecast creation

The dry-spell forecasts are described and validated [in this paper](https://doi.org/10.1007/s00382-025-07920-4). Citing from the abstract, the forecasts are based on 

> a predictive model for dry spells, aiming to provide farmers with actionable information to support agricultural decision-making and enhance resilience. The model, based on a dynamical subseasonal prediction system and validated using reanalysis and satellite-based data, focuses on Malawi as a case study. This model has significant skill in predicting the occurrence of at least one dry spell within the three weeks following initialisation, consistently outperforming a climatology-based reference model. Furthermore, we show that the model is applicable beyond Malawi, specifically in East Africa during both the March–May “long rains” and the October–November “short rains”, highlighting its broader relevance for regions where dry spells pose an agricultural risk.

The model used in the current version of the code is the ECMWF's [subseasonal prediction system](https://www.ecmwf.int/en/forecasts/about-our-forecasts/sub-seasonal-range-forecasts). This model is, as mentioned in the paper, skilful for the rainy season in Malawi, and it has also been validated for the two main rainy seasons across large areas in East Africa (October–December and March–May). In the first version of the code, it is calibrated against ERA5 reanalysis data (the model has biases that should be corrected). I will soon add code for calibrating against satellite-derived data like IMERG.

The calibration is done as follows:
- For a given reference data (initial forecast date), a matching hindcast is searched for. This could be produced within the last week, or it could be produced for the same day of year (+/– 7 days) last year. This allows pre-downloading of hindcasts from the year before, but it is not ideal as the model version used last year was not the same as the current version.
- When a matching hindcast has been found, the code pools all the hindcast data that it can find for each valid time in the forecast. Similarly, it pools observational data (e.g., ERA5) for the same days, but for each day it uses a rolling window of 11 days. This ensures that more data are used for calibration, and it matches the hindcast ensemble size (11).
- Now that the code has matching model/obs data for each day, it can perform a simple quantile mapping. This is done separately for each dry day threshold. The default here is 1 mm, meaning that if there were less than 1 mm of rain on a particular day, it counts as a dry day. First, the code determines the percentile score corresponding to the threshold. Let's say this is 32, which means that 32 percent of the data has less than 1 mm, and similarly 68 percent of the data has more than 1 mm. This is done for each day and each grid point separately. Then the 32th percentile in the model data is computed. If this is, say, 1.3 mm, it means that the model has a positive bias. In other words, the percentile score for 1 mm is lower than 32; the model has too many days with drizzle or light rain. The crucial point here is that we then use the model's threshold (1.3 mm) to define a dry day instead of the original threshold (1 mm).

Having identified proper model-specific thresholds for each day of the year and each grid point, the code now goes through each day and marks it as either dry or non-dry, again separately for each grid point. This is done for all the days in the forecast range: days 1 to 46 or however many lead time days you have available.

Now that the code has a Boolean variable for each grid point and lead time, it can scan through the forecast data and check whether there are any dry spells during the next, e.g., 2 weeks, 3 weeks, or 4 weeks. This can be configured, of course, along with the length of the dry spells.

The scanning of the data is done for each ensemble member, and a probability of dry spells can be estimated. This is done by counting the number of ensemble members that do show a dry spell within the specified time horizon, and then dividing by the total number of ensemble members (101).

The maps presented at the [demo page](https://ns9873k.web.sigma2.no/africa/fcst/index.html) show these probabilities. The first mep shows the model probabilities, the second map the probability according to the climatology (ERA5), and the last one shows the difference between the two.

### Running the script

The Python script is called create_forecast_maps.py. It depends on quite a few Python modules, such as numpy, matplotlib, etc. If these are installed on your system, you shoould be able to run the script. Of course you also need the forecast data, the hindcast data, and the climatological reference (e.g., ERA5).

## Status
Initial working version:
- ECMWF native resolution (0.4 deg) forecast reader (only the version that is pushed from ECMWF for subscribers)
- Matching and calibration using hindcasts
- Bias correction with respect to ERA5 (on the same grid)
- Maps are created based on the forecasts

## Things to do
- Make code more flexible with respect to grid so the observational data don't have to be on the same grid as the forecast data; Erik has code for this that could be included
- Include support for different forecast formats, especially those downloaded in GRIB format from the ECMWF MARS archive
- Tweak the code so it could also be used for wet spells or extreme rainfall
- Include code for making different kinds of maps, and/or tables

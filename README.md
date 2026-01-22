# s2s-py-tools (working title)

Procedural Python code for reading ECMWF S2S forecast data and generating forecast products.

## Status
Initial working version:
- ECMWF native resolution (0.4 deg) forecast reader (cf/pf)
- Accumulation of daily precipitation in data
- Grid loading
- Hindcast date matching
- Bias correction with respect to ERA5 (on the same grid)
- Maps are created based on the forecasts

## Things to do
- Make code more flexible with respect to grid so the observational data don't have to be on the same grid as the forecast data; Erik has code for this that could be included
- Tweak the code so it could also be used for wet spells or extreme rainfall
- Include code for making different kinds of maps, and/or tables

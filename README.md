# FireRadiativePowerEstimater
This project predicts the Fire Radiative Power uusing LR without using any ML library.

Forest fires have become an increasingly common occurrence across the world. Most recently, raging forest fires in Turkey caused huge devastation. In this assignment, you will work with a dataset with fourteen different attributes describing fires in Australia. Given these features, the task would be to predict a real valued frp that refers to "Fire Radiative Power".

1. latitude - Center of 1km re pixel but not necessarily the actual location of the fire  as
one or more fires can be
1. longitude - Center of 1km re pixel but not necessarily the actual location of the fire
as one or more fires can be
1. brightness - Channel 21/22 brightness temperature of the re pixel measured in
Kelvin
1. scan - The algorithm produces 1km re pixels but MODIS pixels get bigger toward
the edge of scan. Sca
1. track - The algorithm produces 1km re pixels but MODIS pixels get bigger toward
the edge of scan. Sca
1. acq_date - Date of MODIS acquisition
1. acq_time - Time of acquisition/overpass of the satellite (in UTC)
1. satellite - A = Aqua and T = Terra
1. instrument - Constant value for MODIS
1. confidence -This value is based on a collection of intermediate algorithm quantities
used in the detection process
1. version - Version identi es the collection (e.g. MODIS Collection 6) and source of
data processing: Nea
1. bright_t31 - Channel 31 brightness temperature of the re pixel measured in Kelvin
1. daynight - D = Daytime, N = Nighttim
1. frp - (Target Variable) Fire Radiative Power: Depicts the pixel-integrated re
radiative power in MW (megawatts)

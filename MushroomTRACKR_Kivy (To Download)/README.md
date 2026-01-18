# MUSHROOMTRACKR AI

## Description
MushroomTrackr AI is a **RandomForest Classifier** model where users enter in a country and province of choice, and the model will generate the Top 10 Basidiomycota (aka mushroom) genera associated 
with the province with corresponding accuracy scores and an overarching one. In training, the model **uses biodiversity data AND temperature data** to determine what mushrooms are associated with the location. Dataset includes a mix of GBIF.org and 
temperature data from "ERA5 hourly data on single levels from 1940 to present" at (https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download). 
**If you cannot find the temperature data in this file (MushroomTRACKR_Kivy (To Download), look for a file called temeprature_2m.csv.zip**

## Requirements
This app runs on MacOs and uses purely Python programming

### Python libraries-
kivy  
pandas  
numpy  
os  
threading  
rapidfuzz  
sklearn  

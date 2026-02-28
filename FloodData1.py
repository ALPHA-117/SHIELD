import ee
import time

# Initialize EE
ee.Authenticate()  # Only needed once per session
ee.Initialize(project="shield-460120")

# Define your zone: Barpeta district approx bounding box
barpeta_bbox = ee.Geometry.BBox(90.7, 26.2, 91.2, 26.6)

# Or use centroid point for sampling
barpeta_point = barpeta_bbox.centroid(maxError=100)

# --- 1. Rainfall Dataset (GPM IMERG Final Run)
rainfall_dataset = (ee.ImageCollection('NASA/GPM_L3/IMERG_V06')
                    .filterBounds(barpeta_bbox)
                    .filterDate('2023-04-15', '2023-06-05')
                    .select('precipitationCal'))

# Function to get daily sum rainfall per point
def daily_rainfall_feature(img):
    date = img.date().format('YYYY-MM-dd')
    # Sum rainfall for the day (IMERG data is 30 min, sum over 48 images)
    daily_img = (rainfall_dataset
                 .filterDate(img.date(), img.date().advance(1, 'day'))
                 .sum())
    
    rainfall_value = daily_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=barpeta_point,
        scale=1000
    ).get('precipitationCal')

    return ee.Feature(None, {
        'date': date,
        'rainfall_mm': rainfall_value,
        'longitude': barpeta_point.coordinates().get(0),
        'latitude': barpeta_point.coordinates().get(1)
    })

# Get unique dates in the period
dates = ee.List(rainfall_dataset.aggregate_array('system:time_start')) \
    .map(lambda t: ee.Date(t).format('YYYY-MM-dd')) \
    .distinct()

# Map to get daily rainfall features
daily_rainfall_features = dates.map(lambda d: daily_rainfall_feature(ee.Image().set('system:time_start', ee.Date(d).millis())))

rainfall_fc = ee.FeatureCollection(daily_rainfall_features)

# --- 2. Elevation Dataset (SRTM)
elevation_img = ee.Image('USGS/SRTMGL1_003')

# Get elevation value for the point
elevation_value = elevation_img.reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=barpeta_point,
    scale=30
).get('elevation')

# --- 3. Soil Texture Dataset (SoilGrids USDA Texture Class)
soil_img = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02")

# Soil type value for the point
soil_value = soil_img.reduceRegion(
    reducer=ee.Reducer.mode(),
    geometry=barpeta_point,
    scale=250
).get('b0')  # 'b0' band contains soil texture class

# --- Combine elevation and soil to each feature in rainfall_fc

def add_static_features(feature):
    return feature.set({
        'elevation': elevation_value,
        'soil_texture': soil_value
    })

final_fc = rainfall_fc.map(add_static_features)

# --- Export combined feature collection to CSV on Google Drive
task = ee.batch.Export.table.toDrive(
    collection=final_fc,
    description='Barpeta_Rainfall_Elevation_Soil_2019_2024',
    folder='GEE_Exports',
    fileNamePrefix='barpeta_features',
    fileFormat='CSV',
    selectors=['date', 'rainfall_mm', 'longitude', 'latitude', 'elevation', 'soil_texture'] 
)

task.start()

print('Export started...')
while task.status()['state'] in ['READY', 'RUNNING']:
    print('Status:', task.status())
    time.sleep(30)

print('Export finished:', task.status())

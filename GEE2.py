import ee
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from datetime import datetime

# Initialize Earth Engine
try:
    ee.Initialize()
except:
    ee.Authenticate()
    ee.Initialize(project="shield-460120")

class FloodDataExporter:
    def __init__(self, root):
        self.root = root
        self.setup_ui()
        self.task = None
        self.river_data = None
        self.checking = False

    def setup_ui(self):
        self.root.title("Flood Risk Data Exporter")
        self.root.geometry("550x550")
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Input Frame
        input_frame = ttk.LabelFrame(main_frame, text="Area Parameters", padding=10)
        input_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Grid configuration
        input_frame.columnconfigure(1, weight=1)
        
        # Coordinates
        ttk.Label(input_frame, text="Min Longitude:").grid(row=0, column=0, sticky='e', padx=5, pady=5)
        self.min_lon_entry = ttk.Entry(input_frame)
        self.min_lon_entry.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        self.min_lon_entry.insert(0, "90.7")

        ttk.Label(input_frame, text="Min Latitude:").grid(row=1, column=0, sticky='e', padx=5, pady=5)
        self.min_lat_entry = ttk.Entry(input_frame)
        self.min_lat_entry.grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        self.min_lat_entry.insert(0, "26.2")

        ttk.Label(input_frame, text="Max Longitude:").grid(row=2, column=0, sticky='e', padx=5, pady=5)
        self.max_lon_entry = ttk.Entry(input_frame)
        self.max_lon_entry.grid(row=2, column=1, sticky='ew', padx=5, pady=5)
        self.max_lon_entry.insert(0, "91.2")

        ttk.Label(input_frame, text="Max Latitude:").grid(row=3, column=0, sticky='e', padx=5, pady=5)
        self.max_lat_entry = ttk.Entry(input_frame)
        self.max_lat_entry.grid(row=3, column=1, sticky='ew', padx=5, pady=5)
        self.max_lat_entry.insert(0, "26.6")

        # Date Frame
        date_frame = ttk.LabelFrame(main_frame, text="Date Range", padding=10)
        date_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        date_frame.columnconfigure(1, weight=1)

        ttk.Label(date_frame, text="Start Date (YYYY-MM-DD):").grid(row=0, column=0, sticky='e', padx=5, pady=5)
        self.start_date_entry = ttk.Entry(date_frame)
        self.start_date_entry.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        self.start_date_entry.insert(0, "2023-05-01")

        ttk.Label(date_frame, text="End Date (YYYY-MM-DD):").grid(row=1, column=0, sticky='e', padx=5, pady=5)
        self.end_date_entry = ttk.Entry(date_frame)
        self.end_date_entry.grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        self.end_date_entry.insert(0, "2023-06-20")

        # Output Frame
        output_frame = ttk.LabelFrame(main_frame, text="Output", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        output_frame.columnconfigure(1, weight=1)

        ttk.Label(output_frame, text="Output Filename:").grid(row=0, column=0, sticky='e', padx=5, pady=5)
        self.output_file_entry = ttk.Entry(output_frame)
        self.output_file_entry.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        self.output_file_entry.insert(0, "barpeta_flood_data")

        # Button Frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Configure button styles
        style = ttk.Style()
        style.configure('TButton', padding=6, font=('Arial', 10))
        
        self.export_button = ttk.Button(
            button_frame, 
            text="Export Data", 
            command=self.start_export,
            style='TButton'
        )
        self.export_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.close_button = ttk.Button(
            button_frame, 
            text="Close", 
            command=self.root.destroy,
            style='TButton'
        )
        self.close_button.pack(side=tk.RIGHT, padx=5, fill=tk.X, expand=True)

        # Status Bar
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to export data")
        status_label = ttk.Label(
            status_frame, 
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=5
        )
        status_label.pack(fill=tk.X)

        self.progress = ttk.Progressbar(
            main_frame, 
            orient=tk.HORIZONTAL, 
            length=400, 
            mode='determinate'
        )
        self.progress.pack(fill=tk.X, pady=5)

    def get_river_data(self, geometry):
        """Extract river information with proper Earth Engine object handling"""
        try:
            water = ee.Image('JRC/GSW1_3/GlobalSurfaceWater')
            
            # Create proper single-band source for distance calculation
            water_mask = water.select('occurrence').gt(0)
            source = water_mask.Not().mask(1)
            
            # Calculate distance to water
            distance = ee.Image().cumulativeCost(
                source=source,
                maxDistance=10000  # 10km maximum
            ).rename('distance_to_water')
            
            # Get water features
            occurrence = water.select('occurrence')
            seasonality = water.select('seasonality')
            
            # Combine results
            result = ee.Image.cat([occurrence, seasonality, distance]).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=30,
                maxPixels=1e9
            ).getInfo()
            
            return {
                'water_occurrence': result.get('occurrence', 0),
                'water_seasonality': result.get('seasonality', 0),
                'distance_to_water': result.get('distance_to_water', 10000)
            }
            
        except Exception as e:
            print(f"River data error: {str(e)}")
            return {
                'water_occurrence': 0,
                'water_seasonality': 0,
                'distance_to_water': 10000
            }

    def get_rainfall_data(self, geometry, start_date, end_date):
        """Get daily rainfall data with proper feature creation"""
        collection = ee.ImageCollection('NASA/GPM_L3/IMERG_V06')\
            .filterBounds(geometry)\
            .filterDate(start_date, end_date)\
            .select('precipitationCal')
        
        # Get unique dates
        dates = ee.List(collection.aggregate_array('system:time_start'))\
            .map(lambda t: ee.Date(t).format('YYYY-MM-dd'))\
            .distinct()
        
        # Create features for each date
        def create_feature(date):
            date_obj = ee.Date(date)
            daily_sum = collection.filterDate(date_obj, date_obj.advance(1, 'day')).sum()
            
            rainfall = daily_sum.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=1000
            ).get('precipitationCal')
            
            return ee.Feature(None, {
                'date': date,
                'rainfall_mm': rainfall
            })
        
        return dates.map(create_feature)

    def get_elevation(self, point):
        """Get elevation data"""
        elevation = ee.Image('USGS/SRTMGL1_003').select('elevation')
        return elevation.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=30
        ).get('elevation')

    def get_soil_texture(self, point):
        """Get soil texture data"""
        soil = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").select('b0')
        return soil.reduceRegion(
            reducer=ee.Reducer.mode(),
            geometry=point,
            scale=250
        ).get('b0')

    def start_export(self):
        """Validate inputs and start export thread"""
        try:
            # Validate coordinates
            min_lon = float(self.min_lon_entry.get())
            min_lat = float(self.min_lat_entry.get())
            max_lon = float(self.max_lon_entry.get())
            max_lat = float(self.max_lat_entry.get())
            
            # Validate dates
            start_date = self.start_date_entry.get()
            end_date = self.end_date_entry.get()
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
            
            # Validate filename
            output_file = self.output_file_entry.get().strip()
            if not output_file:
                raise ValueError("Output filename cannot be empty")

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {str(e)}")
            return

        self.export_button.config(state=tk.DISABLED)
        self.status_var.set("Starting export...")
        self.progress["value"] = 0

        # Start export in separate thread
        export_thread = threading.Thread(
            target=self.run_export,
            args=(min_lon, min_lat, max_lon, max_lat, start_date, end_date, output_file),
            daemon=True
        )
        export_thread.start()

    def run_export(self, min_lon, min_lat, max_lon, max_lat, start_date, end_date, output_file):
        """Main export process running in background thread"""
        try:
            # Create bounding box
            bbox = ee.Geometry.BBox(min_lon, min_lat, max_lon, max_lat)
            centroid = bbox.centroid(maxError=100)

            # Step 1: Get river data
            self.status_var.set("Getting river data...")
            self.root.update()
            self.river_data = self.get_river_data(bbox)
            
            # Step 2: Get rainfall data
            self.status_var.set("Getting rainfall data...")
            self.root.update()
            rainfall_data = self.get_rainfall_data(bbox, start_date, end_date)
            
            # Step 3: Get elevation and soil data
            self.status_var.set("Getting terrain data...")
            self.root.update()
            elevation = self.get_elevation(centroid)
            soil_texture = self.get_soil_texture(centroid)
            
            # Step 4: Prepare export
            self.status_var.set("Preparing export...")
            self.root.update()
            task = self.prepare_export(
                rainfall_data, 
                elevation, 
                soil_texture, 
                self.river_data, 
                output_file
            )
            
            # Monitor export progress
            self.task = task
            self.checking = True
            self.monitor_export(task)
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.export_button.config(state=tk.NORMAL)
            messagebox.showerror("Export Error", str(e))

    def prepare_export(self, rainfall_data, elevation, soil_texture, river_data, output_file):
        """Prepare and start the Earth Engine export task"""
        # Get rainfall data as list
        rainfall_list = rainfall_data.getInfo()
        
        # Prepare features list
        features = []
        for feature_info in rainfall_list:
            feature = ee.Feature(None, {
                'date': feature_info['properties']['date'],
                'rainfall_mm': feature_info['properties']['rainfall_mm'],
                'elevation': elevation,
                'soil_texture': soil_texture,
                'water_occurrence': river_data['water_occurrence'],
                'water_seasonality': river_data['water_seasonality'],
                'distance_to_water': river_data['distance_to_water']
            })
            features.append(feature)
        
        # Create feature collection
        fc = ee.FeatureCollection(features)
        
        # Start export task
        task = ee.batch.Export.table.toDrive(
            collection=fc,
            description='Flood_Risk_Export',
            folder='GEE_Exports',
            fileNamePrefix=output_file,
            fileFormat='CSV',
            selectors=['date', 'rainfall_mm', 'elevation', 'soil_texture',
                      'water_occurrence', 'water_seasonality', 'distance_to_water']
        )
        
        task.start()
        return task

    def monitor_export(self, task):
        """Monitor export task progress"""
        if not self.checking:
            return
            
        try:
            status = task.status()
            state = status['state']
            
            if state == 'COMPLETED':
                self.status_var.set("Export completed successfully!")
                self.progress["value"] = 100
                self.export_button.config(state=tk.NORMAL)
                self.checking = False
                messagebox.showinfo("Success", "Data exported successfully to your Google Drive!")
                
            elif state == 'FAILED':
                self.status_var.set(f"Export failed: {status.get('error_message', 'Unknown error')}")
                self.progress["value"] = 0
                self.export_button.config(state=tk.NORMAL)
                self.checking = False
                messagebox.showerror("Export Failed", status.get('error_message', 'Unknown error'))
                
            else:
                # Still running
                progress = int(status.get('progress', 0) * 100)
                self.status_var.set(f"Exporting... {progress}% complete")
                self.progress["value"] = progress
                self.root.after(5000, lambda: self.monitor_export(task))
                
        except Exception as e:
            self.status_var.set(f"Status check error: {str(e)}")
            self.export_button.config(state=tk.NORMAL)
            self.checking = False

def main():
    root = tk.Tk()
    app = FloodDataExporter(root)
    root.mainloop()

if __name__ == "__main__":
    main()

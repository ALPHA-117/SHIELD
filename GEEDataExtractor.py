import ee
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox

# Your original Earth Engine code wrapped as a function (no logic change)
def run_earthengine_export(min_lon, min_lat, max_lon, max_lat, start_date, end_date, output_file_name):
    # Service account authentication using credentials JSON
    KEY_FILE = r"shield\shield-488115-ad3bb2e0adfc.json"
    SERVICE_ACCOUNT = "gee-service-account@shield-488115.iam.gserviceaccount.com"
    PROJECT_ID = "shield-488115"

    credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_FILE)
    ee.Initialize(credentials=credentials, project=PROJECT_ID)

    barpeta_bbox = ee.Geometry.BBox(min_lon, min_lat, max_lon, max_lat)
    barpeta_point = barpeta_bbox.centroid(maxError=100)

    rainfall_dataset = (ee.ImageCollection('NASA/GPM_L3/IMERG_V06')
                        .filterBounds(barpeta_bbox)
                        .filterDate(start_date, end_date)
                        .select('precipitationCal'))

    def daily_rainfall_feature(img):
        date = img.date().format('YYYY-MM-dd')
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

    dates = ee.List(rainfall_dataset.aggregate_array('system:time_start')) \
        .map(lambda t: ee.Date(t).format('YYYY-MM-dd')) \
        .distinct()

    daily_rainfall_features = dates.map(lambda d: daily_rainfall_feature(ee.Image().set('system:time_start', ee.Date(d).millis())))

    rainfall_fc = ee.FeatureCollection(daily_rainfall_features)

    elevation_img = ee.Image('USGS/SRTMGL1_003')
    elevation_value = elevation_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=barpeta_point,
        scale=30
    ).get('elevation')

    soil_img = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02")
    soil_value = soil_img.reduceRegion(
        reducer=ee.Reducer.mode(),
        geometry=barpeta_point,
        scale=250
    ).get('b0')

    def add_static_features(feature):
        return feature.set({
            'elevation': elevation_value,
            'soil_texture': soil_value
        })

    final_fc = rainfall_fc.map(add_static_features)

    task = ee.batch.Export.table.toDrive(
        collection=final_fc,
        description='Barpeta_Rainfall_Elevation_Soil_2019_2024',
        folder='GEE_Exports',
        fileNamePrefix=output_file_name,
        fileFormat='CSV',
        selectors=['date', 'rainfall_mm', 'longitude', 'latitude', 'elevation', 'soil_texture']
    )

    task.start()
    return task


class EEExportApp:
    def __init__(self, root):
        self.root = root
        root.title("GEE Export GUI")

        tk.Label(root, text="Min Longitude:").grid(row=0, column=0, sticky='e')
        tk.Label(root, text="Min Latitude:").grid(row=1, column=0, sticky='e')
        tk.Label(root, text="Max Longitude:").grid(row=2, column=0, sticky='e')
        tk.Label(root, text="Max Latitude:").grid(row=3, column=0, sticky='e')
        tk.Label(root, text="Start Date (YYYY-MM-DD):").grid(row=4, column=0, sticky='e')
        tk.Label(root, text="End Date (YYYY-MM-DD):").grid(row=5, column=0, sticky='e')
        tk.Label(root, text="Output File Name:").grid(row=6, column=0, sticky='e')

        self.min_lon_entry = tk.Entry(root)
        self.min_lat_entry = tk.Entry(root)
        self.max_lon_entry = tk.Entry(root)
        self.max_lat_entry = tk.Entry(root)
        self.start_date_entry = tk.Entry(root)
        self.end_date_entry = tk.Entry(root)
        self.output_file_entry = tk.Entry(root)

        self.min_lon_entry.grid(row=0, column=1)
        self.min_lat_entry.grid(row=1, column=1)
        self.max_lon_entry.grid(row=2, column=1)
        self.max_lat_entry.grid(row=3, column=1)
        self.start_date_entry.grid(row=4, column=1)
        self.end_date_entry.grid(row=5, column=1)
        self.output_file_entry.grid(row=6, column=1)

        self.min_lon_entry.insert(0, "90.7")
        self.min_lat_entry.insert(0, "26.2")
        self.max_lon_entry.insert(0, "91.2")
        self.max_lat_entry.insert(0, "26.6")
        self.start_date_entry.insert(0, "2023-04-15")
        self.end_date_entry.insert(0, "2023-06-05")
        self.output_file_entry.insert(0, "barpeta_features")

        self.start_button = tk.Button(root, text="Start Export", command=self.start_export_thread)
        self.start_button.grid(row=7, column=0, columnspan=2, pady=10)

        self.status_var = tk.StringVar()
        self.status_var.set("Idle")
        self.status_label = tk.Label(root, textvariable=self.status_var)
        self.status_label.grid(row=8, column=0, columnspan=2)

        self.progress = ttk.Progressbar(root, length=300, mode='determinate')
        self.progress.grid(row=9, column=0, columnspan=2, pady=5)

        self.task = None
        self.checking = False

    def start_export_thread(self):
        try:
            min_lon = float(self.min_lon_entry.get())
            min_lat = float(self.min_lat_entry.get())
            max_lon = float(self.max_lon_entry.get())
            max_lat = float(self.max_lat_entry.get())
            start_date = self.start_date_entry.get()
            end_date = self.end_date_entry.get()
            output_file_name = self.output_file_entry.get().strip()
            if not output_file_name:
                raise ValueError("Output file name cannot be empty")
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")
            return

        self.start_button.config(state='disabled')
        self.status_var.set("Initializing export...")
        self.progress['value'] = 0

        thread = threading.Thread(target=self.run_export_task,
                                  args=(min_lon, min_lat, max_lon, max_lat, start_date, end_date, output_file_name),
                                  daemon=True)
        thread.start()

    def run_export_task(self, min_lon, min_lat, max_lon, max_lat, start_date, end_date, output_file_name):
        try:
            self.task = run_earthengine_export(min_lon, min_lat, max_lon, max_lat, start_date, end_date, output_file_name)
        except Exception as e:
            self.status_var.set(f"Error starting export: {str(e)}")
            self.start_button.config(state='normal')
            return

        self.checking = True
        while self.checking:
            status = self.task.status()
            state = status['state']
            progress_percent = int(status.get('progress', 0) * 100)
            self.status_var.set(f"Export Status: {state} ({progress_percent}%)")
            self.progress['value'] = progress_percent

            if state == 'COMPLETED':
                self.status_var.set("Export completed successfully!")
                messagebox.showinfo("Export Status", "Export finished successfully.")
                self.start_button.config(state='normal')
                self.checking = False
                break
            elif state == 'FAILED':
                self.status_var.set(f"Export failed: {status.get('error_message', '')}")
                messagebox.showerror("Export Status", f"Export failed: {status.get('error_message', '')}")
                self.start_button.config(state='normal')
                self.checking = False
                break

            time.sleep(5)


def main():
    root = tk.Tk()
    app = EEExportApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

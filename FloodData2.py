import ee
import time
import tkinter as tk
from tkinter import messagebox, ttk

# ---------------- Earth Engine Export Function ---------------- #
def run_export(lat, lon, buffer_deg, start_date, end_date, file_name, progress_label, progress_bar, root):
    try:
        ee.Authenticate()
        ee.Initialize(project="shield-460120")
    except Exception as e:
        messagebox.showerror("Earth Engine Auth Error", str(e))
        return

    region = ee.Geometry.BBox(
        lon - buffer_deg, lat - buffer_deg,
        lon + buffer_deg, lat + buffer_deg
    )

    dataset = ee.ImageCollection('NASA/GPM_L3/IMERG_V06') \
        .filterBounds(region) \
        .filterDate(start_date, end_date) \
        .select('precipitationCal')

    def daily_sum(img):
        date = img.date().format('YYYY-MM-dd')
        daily = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=10000,
            maxPixels=1e13
        )
        return ee.Feature(None, {
            'date': date,
            'rainfall_mm': daily.get('precipitationCal')
        })

    rainfall_features = dataset.map(daily_sum)
    rainfall_fc = ee.FeatureCollection(rainfall_features)

    task = ee.batch.Export.table.toDrive(
        collection=rainfall_fc,
        description=f"{file_name}_export",
        fileNamePrefix=file_name,
        fileFormat='CSV',
        selectors=['date', 'rainfall_mm']
    )
    task.start()

    progress_label.config(text="🛰️ Export started...")

    # Simulate progress tracking (GUI-based since GEE doesn't give exact %)
    percent = 0
    while task.status()['state'] in ['READY', 'RUNNING']:
        progress_bar['value'] = percent
        progress_label.config(text=f"⏳ Exporting... {percent}%")
        root.update()
        time.sleep(10)
        percent = min(percent + 10, 95)

    final_state = task.status()['state']
    progress_bar['value'] = 100
    if final_state == 'COMPLETED':
        progress_label.config(text=f"✅ Done! File: {file_name}.csv")
    else:
        progress_label.config(text=f"❌ Failed: {task.status()['error_message']}")

# ---------------- GUI Function ---------------- #
def get_inputs():
    def on_submit():
        try:
            lat = float(lat_entry.get())
            lon = float(lon_entry.get())
            buffer = float(buffer_entry.get())
            start = start_date.get()
            end = end_date.get()
            filename = output_name.get()
            submit_button.config(state=tk.DISABLED)
            run_export(lat, lon, buffer, start, end, filename, progress_label, progress_bar, root)
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric coordinates and buffer.")

    root = tk.Tk()
    root.title("SHIELD Rainfall Export")

    # Entry fields
    labels = ["Latitude:", "Longitude:", "Bounding Box Radius (deg):", "Start Date (YYYY-MM-DD):",
              "End Date (YYYY-MM-DD):", "Output File Name:"]
    entries = []

    for i, label_text in enumerate(labels):
        tk.Label(root, text=label_text).grid(row=i, column=0, sticky="e")
        entry = tk.Entry(root, width=30)
        entry.grid(row=i, column=1)
        entries.append(entry)

    lat_entry, lon_entry, buffer_entry, start_date, end_date, output_name = entries

    # Progress section
    progress_label = tk.Label(root, text="Status: Waiting for input...")
    progress_label.grid(row=6, column=0, columnspan=2, pady=(10, 2))

    progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
    progress_bar.grid(row=7, column=0, columnspan=2, pady=(0, 10))

    submit_button = tk.Button(root, text="Start Export", command=on_submit)
    submit_button.grid(row=8, column=0, columnspan=2, pady=5)

    root.mainloop()

# ---------------- Run ---------------- #
if __name__ == "__main__":
    get_inputs()

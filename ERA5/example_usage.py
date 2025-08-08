import xarray as xr
import cfgrib

# Load the GRIB file
ds = xr.open_dataset('download.grib', engine='cfgrib')

# Explore the dataset
print(ds)  # Shows variables, dimensions, and metadata
print(ds.variables)  # Lists variables like temperature, pressure, etc.

# Access a specific variable (e.g., 2m temperature)
temp = ds['t2m']  # 't2m' is the GRIB short name for 2m temperature
print(temp)

# Plot the data (example for a single time step)
import matplotlib.pyplot as plt
temp.isel(time=0).plot()  # Plot the first time step
plt.show()
import pvlib
import pandas as pd

# 1. Deine Anlagen-Konfiguration
TILT = 30       # Neigungswinkel (Grad)
AZIMUTH = 180   # Ausrichtung (180 = Süden, 90 = Osten, 270 = Westen)
LAT, LON = 52.52, 13.41  # Beispiel: Berlin
TZ = 'Europe/Berlin'

# 2. Beispiel-Daten (Diese würdest du von Open-Meteo beziehen)
# Angenommen, wir haben Werte für einen sonnigen Mittag
times = pd.date_range('2024-06-21 12:00:00', periods=1, freq='H', tz=TZ)
dni = 800.0  # Direct Normal Irradiance
dhi = 150.0  # Diffuse Horizontal Irradiance
ghi = 550.0  # Global Horizontal Irradiance

# 3. Sonnenstand berechnen (Notwendig für den Einstrahlungswinkel)
solpos = pvlib.solarposition.get_solarposition(times, LAT, LON)

# 4. Präzise Berechnung der Einstrahlung auf die geneigte Ebene (POA)
# Wir nutzen das Perez-Modell (anisotropic), da es DNI und DHI separat verarbeitet
poa_irradiance = pvlib.irradiance.get_total_irradiance(
    surface_tilt=TILT,
    surface_azimuth=AZIMUTH,
    dni=dni,
    ghi=ghi,
    dhi=dhi,
    solar_zenith=solpos['zenith'],
    solar_azimuth=solpos['azimuth'],
    model='perez' # Hier liegt der Qualitätsvorteil gegenüber Open-Meteo
)

# 5. Ergebnisse ausgeben
print(f"Einstrahlung auf der schrägen Fläche (Total): {poa_irradiance['poa_global'].values[0]:.2f} W/m²")


Implement Perez POA Calculation
This plan outlines the creation of src/calc_historic_ghi_perez.py to calculate Plane of Array (POA) irradiance using the Perez model in pvlib.

Proposed Changes
Configuration
[MODIFY] 
settings.example.toml
Add f_poa_perez to [fields] section.
Dependencies
[MODIFY] 
requirements.txt
Add pvlib if missing.
Source Code
[NEW] 
src/calc_historic_ghi_perez.py
Imports: pvlib, pandas, influxdb_client, src.config.settings, src.db.InfluxDBWrapper.
Logic:
Connect to InfluxDB: Use 
InfluxDBWrapper
 to connect.
Fetch Data: Query b_regressor_history and b_regressor_future to get diffuse_radiation and direct_normal_irradiance.
Get Location/Orientation: Read latitude, longitude, tilt, and azimuth from settings['open_meteo'].
Azimuth Conversion: Open-Meteo uses 0=South, -90=East, 90=West. pvlib uses 180=South, 90=East, 270=West.
Formula: pvlib_azimuth = (open_meteo_azimuth + 180) % 360.
Example: 0 (South) -> 180. -90 (East) -> 90. 90 (West) -> 270.
Calculate POA: Use pvlib with model='perez'.
Write to InfluxDB: Write result to f_poa_perez.
Verification Plan
Automated Tests
Run pip install -r requirements.txt.
Run the script python src/calc_historic_ghi_perez.py and check for successful execution.
Verify InfluxDB contains the new field.

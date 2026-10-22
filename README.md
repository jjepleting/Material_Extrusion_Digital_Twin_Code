# Material Extrusion Digital Twin

This repository contains a Python-based digital twin developed for a Bambu Lab A1 Mini material-extrusion 3D printer. The application combines machine-learning predictions of dimensional error, live printer telemetry received through MQTT, and layer-by-layer toolpath visualization.

The project was developed by Joyce Jepleting Biwott as part of graduate research at Grand Valley State University.

## Main features

- Predicts length, width, and height errors from six printing parameters.
- Displays the predicted dimensions and a one-standard-deviation uncertainty estimate.
- Receives printer state, layer number, print progress, nozzle temperature, bed temperature, and cooling-fan data through MQTT.
- Displays live nozzle- and bed-temperature graphs.
- Visualizes completed and active toolpath layers in three dimensions.
- Allows the operator to select the `.gcode.3mf` file used for each validation print.
- Saves printer telemetry and post-print validation measurements to CSV files.

## Repository files

- `digital_twin_dynamic_gcode_v5.py` — main digital-twin application.
- `gcode_parser.py` — extracts process settings and layer toolpaths from Bambu Studio `.gcode.3mf` files.
- `Final_Parameter_Only_Random_Forest.joblib` — trained dimensional-error prediction model.
- `Walltest.gcode.3mf` — default startup toolpath.
- `requirements.txt` — required Python packages.

## Software requirements

- Python 3.12
- Bambu Studio
- A Bambu Lab printer configured for LAN-only MQTT access

Install the Python packages with:

```powershell
python -m pip install -r requirements.txt
```

## Running the application

Open PowerShell in the repository folder and run:

```powershell
python digital_twin_dynamic_gcode_v5.py
```

If a virtual environment is stored in the parent project folder, run the application with that environment's Python executable.

## Validation workflow

1. Prepare and slice the validation specimen in Bambu Studio.
2. Save the sliced `.gcode.3mf` file.
3. Start the digital-twin application.
4. Select **Select G-code/3MF** and load the exact file that will be sent to the printer.
5. Confirm the nominal CAD dimensions and the six commanded printing parameters.
6. Enter the printer IP address, serial number, and LAN access code at runtime.
7. Connect the application to the printer and begin printing.
8. After printing, enter the averaged measured dimensions and save the validation result.

## Privacy and data

Printer credentials are entered at runtime and must not be committed to this repository. Generated telemetry logs are excluded through `.gitignore`.

## Research status

This repository contains research software. Results should be interpreted within the printer, material, specimen geometry, and process-parameter ranges used to train and validate the model.

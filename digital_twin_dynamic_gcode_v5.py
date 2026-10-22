"""Material-extrusion digital twin for a Bambu Lab printer.

Main functions
--------------
1. Load a trained Random Forest model and predict length, width, and height
   errors from six commanded printing parameters.
2. Load the exact Bambu Studio .gcode.3mf file used for a validation print.
3. Receive live printer telemetry through MQTT.
4. Synchronize the toolpath display with the physical printer's reported layer.
5. Plot live nozzle and bed temperatures and record one telemetry CSV per print.
6. Save manually entered post-print measurements to one validation-results CSV.

Requirements (Python 3.12):
    pip install PySide6 pyqtgraph PyOpenGL paho-mqtt numpy pandas joblib scikit-learn==1.6.1

Keep this file beside:
    Final_Parameter_Only_Random_Forest.joblib
    gcode_parser.py

The program can start with Walltest.gcode.3mf when it is available, and the
operator can select the .gcode.3mf file for each new validation print from the
interface.

NOTE: walltest.gcode.3mf is the part that was used for training(20*1*10)-all in mm
"""

from __future__ import annotations

# Python standard-library modules.
import csv
import json
import math
import ssl
import sys
import time
from datetime import datetime
from pathlib import Path

# Third-party scientific, communication, and visualization packages.
import joblib
import numpy as np
import paho.mqtt.client as mqtt
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

# Local parser used to extract layers, extrusion segments, and settings from a
# Bambu Studio .gcode.3mf archive.
from gcode_parser import parse_bambu_3mf


# All required project files are resolved relative to this script. This avoids
# dependence on the directory from which PowerShell launches the application.
PROGRAM_FOLDER = Path(__file__).resolve().parent
MODEL_PATH = PROGRAM_FOLDER / "Final_Parameter_Only_Random_Forest.joblib"
DEFAULT_TOOLPATH_PATH = PROGRAM_FOLDER / "Walltest.gcode.3mf"
LOG_FOLDER = PROGRAM_FOLDER / "session_logs"

# The order must match the feature order used when the Random Forest was fitted.
FEATURE_NAMES = [
    "Layer_Height_mm",
    "Line_Width_mm",
    "Print_Speed_mm_s",
    "Flow_Ratio",
    "Nozzle_Temp_C",
    "Cooling_Fan_Speed_pct",
]


def create_box_edges(length: float, width: float, height: float) -> np.ndarray:
    """Return independent line segments forming a centered rectangular box."""
    x0, x1 = -length / 2.0, length / 2.0
    y0, y1 = -width / 2.0, width / 2.0
    z0, z1 = 0.0, max(0.001, height)
    corners = np.array(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ],
        dtype=float,
    )
    pairs = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return np.asarray([[corners[a], corners[b]] for a, b in pairs]).reshape(-1, 3)


def segments_to_lines(segments: np.ndarray) -> np.ndarray:
    """Flatten Nx2x3 independent segments for GL_LINES rendering."""
    if segments is None or not segments.size:
        return np.empty((0, 3), dtype=float)
    return np.asarray(segments, dtype=float).reshape(-1, 3)


BOX_FACES = np.asarray(
    [
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
    ],
    dtype=np.uint32,
)


def box_vertices(center, size) -> np.ndarray:
    cx, cy, cz = center
    sx, sy, sz = size
    x0, x1 = cx - sx / 2.0, cx + sx / 2.0
    y0, y1 = cy - sy / 2.0, cy + sy / 2.0
    z0, z1 = cz - sz / 2.0, cz + sz / 2.0
    return np.asarray(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ],
        dtype=float,
    )


def create_box_item(center, size, color) -> gl.GLMeshItem:
    return gl.GLMeshItem(
        vertexes=box_vertices(center, size),
        faces=BOX_FACES,
        color=color,
        smooth=False,
        drawEdges=True,
        edgeColor=(0.05, 0.07, 0.09, 0.9),
        shader="shaded",
    )


class MqttBridge(QObject):
    """Safely transfer MQTT callbacks from a worker thread to the Qt GUI."""

    connected = Signal(bool, str)
    telemetry = Signal(dict)


class BambuMqttClient:
    """Manage the encrypted LAN-mode MQTT connection to the Bambu printer."""

    def __init__(self, bridge: MqttBridge):
        self.bridge = bridge
        self.client = None
        self.serial = ""
        self._user_disconnect = False

    def connect(self, ip: str, serial_number: str, access_code: str) -> None:
        """Open the MQTT connection using credentials entered in the GUI."""
        self.disconnect()
        self.serial = serial_number.strip()
        self._user_disconnect = False

        callback_version = getattr(mqtt, "CallbackAPIVersion", None)
        if callback_version is not None:
            self.client = mqtt.Client(callback_version.VERSION2)
        else:
            self.client = mqtt.Client()

        self.client.username_pw_set("bblp", access_code)
        self.client.tls_set(cert_reqs=ssl.CERT_NONE)
        self.client.tls_insecure_set(True)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.connect_async(ip.strip(), 8883, keepalive=60)
        self.client.loop_start()

    def disconnect(self) -> None:
        if self.client is not None:
            self._user_disconnect = True
            try:
                self.client.disconnect()
                self.client.loop_stop()
            except Exception:
                pass
            self.client = None

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        try:
            success = int(reason_code) == 0
        except (TypeError, ValueError):
            success = reason_code == 0

        if not success:
            self.bridge.connected.emit(False, f"Connection rejected: {reason_code}")
            return

        client.subscribe(f"device/{self.serial}/report")
        self.bridge.connected.emit(True, "Connected to printer")

        request = {
            "pushing": {
                "sequence_id": "0",
                "command": "pushall",
                "version": 1,
                "push_target": 1,
            }
        }
        client.publish(f"device/{self.serial}/request", json.dumps(request))

    def _on_disconnect(self, client, userdata, *args):
        if not self._user_disconnect:
            self.bridge.connected.emit(False, "Printer connection lost")

    def _on_message(self, client, userdata, message):
        """Extract the printer report and forward it to the Qt main thread."""
        try:
            payload = json.loads(message.payload.decode("utf-8", errors="replace"))
            report = payload.get("print", payload)
            if isinstance(report, dict):
                self.bridge.telemetry.emit(report)
        except Exception:
            return


class DigitalTwinWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Material Extrusion Digital Twin")
        self.resize(1360, 900)

        # Fail early with a readable message if a required local dependency is
        # missing. No printer credentials are stored in this program.
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found:\n{MODEL_PATH}")
        if not DEFAULT_TOOLPATH_PATH.exists():
            raise FileNotFoundError(
                "Default toolpath file not found. Keep Walltest.gcode.3mf beside "
                "the program for startup; after startup, use Select toolpath to "
                "load the file for each validation print.\n\n"
                f"Expected location:\n{DEFAULT_TOOLPATH_PATH}"
            )

        # Load the fitted prediction model and an initial toolpath. The initial
        # file only makes startup possible; the operator selects the exact
        # validation toolpath from the interface before each print.
        self.model = joblib.load(MODEL_PATH)
        self.toolpath_path = DEFAULT_TOOLPATH_PATH
        self.toolpath = self.parse_toolpath_file(self.toolpath_path)
        self.total_layers = self.toolpath.total_layers
        # Latest values received from the printer. The GUI and CSV logger read
        # this shared snapshot; MQTT callbacks update only keys that are present.
        self.telemetry_data = {
            "gcode_state": "OFFLINE",
            "mc_percent": 0.0,
            "layer_num": 0,
            "total_layer_num": self.total_layers,
            "nozzle_temper": None,
            "nozzle_target_temper": None,
            "bed_temper": None,
            "bed_target_temper": None,
            "cooling_fan_speed": None,
            "mc_print_line_number": None,
        }
        # State flags prevent stale FINISH/100% reports from a previous job from
        # appearing as a new print immediately after the printer is connected.
        self.is_connected = False
        self.active_job_seen = False
        self.previous_progress = None
        self.last_reported_layer = None
        self.session_log_path = None
        self.last_logged_at = None
        self.telemetry_logging_active = False
        self.finish_row_written = False
        self.print_started_monotonic = None
        # Flatten the layer structure for optional animation while retaining the
        # layer number associated with every extrusion segment.
        self.preview_segments = [
            (layer_number, segment.copy())
            for layer_number, layer in enumerate(self.toolpath.layers, start=1)
            for segment in layer.segments
        ]
        self.preview_index = 0
        self.preview_running = False
        self.preview_started = False
        self.telemetry_sample = 0
        self.telemetry_history_limit = 240
        # Keep a rolling history for the live temperature plots. The x-values
        # represent elapsed seconds from the beginning of the current print.
        self.telemetry_history = {
            "sample": [], "nozzle": [], "nozzle_target": [],
            "bed": [], "bed_target": [],
        }
        self.fan_angle = 0

        self.bridge = MqttBridge()
        self.bridge.connected.connect(self.on_connection_changed)
        self.bridge.telemetry.connect(self.on_telemetry)
        self.mqtt_client = BambuMqttClient(self.bridge)

        self.parameter_boxes = {}
        self.nominal_boxes = {}
        self.build_interface()
        self.build_3d_scene()
        self.load_toolpath_settings()
        self.update_prediction()
        self.update_telemetry_labels()
        self.update_3d_part()

        self.log_timer = QTimer(self)
        self.log_timer.timeout.connect(self.log_snapshot)
        self.log_timer.start(1000)

        self.dashboard_timer = QTimer(self)
        self.dashboard_timer.timeout.connect(self.update_live_dashboard)
        self.dashboard_timer.start(1000)

    def build_interface(self) -> None:
        """Create the controls, 3D view, telemetry cards, and live plots."""
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        title_row = QHBoxLayout()
        title = QLabel("Material Extrusion Digital Twin")
        title.setStyleSheet("font-size: 22px; font-weight: 700;")
        self.connection_status = QLabel("OFFLINE")
        self.connection_status.setStyleSheet("font-weight: 700; color: #d7191c;")
        title_row.addWidget(title)
        title_row.addStretch()
        title_row.addWidget(self.connection_status)
        outer.addLayout(title_row)

        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter, 1)

        left_container = QWidget()
        left_container.setMinimumWidth(425)
        left_container.setMaximumWidth(470)
        left = QVBoxLayout(left_container)
        splitter.addWidget(left_container)

        # Printer credentials are entered at runtime and are never written to a
        # source file or included in a telemetry/validation CSV.
        connection_group = QGroupBox("Printer connection")
        connection_form = QFormLayout(connection_group)
        self.ip_input = QLineEdit()
        self.ip_input.setPlaceholderText("Example: 192.168.1.100")
        self.serial_input = QLineEdit()
        self.serial_input.setPlaceholderText("Printer serial number")
        self.access_input = QLineEdit()
        self.access_input.setPlaceholderText("LAN access code")
        self.access_input.setEchoMode(QLineEdit.Password)
        self.connect_button = QPushButton("Connect to printer")
        self.connect_button.clicked.connect(self.connect_to_printer)
        connection_form.addRow("Printer IP:", self.ip_input)
        connection_form.addRow("Serial number:", self.serial_input)
        connection_form.addRow("LAN access code:", self.access_input)
        connection_form.addRow(self.connect_button)
        left.addWidget(connection_group)

        # The selected toolpath must match the file sent to the printer. This is
        # what prevents the viewer from remaining tied to Walltest.gcode.3mf.
        toolpath_group = QGroupBox("Print toolpath")
        toolpath_layout = QHBoxLayout(toolpath_group)
        self.toolpath_name = QLabel(self.toolpath_path.name)
        self.toolpath_name.setToolTip(str(self.toolpath_path))
        self.toolpath_name.setWordWrap(True)
        self.select_toolpath_button = QPushButton("Select G-code/3MF")
        self.select_toolpath_button.clicked.connect(self.select_toolpath_file)
        toolpath_layout.addWidget(self.toolpath_name, 1)
        toolpath_layout.addWidget(self.select_toolpath_button)
        left.addWidget(toolpath_group)

        # Six commanded inputs used by the trained machine-learning model.
        process_group = QGroupBox("Commanded process parameters")
        process_form = QFormLayout(process_group)
        parameter_specs = [
            ("Layer_Height_mm", "Layer height (mm):", 0.05, 1.00, 0.01, 2),
            ("Line_Width_mm", "Line width (mm):", 0.10, 2.00, 0.01, 2),
            ("Print_Speed_mm_s", "Print speed (mm/s):", 1.0, 500.0, 1.0, 0),
            ("Flow_Ratio", "Flow ratio:", 0.50, 1.50, 0.01, 2),
            ("Nozzle_Temp_C", "Nozzle temperature (°C):", 150.0, 300.0, 5.0, 0),
            ("Cooling_Fan_Speed_pct", "Cooling-fan command (%):", 0.0, 100.0, 10.0, 0),
        ]
        for key, label, minimum, maximum, step, decimals in parameter_specs:
            box = QDoubleSpinBox()
            box.setRange(minimum, maximum)
            box.setSingleStep(step)
            box.setDecimals(decimals)
            box.valueChanged.connect(self.update_prediction)
            self.parameter_boxes[key] = box
            process_form.addRow(label, box)
        left.addWidget(process_group)

        # Geometry dimensions are independent of the six model inputs and are
        # added to the predicted dimensional errors to obtain final dimensions.
        nominal_group = QGroupBox("Nominal CAD dimensions")
        nominal_form = QFormLayout(nominal_group)
        for key, label, value in [
            ("Length", "Length (mm):", 20.0),
            ("Width", "Width (mm):", 1.0),
            ("Height", "Height (mm):", 10.0),
        ]:
            box = QDoubleSpinBox()
            box.setRange(0.01, 500.0)
            box.setDecimals(3)
            box.setValue(value)
            box.valueChanged.connect(self.update_prediction)
            self.nominal_boxes[key] = box
            nominal_form.addRow(label, box)
        left.addWidget(nominal_group)

        # Display the predicted error, predicted dimension, and the spread of
        # predictions among the individual Random Forest trees.
        prediction_group = QGroupBox("Machine-learning prediction")
        prediction_layout = QGridLayout(prediction_group)
        prediction_layout.addWidget(QLabel("Dimension"), 0, 0)
        prediction_layout.addWidget(QLabel("Error (mm)"), 0, 1)
        prediction_layout.addWidget(QLabel("Predicted (mm)"), 0, 2)
        tolerance_header = QLabel("±1 SD (mm)")
        tolerance_header.setToolTip("Standard deviation of predictions from the 500 Random Forest trees")
        prediction_layout.addWidget(tolerance_header, 0, 3)
        self.prediction_labels = {}
        for row, dimension in enumerate(["Length", "Width", "Height"], start=1):
            error_label = QLabel("—")
            dimension_label = QLabel("—")
            tolerance_label = QLabel("—")
            self.prediction_labels[dimension] = (
                error_label, dimension_label, tolerance_label
            )
            prediction_layout.addWidget(QLabel(dimension), row, 0)
            prediction_layout.addWidget(error_label, row, 1)
            prediction_layout.addWidget(dimension_label, row, 2)
            prediction_layout.addWidget(tolerance_label, row, 3)
        left.addWidget(prediction_group)

        # Measurements remain blank until the user enters averaged micrometer
        # readings after the print has been completed.
        validation_group = QGroupBox("Post-print validation")
        validation_layout = QGridLayout(validation_group)
        validation_layout.addWidget(QLabel("Dimension"), 0, 0)
        validation_layout.addWidget(QLabel("Measured (mm)"), 0, 1)
        validation_layout.addWidget(QLabel("Measured − predicted"), 0, 2)
        self.actual_boxes = {}
        self.validation_labels = {}
        for row, dimension in enumerate(["Length", "Width", "Height"], start=1):
            box = QDoubleSpinBox()
            box.setRange(0.0, 500.0)
            box.setDecimals(4)
            box.setSingleStep(0.001)
            box.setSpecialValueText("Not entered")
            box.setValue(0.0)
            box.valueChanged.connect(self.update_validation_display)
            result = QLabel("—")
            self.actual_boxes[dimension] = box
            self.validation_labels[dimension] = result
            validation_layout.addWidget(QLabel(dimension), row, 0)
            validation_layout.addWidget(box, row, 1)
            validation_layout.addWidget(result, row, 2)
        self.save_validation_button = QPushButton("Save validation result")
        self.save_validation_button.clicked.connect(self.save_validation_result)
        validation_layout.addWidget(self.save_validation_button, 4, 0, 1, 3)
        left.addWidget(validation_group)

        telemetry_group = QGroupBox("Live printer telemetry")
        telemetry_form = QFormLayout(telemetry_group)
        self.state_value = QLabel("—")
        self.layer_value = QLabel("—")
        self.nozzle_value = QLabel("—")
        self.bed_value = QLabel("—")
        self.fan_value = QLabel("—")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        telemetry_form.addRow("Printer state:", self.state_value)
        telemetry_form.addRow("Layer:", self.layer_value)
        telemetry_form.addRow("Print progress:", self.progress_bar)
        left.addWidget(telemetry_group)
        left.addStretch()

        # One coordinated visual: the part/toolpath remains central while live
        # telemetry is always visible in a compact panel on its right.
        visual_container = QWidget()
        visual_layout = QHBoxLayout(visual_container)
        visual_layout.setContentsMargins(0, 0, 0, 0)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor(QColor(8, 10, 14))
        self.view.setCameraPosition(distance=38, elevation=22, azimuth=-55)
        visual_layout.addWidget(self.view, 1)

        dashboard = QWidget()
        dashboard.setMinimumWidth(300)
        dashboard.setMaximumWidth(360)
        dashboard.setStyleSheet("background: #0d1117; color: #dce6f2;")
        dashboard_layout = QVBoxLayout(dashboard)
        dashboard_layout.setContentsMargins(8, 8, 8, 8)
        dashboard_layout.setSpacing(7)

        dashboard_title = QLabel("LIVE PRINTER TELEMETRY")
        dashboard_title.setAlignment(Qt.AlignCenter)
        dashboard_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #8fd3ff;")
        dashboard_layout.addWidget(dashboard_title)

        card_grid = QGridLayout()
        self.printer_state_card = self.make_status_card("STATE", "OFFLINE")
        self.printer_layer_card = self.make_status_card("LAYER", "—")
        self.printer_nozzle_card = self.make_status_card("NOZZLE", "—")
        self.printer_bed_card = self.make_status_card("BED", "—")
        card_grid.addWidget(self.printer_state_card, 0, 0)
        card_grid.addWidget(self.printer_layer_card, 0, 1)
        card_grid.addWidget(self.printer_nozzle_card, 1, 0)
        card_grid.addWidget(self.printer_bed_card, 1, 1)
        dashboard_layout.addLayout(card_grid)

        self.nozzle_plot, self.nozzle_actual_curve, self.nozzle_target_curve = \
            self.make_temperature_plot("Nozzle temperature", "#ff6b35")
        self.bed_plot, self.bed_actual_curve, self.bed_target_curve = \
            self.make_temperature_plot("Bed temperature", "#2da8ff")
        dashboard_layout.addWidget(self.nozzle_plot, 1)
        dashboard_layout.addWidget(self.bed_plot, 1)

        animation_group = QGroupBox("Live process animation")
        animation_group.setStyleSheet(
            "QGroupBox { color: #dce6f2; border: 1px solid #364353; "
            "margin-top: 8px; padding-top: 8px; }"
        )
        animation_layout = QGridLayout(animation_group)
        self.fan_animation = QLabel("✣")
        self.fan_animation.setAlignment(Qt.AlignCenter)
        self.fan_animation.setStyleSheet("font-size: 38px; color: #72d6ff;")
        self.fan_caption = QLabel("Fan: —")
        self.fan_caption.setAlignment(Qt.AlignCenter)
        animation_layout.addWidget(self.fan_animation, 0, 0)
        animation_layout.addWidget(self.fan_caption, 1, 0)
        dashboard_layout.addWidget(animation_group)

        visual_layout.addWidget(dashboard)
        splitter.addWidget(visual_container)
        splitter.setSizes([440, 1200])

    @staticmethod
    def make_status_card(title: str, value: str) -> QLabel:
        card = QLabel(f"{title}\n{value}")
        card.setProperty("title", title)
        card.setAlignment(Qt.AlignCenter)
        card.setMinimumHeight(48)
        card.setStyleSheet(
            "background: #151a22; color: #dce6f2; border: 1px solid #364353; "
            "border-radius: 5px; font-size: 11px; font-weight: 600; padding: 4px;"
        )
        return card

    @staticmethod
    def update_status_card(card: QLabel, value: str, color: str = "#dce6f2") -> None:
        card.setText(f"{card.property('title')}\n{value}")
        card.setStyleSheet(
            f"background: #151a22; color: {color}; border: 1px solid #364353; "
            "border-radius: 5px; font-size: 11px; font-weight: 600; padding: 4px;"
        )

    @staticmethod
    def make_temperature_plot(title: str, actual_color: str):
        plot = pg.PlotWidget()
        plot.setBackground("#0d1117")
        plot.setMinimumHeight(145)
        plot.showGrid(x=True, y=True, alpha=0.18)
        plot.setLabel("left", "Temperature", units="°C")
        plot.setLabel("bottom", "Elapsed print time", units="s")
        plot.setTitle(title, color="#dce6f2", size="10pt")
        plot.addLegend(offset=(8, 8))
        actual = plot.plot(
            [], [], pen=pg.mkPen(actual_color, width=2.2), name="Actual"
        )
        target = plot.plot(
            [], [], pen=pg.mkPen("#f3f5f7", width=1.4, style=Qt.DashLine),
            name="Target",
        )
        return plot, actual, target

    def update_live_dashboard(self) -> None:
        """Record and animate the newest MQTT values once per second."""
        if not self.is_connected or not self.telemetry_logging_active:
            return

        self.telemetry_sample += 1
        data = self.telemetry_data
        elapsed_seconds = (
            0.0 if self.print_started_monotonic is None
            else time.monotonic() - self.print_started_monotonic
        )
        values = {
            "sample": elapsed_seconds,
            "nozzle": self.safe_float(data.get("nozzle_temper"), np.nan),
            "nozzle_target": self.safe_float(data.get("nozzle_target_temper"), np.nan),
            "bed": self.safe_float(data.get("bed_temper"), np.nan),
            "bed_target": self.safe_float(data.get("bed_target_temper"), np.nan),
        }
        for key, value in values.items():
            self.telemetry_history[key].append(value)
            self.telemetry_history[key] = self.telemetry_history[key][
                -self.telemetry_history_limit:
            ]

        x = np.asarray(self.telemetry_history["sample"], dtype=float)
        self.nozzle_actual_curve.setData(x, self.telemetry_history["nozzle"])
        self.nozzle_target_curve.setData(x, self.telemetry_history["nozzle_target"])
        self.bed_actual_curve.setData(x, self.telemetry_history["bed"])
        self.bed_target_curve.setData(x, self.telemetry_history["bed_target"])

        fan_raw = self.safe_float(data.get("cooling_fan_speed"), 0.0)
        fan_color = "#72d6ff" if fan_raw > 0 else "#65707d"
        self.fan_animation.setText("✣")
        self.fan_animation.setStyleSheet(f"font-size: 38px; color: {fan_color};")

    def build_3d_scene(self) -> None:
        grid = gl.GLGridItem()
        grid.setSize(x=40, y=40)
        grid.setSpacing(x=2, y=2)
        grid.setColor((95, 105, 120, 120))
        self.view.addItem(grid)

        self.completed_item = gl.GLLinePlotItem(
            pos=np.empty((0, 3)),
            color=(0.10, 0.55, 1.00, 0.88),
            width=2.0,
            mode="lines",
            antialias=True,
        )
        self.active_item = gl.GLLinePlotItem(
            pos=np.empty((0, 3)),
            color=(1.00, 0.38, 0.08, 1.00),
            width=3.0,
            mode="lines",
            antialias=True,
        )
        self.nominal_outline = gl.GLLinePlotItem(
            pos=np.empty((0, 3)),
            color=(0.78, 0.82, 0.88, 0.55),
            width=1.0,
            mode="lines",
            antialias=True,
        )
        self.predicted_outline = gl.GLLinePlotItem(
            pos=np.empty((0, 3)),
            color=(0.20, 1.00, 0.45, 0.85),
            width=2.0,
            mode="lines",
            antialias=True,
        )
        self.nozzle_item = gl.GLScatterPlotItem(
            pos=np.empty((0, 3)),
            color=(1.00, 0.18, 0.08, 1.00),
            size=10.0,
            pxMode=True,
        )
        for item in [
            self.completed_item,
            self.active_item,
            self.nominal_outline,
            self.predicted_outline,
            self.nozzle_item,
        ]:
            self.view.addItem(item)

    def build_printer_scene(self) -> None:
        """Build a separate schematic view of the physical printer."""
        floor = gl.GLGridItem()
        floor.setSize(x=50, y=44)
        floor.setSpacing(x=2, y=2)
        floor.setColor((75, 85, 100, 95))
        floor.translate(0, 0, -1.45)
        self.printer_view.addItem(floor)

        components = [
            create_box_item((0, 0, -0.2), (43, 35, 2.5), (0.12, 0.15, 0.19, 1.0)),
            create_box_item((0, 0, 1.0), (36, 31, 0.7), (0.25, 0.28, 0.31, 1.0)),
            create_box_item((0, 0, 1.4), (34, 29, 0.18), (0.48, 0.51, 0.54, 1.0)),
            create_box_item((-18.5, 9.5, 17), (2.6, 4.5, 34), (0.70, 0.74, 0.79, 1.0)),
            create_box_item((-18.5, 9.5, 34), (5.0, 6.5, 2.5), (0.13, 0.16, 0.20, 1.0)),
            create_box_item((0, 0, 25), (38, 2.0, 2.0), (0.65, 0.69, 0.74, 1.0)),
        ]
        for component in components:
            self.printer_view.addItem(component)

        self.printer_toolhead = create_box_item(
            (0, 0, 21.5), (4.8, 4.8, 6.2), (0.88, 0.90, 0.92, 1.0)
        )
        self.printer_nozzle_body = create_box_item(
            (0, 0, 17.8), (1.1, 1.1, 1.4), (0.95, 0.36, 0.08, 1.0)
        )
        self.printer_view.addItem(self.printer_toolhead)
        self.printer_view.addItem(self.printer_nozzle_body)

        self.printer_completed = gl.GLLinePlotItem(
            pos=np.empty((0, 3)), color=(0.10, 0.55, 1.00, 0.95),
            width=2.0, mode="lines", antialias=True,
        )
        self.printer_active = gl.GLLinePlotItem(
            pos=np.empty((0, 3)), color=(1.00, 0.38, 0.08, 1.00),
            width=3.0, mode="lines", antialias=True,
        )
        self.printer_nozzle_marker = gl.GLScatterPlotItem(
            pos=np.empty((0, 3)), color=(1.00, 0.15, 0.05, 1.00),
            size=9.0, pxMode=True,
        )
        for item in [self.printer_completed, self.printer_active, self.printer_nozzle_marker]:
            item.translate(0, 0, 1.48)
            self.printer_view.addItem(item)

    def move_printer_head(self, point: np.ndarray | None) -> None:
        if point is None or np.asarray(point).size < 3:
            x, z = 0.0, 21.5
        else:
            xyz = np.asarray(point, dtype=float).reshape(-1, 3)[0]
            x = float(np.clip(xyz[0], -15.0, 15.0))
            z = float(np.clip(xyz[2] + 7.0, 6.0, 21.5))
        self.printer_toolhead.setMeshData(
            vertexes=box_vertices((x, 0, z), (4.8, 4.8, 6.2)), faces=BOX_FACES
        )
        self.printer_toolhead.setColor((0.88, 0.90, 0.92, 1.0))
        self.printer_nozzle_body.setMeshData(
            vertexes=box_vertices((x, 0, z - 3.8), (1.1, 1.1, 1.4)), faces=BOX_FACES
        )
        self.printer_nozzle_body.setColor((0.95, 0.36, 0.08, 1.0))

    def load_toolpath_settings(self) -> None:
        defaults = {
            "Layer_Height_mm": 0.20,
            "Line_Width_mm": 0.44,
            "Print_Speed_mm_s": 65.0,
            "Flow_Ratio": 1.00,
            "Nozzle_Temp_C": 220.0,
            "Cooling_Fan_Speed_pct": 50.0,
        }
        for name, box in self.parameter_boxes.items():
            box.blockSignals(True)
            box.setValue(float(self.toolpath.settings.get(name, defaults[name])))
            box.blockSignals(False)

    @staticmethod
    def parse_toolpath_file(path: Path):
        """Parse one Bambu Studio toolpath and remove implausible jump lines."""
        toolpath = parse_bambu_3mf(path)
        if toolpath.total_layers <= 0:
            raise ValueError("The selected file does not contain printable layers.")

        # The validation specimens are at most 25 mm long. Segments longer than
        # 35 mm are non-deposition jumps occasionally encoded around feature
        # changes and should not be drawn as deposited material.
        for layer in toolpath.layers:
            if layer.segments.size:
                xy_change = layer.segments[:, 1, :2] - layer.segments[:, 0, :2]
                segment_length = np.linalg.norm(xy_change, axis=1)
                layer.segments = layer.segments[segment_length <= 35.0]
        return toolpath

    def select_toolpath_file(self) -> None:
        """Load the exact sliced file that will be sent to the printer."""
        # Changing the digital definition during a physical print would break
        # synchronization, so toolpath changes are accepted only between jobs.
        state = str(self.telemetry_data.get("gcode_state") or "").upper()
        if self.active_job_seen and state in {
            "PREPARE", "PREPARING", "RUNNING", "PAUSE", "PAUSED"
        }:
            QMessageBox.warning(
                self,
                "Print is active",
                "Select the validation toolpath before starting the physical print.",
            )
            return

        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select the sliced validation toolpath",
            str(self.toolpath_path.parent),
            "Bambu Studio files (*.gcode.3mf *.3mf);;G-code files (*.gcode);;All files (*.*)",
        )
        if not selected:
            return

        # Parse into a temporary object first. The working digital twin is not
        # changed if the selected file is invalid or cannot be read.
        selected_path = Path(selected)
        try:
            selected_toolpath = self.parse_toolpath_file(selected_path)
        except Exception as error:
            QMessageBox.critical(
                self,
                "Toolpath could not be loaded",
                f"The selected file could not be parsed:\n\n{error}",
            )
            return

        # Commit the successfully parsed toolpath and rebuild every derived
        # layer/segment structure used by the viewer.
        self.toolpath_path = selected_path
        self.toolpath = selected_toolpath
        self.total_layers = self.toolpath.total_layers
        self.telemetry_data["total_layer_num"] = self.total_layers
        self.preview_segments = [
            (layer_number, segment.copy())
            for layer_number, layer in enumerate(self.toolpath.layers, start=1)
            for segment in layer.segments
        ]
        self.preview_index = 0
        self.preview_running = False
        self.preview_started = False
        self.active_job_seen = False
        self.previous_progress = None
        self.last_reported_layer = None

        self.toolpath_name.setText(self.toolpath_path.name)
        self.toolpath_name.setToolTip(str(self.toolpath_path))
        self.load_toolpath_settings()
        self.update_prediction()
        self.update_telemetry_labels()
        self.update_3d_part()
        QMessageBox.information(
            self,
            "Toolpath loaded",
            f"Loaded: {self.toolpath_path.name}\n"
            f"Printable layers: {self.total_layers}\n\n"
            "Confirm the nominal CAD dimensions, then start the print.",
        )

    def connect_to_printer(self) -> None:
        """Connect or disconnect when the printer-connection button is pressed."""
        if self.is_connected:
            self.mqtt_client.disconnect()
            self.on_connection_changed(False, "Disconnected")
            return

        ip = self.ip_input.text().strip()
        serial = self.serial_input.text().strip()
        access_code = self.access_input.text().strip()
        if not ip or not serial or not access_code:
            QMessageBox.warning(
                self,
                "Missing connection information",
                "Enter the printer IP, serial number, and LAN access code.",
            )
            return
        self.connect_button.setEnabled(False)
        self.connection_status.setText("CONNECTING…")
        self.connection_status.setStyleSheet("font-weight: 700; color: #d98c00;")
        try:
            self.mqtt_client.connect(ip, serial, access_code)
        except Exception as error:
            self.connect_button.setEnabled(True)
            QMessageBox.critical(self, "Printer connection error", str(error))

    def on_connection_changed(self, connected: bool, message: str) -> None:
        self.is_connected = connected
        self.connect_button.setEnabled(True)
        if connected:
            # A Bambu push-all response can contain the previous completed job.
            # Do not visualize it until this connection observes a live layer.
            self.active_job_seen = False
            self.previous_progress = None
            self.last_reported_layer = None
            self.telemetry_sample = 0
            for history in self.telemetry_history.values():
                history.clear()
            self.connection_status.setText("CONNECTED")
            self.connection_status.setStyleSheet("font-weight: 700; color: #16823b;")
            self.connect_button.setText("Disconnect from printer")
        else:
            self.telemetry_logging_active = False
            self.connection_status.setText("OFFLINE")
            self.connection_status.setStyleSheet("font-weight: 700; color: #d7191c;")
            self.connect_button.setText("Connect to printer")
            self.telemetry_data["gcode_state"] = "OFFLINE"
            self.update_telemetry_labels()
            self.update_3d_part()
            if message and message not in {"Disconnected", "Printer connection lost"}:
                QMessageBox.warning(self, "Printer connection", message)

    def on_telemetry(self, report: dict) -> None:
        """Merge one MQTT report and update the print-session state machine."""
        # Bambu reports can be partial; keep earlier values for fields omitted
        # from the newest message.
        aliases = {
            "gcode_state": ["gcode_state"],
            "mc_percent": ["mc_percent"],
            "layer_num": ["layer_num"],
            "total_layer_num": ["total_layer_num"],
            "nozzle_temper": ["nozzle_temper"],
            "nozzle_target_temper": ["nozzle_target_temper", "nozzle_target"],
            "bed_temper": ["bed_temper"],
            "bed_target_temper": ["bed_target_temper", "bed_target"],
            "cooling_fan_speed": ["cooling_fan_speed"],
            "mc_print_line_number": ["mc_print_line_number"],
        }
        for destination, source_names in aliases.items():
            for source in source_names:
                if source in report and report[source] is not None:
                    self.telemetry_data[destination] = report[source]
                    break

        state = str(self.telemetry_data.get("gcode_state") or "").upper()
        layer = int(self.safe_float(self.telemetry_data.get("layer_num"), 0))
        progress = self.safe_float(self.telemetry_data.get("mc_percent"), 0)
        active_states = {"PREPARE", "PREPARING", "RUNNING", "PAUSE", "PAUSED"}
        terminal_states = {
            "FINISH", "FINISHED", "COMPLETED", "COMPLETE",
            "FAILED", "CANCEL", "CANCELLED", "IDLE",
        }

        # A large progress drop indicates that a new job has replaced the old
        # completed job. Keep the viewer empty until the new job reaches layer 1.
        if (
            self.previous_progress is not None
            and self.previous_progress >= 95
            and progress < 90
        ):
            self.active_job_seen = False

        if state in active_states and progress < 100.0:
            if not self.telemetry_logging_active:
                self.start_session_log()
            if layer > 0:
                self.active_job_seen = True
        elif self.telemetry_logging_active and state in terminal_states:
            # Capture exactly one terminal record and immediately close this
            # print session. The periodic timer cannot append further rows.
            if not self.finish_row_written:
                self.log_snapshot(force=True)
                self.finish_row_written = True
            self.telemetry_logging_active = False

        if layer <= 0 and state not in active_states:
            self.active_job_seen = False

        self.previous_progress = progress
        self.last_reported_layer = layer
        self.update_telemetry_labels()
        self.update_3d_part()

    @staticmethod
    def safe_float(value, default=0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def temperature_text(actual, target) -> str:
        if actual is None and target is None:
            return "—"
        actual_text = "—" if actual is None else f"{DigitalTwinWindow.safe_float(actual):.1f}"
        target_text = "—" if target is None else f"{DigitalTwinWindow.safe_float(target):.1f}"
        return f"{actual_text} / {target_text} °C"

    def update_telemetry_labels(self) -> None:
        data = self.telemetry_data
        state = str(data.get("gcode_state") or "UNKNOWN").upper()
        layer = max(0, int(self.safe_float(data.get("layer_num"), 0)))
        total = max(self.total_layers, int(self.safe_float(data.get("total_layer_num"), 0)))
        progress = min(100.0, max(0.0, self.safe_float(data.get("mc_percent"), 0)))

        # On connection, Bambu may report the previous job as FINISH/100%.
        # Present the current digital-twin session instead of stale job status.
        if self.is_connected and not self.active_job_seen:
            if state in {"RUNNING", "PAUSE", "PAUSED"}:
                display_state = "PREPARING"
            else:
                display_state = "IDLE"
            display_layer = "—"
            display_progress = 0
        else:
            display_state = state
            display_layer = f"{layer} / {total}"
            display_progress = int(round(progress))

        self.state_value.setText(display_state)
        self.layer_value.setText(display_layer)
        self.nozzle_value.setText(
            self.temperature_text(data.get("nozzle_temper"), data.get("nozzle_target_temper"))
        )
        self.bed_value.setText(
            self.temperature_text(data.get("bed_temper"), data.get("bed_target_temper"))
        )
        fan = data.get("cooling_fan_speed")
        self.fan_value.setText("—" if fan is None else str(fan))
        self.progress_bar.setValue(display_progress)
        self.fan_caption.setText("Fan: —" if fan is None else f"Fan signal: {fan}")

        state_color = "#52d681" if self.is_connected else "#ff6861"
        if display_state in {"PREPARING", "PAUSE", "PAUSED"}:
            state_color = "#ffc857"
        self.update_status_card(self.printer_state_card, display_state, state_color)
        self.update_status_card(
            self.printer_nozzle_card,
            self.temperature_text(data.get("nozzle_temper"), data.get("nozzle_target_temper")),
            "#ff8b58",
        )
        self.update_status_card(
            self.printer_bed_card,
            self.temperature_text(data.get("bed_temper"), data.get("bed_target_temper")),
            "#75c9ff",
        )
        self.update_status_card(self.printer_layer_card, display_layer, "#c6ffcc")

    def update_prediction(self, *args) -> None:
        """Predict dimensional errors whenever a commanded input changes."""
        features = np.asarray(
            [[self.parameter_boxes[name].value() for name in FEATURE_NAMES]],
            dtype=float,
        )
        prediction = np.asarray(self.model.predict(features), dtype=float).reshape(-1)
        if prediction.size != 3:
            raise ValueError("The model must produce length, width, and height errors.")
        # Prediction tolerance is the one-standard-deviation spread among the
        # individual trees in the fitted Random Forest for this exact input.
        estimators = getattr(self.model, "estimators_", [])
        if estimators:
            tree_predictions = np.asarray(
                [np.asarray(tree.predict(features)).reshape(-1) for tree in estimators],
                dtype=float,
            )
            tolerance = np.std(tree_predictions, axis=0, ddof=1)
        else:
            tolerance = np.zeros(3, dtype=float)
        self.predicted_errors = dict(zip(["Length", "Width", "Height"], prediction))
        self.prediction_std = dict(zip(["Length", "Width", "Height"], tolerance))
        self.predicted_dimensions = {}
        for dimension in ["Length", "Width", "Height"]:
            nominal = self.nominal_boxes[dimension].value()
            error = self.predicted_errors[dimension]
            predicted = nominal + error
            self.predicted_dimensions[dimension] = predicted
            error_label, dimension_label, tolerance_label = self.prediction_labels[dimension]
            error_label.setText(f"{error:+.4f}")
            dimension_label.setText(f"{predicted:.4f}")
            tolerance_label.setText(f"±{self.prediction_std[dimension]:.4f}")
            tolerance_label.setToolTip(
                f"Expected range: {predicted - self.prediction_std[dimension]:.4f} "
                f"to {predicted + self.prediction_std[dimension]:.4f} mm"
            )
        self.update_validation_display()
        if hasattr(self, "completed_item"):
            self.update_3d_part()

    def update_validation_display(self, *args) -> None:
        if not hasattr(self, "predicted_dimensions"):
            return
        for dimension in ["Length", "Width", "Height"]:
            measured = self.actual_boxes[dimension].value()
            if measured <= 0.0:
                self.validation_labels[dimension].setText("Not entered")
                self.validation_labels[dimension].setStyleSheet("color: #65707d;")
                self.validation_labels[dimension].setToolTip(
                    "Enter the averaged post-print measurement."
                )
                continue
            predicted = self.predicted_dimensions[dimension]
            residual = measured - predicted
            tolerance = self.prediction_std[dimension]
            inside = abs(residual) <= tolerance
            color = "#16823b" if inside else "#c62828"
            self.validation_labels[dimension].setText(f"{residual:+.4f} mm")
            self.validation_labels[dimension].setStyleSheet(f"color: {color};")
            self.validation_labels[dimension].setToolTip(
                "Within ±1 SD" if inside else "Outside ±1 SD"
            )

    def save_validation_result(self) -> None:
        """Append the current prediction and manual measurements to one CSV."""
        missing = [
            dimension for dimension in ["Length", "Width", "Height"]
            if self.actual_boxes[dimension].value() <= 0.0
        ]
        if missing:
            QMessageBox.warning(
                self,
                "Measurements required",
                "Enter the measured value for " + ", ".join(missing) +
                " before saving the validation result.",
            )
            return
        LOG_FOLDER.mkdir(exist_ok=True)
        output_path = LOG_FOLDER / "digital_twin_validation_results.csv"
        new_file = not output_path.exists()
        dimensions = ["Length", "Width", "Height"]
        with output_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if new_file:
                writer.writerow(
                    [
                        "Timestamp", *FEATURE_NAMES,
                        *[f"Nominal_{d}_mm" for d in dimensions],
                        *[f"Predicted_{d}_mm" for d in dimensions],
                        *[f"Prediction_SD_{d}_mm" for d in dimensions],
                        *[f"Measured_{d}_mm" for d in dimensions],
                        *[f"Measured_Minus_Predicted_{d}_mm" for d in dimensions],
                        *[f"Within_1SD_{d}" for d in dimensions],
                    ]
                )
            residuals = {
                d: self.actual_boxes[d].value() - self.predicted_dimensions[d]
                for d in dimensions
            }
            writer.writerow(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    *[self.parameter_boxes[name].value() for name in FEATURE_NAMES],
                    *[self.nominal_boxes[d].value() for d in dimensions],
                    *[self.predicted_dimensions[d] for d in dimensions],
                    *[self.prediction_std[d] for d in dimensions],
                    *[self.actual_boxes[d].value() for d in dimensions],
                    *[residuals[d] for d in dimensions],
                    *[abs(residuals[d]) <= self.prediction_std[d] for d in dimensions],
                ]
            )
        QMessageBox.information(
            self, "Validation saved", f"Validation result saved to:\n{output_path}"
        )

    def current_layer_number(self) -> int:
        """Use the physical layer report; never infer layers from job percent."""
        reported = int(self.safe_float(self.telemetry_data.get("layer_num"), 0))
        if reported > 0:
            return min(reported, self.total_layers)
        return 0

    def toggle_printer_preview(self) -> None:
        if not self.preview_segments:
            QMessageBox.warning(self, "Simulation", "No extrusion segments are available.")
            return
        if self.preview_index >= len(self.preview_segments):
            self.reset_printer_preview()
        self.preview_started = True
        self.preview_running = not self.preview_running
        self.preview_play_button.setText("Pause" if self.preview_running else "Play simulation")
        self.update_printer_scene()

    def reset_printer_preview(self) -> None:
        self.preview_running = False
        self.preview_started = False
        self.preview_index = 0
        self.preview_play_button.setText("Play simulation")
        self.preview_progress.setValue(0)
        self.update_printer_scene()

    def advance_printer_preview(self) -> None:
        if not self.preview_running or not self.preview_segments:
            return
        count = max(1, int(round(self.preview_speed.value())))
        self.preview_index = min(len(self.preview_segments), self.preview_index + count)
        self.preview_progress.setValue(
            int(round(1000 * self.preview_index / len(self.preview_segments)))
        )
        if self.preview_index >= len(self.preview_segments):
            self.preview_running = False
            self.preview_play_button.setText("Replay")
        self.update_printer_scene()

    def update_printer_scene(self) -> None:
        """Update only the separate printer-schematic tab."""
        completed = []
        active = np.empty((0, 2, 3), dtype=float)

        if self.preview_started:
            completed = [segment for _, segment in self.preview_segments[: self.preview_index]]
            if self.preview_index < len(self.preview_segments):
                active = self.preview_segments[self.preview_index][1].reshape(1, 2, 3)
        elif self.active_job_seen:
            active_number = self.current_layer_number()
            state = str(self.telemetry_data.get("gcode_state") or "").upper()
            if state in {"FINISH", "FINISHED", "COMPLETED", "COMPLETE"}:
                active_number = self.total_layers + 1
            for index, layer in enumerate(self.toolpath.layers, start=1):
                if index < active_number and layer.segments.size:
                    completed.extend(layer.segments)
                elif index == active_number:
                    active = layer.segments

        completed_points = np.empty((0, 3), dtype=float)
        if completed:
            completed_points = segments_to_lines(np.asarray(completed, dtype=float))
        self.printer_completed.setData(pos=completed_points)
        self.printer_active.setData(pos=segments_to_lines(active))

        if active.size:
            point = active[-1, 1].reshape(1, 3).copy()
            marker = point.copy()
            marker[0, 2] += 0.35
            self.printer_nozzle_marker.setData(pos=marker)
            self.move_printer_head(point)
        elif completed:
            point = np.asarray(completed[-1][1], dtype=float).reshape(1, 3)
            self.printer_nozzle_marker.setData(pos=np.empty((0, 3)))
            self.move_printer_head(point)
        else:
            self.printer_nozzle_marker.setData(pos=np.empty((0, 3)))
            self.move_printer_head(None)

    def update_3d_part(self) -> None:
        """Synchronize deposited layers and the active layer with MQTT data."""
        progress = min(100.0, max(0.0, self.safe_float(self.telemetry_data.get("mc_percent"), 0)))
        state = str(self.telemetry_data.get("gcode_state") or "OFFLINE").upper()
        has_started = self.active_job_seen

        if not has_started:
            self.completed_item.setData(pos=np.empty((0, 3)))
            self.active_item.setData(pos=np.empty((0, 3)))
            self.nozzle_item.setData(pos=np.empty((0, 3)))
            self.nominal_outline.setData(pos=np.empty((0, 3)))
            self.predicted_outline.setData(pos=np.empty((0, 3)))
            return

        active_number = self.current_layer_number()
        print_finished = state in {"FINISH", "FINISHED", "COMPLETED", "COMPLETE"}
        if print_finished:
            # Treat every sliced layer as completed and remove the active-layer
            # marker once the physical job has finished.
            active_number = self.total_layers + 1
        completed_segments = []
        active_segments = np.empty((0, 2, 3), dtype=float)
        for index, layer in enumerate(self.toolpath.layers, start=1):
            if index < active_number and layer.segments.size:
                completed_segments.append(layer.segments)
            elif index == active_number:
                active_segments = layer.segments

        completed_points = np.empty((0, 3), dtype=float)
        if completed_segments:
            completed_points = segments_to_lines(np.concatenate(completed_segments, axis=0))

        # MQTT identifies the active physical layer, but does not provide a
        # reliable segment index. Show the whole active layer in orange instead
        # of inventing within-layer motion from mc_percent.
        visible_active = active_segments
        active_points = segments_to_lines(visible_active)

        self.completed_item.setData(pos=completed_points)
        self.active_item.setData(pos=active_points)

        nominal = {key: self.nominal_boxes[key].value() for key in ["Length", "Width", "Height"]}
        predicted = self.predicted_dimensions
        self.nominal_outline.setData(
            pos=create_box_edges(nominal["Length"], nominal["Width"], nominal["Height"])
        )
        self.predicted_outline.setData(
            pos=create_box_edges(predicted["Length"], predicted["Width"], predicted["Height"])
        )

        if visible_active.size and not print_finished:
            # Copy before raising the visual marker. Without copy(), reshape()
            # returns a view and each MQTT update permanently raises the stored
            # G-code endpoint, producing a long false orange line.
            nozzle_position = visible_active[-1, 1].reshape(1, 3).copy()
            nozzle_position[0, 2] += 0.35
            self.nozzle_item.setData(pos=nozzle_position)
        else:
            self.nozzle_item.setData(pos=np.empty((0, 3)))

    def start_session_log(self) -> None:
        """Create exactly one new telemetry CSV when a new print begins."""
        LOG_FOLDER.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log_path = LOG_FOLDER / f"telemetry_print_{timestamp}.csv"
        self.telemetry_logging_active = True
        self.finish_row_written = False
        self.print_started_monotonic = time.monotonic()
        self.telemetry_sample = 0
        for history in self.telemetry_history.values():
            history.clear()
        self.nozzle_actual_curve.setData([], [])
        self.nozzle_target_curve.setData([], [])
        self.bed_actual_curve.setData([], [])
        self.bed_target_curve.setData([], [])
        with self.session_log_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Timestamp", "Printer_State", "Progress_pct", "Layer", "Total_Layers",
                    "Nozzle_Actual_C", "Nozzle_Target_C", "Bed_Actual_C", "Bed_Target_C",
                    "Cooling_Fan_Raw", "Print_Line", *FEATURE_NAMES,
                    "Predicted_Length_mm", "Predicted_Width_mm", "Predicted_Height_mm",
                ]
            )

    def log_snapshot(self, force: bool = False) -> None:
        """Append one timestamped telemetry record during an active print."""
        if not self.is_connected or self.session_log_path is None:
            return
        if not self.telemetry_logging_active and not force:
            return
        data = self.telemetry_data
        with self.session_log_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    datetime.now().isoformat(timespec="milliseconds"),
                    data.get("gcode_state"), data.get("mc_percent"), data.get("layer_num"),
                    data.get("total_layer_num"), data.get("nozzle_temper"),
                    data.get("nozzle_target_temper"), data.get("bed_temper"),
                    data.get("bed_target_temper"), data.get("cooling_fan_speed"),
                    data.get("mc_print_line_number"),
                    *[self.parameter_boxes[name].value() for name in FEATURE_NAMES],
                    self.predicted_dimensions["Length"], self.predicted_dimensions["Width"],
                    self.predicted_dimensions["Height"],
                ]
            )

    def closeEvent(self, event) -> None:
        """Stop the network loop cleanly before the application closes."""
        self.mqtt_client.disconnect()
        event.accept()


def run_application() -> None:
    application = QApplication(sys.argv)
    application.setStyle("Fusion")
    try:
        window = DigitalTwinWindow()
        window.show()
        sys.exit(application.exec())
    except Exception as error:
        QMessageBox.critical(None, "Digital-twin startup error", str(error))
        raise


if __name__ == "__main__":
    run_application()

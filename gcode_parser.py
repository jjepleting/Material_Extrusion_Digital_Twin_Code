"""Parse extrusion paths from a Bambu .gcode.3mf archive."""

import json
import re
import zipfile

from dataclasses import dataclass
from pathlib import Path

import numpy as np


MOVE_PATTERN = re.compile(
    r"^G([01])\s+(.*)$"
)

VALUE_PATTERN = re.compile(
    r"([XYZEF])(-?(?:\d+(?:\.\d*)?|\.\d+))"
)


@dataclass
class ToolpathLayer:

    number: int
    z_height: float
    segments: np.ndarray


@dataclass
class ParsedToolpath:

    layers: list
    settings: dict
    source_name: str

    @property
    def total_layers(self):

        return len(self.layers)


def first_setting_value(
    settings,
    name,
    default
):

    value = settings.get(
        name,
        default
    )

    if isinstance(value, list):

        if value:

            value = value[0]

        else:

            value = default

    try:

        return float(value)

    except (TypeError, ValueError):

        return float(default)


def read_bambu_archive(file_path):

    with zipfile.ZipFile(
        file_path,
        "r"
    ) as archive:

        names = archive.namelist()

        gcode_files = [
            name
            for name in names
            if name.lower().endswith(".gcode")
        ]

        if not gcode_files:

            raise ValueError(
                "No G-code file was found "
                "inside the 3MF archive."
            )

        gcode_text = archive.read(
            gcode_files[0]
        ).decode(
            "utf-8",
            errors="replace"
        )

        settings = {}

        settings_file = (
            "Metadata/project_settings.config"
        )

        if settings_file in names:

            settings = json.loads(
                archive.read(
                    settings_file
                ).decode("utf-8")
            )

    return gcode_text, settings


def parse_bambu_3mf(file_path):

    file_path = Path(file_path)

    if not file_path.exists():

        raise FileNotFoundError(
            file_path
        )

    gcode_text, project_settings = (
        read_bambu_archive(file_path)
    )

    model_settings = {
        "Layer_Height_mm":
        first_setting_value(
            project_settings,
            "layer_height",
            0.20
        ),

        "Line_Width_mm":
        first_setting_value(
            project_settings,
            "line_width",
            0.44
        ),

        "Print_Speed_mm_s":
        first_setting_value(
            project_settings,
            "outer_wall_speed",
            65.0
        ),

        "Flow_Ratio":
        first_setting_value(
            project_settings,
            "filament_flow_ratio",
            1.0
        ),

        "Nozzle_Temp_C":
        first_setting_value(
            project_settings,
            "nozzle_temperature",
            220.0
        ),

        "Cooling_Fan_Speed_pct":
        first_setting_value(
            project_settings,
            "fan_max_speed",
            50.0
        ),
    }

    layers = []

    current_layer_number = 0
    current_z_height = 0.0
    current_feature = ""
    current_segments = []

    current_x = 0.0
    current_y = 0.0
    current_z = 0.0

    relative_extrusion = True
    absolute_extrusion_value = 0.0

    inside_sliced_layers = False

    def finish_current_layer():

        nonlocal current_segments

        if current_layer_number > 0:

            if current_segments:

                segment_array = np.asarray(
                    current_segments,
                    dtype=float
                )

            else:

                segment_array = np.empty(
                    (0, 2, 3),
                    dtype=float
                )

            layers.append(
                ToolpathLayer(
                    number=current_layer_number,
                    z_height=current_z_height,
                    segments=segment_array
                )
            )

        current_segments = []

    for raw_line in gcode_text.splitlines():

        line = raw_line.strip()

        if line == "M82":

            relative_extrusion = False
            continue

        if line == "M83":

            relative_extrusion = True
            continue

        if line.startswith("G92"):

            values = {
                name: float(value)
                for name, value
                in VALUE_PATTERN.findall(line)
            }

            if "E" in values:

                absolute_extrusion_value = (
                    values["E"]
                )

            continue

        if line == "; CHANGE_LAYER":

            if inside_sliced_layers:

                finish_current_layer()

            inside_sliced_layers = True

            current_layer_number += 1

            current_feature = ""

            continue

        if not inside_sliced_layers:

            continue

        if line.startswith("; Z_HEIGHT:"):

            current_z_height = float(
                line.split(
                    ":",
                    1
                )[1].strip()
            )

            continue

        if line.startswith("; FEATURE:"):

            current_feature = (
                line.split(
                    ":",
                    1
                )[1].strip()
            )

            continue

        move_match = MOVE_PATTERN.match(
            line
        )

        if not move_match:

            continue

        values = {
            name: float(value)
            for name, value
            in VALUE_PATTERN.findall(
                move_match.group(2)
            )
        }

        new_x = values.get(
            "X",
            current_x
        )

        new_y = values.get(
            "Y",
            current_y
        )

        new_z = values.get(
            "Z",
            current_z
        )

        extrusion_change = 0.0

        if "E" in values:

            if relative_extrusion:

                extrusion_change = (
                    values["E"]
                )

            else:

                extrusion_change = (
                    values["E"]
                    - absolute_extrusion_value
                )

                absolute_extrusion_value = (
                    values["E"]
                )

        wanted_feature = (
            current_feature
            in {
                "Outer wall",
                "Gap infill",
                "Brim",
            }
        )

        position_changed = (
            new_x != current_x
            or new_y != current_y
        )

        if (
            extrusion_change > 0
            and wanted_feature
            and position_changed
        ):

            current_segments.append(
                [
                    [
                        current_x,
                        current_y,
                        current_z_height
                    ],
                    [
                        new_x,
                        new_y,
                        current_z_height
                    ]
                ]
            )

        current_x = new_x
        current_y = new_y
        current_z = new_z

    if inside_sliced_layers:

        finish_current_layer()

    if not layers:

        raise ValueError(
            "No extrusion layers were parsed."
        )

    populated_layers = [
        layer
        for layer in layers
        if layer.segments.size
    ]

    if not populated_layers:

        raise ValueError(
            "No extrusion segments were parsed."
        )

    all_points = np.concatenate(
        [
            layer.segments.reshape(-1, 3)
            for layer in populated_layers
        ],
        axis=0
    )

    center_x = (
        all_points[:, 0].min()
        + all_points[:, 0].max()
    ) / 2

    center_y = (
        all_points[:, 1].min()
        + all_points[:, 1].max()
    ) / 2

    for layer in layers:

        if layer.segments.size:

            layer.segments[:, :, 0] -= (
                center_x
            )

            layer.segments[:, :, 1] -= (
                center_y
            )

    return ParsedToolpath(
        layers=layers,
        settings=model_settings,
        source_name=file_path.name
    )


if __name__ == "__main__":

    toolpath = parse_bambu_3mf(
        "Walltest.gcode.3mf"
    )

    segment_count = sum(
        len(layer.segments)
        for layer in toolpath.layers
    )

    print(
        "File:",
        toolpath.source_name
    )

    print(
        "Layers:",
        toolpath.total_layers
    )

    print(
        "Extrusion segments:",
        segment_count
    )

    print("Settings:")

    for name, value in (
        toolpath.settings.items()
    ):

        print(
            f"  {name}: {value}"
        )
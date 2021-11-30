"""
App to run pulse oximeter from Arduino.

To serve the app, run

    bokeh serve --show pulseox_app.py

on the command line.
"""

import asyncio
import re
import sys
import time
from arduino_py import *

import numpy as np
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import scipy.signal

import serial
import serial.tools.list_ports

import bokeh.plotting
import bokeh.io
import bokeh.layouts
import bokeh.driving

# Set up connection
HANDSHAKE = 0
VOLTAGE_REQUEST = 1
ON_REQUEST = 2
STREAM = 3
READ_DAQ_DELAY = 4

port = find_arduino()
arduino = serial.Serial(port, baudrate=115200)
handshake_arduino(arduino)

data = dict(prev_array_length=0, time_ms=[], Voltage1=[], Voltage2=[], mode="stream")


def plot():
    """Build a plot of voltage vs time data"""
    # Set up plot area
    p = bokeh.plotting.figure(
        frame_width=500,
        frame_height=175,
        x_axis_label="time (s)",
        title="streaming data",
        toolbar_location="above",
    )

    # No range padding on x: signal spans whole plot
    p.x_range.range_padding = 0

    # We'll sue whitesmoke backgrounds
    p.border_fill_color = "whitesmoke"

    # Defined the data source
    source = bokeh.models.ColumnDataSource(
        data=dict(time_ms=[], Voltage1=[], Voltage2=[])
    )
    peak_time_source = bokeh.models.ColumnDataSource(
        data=dict(v1_peak_time_ms=[0], v2_peak_time_ms=[0])
    )

    p.line(
        source=source,
        x="time_ms",
        y="Voltage1",
        visible=True,
        line_color="#f46d43",
        legend_label="RED LED",
    )
    p.line(
        source=source, x="time_ms", y="Voltage2", visible=True, legend_label="IR LED"
    )
    p.legend.orientation = "horizontal"
    p.add_layout(p.legend[0], "above")

    # Put a phantom circle so axis labels show before data arrive
    phantom_source = bokeh.models.ColumnDataSource(
        data=dict(time_ms=[0], Voltage1=[7], Voltage2=[7])
    )
    p.circle(source=phantom_source, x="time_ms", y="Voltage1", visible=False)
    p.circle(source=phantom_source, x="time_ms", y="Voltage2", visible=False)

    heartrate_source = bokeh.models.ColumnDataSource(data=dict(heartrate=[]))
    heartrate_display = bokeh.models.Div(
        text="--",
        style={"font-size": "700%", "color": "#3182bd", "font-family": "Helvetica"},
    )

    oxygen_source = bokeh.models.ColumnDataSource(data=dict(oxygen=[]))
    oxygen_display = bokeh.models.Div(
        text="--",
        style={"font-size": "700%", "color": "#3182bd", "font-family": "Helvetica"},
    )
    return (
        p,
        source,
        peak_time_source,
        phantom_source,
        heartrate_source,
        heartrate_display,
        oxygen_source,
        oxygen_display,
    )


def controls(mode):
    acquire = bokeh.models.Toggle(label="stream", button_type="success", width=100)
    save_notice = bokeh.models.Div(text="<p>No streaming data saved.</p>", width=300)
    save = bokeh.models.Button(label="save", button_type="primary", width=100)
    reset = bokeh.models.Button(label="reset", button_type="warning", width=100)
    file_input = bokeh.models.TextInput(
        title="file name", value=f"{mode}.csv", width=160
    )
    # Shut down layout
    shutdown = bokeh.models.Button(label="shut down", button_type="danger", width=100)

    return dict(
        acquire=acquire,
        reset=reset,
        save=save,
        file_input=file_input,
        save_notice=save_notice,
        shutdown=shutdown,
    )


def layout(p, ctrls):

    buttons = bokeh.layouts.row(
        bokeh.models.Spacer(width=30),
        ctrls["acquire"],
        bokeh.models.Spacer(width=295),
        ctrls["shutdown"],
    )

    top = bokeh.layouts.column(p, buttons, spacing=15)
    bottom = bokeh.layouts.row(
        ctrls["file_input"], bokeh.layouts.column(ctrls["save_notice"], ctrls["save"])
    )
    return bokeh.layouts.column(
        top,
        bottom,
        background="whitesmoke",
    )


def stream_callback(arduino, stream_data, new):
    if new:
        stream_data["mode"] = "stream"
    else:
        stream_data["mode"] = "on-demand"
        arduino.write(bytes([ON_REQUEST]))

    arduino.reset_input_buffer()


def reset_callback(
    mode,
    data,
    source,
    peak_time_source,
    phantom_source,
    heartrate_source,
    oxygen_source,
    controls,
):
    # Turn off the stream
    if mode == "stream":
        controls["acquire"].active = False

    # Black out the data dictionaries
    data["time_ms"] = []
    data["Voltage1"] = []
    data["Voltage2"] = []

    # Reset the sources
    source.data = dict(time_ms=[], Voltage1=[], Voltage2=[])
    peak_time_source.data = dict(v1_peak_time_ms=[], v2_peak_time_ms=[])
    phantom_source.data = dict(time_ms=[0], Voltage1=[0], Voltage2=[0])
    heartrate_source.data = dict(heartrate=[])
    oxygen_source.data = dict(oxygen=[])


def disable_controls(controls):
    """Disable all controls."""
    for key in controls:
        controls[key].disabled = True


def save_callback(mode, data, controls):
    # Convert data to data frame and save
    df = pd.DataFrame(
        data={
            "time (ms)": data["time_ms"],
            "Voltage1 (V)": data["Voltage1"],
            "Voltage2 (V)": data["Voltage2"],
        }
    )
    df.to_csv(controls["file_input"].value, index=False)

    # Update notice text
    notice_text = "<p>" + ("Streaming")
    notice_text += f" data was last saved to {controls['file_input'].value}.</p>"
    controls["save_notice"].text = notice_text


def shutdown_callback(
    arduino,
    daq_task,
    stream_data,
    stream_controls,
):
    # Disable controls
    disable_controls(stream_controls)

    # Strop streaming
    stream_data["mode"] = "on-demand"
    arduino.write(bytes([ON_REQUEST]))

    # Stop DAQ async task
    daq_task.cancel()

    # Disconnect from Arduino
    arduino.close()


def stream_update(
    data,
    source,
    peak_time_source,
    phantom_source,
    heartrate_source,
    oxygen_source,
    rollover,
    heartrate_display,
    oxygen_display,
):
    # Update plot by streaming in data
    new_data = {
        "time_ms": np.array(data["time_ms"][data["prev_array_length"] :]) / 1000,
        "Voltage1": data["Voltage1"][data["prev_array_length"] :],
        "Voltage2": data["Voltage2"][data["prev_array_length"] :],
    }

    # print(new_data)
    source.stream(new_data, rollover)
    # _,_,_,_,_, oxygen = take_snapshot(source, n=window)
    # oxygen_source.stream({"oxygen" : [o]})
    window = 200
    dac_delay = 5
    if len(source.data["time_ms"]) > window:
        if source.data["time_ms"][-1] > (
            peak_time_source.data["v2_peak_time_ms"][-1] + 3
        ):
            (
                v1_peak_times,
                v2_peak_times,
                v1_dc,
                v2_dc,
                time_window,
                oxygen,
                heartrates,
            ) = take_snapshot(source, n=window)
            peak_time_source.stream(
                {"v1_peak_time_ms": v1_peak_times, "v2_peak_time_ms": v1_peak_times},
                rollover,
            )

            # if there are peaks
            # DO SOMETHING
            heartrate_source.stream({"heartrate": heartrates})
            heartrate_display.text = (
                str(int(scipy.stats.trim_mean(heartrate_source.data["heartrate"], 0.1)))
                + "BPM"
            )
            if oxygen[-1] != "--":
                oxygen_source.stream({"oxygen": oxygen})
                oxygen_display.text = (
                    str(round(np.average(oxygen_source.data["oxygen"]), 2)) + "%"
                )
    # Adjust new phantom data point if new data arrived
    if len(new_data["time_ms"] > 0):
        phantom_source.data = dict(
            time_ms=[new_data["time_ms"][-1]],
            Voltage1=[new_data["Voltage1"][-1]],
            Voltage2=[new_data["Voltage2"][-1]],
        )
    data["prev_array_length"] = len(data["time_ms"])
    # print(data)


def take_snapshot(datasource, n=50):
    time_window = np.array(datasource.data["time_ms"][-n:])
    v1_window = np.array(datasource.data["Voltage1"][-n:])
    v2_window = np.array(datasource.data["Voltage2"][-n:])

    v1_dc = pd.Series(v1_window).rolling(20).mean().dropna().tolist()
    v2_dc = pd.Series(v2_window).rolling(20).mean().dropna().tolist()

    v1_dc = extend_moving_avg(v1_dc, n)
    v2_dc = extend_moving_avg(v2_dc, n)
    #     rescaled_v1 = rescale(v1_window)
    #     rescaled_v2 = rescale(v2_window)
    smooth_v1_window = savgol_filter(v1_window, window_length=21, polyorder=3)
    smooth_v2_window = savgol_filter(v1_window, window_length=21, polyorder=3)
    v1_peak_inds = scipy.signal.find_peaks(smooth_v1_window, prominence=0.0035)[0]
    v2_peak_inds = scipy.signal.find_peaks(smooth_v2_window, prominence=0.0035)[0]

    v1_peak_times = time_window[v1_peak_inds]
    v2_peak_times = time_window[v2_peak_inds]

    if len(v1_peak_inds) == len(v2_peak_inds):
        v1_peak_ac = np.array(v1_window)[v1_peak_inds]
        v2_peak_ac = np.array(v2_window)[v2_peak_inds]

        v1_peak_dc = np.array(v1_dc)[v1_peak_inds]
        v2_peak_dc = np.array(v2_dc)[v2_peak_inds]

        a = -3.3
        b = -21.1
        c = 109.6
        R = (v1_peak_ac / v1_peak_dc) / (v2_peak_ac / v2_peak_dc)
        oxygen = a * R ** 2 + b * R + c
    else:
        oxygen = ["--"]

    heartrates = np.array(1) / np.ediff1d(v2_peak_times) * 60
    return (v1_peak_times, v2_peak_times, v1_dc, v2_dc, time_window, oxygen, heartrates)


# def rescale(signal):
#     """Rescale signal to go from -1 to 1."""
#     y = signal - signal.mean()
#     y = 1 + 2 / (y.max() - y.min()) * (y - y.max())

#     return y


def extend_moving_avg(l, length):
    """Since moving average returns an array that is smaller than the
    input array, extend the last output of the moving averaged array to
    make it the same length as its pre-filter array"""
    l.extend([l[-1]] * (length - len(l)))
    return l


def pulse_app(
    arduino,
    stream_data,
    daq_task,
    rollover=400,
    stream_plot_delay=1,
):
    def _app(doc):
        # Plots
        (
            p_stream,
            stream_source,
            peak_time_source,
            stream_phantom_source,
            stream_heartrate,
            heartrate_display,
            stream_oxygen,
            oxygen_display,
        ) = plot()

        # Controls
        stream_controls = controls("stream")

        # Shut down
        # shutdown_button = bokeh.models.Button(
        #     label="shut down", button_type="danger", width=100
        # )

        # Layouts
        stream_layout = layout(p_stream, stream_controls)

        # # Shut down layout
        # shutdown_layout = bokeh.layouts.row(
        #     bokeh.models.Spacer(width=670), shutdown_button
        # )

        # results layout
        url = "https://i.ibb.co/rfK829J/health-insights-heart-rate.png"
        results_layout = bokeh.layouts.column(
            bokeh.models.Div(text="<img src=" + url + ' style="width:300px">'),
            heartrate_display,
            bokeh.models.Spacer(height=10),
            oxygen_display,
            background="whitesmoke",
        )

        app_layout = bokeh.layouts.column(
            bokeh.layouts.row(stream_layout, results_layout), background="whitesmoke"
        )

        def _acquire_callback(event=None):
            acquire_callback(
                arduino,
                stream_data,
                rollover,
            )

        def _stream_callback(attr, old, new):
            stream_callback(arduino, stream_data, new)

        def _stream_reset_callback(event=None):
            reset_callback(
                "stream",
                stream_data,
                stream_source,
                peak_time_source,
                stream_phantom_source,
                stream_heartrate,
                stream_oxygen,
                stream_controls,
            )

        def _stream_save_callback(event=None):
            save_callback("stream", stream_data, stream_controls)

        def _shutdown_callback(event=None):
            shutdown_callback(
                arduino,
                daq_task,
                stream_data,
                stream_controls,
            )

        @bokeh.driving.linear()
        def _stream_update(step):
            stream_update(
                stream_data,
                stream_source,
                peak_time_source,
                stream_phantom_source,
                stream_heartrate,
                stream_oxygen,
                rollover,
                heartrate_display,
                oxygen_display,
            )

            # Shut down server if Arduino disconnects (commented out in Jupyter notebook)
            if not arduino.is_open:
                sys.exit()

        # Link callbacks
        stream_controls["acquire"].on_change("active", _stream_callback)
        stream_controls["reset"].on_click(_stream_reset_callback)
        stream_controls["save"].on_click(_stream_save_callback)
        stream_controls["reset"].on_click(_shutdown_callback)
        stream_controls["shutdown"].on_click(_shutdown_callback)

        # Add the layout to the app
        doc.add_root(app_layout)

        # Add a periodic callback, monitor changes in stream data
        pc = doc.add_periodic_callback(_stream_update, stream_plot_delay)

    return _app


daq_task = asyncio.create_task(daq_stream_async2(arduino, data))

# Build app
app = pulse_app(arduino, data, daq_task)
# Build it with curdoc
app(bokeh.plotting.curdoc())

# Copyright (c) 2020-2021 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import io

import numpy as np

from astropy import units as u

import h5py

import toast

from toast.utils import Environment, Logger, object_fullname

from toast.timing import function_timer

from toast.instrument import GroundSite

from toast.spt3g import (
    available,
    to_g3_time,
)

from toast.spt3g.spt3g_export import (
    export_detdata,
    export_shared,
    export_intervals,
)

if available:
    from spt3g import core as c3g


class ExportSimMeta(object):
    """Export metadata to Observation and Calibration frames.

    The telescope and site information will be written to the Observation frame.  The
    focalplane information will be written to the Calibration frame.

    """

    def __init__(self):
        pass

    @function_timer
    def __call__(self, obs):
        # Construct observation frame
        ob = c3g.G3Frame(c3g.G3FrameType.Observation)
        ob["observation_name"] = c3g.G3String(obs.name)
        ob["observation_uid"] = c3g.G3Int(obs.uid)
        ob["observation_detector_sets"] = c3g.G3VectorVectorString(
            obs.all_detector_sets
        )
        ob["telescope_name"] = c3g.G3String(obs.telescope.name)
        ob["telescope_class"] = c3g.G3String(object_fullname(obs.telescope.__class__))
        ob["telescope_uid"] = c3g.G3Int(obs.telescope.uid)
        site = obs.telescope.site
        ob["site_name"] = c3g.G3String(site.name)
        ob["site_class"] = c3g.G3String(object_fullname(site.__class__))
        ob["site_uid"] = c3g.G3Int(site.uid)
        ob["site_lat_deg"] = c3g.G3Double(site.earthloc.lat.to_value(u.degree))
        ob["site_lon_deg"] = c3g.G3Double(site.earthloc.lon.to_value(u.degree))
        ob["site_alt_m"] = c3g.G3Double(site.earthloc.height.to_value(u.meter))
        if site.weather is not None:
            if hasattr(site.weather, "name"):
                # This is a simulated weather object, dump it.
                ob["site_weather_name"] = c3g.G3String(site.weather.name)
                ob["site_weather_realization"] = c3g.G3Int(site.weather.realization)
                if site.weather.max_pwv is None:
                    ob["site_weather_max_pwv"] = c3g.G3String("NONE")
                else:
                    ob["site_weather_max_pwv"] = c3g.G3Double(site.weather.max_pwv)
                ob["site_weather_time"] = to_g3_time(site.weather.time.timestamp())

        # Construct calibration frame
        cal = c3g.G3Frame(c3g.G3FrameType.Calibration)

        # Serialize focalplane to HDF5 bytes and write to frame.
        byte_writer = io.BytesIO()
        with h5py.File(byte_writer, "w") as f:
            obs.telescope.focalplane.write(f)
        cal["focalplane"] = c3g.G3VectorUnsignedChar(byte_writer.getvalue())
        del byte_writer

        # Serialize noise models
        for m_in, m_out in self._noise_models:
            byte_writer = io.BytesIO()
            obs[m_in].save_hdf5(byte_writer)
            cal[m_out] = c3g.G3VectorUnsignedChar(byte_writer.getvalue())
            del byte_writer
            cal[f"{m_out}_class"] = c3g.G3String(object_fullname(obs[m_in].__class__))

        return ob, cal


class ExportSimData(object):
    """Export data to Scan frames

    Shared objects:  The `shared_names` list of tuples specifies the TOAST shared key,
    corresponding Scan frame key, and optionally the G3 datatype to use.  Each process
    will duplicate shared data into their Scan frame stream.  If the G3 datatype is
    None, the closest G3 object type will be chosen.  If the shared object contains
    multiple values per sample, these are reshaped into a flat-packed array.  Only
    sample-wise shared objects are supported at this time (i.e. no other shared objects
    like beams, etc).  One special case:  The `timestamps` field will always be copied
    to each Scan frame as a `G3Timestream`.

    DetData objects:  The `det_names` list of tuples specifies the TOAST detdata key,
    the corresponding Scan frame key, and optionally the G3 datatype to use.  If the G3
    datatype is None, then detdata objects with one value per sample and of type
    float64 or float32 will be converted to a `G3TimestreamMap`.  All other detdata
    objects will be converted to the most appropriate `G3Map*` data type, with all data
    flat-packed.  In this case any processing code will need to interpret this data in
    conjunction with the separate `timestamps` frame key.

    Intervals objects:  The `interval_names` list of tuples specifies the TOAST
    interval name and associated Scan frame key.  We attempt to use an `IntervalsTime`
    object filled with the start / stop times of each interval, but if that is not
    available we flat-pack the start / stop times into a `G3VectorTime` object.

    Args:
        timestamp_names (tuple):  The name of the shared data containing the
            timestamps, and the output frame key to use.
        shared_names (list):  The observation shared objects to export.
        det_names (list):  The observation detdata objects to export.
        interval_names (list):  The observation intervals to export.
        compress (bool):  If True, attempt to use flac compression for all exported
            G3TimestreamMap objects.

    """

    def __init__(
        self,
        timestamp_names=("times", "times"),
        shared_names=list(
            ("boresight_azel", "boresight_azel", c3g.G3VectorQuat),
            ("boresight_radec", "boresight_radec", c3g.G3VectorQuat),
            ("position", "position", None),
            ("velocity", "velocity", None),
            ("azimuth", "azimuth", None),
            ("elevation", "elevation", None),
            ("hwp_angle", "hwp_angle", None),
            ("flags", "telescope_flags", None),
        ),
        det_names=list(
            ("signal", "signal", None),
            ("flags", "detector_flags", None),
        ),
        interval_names=list(
            ("scan_leftright", "intervals_scan_leftright"),
            ("turn_leftright", "intervals_turn_leftright"),
            ("scan_rightleft", "intervals_scan_rightleft"),
            ("turn_rightleft", "intervals_turn_rightleft"),
            ("elnod", "intervals_elnod"),
            ("scanning", "intervals_scanning"),
            ("turnaround", "intervals_turnaround"),
            ("sun_up", "intervals_sun_up"),
            ("sun_close", "intervals_sun_close"),
        ),
        compress=False,
    ):
        self._timestamp_names = timestamp_names
        self._shared_names = shared_names
        self._det_names = det_names
        self._interval_names = interval_names
        self._compress = compress

    @function_timer
    def __call__(self, obs):
        # We are using the sample set distribution for our frame boundaries.
        frame_intervals = "frames"
        timespans = list()
        offset = 0
        n_frames = 0
        first_set = obs.dist.samp_sets[obs.comm_rank].offset
        n_set = obs.dist.samp_sets[obs.comm_rank].n_elem
        for sset in range(first_set, first_set + n_set):
            for chunk in obs.dist.sample_sets[sset]:
                timespans.append(
                    (
                        obs.shared[self._timestamp_names[0]][offset],
                        obs.shared[self._timestamp_names[0]][offset + chunk - 1],
                    )
                )
                n_frames += 1
                offset += chunk
        obs.intervals.create_col(
            frame_intervals, timespans, obs.shared[self._timestamp_names[0]]
        )

        output = list()
        frame_view = obs.view[frame_intervals]
        for ivw, tview in enumerate(frame_view.shared[self._timestamp_names[0]]):
            # Construct the Scan frame
            frame = c3g.G3Frame(c3g.G3FrameType.Scan)
            # Add timestamps
            frame[self._timestamp_names[1]] = export_shared(
                obs,
                self._timestamp_names[0],
                view_name=frame_intervals,
                view_index=ivw,
                g3t=c3g.G3VectorTime,
            )
            for shr_key, shr_val, shr_type in self._shared_names:
                frame[shr_val] = export_shared(
                    obs,
                    shr_key,
                    view_name=frame_intervals,
                    view_index=ivw,
                    g3t=shr_type,
                )
            for det_key, det_val, det_type in self._det_names:
                frame[det_val], gunits, compression = export_detdata(
                    obs,
                    det_key,
                    view_name=frame_intervals,
                    view_index=ivw,
                    g3t=det_type,
                    times=self._timestamp_names[0],
                    compress=self._compress,
                )
                # Record the original detdata type, so that it can be reconstructed
                # later.
                det_type_name = f"{det_val}_dtype"
                frame[det_type_name] = c3g.G3String(obs.detdata[det_key].dtype.char)
                if compression is not None:
                    # Store per-detector compression parameters in the frame.  Also
                    # store the original units, since these are wiped by the
                    # compression.
                    froot = f"compress_{det_val}"
                    for d in obs.local_detectors:
                        frame[f"{froot}_{d}_gain"] = compression[d]["gain"]
                        frame[f"{froot}_{d}_offset"] = compression[d]["offset"]
                        frame[f"{froot}_{d}_units"] = gunits
            for ivl_key, ivl_val in self._interval_names:
                frame[ivl_val] = export_intervals(
                    obs,
                    ivl_key,
                    self._timestamp_names[0],
                    view_name=frame_intervals,
                    view_index=ivw,
                )
            output.append(frame)

        # Delete our temporary frame interval
        del obs.intervals[frame_intervals]

        return output

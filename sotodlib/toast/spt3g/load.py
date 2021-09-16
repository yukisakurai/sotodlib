# Copyright (c) 2020-2021 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import io

import re

import datetime

import numpy as np

from astropy import units as u

import h5py

import toast

from toast.utils import Environment, Logger, import_from_name

from toast.timing import function_timer

from toast.instrument import GroundSite

from toast.weather import SimWeather

from toast.spt3g import (
    available,
    from_g3_array_type,
    from_g3_scalar_type,
    from_g3_unit,
    from_g3_time,
    from_g3_quats,
)

from toast.spt3g.spt3g_import import (
    import_detdata,
    import_shared,
    import_intervals,
)

if available:
    from spt3g import core as c3g

from ..focalplane import SOFocalplane


class ImportSimMeta(object):
    """Import metadata from Observation and Calibration frames.

    A list of frames is processed and the returned information consists of the metadata
    needed to construct an observation.

    """

    def __init__(self):
        pass

    @function_timer
    def __call__(self, frames):
        """Process metadata frames.

        Args:
            frames (list):  The list of frames.

        Returns:
            (tuple):  The (observation name, observation UID, observation meta
                dictionary, observation det sets, Telescope, list of noise models)

        """
        name = None
        uid = None
        meta = dict()
        site = None
        focalplane = None
        telescope_class = None
        telescope_name = None
        telescope_uid = None
        telescope = None
        noise = list()
        detsets = None
        for frm in frames:
            if frm.type == c3g.G3FrameType.Observation:
                name = from_g3_scalar_type(frm["observation_name"])
                uid = from_g3_scalar_type(frm["observation_uid"])
                detsets = list()
                for dset in frm["observation_detector_sets"]:
                    detsets.append(list(dset))
                site_class = import_from_name(from_g3_scalar_type(frm["site_class"]))

                weather_name = None
                weather_realization = None
                weather_max_pwv = None
                weather_time = None
                if "site_weather_name" in frm:
                    weather_name = from_g3_scalar_type(frm["site_weather_name"])
                    weather_realization = from_g3_scalar_type(
                        frm["site_weather_realization"]
                    )
                    weather_max_pwv = from_g3_scalar_type(
                        frm["site_weather_max_pwv"]
                    )
                    weather_time = datetime.datetime.fromtimestamp(
                        from_g3_time(frm["site_weather_time"]),
                        tz=datetime.timezone.utc,
                    )
                site_uid = from_g3_scalar_type(frm["site_uid"])
                weather = None
                if weather_name is not None:
                    weather = SimWeather(
                        name=weather_name,
                        time=weather_time,
                        site_uid=site_uid,
                        realization=weather_realization,
                        max_pwv=weather_max_pwv,
                    )
                site = site_class(
                    from_g3_scalar_type(frm["site_name"]),
                    from_g3_scalar_type(frm["site_lat_deg"]) * u.degree,
                    from_g3_scalar_type(frm["site_lon_deg"]) * u.degree,
                    from_g3_scalar_type(frm["site_alt_m"]) * u.meter,
                    uid=site_uid,
                    weather=weather,
                )

                telescope_class = import_from_name(
                    from_g3_scalar_type(frm["telescope_class"])
                )
                telescope_name = from_g3_scalar_type(frm["telescope_name"])
                telescope_uid = from_g3_scalar_type(frm["telescope_uid"])
                meta = dict()
                unit_match = re.compile(r"^.*_units$")
                for f_key, f_val in frm.items():
                    if f_key in self._obs_reserved:
                        continue
                    if unit_match.match(f_key) is not None:
                        # This is a unit value for some other object
                        continue
                    if f_key in self._meta_arrays:
                        # This is an array we are importing
                        dt = from_g3_array_type(f_val)
                        meta[self._meta_arrays[f_key]] = np.array(f_val, dtype=dt)
                    else:
                        try:
                            l = len(f_val)
                            # This is an array
                        except Exception:
                            # This is a scalar (no len defined)
                            unit_key = f"{f_key}_units"
                            aunit_key = f"{f_key}_astropy_units"
                            unit_val = None
                            aunit = None
                            if unit_key in frm:
                                unit_val = frm[unit_key]
                                aunit = u.Unit(str(frm[aunit_key]))
                            try:
                                meta[f_key] = from_g3_scalar_type(f_val, unit_val).to(
                                    aunit
                                )
                            except Exception:
                                # This is not a datatype we can convert
                                pass

            elif frm.type == c3g.G3FrameType.Calibration:
                # Extract the focalplane and noise models
                byte_reader = io.BytesIO(np.array(frm["focalplane"], dtype=np.uint8))
                with h5py.File(byte_reader, "r") as f:
                    focalplane = SOFocalplane(file=f)
                del byte_reader

        telescope = telescope_class(
            telescope_name, uid=telescope_uid, focalplane=focalplane, site=site
        )
        return (name, uid, meta, detsets, telescope, dict())


class ImportSimData(object):
    """Import Scan frames to an observation.

    Shared objects:  The `shared_names` list of tuples specifies the mapping from Scan
    frame key to TOAST shared key.  The data type will be converted to the most
    appropriate TOAST dtype.  Only sample-wise data is currently supported.  The format
    of each tuple in the list is:

        (frame key, observation shared key)

    DetData objects:  The `det_names` list of tuples specifies the mapping from Scan
    frame values to TOAST detdata objects.  `G3TimestreamMap` objects are assumed to
    have one element per sample.  Any `G3Map*` objects will assumed to be flat-packed
    and will be reshaped so that the leading dimension is number of samples.  The format
    of each tuple in the list is:

        (frame key, observation detdata key)

    Intervals objects:  The `interval_names` list of tuples specifies the mapping from
    Scan frame values to TOAST intervals.  The frame values can be either
    `IntervalsTime` objects filled with the start / stop times of each interval, or
    flat-packed start / stop times in a `G3VectorTime` object.  The format of each
    tuple in the list is:

        (frame key, observation intervals key)

    Args:
        timestamp_names (tuple):  The name of the shared data containing the
            timestamps, and the output frame key to use.
        shared_names (list):  The observation shared objects to import.
        det_names (list):  The observation detdata objects to import.
        interval_names (list):  The observation intervals to import.

    """

    def __init__(
        self,
        timestamp_names=("times", "times"),
        shared_names=list(
            ("boresight_azel", "boresight_azel"),
            ("boresight_radec", "boresight_radec"),
            ("position", "position"),
            ("velocity", "velocity"),
            ("azimuth", "azimuth"),
            ("elevation", "elevation"),
            ("hwp_angle", "hwp_angle"),
            ("telescope_flags", "flags"),
        ),
        det_names=list(
            ("signal", "signal"),
            ("detector_flags", "flags"),
        ),
        interval_names=list(
            ("intervals_scan_leftright", "scan_leftright"),
            ("intervals_turn_leftright", "turn_leftright"),
            ("intervals_scan_rightleft", "scan_rightleft"),
            ("intervals_turn_rightleft", "turn_rightleft"),
            ("intervals_elnod", "elnod"),
            ("intervals_scanning", "scanning"),
            ("intervals_turnaround", "turnaround"),
            ("intervals_sun_up", "sun_up"),
            ("intervals_sun_close", "sun_close"),
        ),
    ):
        self._timestamp_names = timestamp_names
        self._shared_names = shared_names
        self._det_names = det_names
        self._interval_names = interval_names

    @function_timer
    def __call__(self, obs, frames):
        log = Logger.get()
        # Sanity check that the lengths of the frames correspond the number of local
        # samples.
        frame_total = np.sum([len(x[self._timestamp_names[0]]) for x in frames])
        if frame_total != obs.n_local_samples:
            msg = f"Process {obs.comm_rank} has {obs.n_local_samples} local samples, "
            msg += f"but is importing Scan frames with a total length of {frame_total}."
            log.error(msg)
            raise RuntimeError(msg)
        if frame_total == 0:
            return

        # Using the data types from the first frame, create the observation objects that
        # we will populate.

        # Timestamps are required
        frame_times, obs_times = self._timestamp_names
        obs.shared.create(
            obs_times,
            (obs.n_local_samples,),
            dtype=np.float64,
            comm=obs.comm_col,
        )
        frame_zero_samples = len(frames[0][self._timestamp_names[0]])
        for frame_field, obs_field in self._shared_names:
            dt = None
            nnz = None
            if isinstance(frames[0][frame_field], c3g.G3VectorQuat):
                dt = np.float64
                nnz = 4
            else:
                dt = from_g3_array_type(frames[0][frame_field])
                nnz = len(frames[0][frame_field]) // frame_zero_samples
            sshape = (obs.n_local_samples,)
            if nnz > 1:
                sshape = (obs.n_local_samples, nnz)
            obs.shared.create(
                obs_field,
                sshape,
                dtype=dt,
                comm=obs.comm_col,
            )

        for frame_field, obs_field in self._det_names:
            det_type_name = f"{frame_field}_dtype"
            dt = None
            if det_type_name in frames[0]:
                dt = np.dtype(str(frames[0][det_type_name]))
            else:
                dt = from_g3_array_type(frames[0][frame_field])
            units = u.dimensionless_unscaled
            if isinstance(frames[0][frame_field], c3g.G3TimestreamMap):
                check_units_name = (
                    f"compress_{frame_field}_{obs.local_detectors[0]}_units"
                )
                # If the compressed units name for the first detector is in the frame,
                # that means that we are (1) using compression and (2) the original
                # timestream had units (not just counts / dimensionless).
                if check_units_name in frames[0]:
                    units = from_g3_unit(frames[0][check_units_name])
                else:
                    units = from_g3_unit(frames[0][frame_field].units)
            nnz = len(frames[0][frame_field]) // frame_zero_samples
            dshape = None
            if nnz > 1:
                dshape = (nnz,)
            obs.detdata.create(obs_field, sample_shape=dshape, dtype=dt, units=units)

        for frame_field, obs_field in self._interval_names:
            obs.intervals.create_col(obs_field, list(), obs.shared[obs_times])

        # Go through each frame and copy the shared and detector data into the
        # observation.

        offset = 0
        for frm in frames:
            # Copy timestamps and shared data.  Because the data is explicitly
            # distributed in the sample direction, we know that there is only one
            # process accessing the data for each time slice
            import_shared(obs, obs_times, offset, frm[frame_times])
            for frame_field, obs_field in self._shared_names:
                import_shared(obs, obs_field, offset, frm[frame_field])

            # Copy detector data
            for frame_field, obs_field in self._det_names:
                comp = None
                # See if we have compression parameters for this object
                comp_root = f"compress_{frame_field}"
                for d in obs.local_detectors:
                    comp_gain_name = f"{comp_root}_{d}_gain"
                    comp_offset_name = f"{comp_root}_{d}_offset"
                    comp_units_name = f"{comp_root}_{d}_units"
                    if comp_offset_name in frm:
                        # This detector is compressed
                        if comp is None:
                            comp = dict()
                        comp[d] = dict()
                        comp[d]["offset"] = float(frm[comp_offset_name])
                        comp[d]["gain"] = float(frm[comp_gain_name])
                        comp[d]["units"] = c3g.G3TimestreamUnits(frm[comp_units_name])
                import_detdata(
                    obs, obs_field, offset, frm[frame_field], compression=comp
                )
            offset += len(frm[frame_times])

        # Now that we have the full set of timestamps in the observation, we
        # can construct our intervals.

        offset = 0
        sample_spans = list()
        for frm in frames:
            nsamp = len(frm[frame_times])
            sample_spans.append((offset, offset + nsamp - 1))
            for frame_field, obs_field in self._interval_names:
                import_intervals(obs, obs_field, obs_times, frm[frame_field])
            offset += nsamp

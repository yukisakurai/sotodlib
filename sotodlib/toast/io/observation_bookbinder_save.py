# Copyright (c) 2020-2021 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Tools for saving data in bookbinder format.

"""

import os
import sys

import numpy as np

from astropy import units as u

import h5py

# Import so3g first so that it can control the import and monkey-patching
# of spt3g.  Then our import of spt3g_core will use whatever has been imported
# by so3g.
import so3g
from spt3g import core as c3g

import toast
import toast.spt3g as t3g

from ...__init__ import __version__ as sotodlib_version


@toast.timing.function_timer
def save_bookbinder_shared(obs, name, view_name=None, view_index=0, g3t=None):
    """Convert a single shared object to a G3Object.

    If the G3 data type is not specified, a guess will be made at the closest
    appropriate type.

    Args:
        obs (Observation):  The parent observation.
        name (str):  The name of the shared object.
        view_name (str):  If specified, use this view of the shared object.
        view_index (int):  Export this element of the list of data views.
        g3t (G3Object):  The specific G3Object type to use, or None.

    Returns:
        (G3Object):  The resulting G3 object.

    """
    if name not in obs.shared:
        raise KeyError(f"Shared object '{name}' does not exist in observation")
    if g3t is None:
        g3t = t3g.to_g3_array_type(obs.shared[name].dtype)

    sview = obs.shared[name].data
    if view_name is not None:
        sview = np.array(obs.view[view_name].shared[name][view_index], copy=False)

    if g3t == c3g.G3VectorTime:
        return t3g.to_g3_time(sview)
    elif g3t == c3g.G3VectorQuat:
        return t3g.to_g3_quats(sview)
    else:
        return g3t(sview.flatten().tolist())


@toast.timing.function_timer
def save_bookbinder_detdata(
    obs, name, view_name=None, view_index=0, dtype=None, times=None, options=None
):
    """Convert a single detdata object to a G3SuperTimestream.

    If the output dtype is not specified, the best type will be chosen based on the
    existing detdata dtype.

    Args:
        obs (Observation):  The parent observation.
        name (str):  The name of the detdata object.
        view_name (str):  If specified, use this view of the detdata object.
        view_index (int):  Export this element of the list of data views.
        dtype (numpy.dtype):  Override the output dtype.
        times (str):  Use this shared name for the timestamps.
        options (dict):  If not None, these will be passed to the G3SuperTimestream
            options() method.

    Returns:
        (G3SuperTimestream, G3Units):  The resulting G3 object and the units.

    """
    if name not in obs.detdata:
        raise KeyError(f"DetectorData object '{name}' does not exist in observation")

    # Find the G3 equivalent units and scale factor needed to get the data into that
    gunit, scale = t3g.to_g3_unit(obs.detdata[name].units)

    # Find the best supported dtype
    if dtype is None:
        ch = obs.detdata[name].dtype.char
        if ch == "f":
            dtype = np.float32
        elif ch == "d":
            dtype = np.float64
        elif ch == "l" or ch == "L":
            dtype = np.int64
        elif ch in ["i", "I", "h", "H", "b", "B"]:
            dtype = np.int32
        else:
            raise RuntimeError(f"Unsupported timestream data type '{ch}'")

    # Get the view of the data, either the whole observation or one interval
    dview = obs.detdata[name]
    tview = obs.shared[times]
    if view_name is not None:
        dview = obs.view[view_name].detdata[name][view_index]
        tview = np.array(obs.view[view_name].shared[times][view_index], copy=False)

    out = so3g.G3SuperTimestream()
    out.names = dview.detectors
    out.times = t3g.to_g3_time(tview)
    if dtype == np.float32 or dtype == np.float64:
        out.quanta = np.ones(len(dview.detectors))
    out.data = scale * dview[:].astype(dtype)

    # Set any options
    if options is not None:
        out.options(**options)

    return out, gunit


@toast.timing.function_timer
def save_bookbinder_intervals(obs, name, iframe):
    """Convert the named intervals into a G3 object.

    Args:
        obs (Observation):  The parent observation.
        name (str):  The name of the intervals.
        iframe (IntervalList):  An interval list defined for this frame.

    Returns:
        (G3Object):  An IntervalsTime object.

    """
    overlap = iframe & obs.intervals[name]

    out = c3g.IntervalsTime(
        [(t3g.to_g3_time(x.start), t3g.to_g3_time(x.stop)) for x in overlap]
    )
    return out


class save_bookbinder_obs_meta(object):
    """Default class to export Observation metadata.

    In the bookbinder format we have G3 data that consists of a simple observation
    frame and a stream of Scan frames.  The scan frames have detector data indexed
    by readout channel.  The other detector properties and mapping from from detector
    to readout are contained in an HDF5 file that is located in the same directory
    as the frame files for an observation.

    Args:
        out_dir (str):  The output directory.
        meta_file (str):  The base filename to write

    """

    def __init__(self, out_dir=None, meta_file="metadata.h5"):
        self.out_dir = out_dir
        self._meta_file = meta_file

    @toast.timing.function_timer
    def __call__(self, obs):
        log = toast.Logger.get()
        log.verbose(f"Create observation frame and HDF5 file for {obs.name} in {dir}")

        # Construct observation frame
        ob = self._create_obs_frame(obs)

        # Write hdf5 file
        self._create_meta_file(obs, os.path.join(self._out_dir, self._meta_file))

        return ob

    def _create_obs_frame(self, obs):
        # Construct observation frame
        ob = c3g.G3Frame(c3g.G3FrameType.Observation)
        ob["observation_name"] = c3g.G3String(obs.name)
        ob["observation_uid"] = c3g.G3Int(obs.uid)
        ob["observation_n_channels"] = c3g.G3Int(len(obs.all_detectors))
        ob["observation_n_samples"] = c3g.G3Int(obs.n_all_samples)
        ob["telescope_name"] = c3g.G3String(obs.telescope.name)
        ob["telescope_class"] = c3g.G3String(
            toast.utils.object_fullname(obs.telescope.__class__)
        )
        ob["telescope_uid"] = c3g.G3Int(obs.telescope.uid)
        site = obs.telescope.site
        ob["site_name"] = c3g.G3String(site.name)
        ob["site_class"] = c3g.G3String(toast.utils.object_fullname(site.__class__))
        ob["site_uid"] = c3g.G3Int(site.uid)
        ob["site_lat_deg"] = c3g.G3Double(site.earthloc.lat.to_value(u.degree))
        ob["site_lon_deg"] = c3g.G3Double(site.earthloc.lon.to_value(u.degree))
        ob["site_alt_m"] = c3g.G3Double(site.earthloc.height.to_value(u.meter))

        # Export whatever other metadata we can.  Not all information can be
        # easily stored in a frame, so the HDF5 file will have the full set.
        for m_key, m_val in obs.items():
            try:
                l = len(m_val)
                # This is an array
                ob[m_key] = t3g.to_g3_array_type(m_val)
            except Exception:
                # This is a scalar (no len defined)
                try:
                    ob[m_key], m_unit = t3g.to_g3_scalar_type(m_val)
                    if m_unit is not None:
                        ob[f"{m_key}_astropy_units"] = c3g.G3String(f"{m_val.unit}")
                        ob[f"{m_key}_units"] = m_unit
                except Exception:
                    # This is not a datatype we can convert
                    pass
        return ob

    def _create_meta_file(self, obs, path):
        log = toast.Logger.get()
        if os.path.isfile(path):
            raise RuntimeError(f"Metadata file '{path}' already exists")

        path_temp = f"{path}.tmp"
        if os.path.isfile(path_temp):
            os.remove(path_temp)
        with h5py.File(path_temp, "w") as hf:
            # Record the software versions and config
            hf.attrs["software_version_so3g"] = so3g.__version__
            hf.attrs["software_version_sotodlib"] = sotodlib_version
            toast_env = toast.Environment.get()
            hf.attrs["software_version_toast"] = toast_env.version()

            # Observation properties
            hf.attrs["observation_name"] = obs.name
            hf.attrs["observation_uid"] = obs.uid
            hf.attrs["observation_n_channels"] = len(obs.all_detectors)
            hf.attrs["observation_n_samples"] = obs.n_all_samples
            # FIXME:  what other information would be useful at the top
            # level?  Maybe start time?

            # Instrument properties
            inst_group = hf.create_group("instrument")
            inst_group.attrs["telescope_name"] = obs.telescope.name
            inst_group.attrs["telescope_class"] = object_fullname(
                obs.telescope.__class__
            )
            inst_group.attrs["telescope_uid"] = obs.telescope.uid
            site = obs.telescope.site
            inst_group.attrs["site_name"] = site.name
            inst_group.attrs["site_class"] = toast.utils.object_fullname(site.__class__)
            inst_group.attrs["site_uid"] = site.uid
            inst_group.attrs["site_lat_deg"] = site.earthloc.lat.to_value(u.degree)
            inst_group.attrs["site_lon_deg"] = site.earthloc.lon.to_value(u.degree)
            inst_group.attrs["site_alt_m"] = site.earthloc.height.to_value(u.meter)

            obs.telescope.focalplane.save_hdf5(inst_group, comm=None, force_serial=True)
            del inst_group

            # Track metadata that has already been dumped
            meta_done = set()

            # Dump additional simulation data such as noise models, weather model, etc
            # to a separate group.
            sim_group = hf.create_group("simulation")
            if site.weather is not None:
                if hasattr(site.weather, "name"):
                    # This is a simulated weather object, dump it.
                    sim_group.attrs["site_weather_name"] = str(site.weather.name)
                    sim_group.attrs[
                        "site_weather_realization"
                    ] = site.weather.realization
                    if site.weather.max_pwv is None:
                        sim_group.attrs["site_weather_max_pwv"] = "NONE"
                    else:
                        sim_group.attrs["site_weather_max_pwv"] = site.weather.max_pwv
                    sim_group.attrs["site_weather_time"] = site.weather.time.timestamp()
            for k, v in obs.items():
                if isinstance(v, toast.noise.Noise):
                    kgroup = sim_group.create_group(k)
                    kgroup.attrs["class"] = toast.utils.object_fullname(v.__class__)
                    v.save_hdf5(kgroup, comm=None, force_serial=True)
                    del kgroup
                    meta_done.add(k)
            del sim_group

            # Other arbitrary metadata
            meta_group = hf.create_group("metadata")
            for k, v in obs.items():
                if k in meta_done:
                    continue
                if hasattr(v, "save_hdf5"):
                    kgroup = meta_group.create_group(k)
                    kgroup.attrs["class"] = toast.utils.object_fullname(v.__class__)
                    v.save_hdf5(kgroup, comm=None, force_serial=True)
                    del kgroup
                elif isinstance(v, u.Quantity):
                    if isinstance(v.value, np.ndarray):
                        # Array quantity
                        qdata = meta_group.create_dataset(k, data=v.value)
                        qdata.attrs["units"] = v.unit.to_string()
                        del qdata
                    else:
                        # Must be a scalar
                        meta_group.attrs[f"{k}"] = v.value
                        meta_group.attrs[f"{k}_units"] = v.unit.to_string()
                elif isinstance(v, np.ndarray):
                    marr = meta_group.create_dataset(k, data=v)
                    del marr
                else:
                    try:
                        meta_group.attrs[k] = v
                    except ValueError as e:
                        msg = f"Failed to store obs key '{k}' = '{v}' as an attribute "
                        msg += f"({e}).  Try casting it to a supported type when "
                        msg += f"storing in the observation dictionary or implement "
                        msg += f"save_hdf5() and load_hdf5() methods."
                        log.warning(msg)
            del meta_group

        # Move the file into place
        os.rename(path_temp, path)


class save_bookbinder_obs_data(object):
    """Class to export Scan frames.

    Shared objects:  The `shared_names` list of tuples specifies the TOAST shared key,
    corresponding Scan frame key, and optionally the G3 datatype to use.  Each process
    will duplicate shared data into their Scan frame stream.  If the G3 datatype is
    None, the closest G3 object type will be chosen.  If the shared object contains
    multiple values per sample, these are reshaped into a flat-packed array.  Only
    sample-wise shared objects are supported at this time (i.e. no other shared objects
    like beams, etc).  One special case:  The `timestamps` field will always be copied
    to each Scan frame as a `G3Timestream`.

    DetData objects:  The `det_names` list of tuples specifies the TOAST detdata key,
    the corresponding Scan frame key, and optionally the numpy dtype and compression
    options.  The data is exported to G3SuperTimestream objects.

    Intervals objects:  The `interval_names` list of tuples specifies the TOAST
    interval name and associated Scan frame key.  We save these to `IntervalsTime`
    objects filled with the start / stop times of each interval.

    Args:
        timestamp_names (tuple):  The name of the shared data containing the
            timestamps, and the output frame key to use.
        frame_intervals (str):  The name of the intervals to use for frame boundaries.
            If not specified, the observation sample sets are used.
        shared_names (list):  The observation shared objects to export.
        det_names (list):  The observation detdata objects to export.
        interval_names (list):  The observation intervals to export.

    """

    def __init__(
        self,
        timestamp_names=("times", "times"),
        frame_intervals=None,
        shared_names=list(),
        det_names=list(),
        interval_names=list(),
    ):
        self._timestamp_names = timestamp_names
        self._frame_intervals = frame_intervals
        self._shared_names = shared_names
        self._det_names = det_names
        self._interval_names = interval_names

    @toast.timing.function_timer
    def __call__(self, obs):
        log = toast.utils.Logger.get()
        frame_intervals = self._frame_intervals
        if frame_intervals is None:
            # We are using the sample set distribution for our frame boundaries.
            frame_intervals = "frames"
            timespans = list()
            offset = 0
            n_frames = 0
            first_set = obs.dist.samp_sets[obs.comm.group_rank].offset
            n_set = obs.dist.samp_sets[obs.comm.group_rank].n_elem
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
            msg = f"Create scan frame {obs.name}:{ivw} with fields:"
            msg += f"\n  shared:  {self._timestamp_names[1]}"
            nms = ", ".join([y for x, y, z in self._shared_names])
            msg += f", {nms}"
            nms = ", ".join([x for w, x, y, z in self._det_names])
            msg += f"\n  detdata:  {nms}"
            nms = ", ".join([y for x, y in self._interval_names])
            msg += f"\n  intervals:  {nms}"
            log.verbose(msg)
            # Construct the Scan frame
            frame = c3g.G3Frame(c3g.G3FrameType.Scan)
            # Add timestamps
            frame[self._timestamp_names[1]] = save_bookbinder_shared(
                obs,
                self._timestamp_names[0],
                view_name=frame_intervals,
                view_index=ivw,
                g3t=c3g.G3VectorTime,
            )
            for shr_key, shr_val, shr_type in self._shared_names:
                frame[shr_val] = save_bookbinder_shared(
                    obs,
                    shr_key,
                    view_name=frame_intervals,
                    view_index=ivw,
                    g3t=shr_type,
                )
            for det_key, det_val, det_type, det_opts in self._det_names:
                frame[det_val], gunits = save_bookbinder_detdata(
                    obs,
                    det_key,
                    view_name=frame_intervals,
                    view_index=ivw,
                    g3t=det_type,
                    times=self._timestamp_names[0],
                    options=det_opts,
                )
                # Record the original detdata type, so that it can be reconstructed
                # later.
                det_type_name = f"{det_val}_dtype"
                frame[det_type_name] = c3g.G3String(obs.detdata[det_key].dtype.char)

            # If we are exporting intervals, create an interval list with a single
            # interval for this frame.  Then use this repeatedly in the intersection
            # calculation.
            if len(self._interval_names) > 0:
                tview = obs.view[frame_intervals].shared[self._timestamp_names[0]][ivw]
                iframe = toast.intervals.IntervalList(
                    obs.shared[self._timestamp_names[0]],
                    timespans=[(tview[0], tview[-1])],
                )
                for ivl_key, ivl_val in self._interval_names:
                    frame[ivl_val] = save_bookbinder_intervals(
                        obs,
                        ivl_key,
                        iframe,
                    )
            output.append(frame)
        # Delete our temporary frame interval if we created it
        if self._frame_intervals is None:
            del obs.intervals[frame_intervals]

        return output

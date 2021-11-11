# Copyright (c) 2020-2021 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Operator for interfacing with the Maximum Likelihood Mapmaker.

"""

import os

import numpy as np

import traitlets

from astropy import units as u

from pixell import enmap

import toast
from toast.traits import trait_docs, Unicode, Int, Instance, Bool
from toast.ops import Operator
from toast.utils import Logger, Environment, rate_from_times
from toast.timing import function_timer, Timer
from toast.observation import default_values as defaults
from toast.fft import FFTPlanReal1DStore

import so3g

from ... import mapmaking as mm
from ...core import AxisManager, IndexAxis, OffsetAxis, LabelAxis


@trait_docs
class MLMapmaker(Operator):
    """Operator which accumulates data to the Maximum Likelihood Mapmaker."""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    out_dir = Unicode(".", help="The output directory")

    area = Unicode(None, allow_none=True, help="Load the enmap geometry from this file")

    center_at = Unicode(
        None,
        allow_none=True,
        help="The format is [from=](ra:dec|name),[to=(ra:dec|name)],[up=(ra:dec|name|system)]",
    )

    comps = Unicode("T", help="Components (must be 'T', 'QU' or 'TQU')")

    Nmat = Instance(allow_none=True, klass=mm.Nmat, help="The noise matrix to use")

    dtype_map = Instance(
        klass=np.dtype, args=(np.float64,), help="Numpy dtype of map products"
    )

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    boresight = Unicode(
        defaults.boresight_azel, help="Observation shared key for boresight Az/El"
    )

    det_data = Unicode(
        defaults.det_data, help="Observation detdata key for the timestream data"
    )

    det_flags = Unicode(
        None, allow_none=True, help="Observation detdata key for flags to use"
    )

    det_flag_mask = Int(
        defaults.det_mask_invalid, help="Bit mask value for optional detector flagging"
    )

    shared_flags = Unicode(
        None, allow_none=True, help="Observation shared key for telescope flags to use"
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid, help="Bit mask value for optional shared flagging"
    )

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    noise_model = Unicode(
        "noise_model", help="Observation key containing the noise model"
    )

    purge_det_data = Bool(
        False,
        help="If True, clear all observation detector data after accumulating",
    )

    verbose = Int(
        1,
        allow_none=True,
        help="Set verbosity in MLMapmaker.  If None, use toast loglevel",
    )

    @traitlets.validate("comps")
    def _check_mode(self, proposal):
        check = proposal["value"]
        if check not in ["T", "QU", "TQU"]:
            raise traitlets.TraitError("Invalid comps (must be 'T', 'QU' or 'TQU')")
        return check

    @traitlets.validate("shared_flag_mask")
    def _check_shared_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Shared flag mask should be a positive integer")
        return check

    @traitlets.validate("det_flag_mask")
    def _check_det_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det flag mask should be a positive integer")
        return check

    @traitlets.validate("verbose")
    def _check_params(self, proposal):
        check = proposal["value"]
        if check is None:
            # Set verbosity from the toast loglevel
            env = Environment.get()
            level = env.log_level()
            if level == "VERBOSE":
                check = 3
            elif level == "DEBUG":
                check = 2
            else:
                check = 1
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mapmaker = None

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        if self.area is None:
            raise RuntimeError("You must set the 'area' trait before calling exec")

        if self._mapmaker is None:
            # First call- create the mapmaker instance.
            # Get the timestream dtype from the first observation
            self._dtype_tod = data.obs[0].detdata[self.det_data].dtype

            self._shape, self._wcs = enmap.read_map_geometry(self.area)

            self._recenter = None
            if self.center_at is not None:
                self._recenter = mm.parse_recentering(self.center_at)

            self._mapmaker = mm.MLMapmaker(
                self._shape,
                self._wcs,
                comps=self.comps,
                noise_model=mm.NmatDetvecs(
                    verbose=(self.verbose > 1),
                    downweight=[1e-4, 0.25, 0.50],
                    window=0,
                ),
                #noise_model=mm.NmatUncorr(),
                #dtype_tod=self._dtype_tod,
                dtype_tod=np.float32,
                dtype_map=self.dtype_map,
                comm=data.comm.comm_world,
                recenter=self._recenter,
                verbose=self.verbose,
            )

        for ob in data.obs:
            # Get the detectors we are using locally for this observation
            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            # Get the sample rate from the data.  We also have nominal sample rates
            # from the noise model and also from the focalplane.
            (rate, dt, dt_min, dt_max, dt_std) = rate_from_times(
                ob.shared[self.times].data
            )

            # Get the focalplane for this observation
            fp = ob.telescope.focalplane

            # Prepare data for the mapmaker.

            axdets = LabelAxis("dets", dets)

            axsamps = OffsetAxis(
                "samps",
                count=ob.n_local_samples,
                offset=ob.local_index_offset,
                origin_tag=ob.name,
            )

            # Convert the data view into a RangesMatrix
            ranges = so3g.proj.ranges.RangesMatrix.zeros((len(dets), int(ob.n_local_samples)))
            if self.view is not None:
                 view_ranges = np.array([[x.first, x.last + 1] for x in ob.intervals[self.view]])
                 ranges += so3g.proj.ranges.Ranges.from_array(view_ranges, ob.n_local_samples)

            # Convert the focalplane offsets into the expected form
            det_to_row = {y["name"]: x for x, y in enumerate(fp.detector_data)}
            det_quat = np.array([fp.detector_data["quat"][det_to_row[x]] for x in dets])
            det_theta, det_phi, det_pa = toast.qarray.to_angles(det_quat)

            radius = np.sin(det_theta)
            xi = - radius * np.cos(det_phi)
            eta = - radius * np.sin(det_phi)
            gamma = det_pa
            # for d in range(len(det_quat)):
            #     print(f"{d:03d}: {det_quat[d]}")
            #     print(f"  theta = {det_theta[d]}")
            #     print(f"  phi   = {det_phi[d]}")
            #     print(f"  pa    = {det_pa[d]}")
            #     print(f"  xi    = {xi[d]}")
            #     print(f"  eta   = {eta[d]}")
            #     print(f"  gamma = {gamma[d]}")

            axfp = AxisManager()
            axfp.wrap("xi", xi, axis_map=[(0, axdets)])
            axfp.wrap("eta", eta, axis_map=[(0, axdets)])
            axfp.wrap("gamma", gamma, axis_map=[(0, axdets)])

            # Convert Az/El quaternion of the detector back into
            # angles from the simulation.
            theta, phi, pa = toast.qarray.to_angles(ob.shared[self.boresight])

            # Azimuth is measured in the opposite direction from longitude
            az = 2 * np.pi - phi
            el = np.pi / 2 - theta
            roll = pa  # FIXME: double check this...

            axbore = AxisManager()
            axbore.wrap("az", az, axis_map=[(0, axsamps)])
            axbore.wrap("el", el, axis_map=[(0, axsamps)])
            axbore.wrap("roll", roll, axis_map=[(0, axsamps)])

            axobs = AxisManager()
            axobs.wrap("focal_plane", axfp)
            axobs.wrap("timestamps", ob.shared[self.times], axis_map=[(0, axsamps)])
            axobs.wrap(
                "signal",
                ob.detdata[self.det_data][dets, :],
                axis_map=[(0, axdets), (1, axsamps)],
            )
            axobs.wrap("boresight", axbore)
            axobs.wrap("glitch_flags", ranges, axis_map=[(0, axdets), (1, axsamps)])

            # NOTE:  Expected contents look like:
            # >>> tod
            # AxisManager(signal[dets,samps], timestamps[samps], readout_filter_cal[dets],
            # mce_filter_params[6], iir_params[3,5], flags*[samps], boresight*[samps],
            # array_data*[dets], pointofs*[dets], focal_plane*[dets], abscal[dets],
            # timeconst[dets], glitch_flags[dets,samps], source_flags[dets,samps],
            # relcal[dets], dets:LabelAxis(63), samps:OffsetAxis(372680))
            # >>> tod.focal_plane
            # AxisManager(xi[dets], eta[dets], gamma[dets], dets:LabelAxis(63))
            # >>> tod.boresight
            # AxisManager(az[samps], el[samps], roll[samps], samps:OffsetAxis(372680))

            # Accumulate data to mapmaker
            work = self._mapmaker.build_obs(ob.name, axobs)
            self._mapmaker.add_obs(work)
            del axobs

            # Optionally delete the input detector data to save memory, if
            # the calling code knows that no additional operators will be
            # used afterwards.
            if self.purge_det_data:
                del ob.detdata[self.det_data]

        return

    @function_timer
    def _finalize(self, data, **kwargs):
        # After multiple calls to exec, the finalize step will solve for the map.
        log = Logger.get()
        timer = Timer()
        comm = data.comm.comm_world
        timer.start()

        self._mapmaker.prepare()
        log.info_rank(
            f"MLMapmaker finished prepare in",
            comm=comm,
            timer=timer,
        )

        prefix = os.path.join(self.out_dir, f"{self.name}_")

        if comm.rank == 0:
            enmap.write_map(f"{prefix}rhs.fits", self._mapmaker.map_rhs)
            enmap.write_map(f"{prefix}div.fits", self._mapmaker.map_div)
            enmap.write_map(
                f"{prefix}bin.fits",
                enmap.map_mul(self._mapmaker.map_idiv, self._mapmaker.map_rhs),
            )

        if comm is not None:
            comm.barrier()
        log.info_rank(
            f"MLMapmaker finished writing rhs, div, bin in",
            comm=comm,
            timer=timer,
        )

        tstep = Timer()
        tstep.start()

        for step in self._mapmaker.solve():
            dump = step.i % 10 == 0
            dstr = ""
            if dump:
                dstr = "(write)"
            msg = f"CG step {step.i:4d} {step.err:15.7e} {dstr}"
            log.info_rank(
                f"MLMapmaker   {msg} ",
                comm=comm,
                timer=tstep,
            )
            if dump and comm.rank == 0:
                enmap.write_map(f"{prefix}map{step.i:04d}.fits", step.x)

        log.info_rank(
            f"MLMapmaker finished solve in",
            comm=comm,
            timer=timer,
        )

        if comm.rank == 0:
            enmap.write_map(f"{prefix}map.fits", step.x)

        if comm is not None:
            comm.barrier()
        log.info_rank(
            f"MLMapmaker wrote map in",
            comm=comm,
            timer=timer,
        )

    def _requires(self):
        req = {
            "meta": [self.noise_model],
            "shared": [
                self.times,
            ],
            "detdata": [self.det_data],
            "intervals": list(),
        }
        if self.view is not None:
            req["intervals"].append(self.view)
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        return req

    def _provides(self):
        prov = {"detdata": list()}
        if self.det_out is not None:
            prov["detdata"].append(self.det_out)
        return prov

    def _accelerators(self):
        return list()


# class NmatToast(mm.Nmat):
#     """Noise matrix class that uses a TOAST noise model.

#     This takes an existing TOAST noise model and uses it for a MLMapmaker compatible
#     noise matrix.

#     Args:
#         model (toast.Noise):  The toast noise model.
#         det_order (dict):  The mapping from detector order in the AxisManager
#             to name in the Noise object.

#     """
#     def __init__(self, model, n_sample, det_order):
#         self.model = model
#         self.det_order = det_order
#         self.n_sample = n_sample

#         # Compute the radix-2 FFT length to use
#         self.fftlen = 2
#         while self.fftlen <= self.n_sample:
#             self.fftlen *= 2
#         self.npsd = self.fftlen // 2 + 1

#         # Compute the time domain offset that centers our data within the
#         # buffer
#         self.padded_start = (self.fftlen - self.n_sample) // 2

#         # Compute the common frequency values
#         self.nyquist = model.freq(model.keys[0])[-1].to_value(u.Hz)
#         self.rate = 2 * self.nyquist
#         self.freqs = np.fft.rfftfreq(self.fftlen, 1 / self.rate))

#         # Interpolate the PSDs to desired spacing and store for later
#         # application.

#     def build(self, tod, **kwargs):
#         """Build method is a no-op, we do all set up in the constructor."""
#         return self

#     def apply(self, tod, inplace=False):
#         """Apply our noise filter to the TOD.

#         We use our pre-built Fourier domain kernels.

#         """
#         return tod
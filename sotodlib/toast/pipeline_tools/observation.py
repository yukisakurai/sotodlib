# Copyright (c) 2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import copy

import numpy as np

from toast.dist import distribute_uniform, Data
from toast.pipeline_tools import get_breaks, load_schedule
from toast.timing import function_timer, Timer
from toast.todmap import TODGround
from toast.utils import Logger
from toast.weather import Weather

from .hardware import get_hardware, get_focalplane


def add_import_args(parser):
    parser.add_argument(
        "--import-dir",
        required=False,
        help="Directory to load TOD from",
    )
    parser.add_argument(
        "--import-obs",
        required=False,
        help="Comma-separated list of observations to load.  Default is to load "
        "all observations in --import-dir",
    )
    parser.add_argument(
        "--import-prefix",
        required=False,
        help="Prefix for TOD files to import"
    )
    return


@function_timer
def create_observation(args, comm, telescope, ces, verbose=True):
    """ Create a TOAST observation.

    Create an observation for the CES scan defined by all_ces_tot[ices].

    """
    focalplane = telescope.focalplane
    site = telescope.site
    totsamples = int((ces.stop_time - ces.start_time) * args.sample_rate)

    # create the TOD for this observation

    if comm.comm_group is not None:
        ndetrank = comm.comm_group.size
    else:
        ndetrank = 1

    if args.el_nod_deg and (ces.subscan == 0 or args.el_nod_every_scan):
        el_nod = args.el_nod_deg
    else:
        el_nod = None

    try:
        tod = TODGround(
            comm.comm_group,
            focalplane.detquats,
            totsamples,
            detranks=ndetrank,
            firsttime=ces.start_time,
            rate=args.sample_rate,
            site_lon=site.lon,
            site_lat=site.lat,
            site_alt=site.alt,
            azmin=ces.azmin,
            azmax=ces.azmax,
            el=ces.el,
            el_nod=el_nod,
            start_with_elnod=args.start_with_el_nod,
            end_with_elnod=args.end_with_el_nod,
            el_mod_step=args.el_mod_step_deg,
            el_mod_rate=args.el_mod_rate_hz,
            el_mod_amplitude=args.el_mod_amplitude_deg,
            el_mod_sine=args.el_mod_sine,
            scanrate=args.scan_rate,
            scanrate_el=args.scan_rate_el,
            scan_accel=args.scan_accel,
            scan_accel_el=args.scan_accel_el,
            cosecant_modulation=args.scan_cosecant_modulate,
            CES_start=None,
            CES_stop=None,
            sun_angle_min=args.sun_angle_min,
            coord=args.coord,
            sampsizes=None,
            report_timing=args.debug,
            hwprpm=args.hwp_rpm,
            hwpstep=args.hwp_step_deg,
            hwpsteptime=args.hwp_step_time_s,
        )
    except RuntimeError as e:
        raise RuntimeError(
            'Failed to create TOD for {}-{}-{}: "{}"'
            "".format(ces.name, ces.scan, ces.subscan, e)
        )

    # Create the observation

    obs = {}
    obs["name"] = "CES-{}-{}-{}-{}-{}".format(
        site.name, telescope.name, ces.name, ces.scan, ces.subscan
    )
    obs["tod"] = tod
    obs["baselines"] = None
    obs["noise"] = copy.deepcopy(focalplane.noise)
    obs["id"] = int(ces.mjdstart * 10000)
    obs["intervals"] = tod.subscans
    obs["site"] = site
    obs["site_id"] = site.id
    obs["altitude"] = site.alt
    obs["telescope"] = telescope.name
    obs["telescope_id"] = telescope.id
    obs["fpradius"] = focalplane.radius
    obs["weather"] = copy.deepcopy(site.weather)
    obs["start_time"] = ces.start_time
    obs["season"] = ces.season
    obs["date"] = ces.start_date
    obs["MJD"] = ces.mjdstart
    obs["focalplane"] = focalplane.detector_data
    obs["rising"] = ces.rising
    obs["mindist_sun"] = ces.mindist_sun
    obs["mindist_moon"] = ces.mindist_moon
    obs["el_sun"] = ces.el_sun
    if telescope.name == "LAT":
        if args.corotate_lat:
            obs["corotator_angle_deg"] = -ces.el
        else:
            obs["corotator_angle_deg"] = -60
    return obs


@function_timer
def create_observations(args, comm, schedules):
    """ Create and distribute TOAST observations for every CES in schedules.

    """
    log = Logger.get()
    timer = Timer()
    timer.start()

    data = Data(comm)

    # Loop over the schedules, distributing each schedule evenly across
    # the process groups.  For now, we'll assume that each schedule has
    # the same number of operational days and the number of process groups
    # matches the number of operational days.  Relaxing these constraints
    # will cause the season break to occur on different process groups
    # for different schedules and prevent splitting the communicator.

    for schedule in schedules:

        telescope = schedule.telescope
        all_ces = schedule.ceslist
        nces = len(all_ces)

        breaks = get_breaks(comm, all_ces, args)

        groupdist = distribute_uniform(nces, comm.ngroups, breaks=breaks)
        group_firstobs = groupdist[comm.group][0]
        group_numobs = groupdist[comm.group][1]

        for ices in range(group_firstobs, group_firstobs + group_numobs):
            obs = create_observation(args, comm, telescope, all_ces[ices])
            data.obs.append(obs)

    if comm.comm_world is None or comm.comm_group.rank == 0:
        log.info("Group # {:4} has {} observations.".format(comm.group, len(data.obs)))

    if len(data.obs) == 0:
        raise RuntimeError(
            "Too many tasks. Every MPI task must "
            "be assigned to at least one observation."
        )

    if comm.comm_world is not None:
        comm.comm_world.barrier()
    timer.stop()
    if comm.world_rank == 0:
        timer.report("Simulated scans")

    # Split the data object for each telescope for separate mapmaking.
    # We could also split by site.

    if len(schedules) > 1:
        telescope_data = data.split("telescope")
        if len(telescope_data) == 1:
            # Only one telescope available
            telescope_data = []
    else:
        telescope_data = []
    telescope_data.insert(0, ("all", data))
    return data, telescope_data


def load_observations(args, comm):
    """Load existing data and put it in TOAST observations.
    """
    # This import is not at the top of the file to avoid
    # loading spt3g through so3g unnecessarily
    from ...io.toast_load import load_data
    log = Logger.get()
    if args.import_obs is not None:
        import_obs = args.import_obs.split(",")
    else:
        import_obs = None
    hw, telescope, det_index = get_hardware(args, comm, verbose=True)
    focalplane = get_focalplane(args, comm, hw, det_index, verbose=True)
    detweights = focalplane.detweights
    telescope.focalplane = focalplane

    if comm.world_rank == 0:
        log.info("Loading TOD from {}".format(args.import_dir))
    timer = Timer()
    timer.start()
    data = load_data(
        args.import_dir,
        obs=import_obs,
        comm=comm,
        prefix=args.import_prefix,
        dets=hw,
        detranks=comm.group_size,
        )
    if comm.world_rank == 0:
        timer.report_clear("Load data")
    telescope_data = [("all", data)]
    site = telescope.site
    focalplane = telescope.focalplane
    if args.weather is not None:
        weather = Weather(args.weather)
    else:
        weather = None
    for obs in data.obs:
        #obs["baselines"] = None
        obs["noise"] = focalplane.noise
        #obs["id"] = int(ces.mjdstart * 10000)
        #obs["intervals"] = tod.subscans
        obs["site"] = site.name
        obs["site_id"] = site.id
        obs["telescope"] = telescope.name
        obs["telescope_id"] = telescope.id
        obs["fpradius"] = focalplane.radius
        obs["weather"] = weather
        #obs["start_time"] = ces.start_time
        obs["altitude"] = site.alt
        #obs["season"] = ces.season
        #obs["date"] = ces.start_date
        #obs["MJD"] = ces.mjdstart
        obs["focalplane"] = focalplane.detector_data
        #obs["rising"] = ces.rising
        #obs["mindist_sun"] = ces.mindist_sun
        #obs["mindist_moon"] = ces.mindist_moon
        #obs["el_sun"] = ces.el_sun
    return data, telescope_data, detweights

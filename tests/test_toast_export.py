# Copyright (c) 2018-2021 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""Test toast data export.
"""

import os

import numpy as np

from unittest import TestCase

from ._helpers import create_outdir, simulation_test_data

from sotodlib.sim_hardware import get_example

from sotodlib.sim_hardware import sim_telescope_detectors

# Import so3g first so that it can control the import and monkey-patching
# of spt3g.  Then our import of spt3g_core will use whatever has been imported
# by so3g.
import so3g
from spt3g import core as core3g


toast_available = None
if toast_available is None:
    try:
        import toast
        from toast.mpi import get_world
        import toast.ops
        import sotodlib.toast as sotoast
        toast_available = True
    except ImportError:
        toast_available = False


class ToastBookbinderTest(TestCase):

    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        if not toast_available:
            print("TOAST cannot be imported- skipping unit tests", flush=True)
            return
        world, procs, rank = get_world()
        self.outdir = create_outdir(fixture_name, mpicomm=world)


    def test_bookbinder(self):
        if not toast_available:
            return
        world, procs, rank = get_world()
        testdir = os.path.join(self.outdir, "bookbinder")
        if world is None or world.rank == 0:
            os.makedirs(testdir)

        data = simulation_test_data(world)

        # Simulate some noise
        sim_noise = toast.ops.SimNoise()
        sim_noise.apply(data)



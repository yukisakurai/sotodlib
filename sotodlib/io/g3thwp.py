import numpy as np
import scipy.interpolate
import so3g

class G3tHWP(): 
    def __init__(self, archive_path, ratio=0.1):

        """
        Class to manage a HWP HK data.

        Args
        -----
            archive_path: path
                Path to the data directory
            ratio: reference slit parameter
                0.1 - 0.3 (10-30%, adjustment value)
        """

        self._archive_path = archive_path
        self._hwp_keys=[
            'observatory.HBA1.feeds.HWPEncoder.rising_edge_count',
            'observatory.HBA1.feeds.HWPEncoder.irig_time',
            'observatory.HBA1.feeds.HWPEncoder_full.counter',
            'observatory.HBA1.feeds.HWPEncoder_full.counter_index',
        ]
        self._alias=[key.split('.')[-1] for key in self._hwp_keys]
        
        # Size of pakcets sent from the BBB
        self._pkt_size = 120 # maybe 120 in the latest version, 150 in the previous version
        # Number of encoder slits per HWP revolution
        self._num_edges = 570*2
        self._delta_angle = 2 * np.pi / self._num_edges
        self._ref_edges = 2
        # Allowed jitter in slit width
        self._dev = ratio  # 10%-30%  ## fixme!!! sometimes reference point finding doesn't work!
        self._ref_indexes = None
        self._fast = True
        self._status = False
        
    def load_data(self, start, end, fast=True, show_status=False):

        """
        Loads smurf G3 data for a given time range. For the specified time range
        this will return a chunk of data that includes that time range.

        This function returns an array of IRIG timestamp and hwp angle

        Args
        -----
            start : timestamp or DateTime
                start time for data
            end :  timestamp  or DateTime
                end time for data
            show_status : bool, optional: 
                If True, will show status
        """
        if isinstance(start,np.datetime64): start = start.timestamp()
        if isinstance(end,np.datetime64): end = start.timestamp()
        # load housekeeping data with hwp keys
        if show_status: print('loading HK data files ...')
        if fast==False: self._fast = False
        if show_status: self._status = True
        data = so3g.hk.load_range(start, end, fields=self._hwp_keys, alias=self._alias, data_dir=self._archive_path)
        if show_status:	print('calculating HWP angle ...')

        if len(data['counter'][1]) == 0 or len(data['irig_time'][1]) == 0:
            raise ValueError('HWP is not spinning in this time window!')
        return self._hwp_angle_calculator(data['counter'][1], data['counter_index'][1], data['irig_time'][1], data['rising_edge_count'][1])


    def load_file(self, filename, fast=True, return_ref = False, show_status=False):

        """
        Loads smurf G3 data for a given filebames. 
        This function returns an array of IRIG timestamp and hwp angle

        Args
        -----
            filename : str or list
                A filename or list of filenames (to be loaded in order).
            show_status : bool, optional: 
                If True, will show status
            return_ref : bool, optional: 
                If True, will return reference slit indecies

            load_biases : bool, optional 
                If True, will return biases.
            status : SmurfStatus, optional
                If note none, will use this Status on the data load

        """
        
        # load housekeeping files with hwp keys
        if show_status: print('loading HK data files ...')
        if fast==False: self._fast = False
        scanner = so3g.hk.HKArchiveScanner()
        if isinstance(filename, list) or isinstance(filename, np.ndarray):
            for f in filename: scanner.process_file(self._archive_path + '/' + f)
        else: scanner.process_file(self._archive_path + '/' + filename)
        arc = scanner.finalize()
        for i in range(len(self._hwp_keys)):
            if not self._hwp_keys[i] in arc.get_fields()[0].keys():
                raise ValueError('HWP is not spinning in this file!')
        data = arc.simple(self._hwp_keys)
        if show_status:	print('calculating HWP angle ...')
        if len(data[0][1]) == 0 or len(data[2][1]) == 0:
            raise ValueError('HWP is not spinning in this file!')
        print('INFO: hwp angle calculation is finished.')

        return self._hwp_angle_calculator(data[0][1], data[1][1], data[2][1], data[3][1])

    def _hwp_angle_calculator(self, counter, counter_idx, irig_time, rising_edge):

        #   counter: BBB counter values for encoder signal edges
        self._encd_clk = counter
        #   counter_index: index numbers for detected edges by BBB
        self._encd_cnt = counter_idx
        #   irig_time: decoded time in second since the unix epoch
        self._irig_time = irig_time
        #   rising_edge_count: BBB clcok count values for the IRIG on-time reference marker risinge edge
        self._rising_edge = rising_edge

        # return arrays
        self._time = []
        self._angle = []

        # check packet drop
        self._encoder_packet_sort()
        self._find_dropped_packets()

        # reference finding and fill its angle
        self._find_refs()
        if self._fast: self._fill_refs_fast()
        else: self._fill_refs()

        # assign IRIG synched timestamp
        self._time = scipy.interpolate.interp1d(self._rising_edge, self._irig_time, kind='linear',fill_value='extrapolate')(self._encd_clk)
        # calculate hwp angle with IRIG timing
        self._calc_angle_linear()
        
        ###############
        if self._status:
            print('INFO :qualitycheck')
            print('_time:        ', len(self._time))
            print('_angle:       ', len(self._angle))
            print('_encd_cnt:    ', len(self._encd_cnt))
            print('_encd_clk:    ', len(self._encd_clk))
            print('_ref_cnt:     ', len(self._ref_cnt))
            print('_ref_indexes: ', len(self._ref_indexes))
        ###############

        if len(self._time) != len(self._angle):
            raise ValueError('Failed to calculate hwp angle!')
        print('INFO: hwp angle calculation is finished.')
        return self._time, self._angle
    
    def _find_refs(self):
        """ Find reference slits """
        # Calculate spacing between all clock values
        diff = np.ediff1d(self._encd_clk, to_begin=0)
        # Define median value as nominal slit distance
        self._slit_dist = np.median(diff)
        # Conditions for idenfitying the ref slit
        # Slit distance somewhere between 2 slits:
        # 2 slit distances (defined above) +/- 10%
        ref_hi_cond = ((self._ref_edges + 2) * self._slit_dist * (1 + self._dev))
        ref_lo_cond = ((self._ref_edges + 1) * self._slit_dist * (1 - self._dev))
        # Find the reference slit locations (indexes)
        self._ref_indexes = np.argwhere(np.logical_and(diff < ref_hi_cond, diff > ref_lo_cond)).flatten()
        print('INFO: found {} reference points'.format(len(self._ref_indexes)))
        # Define the reference slit line to be the line before
        # the two "missing" lines
        # Store the count and clock values of the reference lines
        self._ref_clk = np.take(self._encd_clk, self._ref_indexes)
        self._ref_cnt = np.take(self._encd_cnt, self._ref_indexes)
            
        return
        
    def _fill_refs(self):
        """ Fill in the reference edges """
        # If no references, say that the first sample is theta = 0
        # This case comes up for testing with a function generator
        if len(self._ref_clk) == 0:
            self._ref_clk = [self._encd_clk[0]]
            self._ref_cnt = [self._encd_cnt[0]]
            return
        # Loop over all of the reference slits
        for ii in range(len(self._ref_indexes)):
            print("\r {:.2f} %".format(100.*ii/len(self._ref_indexes)), end="")
            # Location of this slit
            ref_index = self._ref_indexes[ii]
            # Linearly interpolate the missing slits
            clks_to_add = np.linspace(self._encd_clk[ref_index-1], self._encd_clk[ref_index],self._ref_edges + 2)[1:-1]
            self._encd_clk = np.insert(self._encd_clk, ref_index, clks_to_add)
            # Adjust the encoder count values for the added lines
            # Add 2 to all future counts and interpolate the counts
            # for the two added slits
            self._encd_cnt[ref_index:] += self._ref_edges
            cnts_to_add = np.linspace(self._encd_cnt[ref_index-1], self._encd_cnt[ref_index], self._ref_edges + 2)[1:-1]
            self._encd_cnt = np.insert(self._encd_cnt, ref_index, cnts_to_add)
            # Also adjsut the reference count values in front of
            # this one for the added lines
            self._ref_cnt[ii+1:] += self._ref_edges
            # Adjust the reference index values in front of this one
            # for the added lines
            self._ref_indexes[ii+1:] += self._ref_edges
            #print(clks_to_add)
            #print(cnts_to_add)
            #print(self._ref_cnt, np.diff(self._ref_cnt), print(self._ref_indexes))
        return
    
    def _fill_refs_fast(self):
        """ Fill in the reference edges """
        # If no references, say that the first sample is theta = 0
        # This case comes up for testing with a function generator
        if len(self._ref_clk) == 0:
            self._ref_clk = [self._encd_clk[0]]
            self._ref_cnt = [self._encd_cnt[0]]
            return
        #insert interpolate clk to reference points
        lastsub = np.split(self._encd_clk, self._ref_indexes)[-1]
        self._encd_clk = np.concatenate(
            np.array(
                [[sub_clk, np.linspace(self._encd_clk[ref_index-1], self._encd_clk[ref_index], self._ref_edges + 2)[1:-1]]\
                 for ref_index, sub_clk\
                 in zip(self._ref_indexes, np.split(self._encd_clk, self._ref_indexes))],dtype=object
            ).flatten()
        )
        self._encd_clk = np.append(self._encd_clk, lastsub)
            
        self._encd_cnt = self._encd_cnt[0] + np.arange(self._encd_cnt.size + self._ref_indexes.size * self._ref_edges)
        self._ref_cnt += np.arange(self._ref_cnt.size)*self._ref_edges
        self._ref_indexes += np.arange(self._ref_indexes.size)*self._ref_edges
            
        return
    
    def _flatten_counter(self):
        cnt_diff = np.diff(self._encd_cnt)
        loop_indexes = np.argwhere(cnt_diff <= -(self._max_cnt-1)).flatten()
        for ind in loop_indexes:
            self._encd_cnt[(ind+1):] += -(cnt_diff[ind]-1)
        return
    
    def _calc_angle_linear(self):
        self._angle = (self._encd_cnt - self._ref_cnt[0]) * self._delta_angle % (2*np.pi)
        #self._angle = (self._encd_cnt - self._ref_cnt[0]) * self._delta_angle
        return

    def _find_dropped_packets(self):
        """ Estimate the number of dropped packets """
        cnt_diff = np.diff(self._encd_cnt)
        dropped_samples = np.sum(cnt_diff[cnt_diff >= self._pkt_size])
        self._num_dropped_pkts = dropped_samples // (self._pkt_size - 1)
        if self._num_dropped_pkts > 0:
            print('WARNING: {} dropped packets are found.'.format(self._num_dropped_pkts))
        return
    
    def _encoder_packet_sort(self):
        cnt_diff = np.diff(self._encd_cnt)
        if np.any(cnt_diff != 1):
            print('WARNING: counter is not correct')
            if np.any(cnt_diff < 0): 
                print('WARNING: counter is not incremental') 
                if 1-self._pkt_size in cnt_diff: print('packet flip is found')
                idx = np.argsort(self._encd_cnt)
                self._encd_clk = self._encd_clk[idx]
            else: print('WARNING: maybe packet drop exists')
        else: print('INFO: no need to fix encoder index')
                 
    def interp_smurf(self, smurf_timestamp):
        smurf_angle = scipy.interpolate.interp1d(self._time, self.angle, kind='linear',fill_value='extrapolate')(smurf_timestamp)
        return smurf_angle

def hwpss(angle, tsm, dsm, bins=128):
    hwpss_denom = np.histogram(angle, bins=bins, range=[0, 2*np.pi])[0]
    hwpss_num = np.histogram(angle, bins=bins, range=[0, 2*np.pi],weights=dsm)[0]
    hwpss = hwpss_num / hwpss_denom
    return np.linspace(0, 2*np.pi, bins), hwpss

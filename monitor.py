import numpy as np

from sm_api import monitor_app
from util import xapp_sdk as ric
import time

class monitor():

    def __init__(self,logfolder,uetoue=False):
        print("monior")
        if uetoue:
            self.average_count = 500
        else:
            self.average_count = 500
        self.logfolder=logfolder
        print("handle starting")
        self.monitor_handle = monitor_app.monitor()
        print("handled started") 
        self.mac_cb = self.monitor_handle.mac_cb

    def get_pusch(self,rnti):
        pusch_snr =-1
        if rnti in self.mac_cb.test:
            d = self.mac_cb.test[rnti]['pusch_snr'].copy()
            d = list(d)
            #d.rotate(self.average_count)
            pusch_snr = d[-self.average_count:]
            mean_snr = np.mean(pusch_snr)
        else:
            print(rnti)
        return mean_snr

    def get_pucch(self,rnti):
        pucch_snr = -1
        if rnti in self.mac_cb.test:
            pucch_snr = self.mac_cb.test[rnti]['pucch_snr'][-1]
        return pucch_snr
    def get_ul_mcs(self,rnti):
        ul_mcs1 = -1
        if rnti in self.mac_cb.test:
            ul_mcs1 = self.mac_cb.test[rnti]['ul_mcs1'][-1]
        return ul_mcs1
    
    def shutdown(self):

        self.monitor_handle.shutdown(self.logfolder)
        while ric.try_stop == 0:
            time.sleep(1)

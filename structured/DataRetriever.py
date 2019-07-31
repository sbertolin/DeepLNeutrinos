from enum import Enum
import numpy as np
import pandas as pd
import math as m
from pathlib import Path

class State(Enum):
    EVENT_ID = 1
    RUN_INFO = 2
    WEIGHTS = 3
    NEUTRINO = 4
    MUON = 5
    AAFIT = 6
    BBFIT_TRACK = 7
    BBFIT_BRIGHT = 8
    GRIDFIT = 9
    HITS = 10
    SELECTED_HITS = 11

class Hit:

    def __init__(self, det_line, floor, det_ori_elev, det_ori_az, det_pos_x, det_pos_y, det_pos_z, det_time, det_amp):
        self._order = 0
        self._det_line = det_line
        self._floor = floor
        self._det_ori_elev = det_ori_elev
        self._det_ori_az = det_ori_az
        self._det_pos_x = det_pos_x
        self._det_pos_y = det_pos_y
        self._det_pos_z = det_pos_z
        self._det_time = det_time
        self._det_amp = det_amp

    def convert_to_relative_time(self, hit_0):
        self._det_time = self._det_time - hit_0._det_time

    def convert_to_relative_amplitude(self, hit_0):
        if hit_0._det_amp != 0:
            self._det_amp = self._det_amp / hit_0._det_amp

    def set_order(self, order):
        self._order = order

    def flatten(self):
        ret_array = {}
        ret_array[str(self._order) + "_det_line"] = self._det_line
        ret_array[str(self._order) + "_floor"] = self._floor
        #ret_array[str(self._order) + "_det_ori_elev"] = self._det_ori_elev
        ret_array[str(self._order) + "_det_ori_az"] = self._det_ori_az
        ret_array[str(self._order) + "_det_pos_x"] = self._det_pos_x
        ret_array[str(self._order) + "_det_pos_y"] = self._det_pos_y
        ret_array[str(self._order) + "_det_pos_z"] = self._det_pos_z
        ret_array[str(self._order) + "_det_time"] = self._det_time
        ret_array[str(self._order) + "_det_amp"] = self._det_amp

        return ret_array

class DataRetriever:

    def __init__(self, path, file_number):
        self.__basepath = path
        self.__file_number = file_number
        self.__get_sel_events = True
        self.__filter_bbfit_null = True
        self.__get_det_type = True
        self.__stand_event_amplitude = False
        self.__selected_hits = 5
        self.__current_event = {}
        self.__df_list = []
        self.__target_keys = []
        self.__hit_list = []
        self.__compare_df = []
        self.__current_state = State.EVENT_ID
        self.__state_dict = {
            State.EVENT_ID: self.event_id,
            State.RUN_INFO: self.run_info,
            State.WEIGHTS: self.weights,
            State.NEUTRINO: self.neutrino,
            State.MUON: self.muon,
            State.AAFIT: self.aafit,
            State.BBFIT_TRACK: self.bbfit_track,
            State.BBFIT_BRIGHT: self.bbfit_bright,
            State.GRIDFIT: self.gridfit,
            State.HITS: self.hits,
            State.SELECTED_HITS: self.selected_hits
        }

    def get_det_type(self, det_type):
        self.__get_det_type = det_type

    def set_target(self, target_keys):
        self.__target_keys = target_keys

    def set_selected_hits(self, selected_hits, stand_event_amplitude = False):
        self.__selected_hits = selected_hits
        self.__stand_event_amplitude = stand_event_amplitude

    def load_data (self, get_sel_events = True, filter_bbfit_null = True, get_bbfit_data = True, get_aafit_data = True):
        pathlist = Path(self.__basepath).glob('*.txt')
        counter = 0
        self.__get_sel_events = get_sel_events
        self.__filter_bbfit_null = filter_bbfit_null
        self.__get_bbfit_data = get_bbfit_data
        self.__get_aafit_data = get_aafit_data
        #print(self.__state_dict)
        for path in pathlist:
            # because path is object not string
            path_in_str = str(path)
            #print("\nFILE: " + path_in_str)
            detection_type = 1 if "anumu" in path_in_str else 2
            with open(path_in_str, "r") as f_test:
                line = f_test.readline()
                self.__current_state = State.EVENT_ID
                self.__current_event = {}
                self.__current_event['det_type'] = detection_type
                while line:
                    self.__state_dict[self.__current_state](line)
                    line = f_test.readline()
                    
            counter += 1
            if counter > self.__file_number:
                break
            
        self.__final_df = pd.DataFrame(self.__df_list)
        self.__target = self.__final_df[self.__target_keys]
        self.__data_df = self.__final_df.drop(['m_elev', 'm_az', 'event_id', 'run_id', 'date', 'time', 'm_energy', 'm_pos_x', 'm_pos_y', 'm_pos_z',
                   'm_dir_x', 'm_dir_y', 'm_dir_z', 'frame_id', 'trigger_counter'], axis=1)
        
        if self.__get_bbfit_data:
            self.__compare_df = self.__data_df[['bbfit_elev','bbfit_az']]
        
        if self.__get_bbfit_data == False:
            self.__data_df = self.__data_df.drop(['bbfit_elev', 'bbfit_az', 'bbfit_chi2'], axis=1, errors='ignore')
        if self.__get_aafit_data == False:
            self.__data_df = self.__data_df.drop(['aafit_elev', 'aafit_az', 'aafit_lambda', 'aafit_beta'], axis=1, errors='ignore')
        
        return self.__target, self.__data_df, self.__compare_df

    def event_id (self, line):
        if "start_event" in line:
            self.__current_event["event_id"] = int(line.split()[1])
            self.__current_state = State.RUN_INFO
            #print("START EVENT: " + self.__current_event["event_id"])
        
    def run_info (self, line):
        element_list = line.split()
        if len(element_list) == 7: 
            self.__current_event["run_id"] = int(element_list[0])
            self.__current_event["frame_id"] = int(element_list[1])
            self.__current_event["trigger_counter"] = int(element_list[2])
            self.__current_event["date"] = element_list[4]
            self.__current_event["time"] = element_list[5].split(",")[0]
            self.__current_state = State.WEIGHTS
        else:
            print("ERROR IN STATE: RUN_INFO")

    def weights (self, line):
        element_list = line.split()
        if len(element_list) == 4 and element_list[0] == "weights": 
            self.__current_event["w_1"] = float(element_list[1])
            self.__current_event["w_2"] = float(element_list[2])

            # Removed since all the values are the same (0 variance)
            #self.__current_event["w_3"] = float(element_list[3])
            self.__current_state = State.NEUTRINO
        else:
            print("ERROR IN STATE: WEIGHTS")

    def cart2sph(self, x,y,z):
        x = float(x)
        y = float(y)
        z = float(z)
        XsqPlusYsq = x**2 + y**2
        #r = m.sqrt(XsqPlusYsq + z**2)               # r
        elev = m.atan2(z,m.sqrt(XsqPlusYsq))     # theta
        az = m.atan2(y,x)                           # phi
        return elev, az

    def neutrino (self, line):
        element_list = line.split()
        if len(element_list) == 9 and element_list[0] == "nu": 
        #    self.__current_event["n_elev"], self.__current_event["n_az"] = cart2sph(element_list[1], element_list[2], element_list[3])
            # self.__current_event["n_dir_x"] = element_list[1]
            # self.__current_event["n_dir_y"] = element_list[2]
            # self.__current_event["n_dir_z"] = element_list[3]
        #    self.__current_event["n_pos_x"] = element_list[4]
        #    self.__current_event["n_pos_y"] = element_list[5]
        #    self.__current_event["n_pos_z"] = element_list[6]
        #    self.__current_event["n_energy"] = element_list[7]
            self.__current_state = State.MUON
        else:
            print("ERROR IN STATE: NEUTRINO")
   
    def muon (self, line):
        element_list = line.split()
        if len(element_list) == 9 and element_list[0] == "muon": 
            self.__current_event["m_elev"], self.__current_event["m_az"] = self.cart2sph(element_list[1], element_list[2], element_list[3])
            self.__current_event["m_dir_x"] = float(element_list[1])
            self.__current_event["m_dir_y"] = float(element_list[2])
            self.__current_event["m_dir_z"] = float(element_list[3])
            self.__current_event["m_pos_x"] = float(element_list[4])
            self.__current_event["m_pos_y"] = float(element_list[5])
            self.__current_event["m_pos_z"] = float(element_list[6])
            self.__current_event["m_energy"] = float(element_list[7])
            self.__current_state = State.AAFIT
        else:
            print("ERROR IN STATE: MUON")

    def aafit (self, line):
        element_list = line.split()
        if len(element_list) == 9 and element_list[0] == "aafit":
            if self.__get_aafit_data == True:
                self.__current_event["aafit_elev"], self.__current_event["aafit_az"] = self.cart2sph(element_list[1], element_list[2], element_list[3])
                #self.__current_event["aafit_dir_x"] = element_list[1]
                #self.__current_event["aafit_dir_y"] = element_list[2]
                #self.__current_event["aafit_dir_z"] = element_list[3]
                self.__current_event["aafit_lambda"] = float(element_list[7])
                self.__current_event["aafit_beta"] = float(element_list[8])

            self.__current_state = State.BBFIT_TRACK
        else:
            self.__current_state = State.EVENT_ID
        #    print("ERROR IN STATE: AAFIT -> Event " + self.__current_event["event_id"])
        
    def bbfit_track (self, line):
        element_list = line.split()
        if len(element_list) == 8 and element_list[0] == "bbfit_track":
            if self.__get_bbfit_data == False:
                self.__current_state = State.BBFIT_BRIGHT
            elif self.__filter_bbfit_null and element_list[1] == "nan" or element_list[2] == "nan" or element_list[3] == "nan":
                self.__current_state = State.EVENT_ID
            #  print("BBFIT_TRACK NOT FOUND -> Event " + self.__current_event["event_id"])
            else:
                self.__current_event["bbfit_elev"], self.__current_event["bbfit_az"] = self.cart2sph(
                    0 if element_list[1] == "nan" else element_list[1],
                    0 if element_list[2] == "nan" else element_list[2],
                    0 if element_list[3] == "nan" else element_list[3])
                # self.__current_event["bbfit_dir_x"] = np.nan if element_list[1] == "nan" else element_list[1]
                #self.__current_event["bbfit_dir_x"] = 0 if element_list[1] == "nan" else float(element_list[1])
                #self.__current_event["bbfit_dir_y"] = 0 if element_list[2] == "nan" else float(element_list[2])
                #self.__current_event["bbfit_dir_z"] = 0 if element_list[3] == "nan" else float(element_list[3])
                self.__current_event["bbfit_chi2"] = float(element_list[7])
                self.__current_state = State.BBFIT_BRIGHT
        else:
            print("ERROR IN STATE: BBFIT_TRACK")
            #self.__current_state = State.BBFIT_BRIGHT

    def bbfit_bright (self, line):
        element_list = line.split()
        if len(element_list) == 8 and element_list[0] == "bbfit_bright": 
            self.__current_state = State.GRIDFIT
        else:
            print("ERROR IN STATE: BBFIT_BRIGHT")
            self.__current_state = State.GRIDFIT
    
    def gridfit (self, line):
        element_list = line.split()
        if len(element_list) == 8 and element_list[0] == "gridfit": 
            self.__current_state = State.HITS
        elif element_list[0] == "hit":
            # Some times there are not gridfit measures...
            return self.hits(line)
        else:
            print("ERROR IN STATE: GRIDFIT")
            #self.__current_state = State.HITS

    def hits (self, line):
        element_list = line.split()
        if len(element_list) == 14 and element_list[0] == "hit":
            self.__current_state = State.HITS
        elif len(element_list) == 3 and element_list[0] == "BBFit":
            self.__current_state = State.SELECTED_HITS
        else:
            print("ERROR IN STATE: HITS")
   
    def selected_hits (self, line):
        element_list = line.split()
        if self.__get_sel_events and len(element_list) == 14 and element_list[0] == "hit": 
            self.__current_state = State.SELECTED_HITS
            if int(element_list[1]) < (self.__selected_hits + 1):
                elev, az = self.cart2sph(float(element_list[8]), float(element_list[9]), float(element_list[10]))
                self.__hit_list.append(Hit(int(element_list[2]), int(element_list[3]), elev, az, float(element_list[5]), float(element_list[6]), float(element_list[7]), float(element_list[11]), float(element_list[12])))

        elif "end_event" in line:
            self.__current_state = State.EVENT_ID

            if not self.__get_sel_events or len(self.__hit_list) == self.__selected_hits:

                if self.__stand_event_amplitude:
                    max_amp_hit = max(self.__hit_list, key=lambda item: item._det_amp)
                    for hit_obj in self.__hit_list:
                        hit_obj.convert_to_relative_amplitude(max_amp_hit)

                self.__hit_list.sort(key=lambda x: x._det_time)
                for index, hit_obj in reversed(list(enumerate(self.__hit_list))):
                    hit_obj.convert_to_relative_time(self.__hit_list[0])
                    hit_obj.set_order(index + 1)

                for hit_obj in self.__hit_list:
                    self.__current_event.update(hit_obj.flatten())

                temp_event = self.__current_event.copy()
                self.__df_list.append(temp_event)

                if self.__get_det_type:
                    self.__current_event['det_type'] = temp_event['det_type']
            
            self.__hit_list.clear()

            #print("END EVENT: " + self.__current_event["event_id"])
        elif self.__get_sel_events:
            print("ERROR IN STATE: SELECTED_HITS")
            print(element_list)
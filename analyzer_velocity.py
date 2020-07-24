from imanalyzer import Imanalyzer
import numpy as np
import scipy.signal as sps
from collections import deque
class Velocityanalyzer(Imanalyzer):
    """Object detecting velocity of sludge settling [%/min] and [mL/min] basing on sedimentation curve

    :param type id: Method's iteration number
    :attr type analyzer_type: object type for Liveplot
    :attr type spacing: Order of quotient
    :attr type boundary_pos_list: Remembered curve fragment
    :attr type timestamps_list: Timestamps from boundary_pos_list
    :attr type velocity_list: Array of recorded velocity values
    :attr type filtered_velocity: Array of filtered velocity values
    :attr type max_mean_velocity: Highest mean velocity
    :attr type max_velocity: Highest sedimentation velocity
    :attr type max_velocity_timestamp: Timestamp when highest velocity took place
    :attr type max_vel_found: Flag - True when max_velocity was found or time limit was exceeded, False otherwise

    """

    def __init__(self, id = 4):
        super(Velocityanalyzer,self).__init__()
        self.analyzer_type = id
        self.spacing = 20
        self.boundary_pos_list = deque()
        self.timestamps_list = deque()
        self.velocity_list = np.empty(0)  # currently all history is stored but can be changed to story only last 7 measurments if needed
        self.filtered_velocity = np.empty(0)  # similar as above, but only last 6 measurements are needed
        self.max_mean_velocity = 0
        self.max_velocity = 0
        self.max_velocity_timestamp = 0 # calculated properly only when photos were made once every 10 seconds
        self.max_velocity_lp = 0
        self.max_vel_found = False

    def analyze(self, boundary_pos, timestamp):
        self.boundary_pos_list.appendleft(boundary_pos)
        self.timestamps_list.appendleft(timestamp)
        if self.lp == 0:
            self.filter_1st_iteration_zeros()
        self.lp += 1
        if len(self.boundary_pos_list) > self.spacing:
            self.boundary_pos_list.pop()
            self.timestamps_list.pop()
        if len(self.boundary_pos_list) >= 2:
            vel = (self.boundary_pos_list[-1] - self.boundary_pos_list[0]) / (self.timestamps_list[-1] - self.timestamps_list[0])
        else:
            vel = 0
        vel = abs(vel)
        self.velocity_list = np.append(self.velocity_list, vel)
        if self.lp >= 7:
            # https://en.wikipedia.org/wiki/Savitzky–Golay_filter
            filtered = sps.savgol_filter(self.velocity_list[self.lp-7:self.lp], 7, 1)
            if self.lp == 7:
                self.filtered_velocity = np.append(self.filtered_velocity, filtered[0:4])
            else:
                self.filtered_velocity = np.append(self.filtered_velocity, filtered[3])

            # intentionally avoiding first two measures because disturbance cannot be effectively filtered from them
            # additionally probability that maximum will occur in first 10 second is negligible
            if self.lp == 12:
                mean_velocity = np.mean(self.filtered_velocity[self.lp-10:self.lp-4])
                self.max_mean_velocity = mean_velocity
                self.max_velocity = np.amax(self.filtered_velocity[self.lp-10:self.lp-4])
                max_velocity_ind = np.nonzero(self.filtered_velocity[self.lp-10:self.lp-4] == self.max_velocity)
                self.max_velocity_timestamp = timestamp - ((1/6) * (9 - max_velocity_ind[0][0]))
                self.max_velocity_lp = self.lp - 10 + max_velocity_ind[0][0]
            elif self.lp > 12:
                mean_velocity = np.mean(self.filtered_velocity[self.lp - 10:self.lp - 4])
                if mean_velocity > self.max_mean_velocity:
                    self.max_mean_velocity = mean_velocity
                    self.max_velocity = np.amax(self.filtered_velocity[self.lp - 10:self.lp - 4])
                    max_velocity_ind = np.nonzero(self.filtered_velocity[self.lp - 10:self.lp - 4] == self.max_velocity)
                    self.max_velocity_timestamp = timestamp - ((1 / 6) * (9 - max_velocity_ind[0][0]))
                    self.max_velocity_lp = self.lp - 10 + max_velocity_ind[0][0]
                else:
                    hysteresis_threshold = np.amax(np.array([0.2*self.max_mean_velocity, 0.5]))
                    if mean_velocity < self.max_mean_velocity - hysteresis_threshold and self.max_vel_found is False:
                        self.max_vel_found = True
                        print("Successfully found maximum velocity")
                        print("Max velocity = " + str(self.max_velocity * 10) + "[mL/min] in " + str(self.max_velocity_timestamp) + '[min]') # *10 to convert [%/min] to [mL/min]

                if timestamp >= 30 and self.max_vel_found is False:
                    self.max_vel_found = True
                    # Mean value from 10 minutes wide window around found maximum value is used instead, if it is not
                    # possible to get that wide window then it is cropped to doable width.
                    if self.max_velocity_lp + 30 > self.lp - 4:
                        mean_velocity = np.mean(self.filtered_velocity[self.max_velocity_lp - 30:self.lp-4])
                    else:
                        mean_velocity = np.mean(self.filtered_velocity[self.max_velocity_lp - 30:self.max_velocity_lp + 30])
                    print("Maximum velocity criteria were not met, returning mean velocity.")
                    print("Max velocity at " + str(self.max_velocity_timestamp) + "[min] mean velocity " + str(mean_velocity * 10) + '[mL/min]') # *10 to convert [%/min] to [mL/min]

        self._notify([vel, self.analyzer_type - 3, timestamp])
        return vel, self.max_vel_found

    def diff_progressive(self, array,space):
        out = np.empty(len(array) - space + 1)
        for i in range(0,len(array) - space ):
            out[i] = (array[i + space] - array[i]) / space
        return out

    def filter_1st_iteration_zeros(self):
        if self.boundary_pos_list[0] == 0:
            self.boundary_pos_list.popleft()
            self.timestamps_list.popleft()

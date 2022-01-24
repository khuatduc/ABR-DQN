import os
#import os.path
#import sys
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
#import env
#import load_trace
import gym
from gym import spaces
#from gym.utils import seeding

S_INFO = 5  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
#TRAIN_SEQ_LEN = 100  # take as a train batch
#MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
#RAND_RANGE = 1000


class ABREnv(gym.Env):

    def __init__(self, random_seed=RANDOM_SEED):
        np.random.seed(RANDOM_SEED)
        self.action_space = spaces.Discrete(A_DIM)
        self.observation_space = spaces.Box(
            0, 10.0, [S_INFO, S_LEN], dtype=np.float32)
        all_cooked_time, all_cooked_bw, _ = load_trace()
        self.net_env = Environment(all_cooked_time=all_cooked_time,
                            all_cooked_bw=all_cooked_bw)
        
        self.last_bit_rate = DEFAULT_QUALITY
        self.state = np.zeros((S_INFO, S_LEN))
        self.reset()

    def reset(self):
        self.time_stamp = 0
        self.last_bit_rate = DEFAULT_QUALITY
        self.state = np.zeros((S_INFO, S_LEN))
        return self.state

    def step(self, action):
        bit_rate = action
        # the action is from the last decision
        # this is to make the framework similar to the real buffer_size
        delay, sleep_time, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
            self.net_env.get_video_chunk(bit_rate)

        self.time_stamp += delay  # in ms
        self.time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty - smooth penalty
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
            - REBUF_PENALTY * rebuf \
            - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                      VIDEO_BIT_RATE[self.last_bit_rate]) / M_IN_K

        self.last_bit_rate = bit_rate
        state = np.roll(self.state, -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / \
            float(np.max(VIDEO_BIT_RATE))  # last quality
        #state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = float(video_chunk_size) / \
            float(delay) / M_IN_K  # kilo byte / ms
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = np.array(
            next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = np.minimum(video_chunk_remain,
                                  CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

        self.state = state
        #observation, reward, done, info = env.step(action)
        return state, reward, end_of_video, {}


MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
RANDOM_SEED = 4200
VIDEO_CHUNCK_LEN = 4000.0  # millisec, every time add this amount to buffer
BITRATE_LEVELS = 6
TOTAL_VIDEO_CHUNCK = 48
BUFFER_THRESH = 60.0 * MILLISECONDS_IN_SECOND  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec
PACKET_SIZE = 1500  # bytes
NOISE_LOW = 0.9
NOISE_HIGH = 1.1
VIDEO_SIZE_FILE = './video_size_'


class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw, random_seed=RANDOM_SEED):
        assert len(all_cooked_time) == len(all_cooked_bw)

        np.random.seed(random_seed)

        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw

        self.video_chunk_counter = 0
        self.buffer_size = 0

        # pick a random trace file
        self.trace_idx = np.random.randint(len(self.all_cooked_time))
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.video_size = {}  # in bytes
        for bitrate in range(BITRATE_LEVELS):
            self.video_size[bitrate] = []
            with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))

    def get_video_chunk(self, quality):

        assert quality >= 0
        assert quality < BITRATE_LEVELS

        video_chunk_size = self.video_size[quality][self.video_chunk_counter]
        
        # use the delivery opportunity in mahimahi
        delay = 0.0  # in ms
        video_chunk_counter_sent = 0  # in bytes
        
        while True:  # download video chunk over mahimahi
            throughput = self.cooked_bw[self.mahimahi_ptr] \
                         * B_IN_MB / BITS_IN_BYTE
            duration = self.cooked_time[self.mahimahi_ptr] \
                       - self.last_mahimahi_time
	    
            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_chunk_counter_sent + packet_payload > video_chunk_size:

                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                  throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                assert(self.last_mahimahi_time <= self.cooked_time[self.mahimahi_ptr])
                break

            video_chunk_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0

        delay *= MILLISECONDS_IN_SECOND
        delay += LINK_RTT

    	# add a multiplicative noise to the delay
        delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)

        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

        # add in the new chunk
        self.buffer_size += VIDEO_CHUNCK_LEN

        # sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size > BUFFER_THRESH:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - BUFFER_THRESH
            sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
                         DRAIN_BUFFER_SLEEP_TIME
            self.buffer_size -= sleep_time

            while True:
                duration = self.cooked_time[self.mahimahi_ptr] \
                           - self.last_mahimahi_time
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = 0

        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.buffer_size

        self.video_chunk_counter += 1
        video_chunk_remain = TOTAL_VIDEO_CHUNCK - self.video_chunk_counter

        end_of_video = False
        if self.video_chunk_counter >= TOTAL_VIDEO_CHUNCK:
            end_of_video = True
            self.buffer_size = 0
            self.video_chunk_counter = 0

            # pick a random trace file
            self.trace_idx = np.random.randint(len(self.all_cooked_time))
            self.cooked_time = self.all_cooked_time[self.trace_idx]
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]

            # randomize the start point of the video
            # note: trace file starts with time 0
            self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        next_video_chunk_sizes = []
        for i in range(BITRATE_LEVELS):
            next_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter])

        return delay, \
            sleep_time, \
            return_buffer_size / MILLISECONDS_IN_SECOND, \
            rebuf / MILLISECONDS_IN_SECOND, \
            video_chunk_size, \
            next_video_chunk_sizes, \
            end_of_video, \
            video_chunk_remain


COOKED_TRACE_FOLDER = './cooked_traces/'


def load_trace(cooked_trace_folder=COOKED_TRACE_FOLDER):
    cooked_files = os.listdir(cooked_trace_folder)
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    for cooked_file in cooked_files:
        file_path = cooked_trace_folder + cooked_file
        cooked_time = []
        cooked_bw = []
        # print file_path
        with open(file_path, 'rb') as f:
            for line in f:
                parse = line.split()
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]))
        all_cooked_time.append(cooked_time)
        all_cooked_bw.append(cooked_bw)
        all_file_names.append(cooked_file)

    return all_cooked_time, all_cooked_bw, all_file_names
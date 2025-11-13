import numpy as np

from submodules.EyeTrackingMetrics.Metrics import *
from submodules.EyeTrackingMetrics.transition_matrix import *
from submodules.PyGazeAnalyser.pygazeanalyser.detectors import (
    fixation_detection,
    saccade_detection,
)

TEST_SCREENDIM = [1, 1]


class GazeEntropyProcess:
    """
    Calc gaze entropy features with rolling window
    """

    def __init__(self, ws=10, grid_x=3, grid_y=3, stride=180, fps=10.0):
        self.screen_dimensions = TEST_SCREENDIM
        self.ws_f = int(round(ws * fps))
        self.stride = stride
        # self.aoi_grid = {
        #     "grid1": PolyAOI(TEST_SCREENDIM,[[ 0,0] , [0,0.33], [0.33,0.33], [0.33,0]])
        # } # 3x3

        self.aoi_grid = {}
        for i in range(0, grid_y):
            for j in range(0, grid_x):
                grid_name = f"grid{i * grid_y + j}"
                x1 = j / grid_x
                y1 = i / grid_y
                x2 = (j + 1) / grid_x
                y2 = (i + 1) / grid_y
                self.aoi_grid[grid_name] = PolyAOI(TEST_SCREENDIM, [[x1, y1], [x1, y2], [x2, y2], [x2, y1]])
                # print('AOI:',[[x1, y1], [x1, y2], [x2, y2], [x2, y1]])

    def calc_ent_rolling(self, gaze_x, gaze_y):
        """calc entropy on rolling window"""
        # stride = 180  # every 180 frame
        gaze_array = np.array([gaze_x, gaze_y, np.zeros_like(gaze_x)]).transpose()

        _range = np.arange(self.ws_f, len(gaze_array), self.stride)
        indexes = np.array([i for i in _range])
        entropy_transition = np.empty_like(indexes, dtype=float)
        entropy_stationary = np.empty_like(indexes, dtype=float)
        for out_idx, i in enumerate(_range):
            chunk = gaze_array[i - self.ws_f : i, :]  # take last ws_f frames
            entropy_transition[out_idx] = self.calc_chunk(chunk, entropy="transition")
            entropy_stationary[out_idx] = self.calc_chunk(chunk, entropy="stationary")
        return indexes, entropy_transition, entropy_stationary

    def calc_ent_whole(self, gaze_x, gaze_y):
        """calc entropy on whole data"""
        gaze_array = np.array([gaze_x, gaze_y, np.zeros_like(gaze_x)]).transpose()
        return self.calc_chunk(gaze_array, entropy="transition"), self.calc_chunk(gaze_array, entropy="stationary")

    def calc_chunk(self, gaze_chunk, entropy):
        aoi_dict = self.aoi_grid
        ge_t = GazeEntropy(self.screen_dimensions, aoi_dict, gaze_chunk, entropy)
        return ge_t.compute()


class GazeFixationSaccade:
    def __init__(self):
        self.MISSING_NUM = -1000
        self.FIX_DIST_TH = 650 * np.tan(np.deg2rad(2))  # 2 degree at 650mm distance = 22.7mm
        self.FIX_MINDUR = 100
        self.SAC_MINLEN = 5
        self.SAC_MAXVEL = 650 * np.tan(np.deg2rad(30))
        self.SAC_MAXACC = 650 * np.tan(np.deg2rad(8000))

    def detect_fixations(self, gaze_x, gaze_y, gaze_t, tabular):
        _, fixation_details = fixation_detection(
            gaze_x, gaze_y, gaze_t, missing=self.MISSING_NUM, maxdist=self.FIX_DIST_TH, mindur=self.FIX_MINDUR
        )
        fixation_df = pd.DataFrame(fixation_details, columns=["start", "end", "duration", "endx", "endy"])

        tabular["fixation"] = 0
        for start, end in zip(fixation_df["start"], fixation_df["end"]):
            tabular.loc[
                (tabular.index >= start * 1000000) & (tabular.index <= end * 1000000),
                "fixation",
            ] = 1
        return tabular

    def detect_saccades(self, gaze_x, gaze_y, gaze_t, tabular):
        _, saccades_details = saccade_detection(
            gaze_x,
            gaze_y,
            gaze_t,
            missing=self.MISSING_NUM,
            minlen=self.SAC_MINLEN,
            maxvel=self.SAC_MAXVEL,
            maxacc=self.SAC_MAXACC,
        )
        saccades_df = pd.DataFrame(
            saccades_details, columns=["start", "end", "duration", "startx", "starty", "endx", "endy"]
        )
        tabular["saccade"] = 0
        for start, end in zip(saccades_df["start"], saccades_df["end"]):
            tabular.loc[
                (tabular.index >= start * 1000000) & (tabular.index <= end * 1000000),
                "saccade",
            ] = 1
        return tabular

    def calc_fixation_saccade_ratio(self, tabular, window_size=5, fps=60, step_size=1):
        window_size = int(window_size * fps)  # 1 second window in terms of number of frames
        fixation_saccade_ratio = []
        saccade_fixation_ratio = []

        fixation_saccade_ratio.extend([np.nan] * window_size)
        saccade_fixation_ratio.extend([np.nan] * window_size)

        for start_idx in range(0, len(tabular) - window_size, step_size):
            end_idx = start_idx + window_size
            start_idx = max(0, start_idx)  # Avoid negative index
            end_idx = min(len(tabular), end_idx)  # Avoid index out of bounds

            fixation_duration = tabular.iloc[start_idx:end_idx]["fixation"].sum()
            saccade_duration = tabular.iloc[start_idx:end_idx]["saccade"].sum()

            if saccade_duration == 0:
                ratio = np.nan  # Avoid division by zero
            else:
                ratio = fixation_duration / saccade_duration
            if fixation_duration == 0:
                ratio2 = np.nan
            else:
                ratio2 = saccade_duration / fixation_duration

            fixation_saccade_ratio.extend([ratio] * step_size)
            saccade_fixation_ratio.extend([ratio2] * step_size)

        tabular["fixation_saccade_ratio"] = fixation_saccade_ratio
        tabular["saccade_fixation_ratio"] = saccade_fixation_ratio
        return tabular

    def prepare_data(self, tabular, side="right"):
        # mm (pixel num * pixel pitch)
        gaze_x = tabular[f"tobii_{side}_eye_gaze_pt_in_display_x"].values * 3840 * 0.229
        gaze_y = tabular[f"tobii_{side}_eye_gaze_pt_in_display_y"].values * 1600 * 0.229
        gaze_t = tabular.index / 1000000.0  # convert from ns to milliseconds
        gaze_x = np.nan_to_num(gaze_x, copy=True, nan=self.MISSING_NUM)
        gaze_y = np.nan_to_num(gaze_y, copy=True, nan=self.MISSING_NUM)
        return gaze_x, gaze_y, gaze_t

    def process(self, tobii_df, window_size=5, fps=60, step_size=1, side="right"):
        gaze_x, gaze_y, gaze_t = self.prepare_data(tobii_df, side=side)
        output_df = pd.DataFrame([], index=tobii_df.index)
        output_df = self.detect_saccades(gaze_x, gaze_y, gaze_t, output_df)
        output_df = self.detect_fixations(gaze_x, gaze_y, gaze_t, output_df)
        output_df = self.calc_fixation_saccade_ratio(output_df, window_size=5, fps=60, step_size=1)
        return output_df

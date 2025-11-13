import glob
import cv2
import os
import numpy as np
import copy

from ddd_2024_features.base_feature import BaseFeature, BaseCollatedFeature
from ddd_2024_features.common import get_matrix_from_tabular, format_columns
from ddd_2024_chunk import filter_dataframe_times


class VideoReader:
    def __init__(self, video_path):
        self.video_path = video_path
        self.current_frame_num = 0
        self.current_data = None
        self.get_next()
        
    def get_next(self):
        raise NotImplementedError()
        
    def seek(self, target_frame_num):
        # Check if frame number earlier than current - need to reset
        if target_frame_num < self.current_frame_num:
            raise Exception("Video seeking frame in the past: " + self.video_path)
        
        # Seek forward until at frame
        while self.current_frame_num != target_frame_num:
            try:
                self.get_next()
                self.current_frame_num += 1
                
            except StopIteration:
                raise Exception("Video reached end of file before desired frame: " + self.video_path)
                    
        return self.current_data
    

class VideoFrameReader(VideoReader):
    def __init__(self, video_path, timing_data, feature_attributes):
        self._feature_attributes = feature_attributes
        self.timing_data = timing_data
        self.num_frames = self.timing_data.shape[0]
        self.cap = cv2.VideoCapture(video_path)
        self.current_index = 0
        self.image_size = None
        self.current_data = None
        self.first_frame = True
        super().__init__(video_path)

    class FrameData():
        def __init__(self, time, image_data, timing_row={}):
            self.time = time
            self.image_data = image_data
            self.timing_row = timing_row

    def get_next(self):
        if self.current_data is not None:
            self.first_frame = False
        ret, self.current_data = self.cap.read()
        if not ret:
            raise Exception("Could not read video frame: " + self.video_path)
        if self.first_frame:
            self.image_size = tuple([int(x) for x in self._feature_attributes["resolution"]]) if "resolution" in self._feature_attributes else self.current_data.shape
            
    def close(self):
        self.cap.release()

    def process_frame(self, frame_data):
        video_type = self._feature_attributes["video_type"] if "video_type" in self._feature_attributes else "rgb"
        
        # Process depth frame                        
        if video_type == "depth":
            normalized_depth = np.dot(frame_data[:, :, :3], [65536.0, 256.0, 1.0])
            normalized_depth /= (256.0 * 256.0 * 256.0 - 1.0)
            normalized_depth *= 1000.  # meters
            frame_data = normalized_depth[:,:,None]
            
        # Process segmentation frame
        if video_type == "segmentation":
            frame_data = frame_data[:,:,2,None]

        # Resize the frame, if needed
        if self.image_size[:-1] != frame_data.shape[:-1]:
            interpolation = self._feature_attributes["interpolation"] if "interpolation" in self._feature_attributes else cv2.INTER_NEAREST
            frame_data = cv2.resize(frame_data, (self.image_size[1], self.image_size[0]), interpolation=interpolation)

        # Add third dimension, if missing
        if len(frame_data.shape) == 2:
            frame_data = frame_data[..., None]

        return frame_data

    def read_at_time(self, time):
        self.current_index = self.timing_data.index.get_loc(time)
        return self.read_next()
            
    def read_next(self):
        if self.current_index >= self.timing_data.shape[0]:
            return None
        df = self.timing_data.iloc[[self.current_index]]
        self.current_index += 1
        timing_row = df.to_dict(orient='records')[0]
        self.seek(timing_row["resampled_frame"])
        image_data = self.current_data
        image_data = self.process_frame(image_data)
        return self.FrameData(df.index.values[0], image_data, timing_row)
         
    def read_frames(self):
        while True:
            frame = self.read_next()
            if frame is None:
                return
            else:
                yield frame

    def read_all_frames(self):
        read_all_first_frame = True
        all_frames = None
        for timing_it in range(self.num_frames):
            timing_row = self.timing_data.iloc[[timing_it]].to_dict(orient='records')[0]
            read_frame = timing_row["resampled_frame"]
            self.seek(read_frame)
            image_data = self.current_data
                
            # Initialize the frame output
            if read_all_first_frame:
                read_all_first_frame = False                
                new_shape = (self.num_frames, self.image_size[1], self.image_size[0], self.image_size[2])
                all_frames = np.empty(new_shape, dtype=image_data.dtype)
                
            # Output the frame
            all_frames[timing_it,...] = self.process_frame(image_data)

        return all_frames
    

# Video data - variable in length over time with constant resolution (also includes timing table)
class VideoFeature(BaseFeature):
    def __init__(self, feature_name, feature_attributes, full_scenario_path, sample):
        super().__init__(feature_name, feature_attributes)
        
        # Setup video path and type
        self.frame_reader = None
        self._feature_attributes = copy.deepcopy(feature_attributes)
        if "columns" in self._feature_attributes:
            if "resampled_frame" not in self._feature_attributes["columns"]:
                self._feature_attributes["columns"].append("resampled_frame")
        video_name = self._feature_attributes["file_name"]
        video_dir = self.get_feature_path(full_scenario_path, video_name)
        
        # Determine extension to use
        video_files = glob.glob(os.path.join(video_dir, 'video.*'))
        if len(video_files) == 0:
            raise FileNotFoundError("Missing video file at " + video_dir)
        elif len(video_files) > 1:
            raise Exception("Multiple video files at " + video_dir)
        video_extension = os.path.splitext(video_files[0])[1]

        # Set file paths
        csv_path = os.path.join(video_dir, "frame_timing.csv")
        video_path = os.path.join(video_dir, "video" + video_extension)
        self.video_path = video_path
        self.timings_path = csv_path

        # Determine which inputs to skip
        skip_timings = "skip_timings" in self._feature_attributes and self._feature_attributes["skip_timings"]
        skip_frames = "skip_frames" in self._feature_attributes and self._feature_attributes["skip_frames"]

        # Load timings
        if not skip_timings:
            self.timing_data = self.load_timings_chunk(csv_path, sample.chunk, self._feature_attributes)
            self.num_frames = self.timing_data.shape[0]
        
        # Load the video frames        
        if not skip_frames:
            if not skip_timings:
                timing_data = self.timing_data
            else:
                timing_data = self.load_timings_chunk(csv_path, sample.chunk, self._feature_attributes)
            self.num_frames = timing_data.shape[0]   
            frame_reader = VideoFrameReader(video_path, timing_data, self._feature_attributes)
            self.image_data = frame_reader.read_all_frames()
            self.image_size = frame_reader.image_size
            frame_reader.close()
        else:
            self.image_data = None
            self.image_size = tuple([int(x) for x in self._feature_attributes["resolution"]]) if "resolution" in self._feature_attributes else None

    @classmethod
    def load_timings(cls, csv_path, start_epoch_ns, end_epoch_ns, feature_attributes):
        # Load video timing csv
        csv_columns = feature_attributes["columns"] if "columns" in feature_attributes else None
        timing_data = cls.load_dataframe(csv_path, index="resampled_epoch_ns", column_list=csv_columns)
        
        # Filter by chunk times - copy if not in cache
        timing_data = filter_dataframe_times(timing_data, start_epoch_ns, end_epoch_ns)
        
        # Get chunk-relative positional encoding
        timing_data = timing_data.copy()
        timing_data["frame_in_chunk"] = timing_data["resampled_frame"] - timing_data["resampled_frame"].iloc[0]
        return timing_data

    @classmethod
    def load_timings_chunk(cls, csv_path, chunk, feature_attributes):
        return cls.load_timings(csv_path, chunk.start_epoch_ns, chunk.end_epoch_ns, feature_attributes)

    def init_frame_reader(self, ok_if_open=True):
        if self.frame_reader is not None:
            if ok_if_open:
                return
            else:
                raise Exception("VideoFrameReader already initialized")
        self.frame_reader = VideoFrameReader(self.video_path, self.timing_data, self._feature_attributes)

    def close(self):
        if self.frame_reader is not None:
            self.frame_reader.close()
            self.frame_reader = None

    def get_frame_reader(self):
        return VideoFrameReader(self.video_path, self.timing_data, self._feature_attributes)


class CollatedVideoFeature(BaseCollatedFeature):
    # Populates the following:
    # self.image_data (batch_size, num_frames, height, width, channels)
    # self.image_data_valid (batch_size)
    # self.timing_data (batch_size, num_frames, num_columns)
    # self.timing_data_valid (batch_size, num_frames, num_columns)
    # self.timing_data_columns
       
    def __init__(self, feature_name, feature_attributes, list_of_features, collated_sample):
        super().__init__(feature_name, feature_attributes)

        self._feature_attributes = copy.deepcopy(feature_attributes)
        if "columns" in self._feature_attributes:
            if "resampled_frame" not in self._feature_attributes["columns"]:
                self._feature_attributes["columns"].append("resampled_frame")

        batch_size = collated_sample.batch_size
        fps = collated_sample.config.features[feature_name]["fps"]
        num_frames = collated_sample.feature_num_frames[fps]
        timing_columns = format_columns(feature_attributes) if "columns" in feature_attributes else None
        if "resolution" not in feature_attributes:
            raise Exception(f"Must declare resolution for video {feature_name} when using collation.")
        resolution = feature_attributes["resolution"]
        
        # x and y are swapped
        resolution = (resolution[1], resolution[0], resolution[2])

        # Initialize outputs
        self.video_path = [None if x is None else x.video_path for x in list_of_features]
        self.timings_path = [None if x is None else x.timings_path for x in list_of_features]
        skip_frames = "skip_frames" in feature_attributes and feature_attributes["skip_frames"]
        if not skip_frames:
            self.image_data = np.zeros((batch_size, num_frames) + resolution)
            self.image_data_valid = np.zeros((batch_size,))
        if timing_columns is not None:
            self.timing_data = np.zeros((batch_size, num_frames, len(timing_columns)))
            self.timing_data_valid = np.zeros((batch_size, num_frames, len(timing_columns)))
            self.timing_data_columns = timing_columns

        # Save chunk start and end times for use with get_frame_reader
        self.chunk_start_ns = collated_sample.chunk.start_epoch_ns
        self.chunk_end_ns = collated_sample.chunk.end_epoch_ns

        # Loop through samples and add the video clips
        for sample_on, sample_video in enumerate(list_of_features):
            if sample_video is not None:
                if not skip_frames:
                    self.image_data_valid[sample_on] = 1
                    self.image_data[sample_on] = sample_video.image_data

                # Add timing table, if requested columns
                if timing_columns is not None:
                    tabular_data = get_matrix_from_tabular(sample_video.timing_data, timing_columns)
                    self.timing_data_valid[sample_on] = np.logical_not(np.isnan(tabular_data))
                    tabular_data[np.isnan(tabular_data)] = 0.0
                    self.timing_data[sample_on] = tabular_data

    def get_frame_reader(self, batch_idx):
        if self.timings_path[batch_idx] is None or self.video_path[batch_idx] is None:
            return None
        timing_data = VideoFeature.load_timings(self.timings_path[batch_idx], self.chunk_start_ns[batch_idx], self.chunk_end_ns[batch_idx], self._feature_attributes)
        return VideoFrameReader(self.video_path[batch_idx], timing_data, self._feature_attributes)

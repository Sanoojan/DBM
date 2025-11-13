import numpy as np
import random

class ChunkIndex():
    def __init__(self, scenario_number, chunk_number):
        self.scenario_number = scenario_number
        self.chunk_number = chunk_number
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.scenario_number == other.scenario_number and self.chunk_number == other.chunk_number
        else:
            return False
        
    def __hash__(self):
        return hash((self.scenario_number, self.chunk_number))
    
    def __str__(self):
        return f"(Scenario: {self.scenario_number}, Chunk: {self.chunk_number})"

def filter_dataframe_times(df, start_epoch_ns, end_epoch_ns):
    return df.loc[start_epoch_ns:end_epoch_ns]

class Chunk():
    def __init__(self, index, times):
        self.index = index
        self.start_epoch_ns = times["start_epoch_ns"]
        self.end_epoch_ns = times["end_epoch_ns"]
        self.resample_times = times["resample_times"]

    def times_in_chunk(self, resample_times):
        return np.logical_and(resample_times >=self.start_epoch_ns, resample_times <= self.end_epoch_ns)

    def filter_resample_times(self, resample_times):
        return resample_times[self.times_in_chunk(resample_times)]

    def filter_dataframe(self, df):
        return filter_dataframe_times(df, self.start_epoch_ns, self.end_epoch_ns)
    
    def filter_intersections(self, df):
        if "start_epoch_ns" in df and "end_epoch_ns" in df:
            chunk_start_before_df_end = self.start_epoch_ns <= df["end_epoch_ns"]
            df_start_before_chunk_end = df["start_epoch_ns"] <= self.end_epoch_ns
            intersects = chunk_start_before_df_end & df_start_before_chunk_end
            df = df[intersects]
        return df

    def __str__(self):
        return f"(Index: {self.index}, Start: {self.start_epoch_ns}, End: {self.end_epoch_ns})"

class ChunkFactory():
    def __init__(self, config, scenarios):
        # Set up chunking parameters
        self.config = config
        if self.config.chunk_fps is None:
            feature_fps = [feature["fps"] for feature in self.config.features.values() if "fps" in feature]
            if len(feature_fps) != 0:
                self.config.chunk_fps = int(np.max(feature_fps))
                print(f"chunk_fps not specified - using max feature frame rate of {self.config.chunk_fps}.")
            else:
                print("WARNING: chunk_fps not specified and no feature frame rates - using chunk_fps of 0.")
                self.config.chunk_fps = 0        
        if self.config.chunk_fps == 0:
            print("WARNING: chunk_fps set to 0 - must use full chunking strategy.")
            self.config.chunk_strategy = "full"
        else:
            self.num_chunk_frames = int(round(self.config.chunk_duration * self.config.chunk_fps))
            self.chunk_start_offset_frames = int(round(self.config.chunk_start_offset * self.config.chunk_fps))
            self.chunk_end_offset_frames = int(round(self.config.chunk_end_offset * self.config.chunk_fps))
            self.chunk_offset_frames = int(round(self.config.chunk_stride * self.config.chunk_fps))
    
        # Create the chunk indices    
        self.chunk_indices = []
        self.cached_chunks = {}
        for scenario in scenarios:
            chunk_resample_times = scenario.get_resample_times(self.config.chunk_fps)  
            if self.config.chunk_strategy == "full":
                times = self.get_chunk_times(0, scenario, chunk_resample_times)
                if times is not None:
                    self.chunk_indices.append(ChunkIndex(scenario.chunk_index, 0))
            else:
                # Determine the number of chunks from each scenario based on the length
                total_frames = len(chunk_resample_times)
                start_offset_frames = self.chunk_start_offset_frames if self.chunk_start_offset_frames >= 0 else total_frames + self.chunk_start_offset_frames
                end_offset_frames = self.chunk_end_offset_frames if self.chunk_end_offset_frames >= 0 else total_frames + self.chunk_end_offset_frames
                num_scenario_frames = total_frames - start_offset_frames - end_offset_frames
                if self.config.chunk_strategy == "random":
                    if self.num_chunk_frames <= num_scenario_frames:
                        # Verify that a chunk can be found
                        times = None
                        attempts = 0
                        while attempts < 100 and times is None:
                            times = self.get_chunk_times(0, scenario, chunk_resample_times)
                            attempts += 1
                            if times is not None:
                                break

                        # Create the chunk indices if any valid time is found
                        if times is not None:
                            num_chunks = self.config.chunks_per_scenario if self.config.chunks_per_scenario is not None else num_scenario_frames // self.num_chunk_frames
                            for chunk_number in range(num_chunks):
                                self.chunk_indices.append(ChunkIndex(scenario.chunk_index, chunk_number))
                elif self.config.chunk_strategy == "start" or self.config.chunk_strategy == "end":
                    num_offset_frames = int(round(self.config.chunk_stride * self.config.chunk_fps))
                    max_chunks = max(0, ((num_scenario_frames - self.num_chunk_frames) // num_offset_frames) + 1)
                    if self.config.chunks_per_scenario is not None:
                        max_chunks = min(self.config.chunks_per_scenario, max_chunks)
                    for chunk_number in range(max_chunks):
                        times = self.get_chunk_times(chunk_number, scenario, chunk_resample_times)
                        if times is not None:
                            self.chunk_indices.append(ChunkIndex(scenario.chunk_index, chunk_number))
                else:
                    raise Exception("Unknown chunk strategy " + self.config.chunk_strategy)
        
    def get_chunk_times(self, chunk_number, scenario, chunk_resample_times=None):
        if chunk_resample_times is None:
            chunk_resample_times = scenario.get_resample_times(self.config.chunk_fps)
        times = {}
        if self.config.chunk_strategy == "full":
            end_sec = chunk_resample_times[-1] / 1e9
            start_offset = self.config.chunk_start_offset if self.config.chunk_start_offset >= 0. else end_sec + self.config.chunk_start_offset
            end_offset = self.config.chunk_end_offset if self.config.chunk_end_offset >= 0. else end_sec + self.config.chunk_end_offset
            times["start_epoch_ns"] = int(round(start_offset * int(1e9)))
            times["end_epoch_ns"] = chunk_resample_times[-1] - int(round(end_offset * int(1e9)))
        else:
            # Determine the starting frame for the chunk
            total_frames = len(chunk_resample_times)
            start_offset_frames = self.chunk_start_offset_frames if self.chunk_start_offset_frames >= 0 else total_frames + self.chunk_start_offset_frames
            end_offset_frames = self.chunk_end_offset_frames if self.chunk_end_offset_frames >= 0 else total_frames + self.chunk_end_offset_frames
            num_scenario_frames = len(chunk_resample_times)
            if self.config.chunk_strategy == "random":
                max_range = num_scenario_frames - self.num_chunk_frames - start_offset_frames - end_offset_frames
                start_frame = random.randint(0, max_range) + start_offset_frames
            elif self.config.chunk_strategy == "start":
                start_frame = start_offset_frames + (chunk_number * self.chunk_offset_frames)
            elif self.config.chunk_strategy == "end":
                start_frame = num_scenario_frames - end_offset_frames - self.num_chunk_frames - (chunk_number * self.chunk_offset_frames)
            else:
                raise Exception("Unknown chunk strategy " + self.config.chunk_strategy)
            
            # Actually sample the chunk times
            times["start_epoch_ns"] = chunk_resample_times[start_frame]
            times["end_epoch_ns"] = chunk_resample_times[start_frame + self.num_chunk_frames - 1]

            # Check for overlap with anomaly
            if self.config.chunk_remove_anomalies is not None:
                for anomaly in scenario.anomalies:
                    if anomaly["start_ns"] <= times["end_epoch_ns"] and anomaly["end_ns"] >= times["start_epoch_ns"]:
                        overlap_ns = min(anomaly["end_ns"], times["end_epoch_ns"]) - max(anomaly["start_ns"], times["start_epoch_ns"])
                        overlap_pct = overlap_ns / (times["end_epoch_ns"] - times["start_epoch_ns"])
                        if overlap_pct >= self.config.chunk_remove_anomalies:
                            return None
            
        # Sample times in chunk
        times["resample_times"] = chunk_resample_times[np.logical_and(chunk_resample_times >= times["start_epoch_ns"], chunk_resample_times <= times["end_epoch_ns"])]
        return times
    
    def get_chunk(self, index, scenario):
        # Check for a cached chunk
        if self.config.cache_chunks and index in self.cached_chunks:
            return self.cached_chunks[index]
        
        # Ensure a valid chunk time range can be found
        times = None
        attempts = 0
        while attempts < 100 and times is None:
            times = self.get_chunk_times(index.chunk_number, scenario)
            attempts += 1
            if times is not None:
                break
        if times is None:
            raise Exception(f"No valid chunk time could be found after {attempts} attempts")
        
        # Create the chunk and cache, if requested
        chunk = Chunk(index, times)
        if self.config.cache_chunks:
            self.cached_chunks[index] = chunk
        return chunk
    
    def __len__(self):
        return len(self.chunk_indices)
    
    def get_index(self, key):
        return self.chunk_indices[key]

class CollatedChunk():
    def __init__(self, list_of_chunks, config):
        self.index = np.array([x.index for x in list_of_chunks])
        self.start_epoch_ns = np.array([x.start_epoch_ns for x in list_of_chunks])
        self.end_epoch_ns = np.array([x.end_epoch_ns for x in list_of_chunks])
        self.resample_times = np.array([x.resample_times for x in list_of_chunks])

    def __str__(self):
        return f"(CollatedChunk Batch Size: {self.index.shape[0]})"
    
    def get_chunk(self, idx):
        return Chunk(self.index[idx], {"start_epoch_ns": self.start_epoch_ns[0], "end_epoch_ns": self.end_epoch_ns[0], "resample_times": self.resample_times[0]})

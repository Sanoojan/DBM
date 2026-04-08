# The DBM Dataset
The DBM_Dataset class within dbm_dataset.py inherits the DDD_2024_Dataset from hail-datasets and extends its behavior to support PyTorch. Unless you have a good reason, you should use this version of the class instead of the inherited one. Initializing the class requires populating a DBM_Dataset_Config, which describes what scenarios to use, the chunking strategy, and the features to load. One example of its use is the following:

```
from data.dbm_dataset import DBM_Dataset, DBM_Dataset_Config

config = DBM_Dataset_Config()
config.base_path = "dataset/Vehicle/No-Video"
config.index_relative_path = "Resampled_previous_10"

config.features = {
"tobii": {
        "relative_path": "Resampled_previous_10",
        "file_name": "experiment_tobii_frame",
        "fps": 10,
        "type": "tabular",
        "columns": [
                "tobii_left_eye_gaze_pt_validity",
        ],
},

config.chunk_strategy = "end"
config.chunks_per_scenario = 5
config.chunk_stride = 5.
config.chunk_duration = 1.0
config.scenario_name_filter = "driving/[0-9].*"

dataset = DBM_Dataset(config)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=dataset.collate, num_workers=4)
```

The remainder of this document describes each of the parameters and the expected output data.


# Dataset paths and filtering scenarios

## base_path - string - defaults to "dataset/Vehicle/No-Video/"
Used to specify the base path used by all other dataset paths. This is prepended to all other paths, so leave blank if some paths do not share a common root. This dataset allows loading of multiple directories with different frame rates and feature subsets.

## index_relative_path - string - defaults to "Resampled_previous_10"
Specifies which dataset directory to use for indexing the dataset and getting scenario and participant metadata. This must contain scenario_index.csv for each participant.

## scenario_name_filter - string - defaults to ".*"
The regex filter used to determine which scenarios to load.


# Chunking

## chunk_strategy - string - defaults to "end"
How to sample chunks from scenarios.
- "full": Returns the entire scenario data instead of chunks.
- "random": Provides randomly selected chunks from within the original scenario.
- "start": Provides chunks starting at the beginning of the scenario.
- "end": Provides chunks starting at the end of the scenario.

## chunk_duration - float - defaults to 10.
The duration of the chunks extracted from the scenario. Ignored if chunk strategy is "full".

## chunks_per_scenario - integer - defaults to 1
The maximum number if chunks extracted for each scenario. Setting this to None instead loads a variable number of chunks based on the scenario length. Behaves as follows for each chunk_strategy:
- "full": Ignored.
- "random": Sets the number of chunks loaded. None creates a number of chunks euqal to the scenario length divided by the chunk length (rounded down).
- "start" and "end": First determined to variable number of chunks possible using chunk duration and offset. Uses this if set to None. Otherwise, takes the smaller of the two values.

## chunk_start_offset - float - defaults to 0.
The number of seconds to ignore from the start of the scenario for all chunking strategies.

## chunk_end_offset - float - defaults to 0.
The number of seconds to ignore from the end of the scenario for all chunking strategies.

## chunk_fps - float - defaults to None
The frames per second to assume when creating chunks and the resample timings. If set to None, uses the maximum of all feature frame rates. However, if using None, must at least specify one feature with an associated frame rate of a "full" chunking strategy will be automatically selected.

## cache_chunks - boolean - defaults to False
Whether or not to reuse selected chunks from the first time they are created. This is useful for validation and test sets where you want the same chunks between multiple epochs, especially when using the "random" chunk strategy.

## chunk_stride - float - defaults to 5.
Only valid for "start" and "end" chunking strategies. The time offset or stride between chunks when creating muiltiple from one scenario.


# Features - dictionary (key string, value Feature) - defaults to {}
Features are loaded for each chunk by listing them in the "features" config dictionary. These will populate the "features" output, when loading a sample from the dictionary. Features are divided into four different types - tabular data, object attributes, object tracks, and video. Regardless of type, features can all have the following attributes:

## Dictionary Key - string
When creating a feature dictionary entry, the key specifies the feature name that will be used to refer to it once loaded. This will be the key used to access the feature in the "features" dictionary of the loaded sample.

## relative_path - string
This path is appended to the base_path and points to the subdirectory in the dataset where the feature is stored (e.g. "Resampled_previous_10").

## file_name - string
This specifies the file or directory name within the scenario directory that holds the feature. It should not include the file extension or "sim_bag", as these are automatically determined. If not specified, this is set to the dictionary key.

## fps - float
Gives the frame rate of the associated feature. Not needed for "object_attributes".

## type - string
Specifies the feature type - "tabular", "object_attributes". "object_tracks", or "video". These are explained in more detail below:


## type "tabular"
Tabular data contains a table with a consistent number of columns, but has a varying number of rows based on the duration of the scenario. It is assumed to have a consistent frame rate and has an index named "resampled_epoch_ns", which gives the time since the beginning of the scenario in nanoseconds. The index has type int64. Currently, all tabular data is assumed to be of floating point type.

### columns - list of strings
Specifies the list of columns to be extracted from the loaded CSV dataframe. If a string is given, it is converted to a list of length one. If the columns are not specified, the dataset loads all of the columns available. However, you are required to specify the columns if using the collate function.

### Example
```
"tobii": {
        "relative_path": "Resampled_previous_10",
        "file_name": "experiment_tobii_frame",
        "fps": 10,
        "type": "tabular",
        "columns": ["tobii_left_eye_gaze_pt_validity",],
}
```


## type "object_attributes"
Scenarios can have a varying number of objects at any given time. Object attributes describe the features that are consistent over the course of an entire scenario. This includes things like ado vehicle type. As such, it has a consistent number of columns and a number of rows equal to the amount of objects in the scenario. Because the values do not vary over time, it is not necessary to give these features an FPS. While not required, if the columns "start_epoch_ns" and "end_epoch_ns" are specified, these are compared with the loaded chunk's start and end times and objects that do not intersect with this time are excluded.

### object_id_column - string
The column used to represent the object ID in the stored feature. If not specified, "object_id" is assumed. Either way, after the feature is loaded the index is renamed to "object_id" and has type int64.

### columns - list of strings
Specifies the list of columns to be extracted from the loaded CSV dataframe. If a string is given, it is converted to a list of length one. If the columns are not specified, the dataset loads all of the columns available. Use column_types instead if using the collate function.

### column_types - dictionary (key string, value type)
Gives the columns to be loaded and their expected types. This parameter automatically overrides the "columns" parameter, if specified. The types must be explicitly specified if using collation.

### max_objects - integer
Specifies the number of objects to allocate when performing collation so that features across a batch can have the same size. The features are also provided with validity bits so unused objects can be determined downstream.

### Example
```
"carla_actors": {
        "relative_path": "Resampled_previous_10",
        "file_name": "carla_actor_list",
        "type": "object_attributes",
        "object_id_column": "carla_actor_id",
        "column_types": {"carla_actor_type" : StringDType},
        "max_objects": 32
},
```


## type "object_tracks"
Scenarios can have a varying number of objects at any given time. Object tracks describe the features that are varying over the course of an entire scenario and also vary in number of objects at any given time. This includes things like ado vehicle movement. As such, it has a consistent number of columns and a number of rows equal to summed total of each object's timestamps. These features are loaded into a dictionary where the key is the object ID (int64) and the value is the table of tracks with an index of "resampled_epoch_ns" (int64). When chunking, objects are only included in the dictionary if they have timestamps that intersect the chunk timings. Currently, all object track data is assumed to be of floating point type.

### object_id_column - string
The column used to represent the object ID in the stored feature. If not specified, "object_id" is assumed.

#### columns - list of strings
Specifies the list of columns to be extracted from the loaded CSV dataframe for each object. If a string is given, it is converted to a list of length one. If the columns are not specified, the dataset loads all of the columns available. However, you are required to specify the columns if using the collate function.

### max_objects - integer
Specifies the number of objects to allocate when performing collation so that features across a batch can have the same size. The features are also provided with validity bits so unused objects or timestamps can be determined downstream.

### Example
```
"ado_tracks": {
        "relative_path": "Resampled_previous_10_derived",
        "type": "object_tracks",
        "fps": 10,
        "object_id_column": "object_id",
        "columns": ["carla_objects_pose_x", "carla_objects_pose_y", "carla_objects_pose_z",],
        "max_objects": 32
},
```


## type "video"
Video data is used to specify image frames to load for each timestamp over a chunk. These image frames are specified by a separately loaded timing table, which contains 
an index named "resampled_epoch_ns" and at least one column named "resampled_frame", specifying the associated frame in the file.

#### columns - list of strings
Specifies the list of columns to be extracted from the loaded CSV timing dataframe. If a string is given, it is converted to a list of length one. If the columns are not specified, the dataset loads all of the columns available. If the columns are not specified when using collation, the timing data is not stored in the sample.

### resolution - tuple of length 3
Gives the resolution and number of channels of the video in (width, height, channels) order. If not specified, is determined from the video. However, you are requires to specify when using collation.

### interpolation - OpenCV interpolation type - defaults to cv2.INTER_NEAREST
Specifies the OpenCV interpolation method to use when the loaded frame does not match the above resolution.

### video_type - string - defaults to "rgb"
Gives the video type to determine if any additional image processing is needed when loading (e.g. for "depth" and "segmentation" videos).

### skip_frames - boolean - defaults to False
If specified and True, does not load the video frames. This is useful if you are only interested in the timings or want to manually load the video frame-by-frame during derived feature extraction or visualization.

### Example
```
"cam_front": {
        "relative_path": "Resampled_previous_10",
        "type": "video",
        "fps": 10,
        "columns": ["frame_in_chunk",]
        "resolution": (960//2, 400//2, 3),
        "interpolation": cv2.INTER_AREA,
        "video_type": "depth",
        "skip_frames": False,
},
```

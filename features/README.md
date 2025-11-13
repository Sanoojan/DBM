# Derived feature processing
Derived features are any features that are created based on the original dataset resampled features. Out of convention, these features are output to a new directory with a mirrored structure (e.g. Resampled_previous_10_derived). The features follow the same format at the original data types (see "data/README.md" for more info on each type). Example feature processing scripts can be found in "generate_derived_features.py" and "generate_derived_video_features.py". Feature processing can use input features from multiple sources and frame rates, but requires that the outputs all be of the same frame rate (chunk_fps).

# Defining the dataset
As described in "data/README.md", you need to first declare the DBM_Dataset_Config and create the DBM_Dataset. Make sure to specify all input features needed for processing. A parallel pool is then used to iterate over all samples specified and process each sample. Use num_processes=0 for debugging purposes. Derived feature processing should generally operate on the entire scenario, and as such should use a "full" chunking strategy. As such, it may be inadvisable to load video features without skip_frames set to True, as it would otherwise attempt to load frames from the entire scenario at once. Instead, use init_frame_reader and frame_reader.read_frames to read frames as needed (ensuring that sample.close is called for cleanup). Alternatively, use get_frame_reader and control the reader handle yourself.

# Processing each sample
## Create the OutputWriter
The beginning of each sample processing function should look like the following:

```
def process_sample(dataset, sample_number, base_out_dir):
    sample = dataset[sample_number]
    output_writer = OutputWriter(sample, base_out_dir)

    split_ego_ado_tracks(sample, output_writer)
```

This loads the sample from the dataset and specifies an OutputWriter that handles any writing of derived features to the file system. All input features should be read using the dataset loaded sample and all output features should be sent though the OutputWriter to ensure compatability with the dataset and dataloader. To ensure this encapsulation, it is best to divide derived features into functions that just take "sample" and "output_writer" as inputs, as in the example files. You should also "close" the sample when finished, as this closes any associated video frame readers and other unloading.

## Video features
Video processing is typically done frame by frame so that the entire video input or output does not need to be held in memory at once. The sample below involves loading video frames and shrinking them to a smaller size. The feature setup results in the frames not being loaded until actually requested. Furthermore, the video loader itself will do the image scaling by specifying a lower resolution.

```
config.features = {
    "cam_front": {
        "relative_path": "Resampled_previous_10",
        "type": "video",
        "fps": 10,
        "resolution": (400//4, 960//4, 3),
        "skip_frames": True,
    }
}
```

### Create the VideoFrameWriter
To facilitate this frame-by-frame processing, create the VideoFrameWriter. Here, the writer shared the same encoding as the input video.

```
# Check that video exists
if "cam_front" not in sample.features:
    return

# Set up the video output
input_video = sample.features["cam_front"]
fourcc = int(input_video.frame_reader.cap.get(cv2.CAP_PROP_FOURCC))
fps = input_video.fps
image_size = input_video.image_size
frame_writer = output_writer.get_video_frame_writer("cam_front_scale4", ".avi", fourcc, fps, (image_size[1], image_size[0]))
```

### Creating the video features
Then iterate through each frame time, load the frame, process the frame, and write it out. When writing the frame, it is also neccessary to provide the frame time that corresponds. This will become the "resampled_epoch_ns" in the eventual output timing CSV file. The timing_row can be ignored if no additional information should be stored in the timing CSV.

```
# Loop through video frames one at a time
for frame in input_video.read_frames():
    # Image data stored in frame.image_data - write back out since already scaled
    frame_writer.write(frame.time, frame.image_data, frame.timing_row)
```

A similar loop would still be needed if only using video for input, but not the output (e.g. semantic segmentation value at gaze point). In this case, you would iterate through the input frames and construct an output dataframe in the format of one of the features types below.

### Closing the video
When finished writing the features, close the output video. Closing the output video will also verify that it contains the expected number of frames, based on the chunk_fps and determined resample times. The frame_reader is automatically closed by the sample.close function.

```
frame_writer.close()
```

## Other features 
Unlike video features, all other features are created by simply loading the sample for the entire scenario, processing the features into dataframes, and then outputting the entire dataframes. The sample below involves loading the raw carla actor list and object tracks and splitting the outputs into ego and ado. Ego tracks are output as tabular data, while the ado tracks are re-exported as object_tracks.

### Loading the features
In this example, the carla actor list and actor tracks are loaded from the sample. These features were originally specified in the DBM_Dataset_Config.

```
# Check that the required input features exist and read
if "carla_actors" not in sample.features or "carla_tracks" not in sample.features:
    return
carla_actors = sample.features["carla_actors"].attributes
carla_tracks = sample.features["carla_tracks"].tracks
```

### Determining the ego using the actor list
The ego object_id is first determined using the carla actor list.

```
# Determine the ego - needed to separate
ego_candidates = carla_actors[carla_actors["carla_actor_rolename"] == "hero"].index.values
if len(ego_candidates) != 1:
    return
ego_id = ego_candidates[0]
```

### Writing the ego features (tabular)
We next select the ego from the carla_tracks dictionary and output its data as a tabular feature. Tabular features should contain a dataframe with a single index named resampled_epoch_ns (int64). They should have a number of rows equal to the number of resample times in for the scenario at the chunk_fps.

```
output_writer.save_tabular(carla_tracks[ego_id], "ego_tracks")
```

### Constructing the ado tracks
In this example, we simply remove the ego from the track dictionary and check that a minimal number of ado vehicles are present in the scenario before writing the ado features.

```
# Remove the ego from the tracks
del carla_tracks[ego_id]

# Determine if the ado tracks are valid
if len(carla_tracks) < 10:
    return
```

### Writing the ado tracks
Object track features are formatted as a dictionary, where the key is the object_id and the value is the table of tracks for the object over time. The object does not need to have valid tracks across all timestamps of the scenario. Each object table should contain a single index - resampled_epoch_ns (int64).

```
output_writer.save_object_tracks(carla_tracks, "ado_tracks")
```

### Writing object_attributes
While not shown in the above sample, object_attributes are also supported. As described in "data/README.md", these features must be a dataframe with an index named "object_id" (int64) and have one entry for each object in the scenario. The table can optionally include "start_epoch_ns" and "end_epoch_ns" to specify when the object is valid. Use "output_writer.save_object_attributes" to write to disk.

#!/bin/bash

SOURCE_DIR="/data/motion-simulator-logs/Processed/Clean/Participants"
DEST_DIR="/data/motion-simulator-logs/Processed/Viz/Participants"

if [ -z "$1" ]; then
  camera_name="cam_front"
else
  camera_name="$1"
fi

if [ -z "$2" ]; then
  last_n=600
else
  last_n="$2"
fi


find "$SOURCE_DIR" -type f -name "*video.avi" | grep "$camera_name" | grep driving | grep -v practice | grep -v P701 | while read -r file; do

  relative_path="${file#$SOURCE_DIR/}"
  output_dir="$DEST_DIR/$(dirname "$relative_path")"
  mkdir -p "$output_dir"
  output_file="$output_dir/$(basename "$relative_path")"

  if [ -f "$output_file" ]; then
    echo ""$output_file" exists, skipping."
    continue
  fi

  echo "$output_file"

  # find frame 20s from the end (assuming 30 Hz)
  csv_file="$(dirname "$file")"/frame_timing.csv
  frame_to_clip=`tail -n 600 "$csv_file" | head -n 1 | cut -d "," -f 4`
  fps=30
  starting_time=`echo "scale=2; "$frame_to_clip" / "$fps"" | tr -d $'\r' | bc`

  # cut, downsample 2x, compress
  ffmpeg -y -nostdin -i "$file" -vf "scale=iw/2:ih/2" -ss $starting_time -c:v libx264 -crf 23 -preset fast "$output_file"
  echo "Processed: $file -> $output_file"

done

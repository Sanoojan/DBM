#!/bin/bash

if [ -z "$1" ]; then
  echo "Must provide an arugment for montage type"
  exit 1
else
  montage_type="$1"
fi

if [ "$montage_type" == "scenario_cam_front" ]; then
  echo "Montage front camera, last 20 seconds, for all participants, by scenario"
  for scenario in {1a-,2-,2b-,3c-,5-,6e-,7a-,8a-}; do
    python montage_videos.py hazard-cam_front-"${scenario%-}".avi 7 `find /data/motion-simulator-logs/Processed/Viz/Participants -name "*.avi" | grep $scenario | grep cam_front | sort`;
  done

elif [ "$montage_type" == "scenario_cam_face" ]; then
  echo "Montage cam_face, last 20 seconds, for all participants, by scenario"
  for scenario in {1a-,2-,2b-,3c-,5-,6e-,7a-,8a-}; do
    python montage_videos.py hazard-cam_face-"${scenario%-}".avi 10 `find /data/motion-simulator-logs/Processed/Viz/Participants -name "*.avi" | grep $scenario | grep cam_face | sort`;
  done

elif [ "$montage_type" == "stationary_cam_face" ]; then
  echo "Montage cam_face, for all participants, during stationary tasks"
  for scenario in {fixed_gaze,gaze_tracking,choice_reaction,silent_reading}; do
    python montage_videos.py st-cam_face-R1-"${scenario}".avi 10 `find /data/motion-simulator-logs/Processed/Viz/Participants -name "*.avi" | grep $scenario | grep cam_face | grep R1 | sort`;
    python montage_videos.py st-cam_face-R2-"${scenario}".avi 10 `find /data/motion-simulator-logs/Processed/Viz/Participants -name "*.avi" | grep $scenario | grep cam_face | grep R2 | sort`;
  done

elif [ "$montage_type" == "hazard_dd" ]; then
  echo "Montage cam_front, for P7XX participants, during driving hazard, before and after drinking"
  for scenario in {1a-,2-,2b-,3c-,5-,6e-,7a-,8a-}; do
    python montage_videos.py hazard-cam_front-dd0-"${scenario%-}".avi 3 `find /data/motion-simulator-logs/Processed/Viz/Participants -name "*.avi" | grep $scenario | grep cam_front | grep P7 | grep R1 | sort`;
    python montage_videos.py hazard-cam_front-dd1-"${scenario%-}".avi 3 `find /data/motion-simulator-logs/Processed/Viz/Participants -name "*.avi" | grep $scenario | grep cam_front | grep P7 | grep R2 | sort`;
  done

elif [ "$montage_type" == "hazard_cd" ]; then
  echo "Montage cam_front, for 72XX participants, during driving hazard, with and without cog distraction"
  for scenario in {1a-,2-,2b-,3c-,5-,6e-,7a-,8a-}; do
    python montage_videos.py hazard-cam_front-cd0-"${scenario%-}".avi 3 `find /data/motion-simulator-logs/Processed/Viz/Participants -name "*.avi" | grep $scenario | grep cam_front | grep no_task | grep 72 | sort`;
    python montage_videos.py hazard-cam_front-cd1-"${scenario%-}".avi 3 `find /data/motion-simulator-logs/Processed/Viz/Participants -name "*.avi" | grep $scenario | grep cam_front | grep -v no_task | grep 72 | sort`;
  done

else
  echo "Can't recognize montage type argument"
  exit 1
fi
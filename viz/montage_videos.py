import sys

from moviepy.editor import (
    ColorClip,
    CompositeVideoClip,
    VideoFileClip,
    clips_array,
    concatenate_videoclips,
)
from moviepy.video.VideoClip import TextClip


def create_blank_clip(width, height, fps, duration_s):
    blank_clip = ColorClip(size=(width, height), color=(0, 0, 0), duration=duration_s)
    blank_clip = blank_clip.set_fps(fps)
    return blank_clip


def create_montage(output_file, num_cols, input_videos):
    clips = []

    for video in input_videos:
        clip = VideoFileClip(video)

        filename = video.split("Participants/")[-1].split("cam_")[0]
        if filename.startswith("P"):
            bg_color = "blue"
        else:
            bg_color = "black"
        # Note: you may need to edit /etc/ImageMagick*/policy.xml to add permissions for @*:
        #   <policy domain="path" rights="read|write" pattern="@*"/>
        text = TextClip(filename, fontsize=12, color="white", bg_color=bg_color, method="caption")
        text = text.set_duration(clip.duration).set_position(("center", "top"))
        video_with_text = CompositeVideoClip([clip, text])

        clips.append(video_with_text)

    extra_clips = len(clips) % num_cols
    if extra_clips != 0:
        for ii in range(num_cols - extra_clips):
            w, h = clips[0].w, clips[0].h
            fps = clips[0].fps
            dur = clips[0].duration
            clips.append(create_blank_clip(w, h, fps, dur))

    min_duration = min([clip.duration for clip in clips])
    clips = [clip.subclip(0, min_duration) for clip in clips]
    rows = [clips[i : i + num_cols] for i in range(0, len(clips), num_cols)]
    final_clip = clips_array(rows)

    final_clip.write_videofile(output_file, codec="libx264", fps=30)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("See scripts/gen_montage_videos.sh for example of use")
        print("  python montage_videos.py PATH_TO_OUTPUT_VIDEO NUM_COLUMNS PATH_TO_INPUT_1 ... PATH_TO_INPUT_N")
        sys.exit(1)

    output_file = sys.argv[1]
    num_cols = int(sys.argv[2])
    input_videos = sys.argv[3:]

    create_montage(output_file, num_cols, input_videos)

import csv
import os
import pickle

import cv2
import numpy as np
import torch


class OutputLogger:
    def __init__(self, use_wandb, output_dir):
        self.use_wandb = use_wandb
        self.output_dir = output_dir
        if not self.use_wandb:
            self.output_values_log = {}

    def output_values(self, values, step):
        if self.use_wandb:
            import wandb

            wandb.log(values, step=step)
        else:
            if step not in self.output_values_log:
                self.output_values_log[step] = {}
            for key, value in values.items():
                if torch.is_tensor(value):
                    value = value.item()
                self.output_values_log[step][key] = value

    def finalize(self):
        if not self.use_wandb:
            # Determine value log headers
            output_values_headers = set()
            for epoch_values in self.output_values_log.values():
                for key in epoch_values.keys():
                    output_values_headers.add(key)
            output_values_headers = [
                "epoch",
            ] + list(output_values_headers)

            # Determine epochs logged
            epochs = list(self.output_values_log.keys())
            epochs.sort()

            # Output csv
            csv_path = os.path.join(self.output_dir, "logged_values.csv")
            with open(csv_path, "w") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=output_values_headers)
                writer.writeheader()
                for epoch in epochs:
                    self.output_values_log[epoch]["epoch"] = epoch
                    writer.writerow(self.output_values_log[epoch])

    def save(self, path):
        if self.use_wandb:
            import wandb

            wandb.save(path)

    def output_video(self, video_data, epoch, Fs, vid_name="val"):
        video_data *= 255
        video_data = video_data.astype(np.uint8)
        if video_data.shape[0] != 3:
            # Correct for videos with less than three channels
            video_data = video_data[0]
            video_data = np.repeat(video_data[np.newaxis], 3, axis=0)
        if self.use_wandb:
            import wandb

            video_data = np.transpose(video_data, (1, 0, 2, 3))
            sample_video = wandb.Video(video_data, fps=Fs, format="mp4")
            wandb.log({vid_name + "_video": sample_video}, step=epoch)
        else:
            dir_path = os.path.join(self.output_dir, "video", vid_name)
            os.makedirs(dir_path, exist_ok=True)
            file_path = os.path.join(dir_path, str(epoch) + ".mp4")
            video_data = np.transpose(video_data, (1, 0, 2, 3))
            video_writer = cv2.VideoWriter(
                file_path, cv2.VideoWriter_fourcc(*"mp4v"), Fs, (video_data.shape[3], video_data.shape[2])
            )
            for frame_on in range(video_data.shape[0]):
                img = np.moveaxis(video_data[frame_on], 0, 2)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                video_writer.write(img)
            video_writer.release()

    def output_epoch_stats(self, stats, epoch, phase):
        if self.use_wandb:
            import wandb

        raise NotImplementedError

    def save_best_model_outputs(self, best_val_stat_name, model_outputs):
        best_val_out_path = os.path.join(self.output_dir, best_val_stat_name)
        os.makedirs(best_val_out_path, exist_ok=True)
        for fold_name, fold_model_outputs in model_outputs.items():
            with open(os.path.join(best_val_out_path, f"{fold_name}.pkl"), "wb") as handle:
                pickle.dump(fold_model_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

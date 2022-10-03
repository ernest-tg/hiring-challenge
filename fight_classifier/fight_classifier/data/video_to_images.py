import os
from pathlib import Path
from typing import List

import fire
import pandas as pd
from PIL import Image
import skvideo.io


def save_video_as_frames(video_path: Path, frame_path_pattern: str) -> List[Path]:
    """Saves every frame of a video

    Args:
        video_path (Path): Path to the video
        frame_path_pattern (str):
            String with a unique placeholder, which will be replaced with the
            id of each frame.

    Returns:
        frames_paths (List[Path]):
            The list of the paths of the frames we saved.
    """
    video = skvideo.io.vread(str(video_path))
    frames_paths = []
    for frame_id, frame in enumerate(video):
        frame_path = frame_path_pattern.format(frame_id)
        frame_im = Image.fromarray(frame)
        frame_im.save(frame_path)
        frames_paths.append(Path(frame_path))
    return frames_paths


def save_videos_dataset_as_frames(
        videos_df: pd.DataFrame, videos_dir: Path, frames_dir: Path
):
    # create an empty dataframe with the columns of 'videos_df' (minus index?) +
    # path + 'id_in_video', create index with (video_path, id_in_video)
    # For each row in the
    frames_columns = pd.concat([videos_df.columns.to_series(), pd.Series(['frame_path', 'frame_id'])])
    frames_df = pd.DataFrame(columns=frames_columns)
    frames_rows = []
    
    os.makedirs(frames_dir, exist_ok=True)
    for _, video_row in videos_df.iterrows():
        video_path = videos_dir / video_row['video_path']
        
        # e.g. 'fights_newfi12.avi'
        flat_video_rel_path = video_row.video_path.replace('/', '_')
        # e.g. 'fights_newfi12'
        frame_name_prefix = os.path.splitext(flat_video_rel_path)[0]
        frame_name_pattern = frame_name_prefix + '_{}.jpeg'
        frame_path_pattern = os.path.join(frames_dir, frame_name_pattern)
        
        frames_paths = save_video_as_frames(
            video_path=video_path,
            frame_path_pattern=frame_path_pattern,
        )
        for frame_id, frame_path in enumerate(frames_paths):
            frame_row = video_row.copy()
            frame_row.at['frame_id'] = frame_id
            frame_row.at['frame_path'] = frame_path
            frames_rows.append(frame_row)
    frames_df = pd.DataFrame(frames_rows)
    frames_csv_path = frames_dir / 'frames.csv'
    frames_df.to_csv(frames_csv_path)
    return frames_df

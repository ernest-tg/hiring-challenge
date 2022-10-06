import os
from pathlib import Path
from typing import List

import pandas as pd
from PIL import Image
import skvideo.io
import tqdm


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
) -> pd.DataFrame:
    """Saves every frame of every video in the dataframe in `frames_dir`

    The frames of a video whose path is '<videos_dir>/foo/bar/vid.avi' will be
    saved as '<frames_dir>/foo_bar_vid_0.png', '<frames_dir>/foo_bar_vid_1.png',
    and so on.

    Args:
        videos_df (pd.DataFrame):
            A DataFrame containing a 'video_path' column, the elements of this
            column being paths of videos, relative to <videos_dir>.
        videos_dir (Path):
            Path to a local directory. The paths to the videos will be given
            relative to this path.
        frames_dir (Path):
            Path to the directory where the frames should be saved (it will be
            created if it does not exist yet).

    Returns:
        frames_df (pd.DataFrame):
            A row for every frame we save, with the data of the video it comes
            from, as well as 'frame_id' (the index of the frame within the
            video) and 'frame_path' (the path where the frame is saved).

            This dataframe is also saved at '<frames_dir>/frames.csv'.
    """
    os.makedirs(frames_dir, exist_ok=True)

    # Creates an empty dataframe with the columns of 'videos_df', as well as
    # 'frame_path', and 'frame_id'.
    frames_columns = pd.concat(
        [videos_df.columns.to_series(), pd.Series(['frame_path', 'frame_id'])])
    frames_df = pd.DataFrame(columns=frames_columns)

    frames_rows = []
    for _, video_row in tqdm.tqdm(videos_df.iterrows(), total=len(videos_df)):
        video_path = videos_dir / video_row['video_path']

        # e.g. 'fights_newfi12.avi'
        flat_video_rel_path = video_row.video_path.replace('/', '_')
        # e.g. 'fights_newfi12'
        frame_name_prefix = os.path.splitext(flat_video_rel_path)[0]
        frame_name_pattern = frame_name_prefix + '_{}.png'
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

from pathlib import Path

import numpy as np

import evo.main_ape as main_ape
from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface#, plot
#from .plot_utils import plot_trajectory


def _build_traj_est(traj_est, timestamps):
    return PoseTrajectory3D(
        positions_xyz=traj_est[:, :3],
        orientations_quat_wxyz=traj_est[:, [6, 3, 4, 5]],
        timestamps=np.asarray(timestamps, dtype=np.float64),
    )


def infer_timestamps_from_rgb_list(imagedir, expected_len=None):
    rgb_txt = Path(imagedir).parent / "rgb.txt"
    if not rgb_txt.exists():
        return None

    entries = []
    with rgb_txt.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            entries.append(float(parts[0]))

    if expected_len is not None and len(entries) < expected_len:
        return None
    if expected_len is not None:
        entries = entries[:expected_len]
    return np.asarray(entries, dtype=np.float64)


def eval_tum_stats(traj_est, timestamps, gt_file):

    traj_ref = file_interface.read_tum_trajectory_file(gt_file)
    traj_est = _build_traj_est(traj_est, timestamps)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    result = main_ape.ape(
        traj_ref,
        traj_est,
        est_name='traj',
        pose_relation=PoseRelation.translation_part,
        align=True,
        correct_scale=True,
    )
    return result.stats


def run_eval_tum(traj_est, timestamps, gt_file, plot_file_name=None, save_dir=None, plot=False):
    stats = eval_tum_stats(traj_est, timestamps, gt_file)
    ate_score = stats["rmse"]

    #if plot and plot_file_name is not None and save_dir is not None:
    #    plot_trajectory(traj_est, traj_ref, f"{plot_file_name} (ATE: {ate_score:.03f})",
    #                    f"{save_dir}/{plot_file_name}.pdf", align=True, correct_scale=True)

    return ate_score

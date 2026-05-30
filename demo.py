import os
import json
import sys
import time
import signal
from argparse import ArgumentParser
from multiprocessing import Process, Queue
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from gui import gui_utils, gui
import os.path as osp
from scipy.spatial.transform import Rotation as R

# DPVO imports
from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.stream import image_stream, image_stream_tum, video_stream
from utils.eval_traj import eval_tum_stats, infer_timestamps_from_rgb_list, run_eval_tum
from dpvo.lietorch import SE3
from pi3.utils.geometry import depth_edge
from mapping.io_utils import save_ply
from evaluation.evaluate_gaussian_renderings import evaluate as evaluate_renderings
from evaluation.evaluate_gaussian_nvs import evaluate_nvs

class Keyframe:
    """Simple keyframe class for GUI visualization"""
    def __init__(self, pose_matrix, uid):
        self.pose_matrix = pose_matrix
        self.uid = uid
        self._camera_center = torch.from_numpy(pose_matrix[:3, 3]).cuda()
    
    @property
    def get_inv_RT(self):
        return None, self._camera_center

class Pi_SAM:
    def __init__(self, config):
        self.device = 'cuda'
        self.gui_process = None
        self.is_paused = False
        
        # DPVO data parameters
        self.imagedir = config.get("imagedir", "")
        self.calib = config.get("calib", "")
        self.stride = config.get("stride", 1)
        self.viz = config.get("viz", False)
        self.tum = config.get("tum", False)
        self.edge = config.get("edge", 0)
        self.gt = config.get("gt", None)
        self.save_outputs = config.get("save_outputs", True)
        self.save_rendered_images = config.get("save_rendered_images", True)
        self.save_gaussian_map_enabled = config.get("save_gaussian_map", False)
        self.save_debug_maps = config.get("save_debug_maps", False)
        self.output_dir = Path(config.get("output_dir", "outputs/demo"))
        self.render_dir = self.output_dir / "rendered"
        self.eval_gt_dir = config.get("eval_gt_dir", None)
        self.eval_out_dir = config.get("eval_out_dir", None)

        
        # Initialize parameters
        self.slam_config = cfg
        self.dpvo_network_path = "checkpoints/dpvo.pth"
        self.pi3_network_path = "checkpoints/model.safetensors"
        
        self.slam = None
        
        # For visualizer
        self.historical_poses = []
        self.frame_counter = 0
        self.predicted_points_world = []
        self.predicted_colors = []
        self.background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
        signal.signal(signal.SIGINT, self.signal_handler)

        self.q_main2vis = mp.Queue()
        self.q_vis2main = mp.Queue()
        self.params_gui = gui_utils.ParamsGUI(
            background=self.background,
            q_main2vis=self.q_main2vis,
            q_vis2main=self.q_vis2main,
        )

        if self.viz:
            # Start GUI process
            self.gui_process = mp.Process(target=gui.run, args=(self.params_gui,))
            self.gui_process.start()

            time.sleep(1)  # Wait for GUI initialization

        
    def shutdown_gui(self):
        if self.gui_process is not None and self.gui_process.is_alive():
            print("showdown GUI...")
            self.gui_process.terminate()
            self.gui_process.join(timeout=5)
            if self.gui_process.is_alive():
                self.gui_process.kill()
            self.gui_process.close()

    def signal_handler(self, signum, frame):
        self.shutdown_gui()
        sys.exit(0)

    def save_trajectory_tum(self, poses, tstamps):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        traj_path = self.output_dir / "trajectory_tum.txt"
        with traj_path.open("w") as f:
            for tstamp, pose in zip(tstamps, poses):
                tx, ty, tz, qx, qy, qz, qw = pose.tolist()
                f.write(f"{float(tstamp):.9f} {tx:.9f} {ty:.9f} {tz:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")
        print(f"Saved trajectory to {traj_path}")

    def save_sparse_map(self):
        if self.slam is None or self.slam.m == 0:
            return

        points = self.slam.pg.points_.detach().cpu().numpy()[:self.slam.m]
        colors = self.slam.pg.colors_.view(-1, 3).detach().cpu().numpy()[:self.slam.m]

        if hasattr(self.slam.pg, 'var_') and self.slam.pg.var_ is not None:
            variances = self.slam.pg.var_.view(-1).detach().cpu().numpy()[:self.slam.m]
            keep = variances <= 1.0
            points = points[keep]
            colors = colors[keep]

        if len(points) == 0:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_ply(str(self.output_dir / "sparse_map"), points, colors.astype(np.uint8))

    def save_predicted_map(self):
        if not self.predicted_points_world:
            return

        points = np.concatenate(self.predicted_points_world, axis=0)
        colors = np.concatenate(self.predicted_colors, axis=0)

        if len(points) == 0:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        colors_uint8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
        save_ply(str(self.output_dir / "pi3_predicted_points"), points.astype(np.float32), colors_uint8)

    def save_gaussian_map(self):
        if not self.save_gaussian_map_enabled:
            return
        if self.slam is None or getattr(self.slam, "gaussian_bridge", None) is None:
            return
        self.slam.gaussian_bridge.save(self.output_dir / "gaussian_map")

    def save_gaussian_summary(self):
        if self.slam is None or getattr(self.slam, "gaussian_bridge", None) is None:
            return
        summary = self.slam.gaussian_bridge.mapper.summary()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / "gaussian_summary.json").open("w") as f:
            json.dump(summary, f, indent=2)

    def save_run_config(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        run_cfg = {
            "imagedir": self.imagedir,
            "calib": self.calib,
            "stride": self.stride,
            "tum": self.tum,
            "edge": self.edge,
            "output_dir": str(self.output_dir),
            "save_rendered_images": self.save_rendered_images,
            "save_gaussian_map": self.save_gaussian_map_enabled,
            "gaussian": {
                "enabled": bool(getattr(self.slam_config, "GAUSSIAN_MAPPING", False)),
                "window_size": int(getattr(self.slam_config, "GAUSSIAN_WINDOW_SIZE", 0)),
                "init_window_size": int(getattr(self.slam_config, "GAUSSIAN_INIT_WINDOW_SIZE", 0)),
                "motion_score": float(getattr(self.slam_config, "GAUSSIAN_MIN_MOTION_SCORE", 0.0)),
                "dynamic_thresh": float(getattr(self.slam_config, "GAUSSIAN_DYNAMIC_THRESH", 0.0)),
                "confidence_thresh": float(getattr(self.slam_config, "GAUSSIAN_INIT_CONF_THRESH", 0.0)),
                "final_refine_iters": int(getattr(self.slam_config, "GAUSSIAN_FINAL_REFINEMENT_ITERS", 0)),
            },
        }
        with (self.output_dir / "run_config.json").open("w") as f:
            json.dump(run_cfg, f, indent=2)

    def save_rendered_frame(self, frame_idx, input_img):
        if not self.save_rendered_images:
            return None
        if self.slam is None or getattr(self.slam, "gaussian_bridge", None) is None:
            return None

        render_pkg = self.slam.gaussian_bridge.render_frame(frame_idx)
        if render_pkg is None:
            return None

        rendered = torch.clamp(render_pkg["render"], 0.0, 1.0)
        rendered_np = (rendered.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        self.render_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.render_dir / f"frame_{frame_idx:05d}.png"), cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR))
        return rendered

    def save_outputs_to_disk(self, poses, tstamps):
        if not self.save_outputs:
            return

        self.save_run_config()
        self.save_trajectory_tum(poses, tstamps)
        if self.save_debug_maps:
            self.save_sparse_map()
            self.save_predicted_map()
        self.save_gaussian_map()
        self.save_gaussian_summary()

    def resolve_eval_timestamps(self, poses, tstamps):
        used_timestamps = np.asarray(tstamps, dtype=np.float64)
        mode = "estimated_timestamps"
        initial_error = None

        if self.gt is None:
            return used_timestamps, mode, initial_error

        try:
            eval_tum_stats(poses, used_timestamps, self.gt)
        except Exception as exc:
            initial_error = str(exc)
            inferred = infer_timestamps_from_rgb_list(self.imagedir, expected_len=len(poses))
            if inferred is not None:
                used_timestamps = inferred
                mode = "rgb_txt_timestamps"
        return used_timestamps, mode, initial_error

    def infer_nvs_dir(self):
        image_dir = Path(self.imagedir)
        if image_dir.is_dir():
            candidate = image_dir.parent / "nvs"
            if candidate.exists():
                return candidate
        return None

    def evaluate_trajectory_and_save(self, poses, tstamps):
        if self.gt is None:
            return None

        stats = None
        used_timestamps, mode, error_message = self.resolve_eval_timestamps(poses, tstamps)

        try:
            stats = eval_tum_stats(poses, used_timestamps, self.gt)
        except Exception as exc:
            if error_message is None:
                error_message = str(exc)
            raise

        result = {
            "ate_rmse_m": float(stats["rmse"]),
            "ate_mean_m": float(stats["mean"]),
            "ate_median_m": float(stats["median"]),
            "ate_std_m": float(stats["std"]),
            "ate_min_m": float(stats["min"]),
            "ate_max_m": float(stats["max"]),
            "num_frames": int(len(poses)),
            "timestamp_mode": mode,
        }
        if error_message is not None:
            result["initial_error"] = error_message

        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / "trajectory_eval.json").open("w") as f:
            json.dump(result, f, indent=2)
        return result

    def accumulate_predicted_map(self, img, predict_points, dynamic_mask, confidences, estimated_pose):
        if predict_points is None:
            return None, None, None

        dynamic_mask_for_gui = dynamic_mask.cpu()
        Hpi, Wpi = predict_points.shape[0], predict_points.shape[1]
        device = predict_points.device

        img_for_gui = img.float() / 255.0
        img_for_gui = img_for_gui[[2, 1, 0], :, :]
        img_resized = torch.nn.functional.interpolate(
            img_for_gui.unsqueeze(0), (Hpi, Wpi), mode="bilinear", align_corners=False
        ).squeeze(0)
        all_colors = img_resized.permute(1, 2, 0).reshape(-1, 3)

        static_mask_thresh = self.slam_config.STATIC_MASK_THRESH
        if dynamic_mask.shape[:2] != (Hpi, Wpi):
            dynamic_mask = torch.nn.functional.interpolate(
                dynamic_mask.unsqueeze(0).unsqueeze(0).to(device),
                (Hpi, Wpi), mode="bilinear", align_corners=False
            ).squeeze(0).squeeze(0)
        else:
            dynamic_mask = dynamic_mask.to(device)

        if confidences.shape[:2] != (Hpi, Wpi):
            confidences = torch.nn.functional.interpolate(
                confidences.unsqueeze(0).unsqueeze(0).to(device),
                (Hpi, Wpi), mode="bilinear", align_corners=False
            ).squeeze(0).squeeze(0)
        else:
            confidences = confidences.to(device)

        is_dynamic = dynamic_mask.view(-1) >= static_mask_thresh
        if torch.any(is_dynamic):
            red_color = torch.tensor([1.0, 0.0, 0.0], device=device)
            all_colors[is_dynamic] = red_color * 0.7 + all_colors[is_dynamic] * 0.3

        depth_map = predict_points[:, :, 2]
        edge_mask = depth_edge(depth_map, atol=0.5, rtol=0.1, kernel_size=3)
        stable_mask = ~edge_mask.view(-1)
        valid_mask = (confidences.view(-1) >= 0.01) & stable_mask
        sel = (torch.rand(Hpi * Wpi, device=device) < 0.3) & valid_mask

        T = torch.from_numpy(estimated_pose).to(device)
        pts_w = (predict_points.view(-1, 3)[sel] @ T[:3, :3].T) + T[:3, 3]

        pred_points_np = pts_w.cpu().numpy()
        pred_colors_np = all_colors[sel].cpu().numpy()
        self.predicted_points_world.append(pred_points_np)
        self.predicted_colors.append(pred_colors_np)

        return pred_points_np, pred_colors_np, dynamic_mask_for_gui
    
    def check_gui_commands(self):
        """Check for pause/resume commands from GUI"""
        try:
            while not self.q_vis2main.empty():
                packet = self.q_vis2main.get_nowait()
                if hasattr(packet, 'flag_pause') and packet.flag_pause is not None:
                    self.is_paused = packet.flag_pause
                    if self.is_paused:
                        print("SLAM paused by user")
                    else:
                        print("SLAM resumed by user")
        except:
            pass  # Queue is empty or other error, continue normally

    @torch.no_grad()
    def run(self):
        queue = Queue(maxsize=8)
        print("Starting Pi3MOS-SLAM + Gaussian mapping run...", flush=True)
        # Start image reader process
        if os.path.isdir(self.imagedir):
            if self.tum:
                stream = image_stream_tum
            else:
                stream = image_stream
            reader = Process(target=stream, args=(queue, self.imagedir, self.calib, self.stride, self.edge))
        else:
            reader = Process(target=video_stream, args=(queue, self.imagedir, self.calib, self.stride))
        
        reader.start()

        while True:
            # Check for GUI commands (pause/resume)
            self.check_gui_commands()
            
            # If paused, skip processing but still check for commands
            if self.viz and self.is_paused:
                time.sleep(0.1)
                continue
            
            (t, image, intrinsics) = queue.get()
            if t < 0:
                break
    
            img = torch.from_numpy(image).permute(2,0,1).cuda()
            intrinsics_tensor = None if intrinsics is None else torch.from_numpy(intrinsics).cuda()
            
            _, H, W = img.shape
            
            if self.slam is None:
                self.slam = DPVO(self.slam_config, self.dpvo_network_path, self.pi3_network_path, ht=H, wd=W)
                print(f"Loaded SLAM backend for frames of size {H}x{W}", flush=True)
            
            # When intrinsics are provided, pass them; otherwise estimates K internally
            if intrinsics_tensor is not None:
                predict_points, dynamic_mask, confidences = self.slam(t, img, intrinsics_tensor)
            else:
                predict_points, dynamic_mask, confidences = self.slam(t, img)

            frame_idx = self.slam.n - 1
            if frame_idx >= 0:
                pose_matrix = SE3(self.slam.poses[0, frame_idx]).matrix().cpu().numpy()
                estimated_pose = np.linalg.inv(pose_matrix)
                render_frame_id = int(self.slam.pg.tstamps_[frame_idx].item())
            else:
                estimated_pose = np.eye(4, dtype=np.float32)
                render_frame_id = frame_idx

            pred_points_np = pred_colors_np = dynamic_mask_for_gui = None
            if predict_points is not None:
                pred_points_np, pred_colors_np, dynamic_mask_for_gui = self.accumulate_predicted_map(
                    img, predict_points, dynamic_mask, confidences, estimated_pose
                )
            rendered = self.save_rendered_frame(render_frame_id, img)
            if frame_idx % 10 == 0:
                print(
                    f"Frame {frame_idx}: tracked, rendered={rendered is not None}, "
                    f"gaussian_backend={'on' if getattr(self.slam, 'gaussian_bridge', None) is not None else 'off'}",
                    flush=True,
                )

            if self.viz:
                points = self.slam.pg.points_.cpu().numpy()[:self.slam.m]
                colors = self.slam.pg.colors_.view(-1, 3).cpu().numpy()[:self.slam.m]
                colors = colors.astype(np.float32) / 255.0
                
                # Filter out points with high variance if available
                if hasattr(self.slam.pg, 'var_') and self.slam.pg.var_ is not None:
                    var_flat = self.slam.pg.var_.view(-1, 1).cpu().numpy()[:self.slam.m]
                    low_var_mask = var_flat.flatten() <= 1.0
                    
                    points = points[low_var_mask]
                    colors = colors[low_var_mask]

                if frame_idx >= 0:
                    if self.frame_counter % 3 == 0:
                        keyframe = Keyframe(estimated_pose, self.frame_counter)
                        self.historical_poses.append(keyframe)
                    self.frame_counter += 1
                img_for_gui = img.float() / 255.0
                img_for_gui = img_for_gui[[2, 1, 0], :, :]

                self.q_main2vis.put(
                    gui_utils.DatePacket(
                        points = points,
                        point_colors = colors,
                        pred_points = pred_points_np,
                        pred_point_colors = pred_colors_np,
                        current_pose = estimated_pose,
                        keyframes = self.historical_poses,
                        gtframes = None,
                        gtcolor = img_for_gui,
                        dynamicmask = dynamic_mask_for_gui
                    )
                )
                
                time.sleep(0.01)
        
        reader.join()
        print("Sequence finished, starting backend finalization...", flush=True)
        poses, tstamps = self.slam.terminate()
        if self.slam is not None and getattr(self.slam, "gaussian_bridge", None) is not None:
            self.slam.gaussian_bridge.wait_idle()
            print("Gaussian backend idle, running final refine...", flush=True)
            self.slam.gaussian_bridge.final_refine(getattr(cfg, "GAUSSIAN_FINAL_REFINEMENT_ITERS", None))
            if self.save_rendered_images:
                print("Rendering all frames for evaluation...", flush=True)
                self.slam.gaussian_bridge.render_all_frames(self.render_dir)
        self.save_outputs_to_disk(poses, tstamps)
        print(f"Outputs saved to {self.output_dir}", flush=True)

        if self.gt is not None:
            print("Evaluating trajectory...", flush=True)
            traj_result = self.evaluate_trajectory_and_save(poses, tstamps)
            print(f"ATE RMSE: {traj_result['ate_rmse_m']:.4f} m", flush=True)

        if self.slam is not None and getattr(self.slam, "gaussian_bridge", None) is not None:
            nvs_dir = self.infer_nvs_dir()
            if nvs_dir is not None and self.gt is not None:
                eval_out = self.eval_out_dir or (self.output_dir / "render_eval_nvs")
                used_timestamps, _, _ = self.resolve_eval_timestamps(poses, tstamps)
                print(f"Evaluating rendered images on NVS protocol from {nvs_dir}...", flush=True)
                evaluate_nvs(
                    self.slam.gaussian_bridge,
                    poses=poses,
                    timestamps=used_timestamps,
                    tracking_gt_file=self.gt,
                    nvs_dir=nvs_dir,
                    out_dir=eval_out,
                    save_vis=True,
                )
            elif self.eval_gt_dir is not None:
                eval_out = self.eval_out_dir or (self.output_dir / "render_eval")
                print("Evaluating rendered images on legacy RGB protocol...", flush=True)
                evaluate_renderings(self.render_dir, self.eval_gt_dir, eval_out, save_vis=True)

        if self.slam is not None and getattr(self.slam, "gaussian_bridge", None) is not None:
            self.slam.gaussian_bridge.close()

        self.shutdown_gui()



if __name__ == "__main__":
    parser = ArgumentParser(description="PI-SAM with DPVO data reading")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--imagedir", type=str, help="Path to image directory or video file")
    parser.add_argument("--calib", type=str, help="Path to calibration file")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument('--opts', nargs='+', default=[])
    parser.add_argument("--viz", action="store_true", help="Enable GUI visualization")
    parser.add_argument("--tum", action="store_true", help="is TUM-format datasets?")
    parser.add_argument("--edge", type=int, default=0, help="The edge need to cut in raw image")
    parser.add_argument("--gt", type=str, default=None, help="TUM-format ground truth file (timestamp tx ty tz qx qy qz qw)")
    parser.add_argument("--output_dir", type=str, default="outputs/demo", help="Directory for saved trajectory and map outputs")
    parser.add_argument("--eval_gt_dir", type=str, default=None, help="Directory containing GT RGB frames for render evaluation")
    parser.add_argument("--eval_out_dir", type=str, default=None, help="Directory for render evaluation outputs")
    parser.add_argument("--no_save_outputs", action="store_true", help="Disable saving trajectory and map outputs")

    args = parser.parse_args(sys.argv[1:])

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    
    config = {
        "imagedir": args.imagedir,
        "calib": args.calib,
        "stride": args.stride,
        "viz": args.viz,
        "tum": args.tum,
        "edge": args.edge,
        "gt": args.gt,
        "output_dir": args.output_dir,
        "eval_gt_dir": args.eval_gt_dir,
        "eval_out_dir": args.eval_out_dir,
        "save_outputs": not args.no_save_outputs,
        "save_rendered_images": bool(cfg.SAVE_RENDERED_IMAGES),
        "save_gaussian_map": bool(cfg.SAVE_GAUSSIAN_MAP),
        "save_debug_maps": False,
    }

    pisam = Pi_SAM(config)
    pisam.run()

from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Callable, Iterable, Optional

import torch

from .gaussian_frame import GaussianFrame
from .gaussian_mapper import GaussianMapper


class GaussianBackendBridge:
    def __init__(self, cfg, device: str = "cuda", output_dir: Optional[str] = None):
        self.cfg = cfg
        self.device = device
        self.mapper = GaussianMapper(cfg, device=device, output_dir=output_dir)
        self._queue: Queue = Queue(maxsize=int(getattr(cfg, "GAUSSIAN_QUEUE_SIZE", 64)))
        self._stop = Event()
        self._idle = Event()
        self._idle.set()
        self._lock = Lock()
        self._last_error = None
        self._worker = Thread(target=self._run, daemon=True)
        self._worker.start()

    def _run(self):
        while not self._stop.is_set() or not self._queue.empty():
            try:
                event = self._queue.get(timeout=0.1)
            except Empty:
                continue
            self._idle.clear()
            try:
                etype = event.get("type")
                if etype == "frame":
                    self.mapper.submit_frame(event["frame"])
                elif etype == "pose":
                    self.mapper.sync_pose(event["frame_id"], event["pose_w2c"])
                elif etype == "window":
                    self.mapper.sync_window(event["pose_provider"], event["frame_ids"])
                    if event.get("optimize", True):
                        refine = bool(event.get("refine", False))
                        self.mapper.optimize_window(event["frame_ids"], refine=refine, cap_window=not refine)
                elif etype == "save":
                    self.mapper.save(event.get("output_dir"))
                elif etype == "render":
                    event["result"][0] = self.mapper.render_frame(event["frame_id"])
                elif etype == "final_refine":
                    self.mapper.final_refine(event.get("iters"))
                elif etype == "render_all":
                    event["result"][0] = self.mapper.render_all_frames(event.get("output_dir"))
                elif etype == "shutdown":
                    break
            except Exception as exc:
                self._last_error = exc
            finally:
                self._queue.task_done()
                if self._queue.empty():
                    self._idle.set()

    def flush(self, timeout: Optional[float] = None) -> bool:
        self._queue.join()
        ok = self._idle.wait(timeout=timeout) if timeout is not None else True
        if self._last_error is not None:
            raise RuntimeError(f"Gaussian backend error: {self._last_error}") from self._last_error
        return ok

    def shutdown(self) -> None:
        self._stop.set()
        self._queue.put({"type": "shutdown"})
        self._worker.join(timeout=5.0)

    def submit_frame_packet(self, packet: dict) -> bool:
        if not self.mapper.enabled:
            return False
        frame = GaussianFrame(
            frame_id=int(packet["frame_id"]),
            timestamp=float(packet.get("timestamp", 0.0)),
            image=packet["image"],
            pose_w2c=packet["pose_w2c"],
            intrinsics=packet["intrinsics"],
            pi3_points=packet.get("pi3_points"),
            pi3_depth=packet.get("pi3_depth"),
            dynamic_mask=packet.get("dynamic_mask"),
            confidence=packet.get("confidence"),
            is_keyframe_candidate=bool(packet.get("is_keyframe_candidate", True)),
            metadata=dict(packet.get("metadata", {})),
        )
        self._queue.put({"type": "frame", "frame": frame})
        return True

    def notify_new_keyframe(self, packet: dict) -> bool:
        return self.submit_frame_packet(packet)

    def sync_pose(self, frame_id: int, pose_w2c: torch.Tensor) -> bool:
        self._queue.put({"type": "pose", "frame_id": int(frame_id), "pose_w2c": pose_w2c.detach().cpu()})
        return True

    def notify_pose_update(self, frame_id: int, pose_w2c: torch.Tensor) -> bool:
        return self.sync_pose(frame_id, pose_w2c)

    def sync_window(self, pose_provider: Callable[[int], Optional[torch.Tensor]], frame_ids: Iterable[int], refine: bool = False) -> int:
        frame_ids = list(frame_ids)
        self._queue.put({"type": "window", "pose_provider": pose_provider, "frame_ids": frame_ids, "optimize": True, "refine": refine})
        return len(frame_ids)

    def notify_loop_closure(self, pose_provider: Callable[[int], Optional[torch.Tensor]], frame_ids: Iterable[int], refine: bool = True) -> int:
        return self.sync_window(pose_provider, frame_ids, refine=refine)

    def render_frame(self, frame_id: int):
        self.flush()
        return self.mapper.render_frame(frame_id)

    def render_custom_view(self, pose_w2c: torch.Tensor, intrinsics: torch.Tensor, image_hw, frame_id: int = -1):
        self.flush()
        return self.mapper.render_custom_view(
            pose_w2c=pose_w2c,
            intrinsics=intrinsics,
            image_hw=image_hw,
            frame_id=frame_id,
        )

    def save(self, output_dir: Optional[str] = None) -> None:
        self.flush()
        self.mapper.save(output_dir=output_dir)

    def final_refine(self, iters: Optional[int] = None) -> None:
        self.flush()
        with torch.enable_grad():
            self.mapper.final_refine(iters=iters)

    def render_all_frames(self, output_dir: Optional[str] = None) -> int:
        self.flush()
        return self.mapper.render_all_frames(output_dir=output_dir)

    def summary(self) -> dict:
        return self.mapper.summary()

    def wait_idle(self, timeout: Optional[float] = None) -> bool:
        return self.flush(timeout=timeout)

    def close(self) -> None:
        self.shutdown()

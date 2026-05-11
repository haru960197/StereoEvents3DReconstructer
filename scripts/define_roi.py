import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class EventWindow:
	start_ts: int
	end_ts: int


def load_events_csv(csv_path: Path) -> np.ndarray:
	"""Load events from x,y,polarity,timestamp CSV while skipping '%' comments."""
	rows: List[Tuple[int, int, int, int]] = []
	with csv_path.open("r", encoding="utf-8") as f:
		for raw in f:
			line = raw.strip()
			if not line or line.startswith("%"):
				continue

			parts = line.split(",")
			if len(parts) < 4:
				continue

			try:
				x = int(parts[0])
				y = int(parts[1])
				p = int(parts[2])
				ts = int(parts[3])
			except ValueError:
				continue

			rows.append((x, y, p, ts))

	if not rows:
		raise ValueError(f"No valid events found in {csv_path}")

	return np.array(rows, dtype=np.int64)


def infer_sensor_size(events: np.ndarray, width: int, height: int) -> Tuple[int, int]:
	if width > 0 and height > 0:
		return width, height

	max_x = int(events[:, 0].max())
	max_y = int(events[:, 1].max())
	inferred_width = max_x + 1
	inferred_height = max_y + 1
	return inferred_width, inferred_height


def choose_windows(events: np.ndarray, window_us: int, num_windows: int) -> List[EventWindow]:
	ts = events[:, 3]
	min_ts = int(ts.min())
	max_ts = int(ts.max())

	if max_ts <= min_ts:
		return [EventWindow(min_ts, min_ts + max(window_us, 1))]

	span = max_ts - min_ts
	if span <= window_us or num_windows <= 1:
		return [EventWindow(min_ts, min_ts + window_us)]

	starts = np.linspace(min_ts, max_ts - window_us, num=num_windows, dtype=np.int64)
	return [EventWindow(int(s), int(s + window_us)) for s in starts]


def events_to_preview(
	events: np.ndarray,
	width: int,
	height: int,
	windows: Sequence[EventWindow],
	polarity_mode: str,
) -> np.ndarray:
	img = np.zeros((height, width), dtype=np.float32)

	for w in windows:
		in_window = (events[:, 3] >= w.start_ts) & (events[:, 3] < w.end_ts)
		if polarity_mode == "on":
			in_window &= events[:, 2] == 1
		elif polarity_mode == "off":
			in_window &= events[:, 2] == 0

		subset = events[in_window]
		if len(subset) == 0:
			continue

		xs = subset[:, 0]
		ys = subset[:, 1]
		valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
		xs = xs[valid]
		ys = ys[valid]
		np.add.at(img, (ys, xs), 1.0)

	if img.max() <= 0:
		return np.zeros((height, width), dtype=np.uint8)

	upper = np.percentile(img, 99.5)
	upper = max(upper, 1.0)
	img = np.clip(img / upper * 255.0, 0.0, 255.0)
	img = cv2.GaussianBlur(img.astype(np.uint8), (5, 5), 0)
	return img


class RoiEditor:
	def __init__(self, preview_gray: np.ndarray):
		self.preview_gray = preview_gray
		self.preview_bgr = cv2.cvtColor(preview_gray, cv2.COLOR_GRAY2BGR)
		self.points: List[Tuple[int, int]] = []
		self.closed = False
		self.window_name = "ROI Editor"

	def _draw_overlay(self) -> np.ndarray:
		canvas = self.preview_bgr.copy()

		if len(self.points) >= 2:
			pts = np.array(self.points, dtype=np.int32)
			cv2.polylines(canvas, [pts], self.closed, (0, 255, 255), 2)

		for pt in self.points:
			cv2.circle(canvas, pt, 3, (0, 255, 0), -1)

		if self.closed and len(self.points) >= 3:
			roi_vis = np.zeros_like(self.preview_gray, dtype=np.uint8)
			cv2.fillPoly(roi_vis, [np.array(self.points, dtype=np.int32)], 255)
			green = np.zeros_like(canvas)
			green[:, :, 1] = 255
			alpha = 0.25
			canvas = np.where(roi_vis[:, :, None] > 0, (1 - alpha) * canvas + alpha * green, canvas)
			canvas = canvas.astype(np.uint8)

		help_lines = [
			"Left click: add vertex",
			"Right click: undo last vertex",
			"ENTER: close polygon",
			"c: clear  s: save  q/Esc: quit",
		]
		y = 18
		for line in help_lines:
			cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
			y += 20

		return canvas

	def _mouse_cb(self, event, x, y, _flags, _userdata):
		if event == cv2.EVENT_LBUTTONDOWN and not self.closed:
			self.points.append((int(x), int(y)))
		elif event == cv2.EVENT_RBUTTONDOWN and self.points:
			self.points.pop()
			self.closed = False

	def run(self) -> List[Tuple[int, int]]:
		cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
		cv2.setMouseCallback(self.window_name, self._mouse_cb)

		while True:
			cv2.imshow(self.window_name, self._draw_overlay())
			key = cv2.waitKey(20) & 0xFF

			if key in (27, ord("q")):
				self.points = []
				break
			if key == ord("c"):
				self.points = []
				self.closed = False
			elif key in (13, 10):
				if len(self.points) >= 3:
					self.closed = True
			elif key == ord("s"):
				if len(self.points) >= 3:
					self.closed = True
					break

		cv2.destroyWindow(self.window_name)
		if len(self.points) < 3:
			return []
		return self.points


def polygon_to_mask(width: int, height: int, polygon: Sequence[Tuple[int, int]]) -> np.ndarray:
	mask = np.zeros((height, width), dtype=np.uint8)
	if len(polygon) < 3:
		return mask

	pts = np.array(polygon, dtype=np.int32)
	cv2.fillPoly(mask, [pts], 255)
	return mask


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Build a binary ROI mask from event CSV by drawing polygon on event preview image.")
	parser.add_argument("--input", default="input/events_master.csv", help="Input event CSV path")
	parser.add_argument("--output", default="config/roi_mask.pgm", help="Output binary mask image path")
	parser.add_argument("--meta", default="config/roi_mask_meta.json", help="Output metadata JSON path")
	parser.add_argument("--width", type=int, default=320, help="Sensor width (320 = infer from data)")
	parser.add_argument("--height", type=int, default=320, help="Sensor height (320 = infer from data)")
	parser.add_argument("--window-us", type=int, default=30000, help="Accumulation window in microseconds")
	parser.add_argument("--num-windows", type=int, default=5, help="Number of sampled windows")
	parser.add_argument(
		"--polarity",
		default="both",
		choices=["both", "on", "off"],
		help="Which event polarity to use for preview image",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	input_path = Path(args.input)
	output_path = Path(args.output)
	meta_path = Path(args.meta)

	events = load_events_csv(input_path)
	width, height = infer_sensor_size(events, args.width, args.height)

	windows = choose_windows(events, args.window_us, args.num_windows)
	preview = events_to_preview(events, width, height, windows, args.polarity)

	editor = RoiEditor(preview)
	polygon = editor.run()
	if not polygon:
		print("ROI was not saved (no polygon).")
		return

	mask = polygon_to_mask(width, height, polygon)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	meta_path.parent.mkdir(parents=True, exist_ok=True)

	ok = cv2.imwrite(str(output_path), mask)
	if not ok:
		raise RuntimeError(f"Failed to write mask image: {output_path}")

	meta = {
		"input_csv": str(input_path),
		"mask_image": str(output_path),
		"width": width,
		"height": height,
		"polygon": [{"x": int(x), "y": int(y)} for x, y in polygon],
		"windows": [{"start_ts": w.start_ts, "end_ts": w.end_ts} for w in windows],
		"polarity": args.polarity,
	}
	with meta_path.open("w", encoding="utf-8") as f:
		json.dump(meta, f, ensure_ascii=False, indent=2)

	print(f"Saved mask: {output_path}")
	print(f"Saved metadata: {meta_path}")
	print("C++ integration: if (mask_image[y][x] == 0) continue;")


if __name__ == "__main__":
	main()

"""
Automatic License Plate Recognition on a video file.

Pipeline:
  1. YOLO (best.pt) detects plate regions in every frame.
  2. Detections are filtered (confidence, aspect ratio, size) to reject
     non-plate rectangles such as signboards.
  3. A simple IoU tracker follows each physical plate across frames so it
     only needs to be OCR'd until a confident reading is found, instead of
     re-guessing every frame.
  4. EasyOCR (Bengali + English) reads the plate crop.
  5. Output video = original frame with detection boxes and OCR readings
     drawn directly on each plate.
  6. Every plate reading is logged to output/plates_log.csv.
"""

import csv
import os
import subprocess

import cv2
import easyocr
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

MODEL_PATH = "best.pt"
INPUT_VIDEO = "16385623_3840_2160_30fps.mp4"
RAW_OUTPUT_VIDEO = "output/_raw_alpr_output.mp4"
OUTPUT_VIDEO = "output/alpr_output.mp4"
LOG_CSV = "output/plates_log.csv"

CONF_THRESHOLD = 0.40
INFER_IMGSZ = 1920
TARGET_FPS = 30  # source may be 60fps; process/output at 30fps to halve runtime

# Real plates are small rectangles, not huge banners/signboards.
MIN_ASPECT_RATIO = 1.3
MAX_ASPECT_RATIO = 4.5
MAX_BOX_AREA_FRACTION = 0.03  # a plate should not be more than 3% of the frame

# Tracking / OCR
IOU_MATCH_THRESHOLD = 0.3
TRACK_MAX_MISSES = 20              # drop a track after this many consecutive missed frames
OCR_CONFIDENCE_GOOD_ENOUGH = 0.45  # stop re-running OCR once a track hits this
OCR_MIN_KEEP_CONFIDENCE = 0.0      # plates are tiny/blurry in this footage, so even a
                                    # low-confidence guess is worth surfacing (with its
                                    # real confidence shown) rather than hiding it

BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 2

# cv2.putText only renders ASCII/Latin glyphs; plates here are read in Bengali
# script, so text is drawn with PIL + a Unicode font and composited back in.
_UNICODE_FONT_PATH = r"C:\Windows\Fonts\Nirmala.ttc"
_font_cache = {}


def get_font(size):
    if size not in _font_cache:
        _font_cache[size] = ImageFont.truetype(_UNICODE_FONT_PATH, size)
    return _font_cache[size]


def render_text_batch(canvas, text_specs):
    """Draw many text strings (any script, e.g. Bengali) onto a BGR numpy
    image in a single PIL round-trip, for speed at 4K resolution."""
    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    for spec in text_specs:
        x, y = spec["org"]
        font = get_font(spec["size"])
        if spec.get("bg"):
            bbox = draw.textbbox((x, y), spec["text"], font=font)
            pad = spec.get("padding", 4)
            bg = spec["bg"]
            draw.rectangle(
                (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
                fill=(bg[2], bg[1], bg[0]),
            )
        color = spec["color"]
        draw.text((x, y), spec["text"], font=font, fill=(color[2], color[1], color[0]))
    canvas[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def looks_like_plate(x1, y1, x2, y2, frame_area):
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return False
    aspect = max(w, h) / min(w, h)
    if not (MIN_ASPECT_RATIO <= aspect <= MAX_ASPECT_RATIO):
        return False
    if (w * h) / frame_area > MAX_BOX_AREA_FRACTION:
        return False
    return True


def preprocess_crop(frame, x1, y1, x2, y2, pad=6, scale=4):
    # Plates are tiny (~70-90px wide) in this 4K footage. Grayscale + CLAHE
    # made EasyOCR's detector find nothing on most crops; a color upscale +
    # mild unsharp mask reliably surfaces at least a low-confidence reading.
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None
    upscaled = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(upscaled, (0, 0), 2)
    sharpened = cv2.addWeighted(upscaled, 1.5, blurred, -0.5, 0)
    return crop, sharpened


def read_plate_text(reader, ocr_crop):
    if ocr_crop is None:
        return "", 0.0
    results = reader.readtext(ocr_crop)
    if not results:
        return "", 0.0
    _, text, conf = max(results, key=lambda r: r[2])
    text = text.strip()
    if conf < OCR_MIN_KEEP_CONFIDENCE or not text:
        return "", 0.0
    return text, float(conf)


class Track:
    _next_id = 1

    def __init__(self, box, yolo_conf, frame_idx):
        self.id = Track._next_id
        Track._next_id += 1
        self.box = box
        self.yolo_conf = yolo_conf
        self.ocr_text = ""
        self.ocr_conf = 0.0
        self.ocr_attempted = False
        self.thumbnail = None
        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.misses = 0

    def update(self, box, yolo_conf, frame_idx):
        self.box = box
        self.yolo_conf = max(self.yolo_conf, yolo_conf)
        self.last_frame = frame_idx
        self.misses = 0

    def needs_ocr(self):
        return self.ocr_conf < OCR_CONFIDENCE_GOOD_ENOUGH

    def apply_ocr_result(self, text, conf, thumbnail):
        self.ocr_attempted = True
        if thumbnail is not None:
            self.thumbnail = thumbnail
        if conf > self.ocr_conf:
            self.ocr_text = text
            self.ocr_conf = conf


def match_detections_to_tracks(detections, active_tracks, frame_idx):
    unmatched_tracks = list(active_tracks)
    for box, yolo_conf in detections:
        best_track, best_iou = None, 0.0
        for t in unmatched_tracks:
            score = iou(box, t.box)
            if score > best_iou:
                best_track, best_iou = t, score
        if best_track is not None and best_iou >= IOU_MATCH_THRESHOLD:
            best_track.update(box, yolo_conf, frame_idx)
            unmatched_tracks.remove(best_track)
        else:
            active_tracks.append(Track(box, yolo_conf, frame_idx))

    for t in unmatched_tracks:
        t.misses += 1


def run_ocr_on_tracks(active_tracks, frame, reader):
    for t in active_tracks:
        if t.misses == 0 and t.needs_ocr():
            x1, y1, x2, y2 = t.box
            raw_crop, ocr_crop = preprocess_crop(frame, x1, y1, x2, y2)
            text, conf = read_plate_text(reader, ocr_crop)
            t.apply_ocr_result(text, conf, raw_crop)


def draw_detections(frame, active_tracks):
    text_specs = []
    for t in active_tracks:
        if t.misses > 0:
            continue
        x1, y1, x2, y2 = t.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
        plate_label = t.ocr_text or ("unreadable" if t.ocr_attempted else "...")
        label = f"#{t.id} {plate_label} ({t.yolo_conf:.2f})"
        text_specs.append({"text": label, "org": (x1 + 3, y1 - 26), "size": 20,
                            "color": (0, 0, 0), "bg": BOX_COLOR})
    return text_specs


def flush_stale_tracks(active_tracks, log_writer, log_file, logged_ids, force_all=False):
    still_active = []
    for t in active_tracks:
        stale = force_all or t.misses > TRACK_MAX_MISSES
        if stale and t.id not in logged_ids and t.ocr_attempted:
            log_writer.writerow([t.id, t.first_frame, t.last_frame,
                                  f"{t.yolo_conf:.3f}", t.ocr_text or "UNREADABLE", f"{t.ocr_conf:.3f}"])
            if log_file is not None:
                log_file.flush()
            logged_ids.add(t.id)
        if not stale:
            still_active.append(t)
    return still_active


def reencode_to_h264(raw_path, out_path):
    # OpenCV's mp4v output isn't playable in most Windows video players,
    # so re-encode to H.264 and drop the intermediate file.
    print("Re-encoding to H.264 for playback compatibility...")
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg_exe, "-y", "-i", raw_path, "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-crf", "20", out_path],
        check=True, capture_output=True,
    )
    os.remove(raw_path)


def main():
    os.makedirs("output", exist_ok=True)

    model = YOLO(MODEL_PATH)
    reader = easyocr.Reader(["bn", "en"], gpu=False, verbose=False)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {INPUT_VIDEO}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_step = max(1, round(source_fps / TARGET_FPS))
    output_fps = source_fps / frame_step
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // frame_step
    frame_area = width * height

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(RAW_OUTPUT_VIDEO, fourcc, output_fps, (width, height))

    log_file = open(LOG_CSV, "w", newline="", encoding="utf-8-sig")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["plate_id", "first_frame", "last_frame", "yolo_confidence",
                          "ocr_text", "ocr_confidence"])

    active_tracks = []
    logged_ids = set()
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # keep 1 of every frame_step frames so the video is effectively
        # processed/output at TARGET_FPS instead of the source's native fps
        for _ in range(frame_step - 1):
            if not cap.grab():
                break

        results = model.predict(frame, conf=CONF_THRESHOLD, imgsz=INFER_IMGSZ, verbose=False)
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            if looks_like_plate(x1, y1, x2, y2, frame_area):
                detections.append(((x1, y1, x2, y2), float(box.conf[0])))

        match_detections_to_tracks(detections, active_tracks, frame_idx)
        run_ocr_on_tracks(active_tracks, frame, reader)

        text_specs = draw_detections(frame, active_tracks)
        render_text_batch(frame, text_specs)
        writer.write(frame)

        active_tracks = flush_stale_tracks(active_tracks, log_writer, log_file, logged_ids)

        frame_idx += 1
        if frame_idx % 30 == 0 or frame_idx == total_frames:
            print(f"Processed {frame_idx}/{total_frames} frames")

    flush_stale_tracks(active_tracks, log_writer, log_file, logged_ids, force_all=True)

    cap.release()
    writer.release()
    log_file.close()

    reencode_to_h264(RAW_OUTPUT_VIDEO, OUTPUT_VIDEO)

    print(f"Done. {len(logged_ids)} distinct plates logged.")
    print(f"Output video saved to: {OUTPUT_VIDEO}")
    print(f"Plate log saved to: {LOG_CSV}")


if __name__ == "__main__":
    main()

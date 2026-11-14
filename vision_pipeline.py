"""

Input:
    - BGR color image from RealSense
    - aligned depth image from RealSense
    - RealSense color intrinsics

Output:
    - candidate block center in pixels
    - candidate 3D position in the camera frame, in meters
    - gripper yaw suggestion, in degrees
"""

import cv2
import numpy as np


EXPECTED_LONG_PX = 110.0
EXPECTED_SHORT_PX = 36.0
EXPECTED_RATIO = EXPECTED_LONG_PX / EXPECTED_SHORT_PX


def normalize_angle(deg):
    """Normalize a gripper angle to [-90, 90)."""
    deg = deg % 360.0
    if deg > 180.0:
        deg -= 360.0
    if deg >= 90.0:
        deg -= 180.0
    if deg < -90.0:
        deg += 180.0
    return deg


def _odd_kernel_size(value, fallback):
    value = int(value or fallback)
    if value < 3:
        value = 3
    if value % 2 == 0:
        value += 1
    return value


def median_depth_raw(depth_image, u, v, half=3):
    """Return the median valid raw depth value around a pixel."""
    h, w = depth_image.shape[:2]
    values = []
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            ny = int(v) + dy
            nx = int(u) + dx
            if 0 <= ny < h and 0 <= nx < w:
                d = int(depth_image[ny, nx])
                if d > 0:
                    values.append(d)
    if not values:
        return 0
    values.sort()
    return values[len(values) // 2]


def contour_depth_raw(depth_image, contour, max_samples=2500):
    """Return a robust valid raw depth from inside a contour."""
    if contour is None:
        return 0

    h, w = depth_image.shape[:2]
    x, y, bw, bh = cv2.boundingRect(contour)
    x0 = max(int(x), 0)
    y0 = max(int(y), 0)
    x1 = min(int(x + bw), w)
    y1 = min(int(y + bh), h)
    if x0 >= x1 or y0 >= y1:
        return 0

    roi_depth = depth_image[y0:y1, x0:x1]
    roi_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    shifted = contour.copy()
    shifted[:, 0, 0] -= x0
    shifted[:, 0, 1] -= y0
    cv2.drawContours(roi_mask, [shifted], -1, 255, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded_mask = cv2.erode(roi_mask, kernel, iterations=1)
    valid_depth = (roi_depth > 0) & (roi_depth < 65535)
    values = roi_depth[(eroded_mask > 0) & valid_depth]
    if values.size == 0:
        values = roi_depth[(roi_mask > 0) & valid_depth]
    if values.size == 0:
        return 0

    if values.size > max_samples:
        step = int(np.ceil(values.size / float(max_samples)))
        values = values[::step]

    return int(np.median(values))


def pixel_to_3d(u, v, depth_m, fx, fy, cx, cy):
    """Convert pixel + depth to camera-frame XYZ in meters."""
    x = (float(u) - cx) / fx * depth_m
    y = (float(v) - cy) / fy * depth_m
    z = depth_m
    return x, y, z


class JengaDetector:
    """Contour detector tuned for a Jenga block under uneven light."""

    def __init__(self, min_area_px=500, min_score=0.3,
                 blur_ksize=5, morph_ksize=5, canny_low=30,
                 canny_high=70, watershed_fg_ratio=0.45,
                 aspect_min=1.8, aspect_max=5.2,
                 max_area_fraction=0.25, max_long_px=180.0,
                 max_short_px=65.0):
        self.min_area_px = float(min_area_px)
        self.min_score = float(min_score)
        self.blur_ksize = _odd_kernel_size(blur_ksize, 5)
        self.morph_ksize = _odd_kernel_size(morph_ksize, 5)
        self.canny_low = int(canny_low)
        self.canny_high = int(canny_high)
        self.watershed_fg_ratio = float(watershed_fg_ratio)
        self.aspect_min = float(aspect_min)
        self.aspect_max = float(aspect_max)
        self.max_area_fraction = float(max_area_fraction)
        self.max_long_px = float(max_long_px)
        self.max_short_px = float(max_short_px)

    def _shadow_normalized_luma(self, image_bgr):
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(l_channel)
        background = cv2.GaussianBlur(equalized, (0, 0), 21)
        return cv2.divide(equalized, background, scale=255)

    def _color_contrast_mask(self, image_bgr):
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        b_channel = cv2.GaussianBlur(lab[:, :, 2], (5, 5), 0)
        saturation = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)

        _, b_mask = cv2.threshold(
            b_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        _, sat_mask = cv2.threshold(
            saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return cv2.bitwise_or(b_mask, sat_mask)

    def _edge_fill_mask(self, image_bgr):
        normalized = self._shadow_normalized_luma(image_bgr)
        blur = cv2.GaussianBlur(normalized, (self.blur_ksize, self.blur_ksize), 0)
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        mask = np.zeros(edges.shape, dtype=np.uint8)
        for contour in contours:
            if cv2.contourArea(contour) >= self.min_area_px * 0.25:
                cv2.drawContours(mask, [contour], -1, 255, -1)
        return mask

    def _watershed_refine(self, image_bgr, mask):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.morph_ksize, self.morph_ksize),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        if dist.max() <= 0:
            return mask

        fg_ratio = min(max(self.watershed_fg_ratio, 0.25), 0.75)
        _, sure_fg = cv2.threshold(dist, fg_ratio * dist.max(), 255, cv2.THRESH_BINARY)
        sure_fg = np.uint8(sure_fg)
        sure_bg = cv2.dilate(mask, kernel, iterations=3)
        unknown = cv2.subtract(sure_bg, sure_fg)

        _, markers = cv2.connectedComponents(sure_fg)
        markers += 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(image_bgr.copy(), markers)

        refined = np.zeros(mask.shape, dtype=np.uint8)
        refined[markers > 1] = 255
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
        return refined

    def _segment(self, image_bgr):
        edge_mask = self._edge_fill_mask(image_bgr)
        color_mask = self._color_contrast_mask(image_bgr)
        combined = cv2.bitwise_or(edge_mask, color_mask)
        return self._watershed_refine(image_bgr, combined)

    def _score(self, long_px, short_px, contour_area, image_bgr=None, center=None):
        rect_area = float(long_px * short_px)
        if rect_area <= 0.0 or short_px <= 0.0:
            return 0.0

        fill_score = min(float(contour_area) / rect_area, 1.0)
        aspect = float(long_px) / float(short_px)
        aspect_score = np.exp(-((aspect - EXPECTED_RATIO) ** 2) / 4.0)
        long_score = np.exp(-((long_px - EXPECTED_LONG_PX) ** 2) / 5000.0)
        short_score = np.exp(-((short_px - EXPECTED_SHORT_PX) ** 2) / 700.0)
        size_score = (long_score + short_score) / 2.0

        color_score = 0.5
        if image_bgr is not None and center is not None:
            u, v = center
            h, w = image_bgr.shape[:2]
            if 0 <= u < w and 0 <= v < h:
                lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
                hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
                y0, y1 = max(0, v - 3), min(h, v + 4)
                x0, x1 = max(0, u - 3), min(w, u + 4)
                b_mean = float(lab[y0:y1, x0:x1, 2].mean())
                s_mean = float(hsv[y0:y1, x0:x1, 1].mean())
                yellow_score = min(max((b_mean - 125.0) / 25.0, 0.0), 1.0)
                saturation_score = min(max(s_mean / 80.0, 0.0), 1.0)
                color_score = max(yellow_score, saturation_score)

        score = (
            0.30 * fill_score
            + 0.35 * aspect_score
            + 0.20 * size_score
            + 0.15 * color_score
        )
        return round(float(score), 3)

    def detect(self, image_bgr):
        mask = self._segment(image_bgr)
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        detections = []
        image_area = float(image_bgr.shape[0] * image_bgr.shape[1])
        max_area = image_area * min(max(self.max_area_fraction, 0.01), 0.95)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area_px:
                continue
            if area > max_area:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue

            center_u = int(moments["m10"] / moments["m00"])
            center_v = int(moments["m01"] / moments["m00"])
            (_, _), (w, h), angle = cv2.minAreaRect(contour)

            if w < h:
                w, h = h, w
                angle += 90.0

            if h <= 0.0:
                continue
            if w > self.max_long_px or h > self.max_short_px:
                continue
            aspect = float(w) / float(h)
            if aspect < self.aspect_min or aspect > self.aspect_max:
                continue

            score = self._score(w, h, area, image_bgr, (center_u, center_v))
            if score < self.min_score:
                continue

            detections.append({
                "center_uv": (center_u, center_v),
                "angle_deg": round(float(angle), 1),
                "width_px": round(float(w), 1),
                "height_px": round(float(h), 1),
                "score": score,
                "contour": contour,
            })

        detections.sort(key=lambda item: item["score"], reverse=True)
        return detections, mask


class VisionPipeline:
    """Convert a RealSense frame into sorted grasp candidates."""

    def __init__(self, intrinsics, depth_scale_m=0.001, config=None):
        config = config or {}
        self.fx = float(intrinsics["fx"])
        self.fy = float(intrinsics["fy"])
        self.cx = float(intrinsics["ppx"])
        self.cy = float(intrinsics["ppy"])
        self.img_w = int(intrinsics["width"])
        self.img_h = int(intrinsics["height"])
        self.depth_scale_m = float(depth_scale_m)

        self.min_score = float(config.get("min_score", 0.5))
        self.min_depth_m = float(config.get("min_depth_m", 0.2))
        self.max_depth_m = float(config.get("max_depth_m", 1.0))
        self.edge_margin_px = int(config.get("edge_margin_px", 40))

        self.detector = JengaDetector(
            min_area_px=config.get("min_area_px", 500),
            min_score=config.get("detector_min_score", 0.3),
            blur_ksize=config.get("blur_ksize", 5),
            morph_ksize=config.get("morph_ksize", 5),
            canny_low=config.get("canny_low", 30),
            canny_high=config.get("canny_high", 70),
            watershed_fg_ratio=config.get("watershed_fg_ratio", 0.45),
            aspect_min=config.get("aspect_min", 1.8),
            aspect_max=config.get("aspect_max", 5.2),
            max_area_fraction=config.get("max_area_fraction", 0.25),
            max_long_px=config.get("max_long_px", 180.0),
            max_short_px=config.get("max_short_px", 65.0),
        )
        self.last_mask = None

    def process(self, color_image, depth_image):
        detections, mask = self.detector.detect(color_image)
        self.last_mask = mask

        candidates = []
        for item in detections:
            u, v = item["center_uv"]
            depth_raw = median_depth_raw(depth_image, u, v, half=3)
            if depth_raw <= 0:
                depth_raw = contour_depth_raw(depth_image, item.get("contour"))
            depth_m = float(depth_raw) * self.depth_scale_m
            if depth_m <= 0.0:
                continue
            if depth_m < self.min_depth_m or depth_m > self.max_depth_m:
                continue
            if item["score"] < self.min_score:
                continue
            if u < self.edge_margin_px or u > self.img_w - self.edge_margin_px:
                continue
            if v < self.edge_margin_px or v > self.img_h - self.edge_margin_px:
                continue

            xyz_cam = pixel_to_3d(u, v, depth_m, self.fx, self.fy, self.cx, self.cy)
            raw_yaw = item["angle_deg"] + 90.0
            candidate = dict(item)
            candidate["depth_m"] = round(depth_m, 4)
            candidate["xyz_cam"] = tuple(round(float(value), 4) for value in xyz_cam)
            candidate["gripper_yaw_deg"] = round(normalize_angle(raw_yaw), 1)
            candidates.append(candidate)

        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates

    def get_best(self, color_image, depth_image):
        candidates = self.process(color_image, depth_image)
        if not candidates:
            return None
        return candidates[0]


def draw_detection(color_image, candidate):
    """Return a debug image with one candidate drawn on top."""
    output = color_image.copy()
    if candidate is None:
        return output

    contour = candidate.get("contour")
    if contour is not None:
        cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)

    u, v = candidate["center_uv"]
    cv2.circle(output, (int(u), int(v)), 5, (0, 0, 255), -1)
    text = (
        f"score={candidate['score']:.2f} "
        f"z={candidate['depth_m']:.3f}m "
        f"yaw={candidate['gripper_yaw_deg']:.1f}"
    )
    cv2.putText(
        output,
        text,
        (int(u) + 10, int(v) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return output

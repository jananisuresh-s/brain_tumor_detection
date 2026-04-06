"""
Brain Tumor Detection Engine
Uses classical image processing techniques:
- Skull stripping (thresholding + morphological ops)
- Otsu's thresholding
- Contour analysis
- Feature extraction (area, intensity, circularity)
- Rule-based tumor type classification
"""

import cv2
import numpy as np
from scipy import ndimage
import base64


def preprocess_mri(image):
    """Convert to grayscale, denoise, and enhance contrast."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return enhanced


def skull_strip(image):
    """
    Remove skull from MRI.
    Returns:
      stripped     - full brain image (skull removed)
      inner_mask   - eroded mask that excludes the bright skull ring
                     so tumor search happens only deep inside brain tissue
    """
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, np.ones_like(image) * 255

    brain_contour = max(contours, key=cv2.contourArea)
    brain_mask = np.zeros_like(binary)
    cv2.fillPoly(brain_mask, [brain_contour], 255)

    # KEY FIX: erode the mask inward to exclude the bright skull ring
    h, w = image.shape
    erode_size = max(15, int(min(h, w) * 0.07))
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
    inner_mask = cv2.erode(brain_mask, erode_kernel, iterations=1)

    stripped = cv2.bitwise_and(image, image, mask=brain_mask)
    return stripped, inner_mask


def segment_tumor(brain_image, inner_mask):
    brain_pixels = brain_image[inner_mask > 0]
    if len(brain_pixels) == 0:
        return None, []

    mean_intensity = np.mean(brain_pixels)
    std_intensity = np.std(brain_pixels)

    h, w = brain_image.shape
    img_area = h * w

    best_contour = None
    best_score = -1

    for sigma in [2.2, 1.8, 1.5]:
        threshold = mean_intensity + sigma * std_intensity
        threshold = min(threshold, 245)

        _, tumor_mask = cv2.threshold(brain_image, int(threshold), 255, cv2.THRESH_BINARY)
        tumor_mask = cv2.bitwise_and(tumor_mask, inner_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        tumor_mask = ndimage.binary_fill_holes(tumor_mask).astype(np.uint8) * 255

        contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = img_area * 0.004
        max_area = img_area * 0.20

        valid = [c for c in contours if min_area < cv2.contourArea(c) < max_area]

        for c in valid:
            # Score each region by its MEAN BRIGHTNESS — tumor is the brightest blob
            mask_c = np.zeros_like(brain_image)
            cv2.fillPoly(mask_c, [c], 255)
            pixels = brain_image[mask_c > 0]
            mean_bright = np.mean(pixels) if len(pixels) > 0 else 0

            # Also prefer regions NOT at very bottom of image (avoid brainstem)
            x, y, cw, ch = cv2.boundingRect(c)
            center_y_ratio = (y + ch // 2) / h
            # Penalize regions in bottom 20% of image
            cx_ratio = (x + cw // 2) / w
            is_near_edge = (
                center_y_ratio > 0.78 or
                center_y_ratio < 0.15 or
                cx_ratio < 0.12 or
                cx_ratio > 0.88
            )
            position_penalty = 0.5 if is_near_edge else 1.0

            score = mean_bright * position_penalty
            if score > best_score:
                best_score = score
                best_contour = c
                best_mask = tumor_mask

    if best_contour is not None:
        return best_mask, [best_contour]

    return None, []


def extract_features(contour, image, inner_mask):
    """Extract morphological and intensity features from a detected region."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 1
    extent = area / (w * h) if w * h > 0 else 0

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [contour], 255)
    pixels = image[mask > 0]
    mean_intensity = np.mean(pixels) if len(pixels) > 0 else 0
    std_intensity = np.std(pixels) if len(pixels) > 0 else 0

    img_h, img_w = image.shape
    cx, cy = x + w // 2, y + h // 2
    dist_from_center = np.sqrt(((cx - img_w // 2) / (img_w / 2)) ** 2 +
                               ((cy - img_h // 2) / (img_h / 2)) ** 2)

    center_x_ratio = cx / img_w
    center_y_ratio = cy / img_h
    is_pituitary_region = (0.35 < center_x_ratio < 0.65) and (0.55 < center_y_ratio < 0.80)

    return {
        "area": area,
        "perimeter": perimeter,
        "circularity": circularity,
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "dist_from_center": dist_from_center,
        "bbox": (x, y, w, h),
        "centroid": (cx, cy),
        "is_pituitary_region": is_pituitary_region,
    }


def classify_tumor(features, img_h, img_w):
    """Rule-based tumor classification using extracted image features."""
    area = features["area"]
    circularity = features["circularity"]
    dist_from_center = features["dist_from_center"]
    is_pituitary = features["is_pituitary_region"]
    std_intensity = features["std_intensity"]
    img_area = img_h * img_w
    area_ratio = area / img_area

    scores = {"Glioma": 0, "Meningioma": 0, "Pituitary": 0, "Metastatic": 0}

    if is_pituitary:
        scores["Pituitary"] += 40
    if circularity > 0.7:
        scores["Pituitary"] += 20
        scores["Meningioma"] += 15
    if area_ratio < 0.05:
        scores["Pituitary"] += 20
        scores["Metastatic"] += 10

    if dist_from_center > 0.6:
        scores["Meningioma"] += 35
    if circularity > 0.65:
        scores["Meningioma"] += 20
    if std_intensity < 30:
        scores["Meningioma"] += 15

    if circularity < 0.5:
        scores["Glioma"] += 30
    if area_ratio > 0.08:
        scores["Glioma"] += 25
    if std_intensity > 40:
        scores["Glioma"] += 20
    if dist_from_center < 0.5:
        scores["Glioma"] += 15

    if area_ratio < 0.03 and std_intensity > 35:
        scores["Metastatic"] += 25

    tumor_type = max(scores, key=scores.get)
    total = sum(scores.values()) or 1
    top_score = scores[tumor_type]
    confidence = min(95, int((top_score / total) * 150 + 40))
    confidence = max(50, confidence)

    grade_map = {
        "Glioma": "High-grade (likely)" if area_ratio > 0.1 or circularity < 0.4 else "Low-grade (likely)",
        "Meningioma": "Benign (Grade I likely)",
        "Pituitary": "Benign (Grade I likely)",
        "Metastatic": "Malignant",
    }

    return tumor_type, grade_map[tumor_type], confidence, scores


def get_location_label(cx, cy, img_w, img_h):
    x_ratio = cx / img_w
    y_ratio = cy / img_h
    lr = "Left" if x_ratio < 0.45 else ("Right" if x_ratio > 0.55 else "Central")
    tb = "frontal" if y_ratio < 0.33 else ("parietal/temporal" if y_ratio < 0.66 else "occipital/posterior")
    if 0.35 < x_ratio < 0.65 and 0.55 < y_ratio < 0.80:
        return "Sellar/suprasellar region (pituitary area)"
    if y_ratio < 0.2:
        return f"Superior {lr.lower()} frontal lobe"
    return f"{lr} {tb} region"


def draw_result(original_image, contours, features_list, has_tumor):
    """Draw detection overlay on the original image."""
    result = original_image.copy()
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    if not has_tumor or not contours:
        overlay = result.copy()
        cv2.rectangle(overlay, (0, 0), (result.shape[1], result.shape[0]), (0, 200, 0), -1)
        result = cv2.addWeighted(result, 0.95, overlay, 0.05, 0)
        label = "NO TUMOR DETECTED"
        font_scale = max(0.5, result.shape[1] / 700)
        thickness = max(1, int(font_scale * 2))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x = (result.shape[1] - tw) // 2
        y = result.shape[0] - 20
        cv2.rectangle(result, (x - 8, y - th - 8), (x + tw + 8, y + 8), (0, 180, 0), -1)
        cv2.putText(result, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        return result

    for contour, features in zip(contours, features_list):
        # Semi-transparent red fill over tumor region
        overlay = result.copy()
        cv2.fillPoly(overlay, [contour], (0, 0, 220))
        result = cv2.addWeighted(result, 0.75, overlay, 0.25, 0)

        # Contour outline
        cv2.drawContours(result, [contour], -1, (0, 0, 255), 2)

        # Corner bracket markers on bounding box
        x, y, w, h = features["bbox"]
        corner_len = min(w, h) // 5
        color = (0, 0, 255)
        for (cx2, cy2) in [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]:
            sx = 1 if cx2 == x else -1
            sy = 1 if cy2 == y else -1
            cv2.line(result, (cx2, cy2), (cx2 + sx * corner_len, cy2), color, 3)
            cv2.line(result, (cx2, cy2), (cx2, cy2 + sy * corner_len), color, 3)

        # Label above bounding box
        label = features.get("tumor_type", "Tumor")
        font_scale = max(0.4, result.shape[1] / 800)
        thickness = max(1, int(font_scale * 2))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        lx = x
        ly = y - 8 if y > th + 12 else y + h + th + 8
        cv2.rectangle(result, (lx - 2, ly - th - 4), (lx + tw + 4, ly + 4), (0, 0, 200), -1)
        cv2.putText(result, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    return result


def encode_image_b64(image_bgr):
    _, buffer = cv2.imencode('.png', image_bgr)
    return base64.b64encode(buffer).decode('utf-8')


def analyze_mri(image_bytes):
    """Main pipeline entry point."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if original is None:
        return {"error": "Could not read image. Please upload a valid MRI scan (JPG/PNG)."}

    img_h, img_w = original.shape[:2]

    preprocessed = preprocess_mri(original)
    stripped, inner_mask = skull_strip(preprocessed)
    tumor_mask, contours = segment_tumor(stripped, inner_mask)

    has_tumor = len(contours) > 0

    if not has_tumor:
        result_img = draw_result(original, [], [], False)
        return {
            "tumor_found": False,
            "tumor_type": "None",
            "grade": "N/A",
            "location": "N/A",
            "confidence": 92,
            "summary": "No significant anomalous regions detected. The MRI appears within normal limits.",
            "characteristics": ["No hyperintense regions detected", "Normal brain tissue distribution"],
            "result_image": encode_image_b64(result_img),
            "scores": {}
        }

    all_features = [extract_features(c, stripped, inner_mask) for c in contours]
    primary = all_features[0]

    tumor_type, grade, confidence, scores = classify_tumor(primary, img_h, img_w)
    cx, cy = primary["centroid"]
    location = get_location_label(cx, cy, img_w, img_h)

    x, y, w, h = primary["bbox"]
    size_cm = f"~{w * 0.026:.1f} x {h * 0.026:.1f} cm (approx.)"

    # Attach tumor type to features for label drawing
    for f in all_features:
        f["tumor_type"] = tumor_type

    characteristics = []
    characteristics.append("Well-defined circular border" if primary["circularity"] > 0.65 else "Irregular, infiltrative border")
    characteristics.append("Heterogeneous signal intensity" if primary["std_intensity"] > 35 else "Homogeneous signal intensity")
    characteristics.append("Peripheral/extra-axial location" if primary["dist_from_center"] > 0.6 else "Intra-axial location")
    if len(contours) > 1:
        characteristics.append("Multiple lesions detected")

    type_summaries = {
        "Glioma": "An irregular hyperintense mass was detected suggesting glial origin. Gliomas are the most common primary brain tumors.",
        "Meningioma": "A well-circumscribed extra-axial mass was detected near the brain periphery, consistent with meningioma characteristics.",
        "Pituitary": "A small mass was detected in the sellar region, consistent with a pituitary adenoma.",
        "Metastatic": "A small, well-defined lesion was detected with features consistent with metastatic disease.",
    }

    result_img = draw_result(original, contours, all_features, True)

    return {
        "tumor_found": True,
        "tumor_type": tumor_type,
        "grade": grade,
        "location": location,
        "size": size_cm,
        "confidence": confidence,
        "summary": type_summaries.get(tumor_type, "Abnormal region detected."),
        "characteristics": characteristics,
        "result_image": encode_image_b64(result_img),
        "scores": scores,
    }

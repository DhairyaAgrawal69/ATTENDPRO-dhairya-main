import os
import cv2
import dlib
import numpy as np
import face_recognition_models
from PIL import Image, ImageOps
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import streamlit as st

from src.database.db import get_all_students

# Path for YuNet model
YUNET_MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
YUNET_MODEL_PATH = os.path.join(YUNET_MODEL_DIR, YUNET_MODEL_FILENAME)
YUNET_DOWNLOAD_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


def ensure_yunet_model():
    """Ensure the YuNet ONNX model is available locally; download if missing."""
    if os.path.exists(YUNET_MODEL_PATH) and os.path.getsize(YUNET_MODEL_PATH) > 10000:
        return YUNET_MODEL_PATH
    
    os.makedirs(YUNET_MODEL_DIR, exist_ok=True)
    try:
        import urllib.request
        urllib.request.urlretrieve(YUNET_DOWNLOAD_URL, YUNET_MODEL_PATH)
        if os.path.exists(YUNET_MODEL_PATH):
            return YUNET_MODEL_PATH
    except Exception as e:
        print(f"Warning: Could not download YuNet model: {e}")
    return None


def resize_image_if_large(image_rgb, max_width=1000):
    """
    Downscale large classroom images if width (or height) exceeds max_width using cv2.INTER_AREA.
    Prevents C++ memory overflow, stack exhaustion, and segfaults during heavy multi-face extraction.
    """
    if image_rgb is None or not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
        return image_rgb

    h, w = image_rgb.shape[:2]
    if w > max_width or h > max_width:
        scale = min(max_width / float(w), max_width / float(h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        print(f"[RESIZE] Downscaling image from {w}x{h} -> {new_w}x{new_h} (scale={scale:.3f}) using cv2.INTER_AREA", flush=True)
        resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(resized, dtype=np.uint8)
    return np.ascontiguousarray(image_rgb, dtype=np.uint8)


def preprocess_image(image_input, max_width=1000):
    """
    Standardize any image input (PIL Image, numpy array, file stream) into a
    strictly contiguous 3-channel uint8 RGB numpy array with correct EXIF orientation,
    and downscale if dimensions exceed max_width to prevent C++ memory crashes.
    """
    if image_input is None:
        return None

    try:
        # If it's a PIL Image
        if isinstance(image_input, Image.Image):
            img_pil = ImageOps.exif_transpose(image_input)
            img_rgb = img_pil.convert("RGB")
            arr = np.array(img_rgb, dtype=np.uint8)
            arr = resize_image_if_large(arr, max_width=max_width)
            return np.ascontiguousarray(arr, dtype=np.uint8)

        # If it has a read method (e.g. UploadedFile or BytesIO)
        if hasattr(image_input, "read"):
            if hasattr(image_input, "seek"):
                image_input.seek(0)
            img_pil = Image.open(image_input)
            img_pil = ImageOps.exif_transpose(img_pil)
            img_rgb = img_pil.convert("RGB")
            arr = np.array(img_rgb, dtype=np.uint8)
            arr = resize_image_if_large(arr, max_width=max_width)
            return np.ascontiguousarray(arr, dtype=np.uint8)

        # If it's already a numpy array
        if isinstance(image_input, np.ndarray):
            arr = np.ascontiguousarray(image_input)
            if arr.size == 0:
                return None
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)

            # Handle grayscale (2D or 3D with 1 channel)
            if len(arr.shape) == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif len(arr.shape) == 3:
                channels = arr.shape[2]
                if channels == 4:
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
                elif channels == 3:
                    pass
                elif channels == 1:
                    arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
                else:
                    return None
            else:
                return None

            arr = resize_image_if_large(arr, max_width=max_width)
            return np.ascontiguousarray(arr, dtype=np.uint8)
    except Exception as e:
        print(f"Error in preprocess_image: {e}", flush=True)
        return None

    return None


def enhance_lighting(image_rgb):
    """
    Enhance lighting using CLAHE on the luminance channel (LAB color space).
    Resolves poor lighting, heavy shadows, and backlighting.
    """
    try:
        if image_rgb is None or not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
            return image_rgb
        contiguous_rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)
        lab = cv2.cvtColor(contiguous_rgb, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        enhanced_lab = cv2.merge((cl, a_channel, b_channel))
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        return np.ascontiguousarray(enhanced_rgb, dtype=np.uint8)
    except (RuntimeError, cv2.error, Exception) as e:
        print(f"Warning: enhance_lighting failed: {e}")
        return np.ascontiguousarray(image_rgb, dtype=np.uint8) if image_rgb is not None else None


@st.cache_resource
def load_face_models():
    """
    Load face detector and recognition models with caching.
    Includes YuNet, Dlib Shape Predictor (68 landmarks), Dlib ResNet Face Recognition,
    Dlib Frontal Face Detector, and Haar cascades.
    """
    # 1. Dlib 68-point shape predictor
    sp = dlib.shape_predictor(
        face_recognition_models.pose_predictor_model_location()
    )

    # 2. Dlib 128-d face recognition model
    facerec = dlib.face_recognition_model_v1(
        face_recognition_models.face_recognition_model_location()
    )

    # 3. Dlib HOG frontal face detector
    dlib_detector = dlib.get_frontal_face_detector()

    # 4. OpenCV YuNet deep learning face detector (extreme angles & varying lighting)
    yunet_path = ensure_yunet_model()
    yunet_detector = None
    if yunet_path and os.path.exists(yunet_path):
        try:
            yunet_detector = cv2.FaceDetectorYN.create(
                model=yunet_path,
                config="",
                input_size=(320, 320),
                score_threshold=0.5,
                nms_threshold=0.3,
                top_k=5000
            )
        except Exception as e:
            print(f"Warning: Failed to initialize YuNet detector: {e}")

    # 5. OpenCV Haar Cascades for frontal and profile faces as backup
    haar_frontal = None
    haar_profile = None
    try:
        cascade_dir = cv2.data.haarcascades
        haar_frontal = cv2.CascadeClassifier(os.path.join(cascade_dir, 'haarcascade_frontalface_default.xml'))
        haar_profile = cv2.CascadeClassifier(os.path.join(cascade_dir, 'haarcascade_profileface.xml'))
    except Exception:
        pass

    return {
        "sp": sp,
        "facerec": facerec,
        "dlib_detector": dlib_detector,
        "yunet": yunet_detector,
        "haar_frontal": haar_frontal,
        "haar_profile": haar_profile
    }


def compute_iou(boxA, boxB):
    """Calculate Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    return interArea / float(boxAArea + boxBArea - interArea)


def non_maximum_suppression(boxes, iou_threshold=0.35):
    """Deduplicate overlapping bounding boxes."""
    if not boxes:
        return []
    
    # Sort boxes by area descending
    boxes_sorted = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    kept_boxes = []

    for box in boxes_sorted:
        should_keep = True
        for kept in kept_boxes:
            if compute_iou(box, kept) > iou_threshold:
                should_keep = False
                break
        if should_keep:
            kept_boxes.append(box)

    return kept_boxes


def detect_face_rectangles(image_rgb):
    """
    Multi-stage, angle-robust, lighting-invariant face detection pipeline.
    Combines:
    1. YuNet Deep Learning detector (handles yaw/pitch/roll up to ±90°, scale, harsh lighting)
    2. Adaptive CLAHE lighting enhancement
    3. Dlib frontal detector with multi-upsampling
    4. OpenCV Haar cascades (frontal + profile) for angled profiles
    5. Non-Maximum Suppression deduplication

    Returns list of [x1, y1, x2, y2] bounding boxes strictly clipped inside image boundaries.
    """
    if image_rgb is None or not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
        return []

    # Downscale image if large to prevent C++ memory overflow in native extensions
    image_rgb = resize_image_if_large(image_rgb, max_width=1000)
    image_rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)
    if len(image_rgb.shape) != 3 or image_rgb.shape[2] != 3:
        return []

    h, w = image_rgb.shape[:2]
    if h < 20 or w < 20:
        return []

    print(f"[DETECTION] Starting face detection pipeline on image ({w}x{h})...", flush=True)
    models = load_face_models()
    all_boxes = []

    # Strictly clip coordinates inside image boundaries (0 <= x < width, 0 <= y < height)
    def clip_box(x1, y1, x2, y2):
        cx1 = max(0, min(w - 1, int(x1)))
        cy1 = max(0, min(h - 1, int(y1)))
        cx2 = max(0, min(w - 1, int(x2)))
        cy2 = max(0, min(h - 1, int(y2)))
        if (cx2 - cx1) >= 15 and (cy2 - cy1) >= 15:
            return [cx1, cy1, cx2, cy2]
        return None

    # Stage 1: YuNet Deep Learning Face Detector
    yunet = models.get("yunet")
    if yunet is not None:
        try:
            print("[DETECTION] Stage 1: Running YuNet deep learning detector...", flush=True)
            image_bgr = np.ascontiguousarray(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), dtype=np.uint8)
            yunet.setInputSize((int(w), int(h)))
            
            # Primary pass
            yunet.setScoreThreshold(0.50)
            _, faces = yunet.detect(image_bgr)
            
            # If no faces, lower threshold for low-light or angled faces
            if faces is None or len(faces) == 0:
                yunet.setScoreThreshold(0.35)
                _, faces = yunet.detect(image_bgr)

            if faces is not None:
                for face in faces:
                    fx, fy, fw, fh = face[:4]
                    cb = clip_box(fx, fy, fx + fw, fy + fh)
                    if cb:
                        all_boxes.append(cb)
            print(f"[DETECTION] Stage 1 (YuNet): found {len(all_boxes)} candidate(s)", flush=True)
        except (RuntimeError, cv2.error, Exception) as e:
            print(f"[DETECTION] YuNet detection exception: {e}", flush=True)

    # Stage 2: CLAHE lighting enhancement pass if few or no faces found
    if len(all_boxes) == 0:
        try:
            print("[DETECTION] Stage 2: Running CLAHE lighting enhancement pass...", flush=True)
            enhanced_rgb = enhance_lighting(image_rgb)
            if yunet is not None and enhanced_rgb is not None:
                enhanced_bgr = np.ascontiguousarray(cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR), dtype=np.uint8)
                yunet.setInputSize((int(w), int(h)))
                yunet.setScoreThreshold(0.30)
                _, faces = yunet.detect(enhanced_bgr)
                if faces is not None:
                    for face in faces:
                        fx, fy, fw, fh = face[:4]
                        cb = clip_box(fx, fy, fx + fw, fy + fh)
                        if cb:
                            all_boxes.append(cb)
            print(f"[DETECTION] Stage 2 (CLAHE + YuNet): found {len(all_boxes)} candidate(s)", flush=True)
        except (RuntimeError, cv2.error, Exception) as e:
            print(f"[DETECTION] YuNet CLAHE detection exception: {e}", flush=True)

    # Stage 3: Dlib Frontal Detector fallback (upsample 0 and 1)
    if len(all_boxes) == 0:
        dlib_det = models.get("dlib_detector")
        if dlib_det is not None:
            try:
                print("[DETECTION] Stage 3: Running Dlib frontal detector fallback...", flush=True)
                dlib_img = np.ascontiguousarray(image_rgb, dtype=np.uint8)
                dlib_faces = dlib_det(dlib_img, 1)
                for rect in dlib_faces:
                    cb = clip_box(rect.left(), rect.top(), rect.right(), rect.bottom())
                    if cb:
                        all_boxes.append(cb)
                print(f"[DETECTION] Stage 3 (Dlib Frontal): found {len(all_boxes)} candidate(s)", flush=True)
            except (RuntimeError, Exception) as e:
                print(f"[DETECTION] Dlib detector exception: {e}", flush=True)

    # Stage 4: OpenCV Haar Cascades fallback (frontal + profile for side angles)
    if len(all_boxes) == 0:
        try:
            print("[DETECTION] Stage 4: Running Haar Cascades fallback...", flush=True)
            gray = np.ascontiguousarray(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY), dtype=np.uint8)
            gray_eq = np.ascontiguousarray(cv2.equalizeHist(gray), dtype=np.uint8)

            hf = models.get("haar_frontal")
            if hf is not None:
                try:
                    detected_frontal = hf.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                    if detected_frontal is not None:
                        for (hx, hy, hw, hh) in detected_frontal:
                            cb = clip_box(hx, hy, hx + hw, hy + hh)
                            if cb:
                                all_boxes.append(cb)
                except (RuntimeError, cv2.error, Exception) as e:
                    print(f"[DETECTION] Haar frontal exception: {e}", flush=True)

            hp = models.get("haar_profile")
            if hp is not None and len(all_boxes) == 0:
                try:
                    # Profile faces (left and right)
                    detected_prof = hp.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                    if detected_prof is not None:
                        for (hx, hy, hw, hh) in detected_prof:
                            cb = clip_box(hx, hy, hx + hw, hy + hh)
                            if cb:
                                all_boxes.append(cb)

                    # Flip image horizontally to detect right profiles
                    gray_flipped = np.ascontiguousarray(cv2.flip(gray_eq, 1), dtype=np.uint8)
                    detected_prof_flip = hp.detectMultiScale(gray_flipped, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                    if detected_prof_flip is not None:
                        for (hx, hy, hw, hh) in detected_prof_flip:
                            cb = clip_box(w - (hx + hw), hy, w - hx, hy + hh)
                            if cb:
                                all_boxes.append(cb)
                except (RuntimeError, cv2.error, Exception) as e:
                    print(f"[DETECTION] Haar profile exception: {e}", flush=True)
            print(f"[DETECTION] Stage 4 (Haar): found {len(all_boxes)} candidate(s)", flush=True)
        except (RuntimeError, cv2.error, Exception) as e:
            print(f"[DETECTION] Haar detection exception: {e}", flush=True)

    # Deduplicate overlapping detections
    kept_boxes = non_maximum_suppression(all_boxes, iou_threshold=0.35)
    print(f"[DETECTION] NMS completed: {len(kept_boxes)} unique face(s) identified", flush=True)
    return kept_boxes


def get_face_embeddings(image_input):
    """
    Extract 128-dimensional face embedding vectors for all detected faces.
    100% compatible with existing Supabase face embeddings!
    """
    encodings, _ = get_face_embeddings_with_boxes(image_input)
    return encodings


def get_face_embeddings_with_boxes(image_input):
    """
    Extract 128-d embeddings and their corresponding bounding boxes [x1, y1, x2, y2].
    Downscales large images, strictly clips coordinates, processes each face in an isolated try-except block,
    and logs progress at every step to eliminate segfaults.
    """
    image_rgb = preprocess_image(image_input, max_width=1000)
    if image_rgb is None or not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
        return [], []

    # Strictly contiguous uint8 array for OpenCV & dlib
    image_rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)
    if len(image_rgb.shape) != 3 or image_rgb.shape[2] != 3:
        return [], []

    models = load_face_models()
    sp = models.get("sp")
    facerec = models.get("facerec")
    if sp is None or facerec is None:
        print("[ERROR] Face models (sp or facerec) not loaded!", flush=True)
        return [], []

    boxes = detect_face_rectangles(image_rgb)
    encodings = []
    valid_boxes = []

    h, w = image_rgb.shape[:2]
    print(f"[ENCODING] Processing {len(boxes)} detected face candidate(s) in image ({w}x{h})...", flush=True)

    for idx, box in enumerate(boxes):
        try:
            bx1, by1, bx2, by2 = box
            bw = bx2 - bx1
            bh = by2 - by1

            if bw < 10 or bh < 10:
                print(f"[CROP] Face #{idx+1}: Box too small ({bw}x{bh}), skipping", flush=True)
                continue

            # Add 10% proportional padding to ensure chin and forehead are included for dlib's 68 landmarks
            pad_x = int(bw * 0.10)
            pad_y = int(bh * 0.10)

            # Strictly clip within 0 <= px1 < px2 <= w - 1 and 0 <= py1 < py2 <= h - 1
            px1 = max(0, min(w - 1, int(bx1 - pad_x)))
            py1 = max(0, min(h - 1, int(by1 - pad_y)))
            px2 = max(0, min(w - 1, int(bx2 + pad_x)))
            py2 = max(0, min(h - 1, int(by2 + pad_y)))

            if (px2 - px1) < 15 or (py2 - py1) < 15:
                print(f"[CROP] Face #{idx+1}: Clipped box too small ({px2 - px1}x{py2 - py1}), skipping", flush=True)
                continue

            print(f"[CROP] Face #{idx+1}/{len(boxes)}: raw box={box} -> clipped rect=[left={px1}, top={py1}, right={px2}, bottom={py2}] in ({w}x{h})", flush=True)

            dlib_rect = dlib.rectangle(int(px1), int(py1), int(px2), int(py2))
            contiguous_img = np.ascontiguousarray(image_rgb, dtype=np.uint8)

            # Isolated landmark prediction
            print(f"[LANDMARK] Face #{idx+1}: computing 68 shape landmarks...", flush=True)
            shape = None
            try:
                shape = sp(contiguous_img, dlib_rect)
            except (RuntimeError, Exception) as le:
                print(f"[LANDMARK] Face #{idx+1} landmark extraction failed: {le}", flush=True)
                continue

            if shape is None or shape.num_parts < 68:
                print(f"[LANDMARK] Face #{idx+1}: Invalid landmarks shape, skipping", flush=True)
                continue
            print(f"[LANDMARK] Face #{idx+1}: 68 shape landmarks extracted successfully", flush=True)

            # Isolated face descriptor encoding
            print(f"[ENCODING] Face #{idx+1}: computing 128-d face descriptor...", flush=True)
            face_descriptor = None
            try:
                face_descriptor = facerec.compute_face_descriptor(contiguous_img, shape, 1)
            except (RuntimeError, Exception) as ee:
                print(f"[ENCODING] Face #{idx+1} descriptor computation failed: {ee}", flush=True)
                continue

            if face_descriptor is None:
                print(f"[ENCODING] Face #{idx+1}: Empty face descriptor, skipping", flush=True)
                continue

            emb_arr = np.array(face_descriptor, dtype=np.float64)
            if emb_arr.shape == (128,):
                encodings.append(emb_arr)
                valid_boxes.append([bx1, by1, bx2, by2])
                print(f"[ENCODING] Face #{idx+1}: 128-d descriptor generated successfully", flush=True)
            else:
                print(f"[ENCODING] Face #{idx+1}: Unexpected embedding shape {emb_arr.shape}, skipping", flush=True)

        except (RuntimeError, cv2.error, Exception) as e:
            print(f"[ERROR] Face #{idx+1} failed in isolated loop block: {e}", flush=True)
            continue

    print(f"[ENCODING] Successfully generated {len(encodings)} embeddings from {len(boxes)} face candidate(s)", flush=True)
    return encodings, valid_boxes


@st.cache_resource
def get_trained_model():
    """
    Fetch all enrolled student embeddings from database and prepare nearest neighbor model.
    """
    X = []
    y = []

    student_db = get_all_students()

    if not student_db:
        return None

    for student in student_db:
        embedding = student.get('face_embedding')
        if embedding:
            emb_arr = np.array(embedding, dtype=np.float64)
            if emb_arr.shape == (128,):
                X.append(emb_arr)
                y.append(student.get('student_id'))

    if len(X) == 0:
        return None

    clf = None
    distinct_classes = len(set(y))
    if distinct_classes >= 2:
        base_svc = SVC(kernel='linear', class_weight='balanced')
        class_counts = [y.count(c) for c in set(y)]
        min_class_count = min(class_counts)
        if min_class_count >= 2:
            clf = CalibratedClassifierCV(estimator=base_svc, cv=min(5, min_class_count), ensemble=False)
        else:
            clf = CalibratedClassifierCV(estimator=base_svc, ensemble=False)

        try:
            clf.fit(X, y)
        except Exception:
            try:
                base_svc.fit(X, y)
                clf = base_svc
            except Exception:
                clf = None

    return {'clf': clf, 'X': X, 'y': y}


def train_classifier():
    """Clear cached model and reload fresh embeddings from Supabase."""
    st.cache_resource.clear()
    model_data = get_trained_model()
    return bool(model_data)


def compute_match_confidence(dist, cos):
    """
    Computes a calibrated confidence percentage (0-100%) from Euclidean distance and Cosine similarity.
    Threshold boundary is calibrated around Euclidean 0.44 and Cosine 0.925.
    """
    if dist < 0.44:
        dist_conf = 60.0 + (0.44 - dist) / (0.44 - 0.20) * 40.0
    else:
        dist_conf = max(0.0, 60.0 - (dist - 0.44) / (0.60 - 0.44) * 60.0)

    if cos > 0.925:
        cos_conf = 60.0 + (cos - 0.925) / (0.99 - 0.925) * 40.0
    else:
        cos_conf = max(0.0, 60.0 - (0.925 - cos) / (0.925 - 0.85) * 60.0)

    combined = 0.6 * dist_conf + 0.4 * cos_conf
    return round(float(np.clip(combined, 0.0, 99.9)), 1)


def authenticate_student_face(
    image_input,
    resemblance_threshold=0.44,
    min_cosine_similarity=0.925,
    min_confidence=60.0
):
    """
    Strict 1-to-N FaceID authentication for student login.
    Guarantees that unregistered faces, impostors, or borderline matches are rejected
    with status 'unknown' instead of misidentifying as another student.

    Returns:
        dict with:
            'authenticated': bool,
            'student_id': int or None,
            'confidence': float,
            'distance': float or None,
            'cosine': float or None,
            'num_faces': int,
            'status': 'success' | 'no_face' | 'multiple_faces' | 'unknown' | 'no_db',
            'message': str
    """
    encodings, _ = get_face_embeddings_with_boxes(image_input)
    num_faces = len(encodings)

    if num_faces == 0:
        return {
            'authenticated': False,
            'student_id': None,
            'confidence': 0.0,
            'distance': None,
            'cosine': None,
            'num_faces': 0,
            'status': 'no_face',
            'message': 'No face detected! Please ensure your face is clearly visible, well-lit, and facing the camera.'
        }

    if num_faces > 1:
        return {
            'authenticated': False,
            'student_id': None,
            'confidence': 0.0,
            'distance': None,
            'cosine': None,
            'num_faces': num_faces,
            'status': 'multiple_faces',
            'message': 'Multiple faces detected! Please ensure only you are in the camera frame.'
        }

    encoding = encodings[0]
    model_data = get_trained_model()

    if not model_data or len(model_data.get('X', [])) == 0:
        return {
            'authenticated': False,
            'student_id': None,
            'confidence': 0.0,
            'distance': None,
            'cosine': None,
            'num_faces': 1,
            'status': 'no_db',
            'message': 'No registered students found in database. Please register your profile below.'
        }

    X_train = model_data['X']
    y_train = model_data['y']

    norm_enc = np.linalg.norm(encoding)
    if norm_enc == 0:
        norm_enc = 1e-7

    matches = []
    for idx, student_emb in enumerate(X_train):
        dist = float(np.linalg.norm(student_emb - encoding))
        norm_st = np.linalg.norm(student_emb)
        if norm_st == 0:
            norm_st = 1e-7
        cos = float(np.dot(student_emb, encoding) / (norm_st * norm_enc))
        conf = compute_match_confidence(dist, cos)
        matches.append({
            'student_id': y_train[idx],
            'distance': dist,
            'cosine': cos,
            'confidence': conf
        })

    matches.sort(key=lambda m: m['distance'])
    best = matches[0]

    # Margin check: if top 2 candidates are nearly equidistant, reject as ambiguous
    is_ambiguous = False
    if len(matches) >= 2:
        second_best = matches[1]
        if (second_best['distance'] - best['distance']) < 0.02 and best['distance'] > 0.40:
            is_ambiguous = True

    # Strict multi-criteria check
    passes_distance = best['distance'] <= resemblance_threshold
    passes_cosine = best['cosine'] >= min_cosine_similarity
    passes_confidence = best['confidence'] >= min_confidence

    if passes_distance and passes_cosine and passes_confidence and not is_ambiguous:
        return {
            'authenticated': True,
            'student_id': best['student_id'],
            'confidence': best['confidence'],
            'distance': best['distance'],
            'cosine': best['cosine'],
            'num_faces': 1,
            'status': 'success',
            'message': f"Face verified successfully ({best['confidence']:.1f}% confidence)."
        }
    else:
        return {
            'authenticated': False,
            'student_id': None,
            'confidence': best['confidence'],
            'distance': best['distance'],
            'cosine': best['cosine'],
            'num_faces': 1,
            'status': 'unknown',
            'message': (
                f"Face not recognized as an enrolled student (confidence: {best['confidence']:.1f}% - "
                f"minimum 60% required). If you are a new student, please register below."
            )
        }


def predict_attendance(
    class_image_input,
    resemblance_threshold=0.45,
    min_cosine_similarity=0.91,
    min_confidence=58.0
):
    """
    Detect faces in classroom / camera image and match against registered students
    using strict dual-metric verification (Euclidean distance + Cosine similarity + confidence threshold)
    to eliminate false positives.

    Returns:
    - detected_student: dict of {student_id: {'distance': float, 'cosine': float, 'confidence': float}}
    - all_students: list of all registered student IDs
    - num_faces: count of detected faces in the image
    """
    print(f"[PREDICTION] Starting predict_attendance...", flush=True)
    encodings, _ = get_face_embeddings_with_boxes(class_image_input)
    detected_student = {}

    model_data = get_trained_model()

    if not model_data or len(model_data.get('X', [])) == 0:
        print("[PREDICTION] No enrolled students found in database!", flush=True)
        return detected_student, [], len(encodings)

    X_train = model_data['X']
    y_train = model_data['y']

    all_students = sorted(list(set(y_train)))
    print(f"[PREDICTION] Matching {len(encodings)} detected face(s) against {len(all_students)} enrolled student(s)...", flush=True)

    for i, encoding in enumerate(encodings):
        try:
            norm_enc = np.linalg.norm(encoding)
            if norm_enc == 0:
                norm_enc = 1e-7

            best_dist = float('inf')
            best_cos = -1.0
            best_id = None
            best_conf = 0.0

            for student_emb, sid in zip(X_train, y_train):
                dist = float(np.linalg.norm(student_emb - encoding))
                norm_st = np.linalg.norm(student_emb)
                if norm_st == 0:
                    norm_st = 1e-7
                cos = float(np.dot(student_emb, encoding) / (norm_st * norm_enc))
                conf = compute_match_confidence(dist, cos)

                if dist < best_dist:
                    best_dist = dist
                    best_cos = cos
                    best_id = sid
                    best_conf = conf

            # Strict check: MUST satisfy distance, cosine similarity, and confidence
            if (best_id is not None and
                best_dist <= resemblance_threshold and
                best_cos >= min_cosine_similarity and
                best_conf >= min_confidence):
                detected_student[best_id] = {
                    'distance': best_dist,
                    'cosine': best_cos,
                    'confidence': best_conf
                }
                print(f"[PREDICTION] Face #{i+1}: MATCH -> Student ID {best_id} (conf={best_conf:.1f}%, dist={best_dist:.3f}, cos={best_cos:.3f})", flush=True)
            else:
                print(f"[PREDICTION] Face #{i+1}: NO MATCH -> Closest Student ID {best_id} (conf={best_conf:.1f}%, dist={best_dist:.3f}, cos={best_cos:.3f}) below threshold", flush=True)
        except Exception as pe:
            print(f"[PREDICTION] Error evaluating face #{i+1}: {pe}", flush=True)

    print(f"[PREDICTION] Attendance prediction complete: {len(detected_student)} student(s) identified from {len(encodings)} face(s)", flush=True)
    return detected_student, all_students, len(encodings)

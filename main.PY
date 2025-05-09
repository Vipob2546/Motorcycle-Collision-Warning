import numpy as np
import cv2
import time
import pandas as pd
import onnxruntime as ort
from ultralytics import YOLO
import supervision as sv
from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment
from pydub.playback import play
import threading

# Constants and Global Variables
MAX_DEPTH = 80
DEPTH_MODEL_PATH = "models/depth_anything_v2_vits_outdoor_dynamic.onnx"
ORIGINAL_CAMERA_MATRIX = np.array([
    [1066, 0, 960],
    [0, 1066, 540],
    [0, 0, 1]
])
TARGET_CLASSES = [3]
prev_positions = {}
WARNING_DURATION = 1

z_to_real_world = {
    0:0.2,
    0.8:1.2,
    2.5:3.2,
    3.8:5.2,
    4.8:7.2,
    6.1:10.2
}

# Global variables for thread safety and audio preloading
_tracker_last_warning = {}
_tracker_last_warning_lock = threading.Lock()

_preloaded_sound = None  # Single preloaded audio, panning applied dynamically

_warning_playing_lock = threading.Lock()
_warning_playing = False

warning_thread_pool = ThreadPoolExecutor(max_workers=1)

# Initialize audio sound
def initialize_audio():
    global _preloaded_sound
    try:
        sound = AudioSegment.from_mp3("sound2.mp3")
        sound = sound + 10  # Increase volume by 10 dB
        # Ensure the sound is stereo
        sound = sound.set_channels(2)
        _preloaded_sound = sound
    except Exception as e:
        print("Error preloading audio:", e)

def map_z_to_real_world(z_value):
    z_values = np.array(list(z_to_real_world.keys()))
    real_world_values = np.array(list(z_to_real_world.values()))

    # Perform linear interpolation if the z_value is between the known z values
    if z_value <= z_values[0]:
        return real_world_values[0]
    elif z_value >= z_values[-1]:
        return real_world_values[-1]
    
    for i in range(1, len(z_values)):
        if z_value < z_values[i]:
            x0, x1 = z_values[i-1], z_values[i]
            y0, y1 = real_world_values[i-1], real_world_values[i]
            real_world_distance = y0 + (z_value - x0) * (y1 - y0) / (x1 - x0)
            return real_world_distance

def trigger_collision_warning(side="left"):
    global _warning_playing
    with _warning_playing_lock:
        if _warning_playing:
            return
        _warning_playing = True
    try:
        if side.lower() == "left":
            audio = _preloaded_sound.pan(-1.0)
        elif side.lower() == "right":
            audio = _preloaded_sound.pan(1.0)
        elif side.lower() == "both":
            audio = _preloaded_sound
        else:
            print("Invalid side for warning")
            audio = _preloaded_sound
        play(audio)
    except Exception as e:
        print("Error playing collision warning sound:", e)
    finally:
        with _warning_playing_lock:
            _warning_playing = False

# Preprocess frame
def preprocess_frame(frame, target_height, target_width):
    frame = cv2.resize(frame, (target_width, target_height))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0
    frame = np.transpose(frame, (2, 0, 1))
    frame = np.expand_dims(frame, axis=0)
    return frame

# Postprocess depth map
def postprocess_depth_map(depth):
    depth = np.squeeze(depth)
    depth = np.clip(depth, 0, MAX_DEPTH)
    depth_scaled = (depth / MAX_DEPTH) * 255.0
    return depth_scaled.astype(np.uint8)

# Run depth model
def run_depth_model(ort_session, input_tensor):
    start_time = time.time()
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
    ort_outs = ort_session.run(None, ort_inputs)
    latency = (time.time() - start_time) * 1000  # in ms
    return ort_outs[0], latency

# Run YOLO with tracking
def run_yolo_with_tracking(byte_track, yolo_model, frame, target_classes):
    start_time = time.time()
    results = yolo_model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    mask = np.isin(detections.class_id, target_classes)
    detections = detections[mask]
    detections = detections[detections.confidence > 0.3]
    detections = detections.with_nms(threshold=0.5)
    detections = byte_track.update_with_detections(detections=detections)
    latency = (time.time() - start_time) * 1000
    return detections, latency

# Depth map to point cloud conversion
def depth_map_to_point_cloud_full(depth_map, camera_matrix):
    depth_map = np.squeeze(depth_map)
    h, w = depth_map.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    f_x = camera_matrix[0, 0]
    f_y = camera_matrix[1, 1]
    c_x = camera_matrix[0, 2]
    c_y = camera_matrix[1, 2]
    X = (u - c_x) * depth_map / f_x
    Y = (v - c_y) * depth_map / f_y
    Z = depth_map
    point_cloud = np.stack((X, Y, Z), axis=-1)
    return point_cloud

# Update z, vz, ttc values every second
def yolo2velocity_with_point_cloud(depth_map, detections, camera_matrix, original_frame_size, 
                                   current_timestamp, prev_positions, car_speed_kmh, is_initializing, last_update_timestamp):
    point_cloud = depth_map_to_point_cloud_full(depth_map, camera_matrix)
    valid_labels = []
    valid_indices = []
    
    frame_width, frame_height = original_frame_size
    depth_height, depth_width = point_cloud.shape[:2]
    
    x_scale = depth_width / frame_width
    y_scale = depth_height / frame_height

    for i, (bbox, tracker_id) in enumerate(zip(detections.xyxy, detections.tracker_id)):
        x1, y1, x2, y2 = (bbox * np.array([x_scale, y_scale, x_scale, y_scale])).astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(depth_width - 1, x2), min(depth_height - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        center_x = (x1 + x2) // 2
        center_y = y2
        if not (0 <= center_x < depth_width and 0 <= center_y < depth_height):
            continue

        current_x = point_cloud[center_y, center_x, 0]
        current_z = point_cloud[center_y, center_x, 2]
        x_mapped = current_x * 0.7
        z_mapped = current_z / 5
        z_mapped_table = map_z_to_real_world(z_mapped)

        if not (0 <= z_mapped_table <= 10):
            continue

        if not (-7.25 <= x_mapped <= 7.25):
            continue

        if -1.75 <= x_mapped <= 1.75:
            lane_label = "Same lane"
        elif x_mapped < -1.75:
            lane_label = "Right lane"
        elif x_mapped > 1.75:
            lane_label = "Left lane"
        else:
            lane_label = "Unknown"

        # Initialize label for the first second
        if is_initializing:
            label = f"#{tracker_id} | Initializing..."
        else:
            label = f"#{tracker_id} | depth:{z_mapped_table:.1f}m | speed:{prev_positions.get(tracker_id, [0, 0, 0])[2]:.1f}km/h | TTC:{prev_positions.get(tracker_id, [0, 0, 0, 0])[3]:.1f}s"

        if tracker_id not in prev_positions:
            prev_positions[tracker_id] = (current_timestamp, z_mapped_table, 0.0, np.nan)  # Initialize with first z value
            ttc = np.nan
            object_vz = 0  # Set object_vz to 0 on initialization
        else:
            prev_timestamp, prev_z, prev_speed, prev_ttc = prev_positions[tracker_id]
            dt = current_timestamp - prev_timestamp

            # Debug: Print prev_positions content

            # Only update once per second
            if (current_timestamp - last_update_timestamp) >= 1:
                if not is_initializing:
                    # Calculate object velocity (vz) using the formula you provided
                    object_vz = (((prev_z - z_mapped_table) * 3.6) / dt) + car_speed_kmh



                    # Calculate TTC using the new formula
                    ttc = (z_mapped_table * 3.6) / (object_vz - car_speed_kmh)

                    # Debug: Print the updated prev_positions for tracker
                    prev_positions[tracker_id] = (current_timestamp, z_mapped_table, object_vz, ttc)
                    
                    label = f"#{tracker_id} | depth:{z_mapped_table:.1f}m | speed:{object_vz:.1f}km/h | TTC:{ttc:.1f}s"

                    # Trigger collision warning only if TTC is positive and less than 1 second
                    if ttc > 0 and ttc < 1 and z_mapped_table < 2.5:
                        with _tracker_last_warning_lock:
                            last_warning = _tracker_last_warning.get(tracker_id, 0)
                            if (current_timestamp - last_warning) >= WARNING_DURATION:
                                if lane_label == "Same lane":
                                    warning_side = "both"
                                elif lane_label == "Left lane":
                                    warning_side = "left"
                                elif lane_label == "Right lane":
                                    warning_side = "right"
                                else:
                                    warning_side = "both"
                                _tracker_last_warning[tracker_id] = current_timestamp
                                warning_thread_pool.submit(trigger_collision_warning, warning_side)

                last_update_timestamp = current_timestamp

        # Print label for debugging
        print(f"Frame {i}, Tracker #{tracker_id}: {label}")

        valid_labels.append(label)
        valid_indices.append(i)
    
    filtered_detections = detections[valid_indices]
    return valid_labels, filtered_detections, last_update_timestamp



def postprocess_yolo(frame, labels, detections, box_annotator, label_annotator):
    annotated_frame = frame.copy()
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame

def load_gps_data(csv_path):
    gps_data = pd.read_csv(csv_path)
    if 'timestamp' not in gps_data.columns or 'speed' not in gps_data.columns:
        raise ValueError("The CSV must contain 'timestamp' and 'speed' columns.")
    gps_data.set_index('timestamp', inplace=True)
    return gps_data

def get_speed_from_timestamp(gps_data, timestamp):
    video_second = int(timestamp)
    if video_second in gps_data.index:
        return gps_data.loc[video_second, 'speed']
    return 0.0

def main():
    initialize_audio()

    video_path = 'test_video/final.mp4'
    yolo_path = 'models/yolov8l.pt'
    gps_path = 'gps_data_final.csv'

    depth_height, depth_width = 518, 518

    byte_track = sv.ByteTrack(track_activation_threshold=0.5)
    box_annotator = sv.BoxAnnotator(thickness=3)
    label_annotator = sv.LabelAnnotator(text_scale=0.75, text_thickness=2, text_position=sv.Position.TOP_CENTER)

    depth_ort_session = ort.InferenceSession(DEPTH_MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    yolo_model = YOLO(yolo_path)

    gps_data = load_gps_data(gps_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_frame_size = (orig_width, orig_height)
    print(f"Camera resolution: {orig_width} x {orig_height}")

    scale_x = depth_width / orig_width
    scale_y = depth_height / orig_height
    CAMERA_MATRIX_SCALED = np.array([
        [ORIGINAL_CAMERA_MATRIX[0, 0] * scale_x, 0, ORIGINAL_CAMERA_MATRIX[0, 2] * scale_x],
        [0, ORIGINAL_CAMERA_MATRIX[1, 1] * scale_y, ORIGINAL_CAMERA_MATRIX[1, 2] * scale_y],
        [0, 0, 1]
    ])

    # Initialize video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    depth_out = cv2.VideoWriter('output_depth_map.mp4', fourcc, 60, (1920, 1080), isColor=False)
    annotated_out = cv2.VideoWriter('output_annotated_frame.mp4', fourcc, 60, (1920, 1080))

    executor = ThreadPoolExecutor(max_workers=16)
    frame_index = 0

    is_initializing = True
    last_update_timestamp = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        video_timestamp = frame_index / original_fps
        car_speed_kmh = get_speed_from_timestamp(gps_data, video_timestamp)

        resized_depth_frame = cv2.resize(frame, (depth_width, depth_height))
        depth_input_tensor = preprocess_frame(resized_depth_frame, depth_height, depth_width)
        depth_future = executor.submit(run_depth_model, depth_ort_session, depth_input_tensor)
        yolo_future = executor.submit(run_yolo_with_tracking, byte_track, yolo_model, frame, TARGET_CLASSES)

        depth_map, depth_latency = depth_future.result()
        detections, yolo_latency = yolo_future.result()

        depth_vis = postprocess_depth_map(depth_map)
        labels, filtered_detections, last_update_timestamp = yolo2velocity_with_point_cloud(
            depth_map, detections, CAMERA_MATRIX_SCALED, original_frame_size,
            video_timestamp, prev_positions, car_speed_kmh, is_initializing, last_update_timestamp)

        # After 2 seconds, stop initializing
        if video_timestamp >= 2:
            is_initializing = False

        annotated_frame = postprocess_yolo(frame, labels, filtered_detections, box_annotator, label_annotator)

        # Resize depth map and annotated frame to FHD resolution (1920x1080)
        resized_depth_map = cv2.resize(depth_vis, (1920, 1080))
        resized_annotated_frame = cv2.resize(annotated_frame, (1920, 1080))

        cv2.putText(
            resized_annotated_frame,
            f"Speed: {car_speed_kmh:.1f} km/h",
            (orig_width - 620, orig_height - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 0),
            6,
            cv2.LINE_AA
        )

        # Write the frames to video
        depth_out.write(resized_depth_map)
        annotated_out.write(resized_annotated_frame)

        total_latency = (time.time() - start_time) * 1000
        fps = 1000 / total_latency if total_latency > 0 else 0

        # Resize for display
        annotated_frame = cv2.resize(resized_annotated_frame, (960, 540))
        depth_map = cv2.resize(resized_depth_map, (960,540))
        cv2.imshow("Tracked Video", annotated_frame)
        cv2.imshow("Depth Map", depth_map)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_index += 1

    cap.release()
    depth_out.release()
    annotated_out.release()
    executor.shutdown()
    warning_thread_pool.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

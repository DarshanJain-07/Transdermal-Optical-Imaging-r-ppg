import cv2
import numpy as np
import mediapipe as mp
import os # Added for path checking
import argparse # Add argparse import
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2 # Import protobuf format
# Import solutions module for drawing utils etc.
import mediapipe.python.solutions as mp_solutions
from feature_extraction import (
    extract_wavelet_features_from_diff, construct_and_resize_sti, # Path A
    apply_msr, fuse_rgb_msr, extract_block_features # Path B
)

# --- MediaPipe Setup ---
# Standard aliases for MediaPipe Task API components
BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

# Global variable to hold the FaceLandmarker instance
# Necessary because the model needs to persist between video frames
face_landmarker = None

def initialize_face_landmarker():
    """Initializes the MediaPipe FaceLandmarker."""
    global face_landmarker
    try:
        # Define options: model path, running mode, confidence thresholds, etc.
        # Using VIDEO mode implies MediaPipe handles some inter-frame tracking.
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task'),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1, # Optimize for single face detection
            min_face_detection_confidence=0.85,
            min_face_presence_confidence=0.85,
            min_tracking_confidence=0.85,
            output_face_blendshapes=False, # Disabled as not needed
            output_facial_transformation_matrixes=False # Disabled as not needed
        )
        face_landmarker = FaceLandmarker.create_from_options(options)
        print("MediaPipe FaceLandmarker initialized successfully.")
    except Exception as e:
        # Catch potential errors during initialization (e.g., model file not found)
        print(f"Error initializing MediaPipe FaceLandmarker: {e}")
        face_landmarker = None # Ensure it's None if initialization failed

# Initialize the landmarker when the module is loaded
initialize_face_landmarker()

# --- Constants ---
# Define specific MediaPipe landmark indices for ROIs
# These are based on the 478-landmark model provided by MediaPipe.
FOREHEAD_POINTS = [103, 67, 109, 10, 338, 297, 332, 333, 334, 296, 336, 285, 417, 351, 419, 197, 196, 122, 193, 55, 107, 66, 105, 104, 103]
LEFT_CHEEK_POINTS = [345, 340, 346, 347, 348, 349, 329, 371, 266, 425, 411, 352, 345] # Viewer's right
RIGHT_CHEEK_POINTS = [116, 111, 117, 118, 119, 120, 100, 142, 36, 205, 187, 123, 116] # Viewer's left

# Standard ROI size for resizing (Width, Height)
ROI_SIZE = (64, 64)

# --- Face and Landmark Detection (MediaPipe) ---

def detect_face_and_landmarks_mediapipe(frame, timestamp_ms):
    """
    Detects a face and its 478 landmarks using MediaPipe FaceLandmarker.

    Args:
        frame: Input image (numpy array in BGR format).
        timestamp_ms: The timestamp of the frame in milliseconds.

    Returns:
        mediapipe.tasks.python.vision.FaceLandmarkerResult: Detection results containing landmarks.
               Returns None if no face is detected or if the landmarker isn't initialized.
    Requires the global face_landmarker to be initialized.
    """
    if face_landmarker is None: # Check if initialization succeeded
        print("MediaPipe FaceLandmarker not initialized.")
        return None

    # MediaPipe expects RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    try:
        # Perform detection for the current video frame
        detection_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
        # Return result only if landmarks were actually found
        if detection_result and detection_result.face_landmarks:
            return detection_result
        else:
            print("No face landmarks detected in the frame.") # Optional: uncomment for debugging
            return None
    except Exception as e:
        print(f"Error during MediaPipe face landmark detection: {e}")
        return None


def extract_landmarks_coordinates_mediapipe(detection_result, frame_shape):
    """
    Converts MediaPipe landmarks result into a numpy array of (x, y) pixel coordinates.

    Args:
        detection_result: The result object from face_landmarker.detect_for_video.
        frame_shape: The shape of the original frame (height, width).

    Returns:
        np.ndarray: Array of shape (478, 2) containing landmark coordinates in pixels.
                    Returns None if input is invalid or no landmarks are present.
    """
    # Validate input
    if not detection_result or not detection_result.face_landmarks:
        return None

    # Assumes num_faces=1 was used during initialization
    landmarks = detection_result.face_landmarks[0]
    height, width = frame_shape[:2] # Get frame dimensions for denormalization

    # Pre-allocate numpy array for efficiency
    coords = np.zeros((len(landmarks), 2), dtype=int)

    # Iterate and convert normalized coordinates to pixel coordinates
    for i, landmark in enumerate(landmarks):
        coords[i] = (int(landmark.x * width), int(landmark.y * height))

    return coords

# --- ROI Extraction (MediaPipe) ---

def get_bounding_box(points):
    """Calculates the tight bounding box for a set of landmark points (pixel coords)."""
    if points is None or len(points) == 0:
        return None
    # Use numpy min/max for efficient calculation
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    x = np.min(x_coords)
    y = np.min(y_coords)
    w = np.max(x_coords) - x
    h = np.max(y_coords) - y
    # Return as integers for use with OpenCV drawing/cropping
    return int(x), int(y), int(w), int(h)

def define_roi_forehead_mediapipe(landmarks_coords):
    """
    Defines the forehead ROI based on predefined MediaPipe landmark indices.

    Args:
        landmarks_coords (np.ndarray): Array of shape (478, 2) with landmark coordinates.

    Returns:
        tuple: (x, y, w, h) defining the ROI rectangle, or None if landmarks are invalid.
    """
    if landmarks_coords is None or landmarks_coords.shape[0] != 478:
        return None
    try:
        # Select the coordinates corresponding to the forehead points
        forehead_landmarks = landmarks_coords[FOREHEAD_POINTS]
        # Calculate the bounding box
        return get_bounding_box(forehead_landmarks)
    except IndexError:
        # Catch error if landmark indices are out of bounds (shouldn't happen with 478)
        print("Error: Forehead landmark indices out of bounds.")
        return None

def define_roi_cheeks_mediapipe(landmarks_coords):
    """
    Defines ROIs for left and right cheeks based on MediaPipe landmarks.

    Args:
        landmarks_coords (np.ndarray): Array of shape (478, 2) with landmark coordinates.

    Returns:
        tuple: (roi_left, roi_right) where each roi is (x, y, w, h), or (None, None).
        Returns two bounding boxes: (roi_left, roi_right).
    """
    if landmarks_coords is None or landmarks_coords.shape[0] != 478:
        return None, None
    try:
        # Select coordinates for left and right cheek points
        left_cheek_landmarks = landmarks_coords[LEFT_CHEEK_POINTS]
        right_cheek_landmarks = landmarks_coords[RIGHT_CHEEK_POINTS]

        # Calculate bounding boxes
        roi_left = get_bounding_box(left_cheek_landmarks)
        roi_right = get_bounding_box(right_cheek_landmarks)

        # Basic validation: Ensure ROIs have positive width and height
        # get_bounding_box should handle empty lists, this handles degenerate cases
        if roi_left and (roi_left[2] <= 0 or roi_left[3] <= 0):
            print("Warning: Calculated left cheek ROI has non-positive width or height.")
            roi_left = None # Invalidate if dimensions are zero or negative
        if roi_right and (roi_right[2] <= 0 or roi_right[3] <= 0):
            print("Warning: Calculated right cheek ROI has non-positive width or height.")
            roi_right = None # Invalidate if dimensions are zero or negative

        return roi_left, roi_right
    except IndexError:
        print("Error: Cheek landmark indices out of bounds.")
        return None, None


def extract_roi_frame(frame, roi_rect):
    """
    Extracts the ROI sub-image (pixels) from the frame using a bounding box.
    Handles boundary conditions.

    Args:
        frame: The input frame (numpy array).
        roi_rect: Tuple (x, y, w, h) defining the ROI.

    Returns:
        numpy.ndarray: The extracted ROI image, or None if roi_rect is invalid.
    """
    if roi_rect is None:
        return None
    x, y, w, h = roi_rect

    # Ensure ROI coordinates are within the actual frame dimensions
    h_frame, w_frame = frame.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_frame, x + w) # Use min with frame boundary
    y2 = min(h_frame, y + h) # Use min with frame boundary

    # Check if the resulting ROI is valid (has actual area)
    if x2 <= x1 or y2 <= y1:
        # This can happen if the calculated ROI is completely outside the frame
        print("Warning: ROI rectangle is outside frame boundaries or invalid after clipping.")
        return None

    # Extract the ROI using numpy slicing
    return frame[y1:y2, x1:x2]

# --- Video FPS Conversion ---
def convert_video_to_30fps(input_video_path, output_video_path):
    """
    Reads an input video, preserves all frames, and writes them
    to a new video file encoded at exactly 30 FPS.

    Note: This might change the playback speed and duration compared
          to the original video if its FPS was different from 30.

    Args:
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path where the 30 FPS output video will be saved.

    Returns:
        bool: True if conversion was successful, False otherwise.
    """
    if not os.path.exists(input_video_path):
        print(f"Error: Input video not found at {input_video_path}")
        return False

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open input video {input_video_path}")
        return False

    # Get original video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Input video: {input_video_path}")
    print(f"Original FPS: {original_fps:.2f}, Dimensions: {frame_width}x{frame_height}")

    # Prepare VideoWriter
    output_dir = os.path.dirname(output_video_path)
    if output_dir and not os.path.exists(output_dir): # Create output dir if needed
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4
    target_fps = 30.0 # Target FPS
    out = cv2.VideoWriter(output_video_path, fourcc, target_fps, (frame_width, frame_height))

    if not out.isOpened():
        print(f"Error: Could not open VideoWriter for output file {output_video_path}")
        cap.release()
        return False

    print(f"Converting to 30 FPS. Output video: {output_video_path}")

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret: # End of video or error
            break
        # Write frame without modification
        out.write(frame)
        frame_num += 1
        # Optional progress print
        # if frame_num % 100 == 0: print(f"Processed {frame_num} frames...")

    # Release resources
    cap.release()
    out.release()
    print(f"Conversion complete. Total frames processed: {frame_num}")
    return True


# --- Example Usage (Updated for Combined Features) ---
# This block demonstrates how to use the functions above for either
# a single image or a video file.
if __name__ == '__main__':
    # --- Argument Parser Setup ---
    parser = argparse.ArgumentParser(description="Preprocess facial video/image for HR estimation: detect landmarks, extract ROIs, combine features (Wavelet Diff + Block Avg), and create final STI.")
    parser.add_argument('-i', '--input', required=True, help="Path to the input video or image file.")
    parser.add_argument('--visualize', action='store_true', help="Enable visualization of processed frames and STIs.") # Add optional visualize flag
    args = parser.parse_args()

    # --- Configuration ---
    # Use the input from command line
    INPUT_SOURCE = args.input
    # Target FPS for video processing and timestamp calculation
    TARGET_FPS = 30.0
    # Supported file extensions
    VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv']
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    # Control visualization based on command line flag
    VISUALIZE = args.visualize # Use the parsed argument
    # --- End Configuration ---

    # ROI Standardization size (Width, Height) - Defined globally or here
    standard_roi_size = ROI_SIZE # Use the global constant
    final_sti_size = (224, 224) # Target size for CNN input

    cap = None # Initialize video capture object outside loop
    # Store combined feature vectors
    combined_forehead_features = []
    combined_left_cheek_features = []
    combined_right_cheek_features = []

    # Variables to store previous frame's state for feature calculation
    prev_forehead_roi = None
    prev_left_cheek_roi = None
    prev_right_cheek_roi = None
    prev_block_features_forehead = None
    prev_block_features_left_cheek = None
    prev_block_features_right_cheek = None

    try:
        # --- Input Type Handling ---
        file_extension = os.path.splitext(INPUT_SOURCE)[1].lower()
        is_video = file_extension in VIDEO_EXTENSIONS
        is_image = file_extension in IMAGE_EXTENSIONS

        if is_video:
            # --- Video Processing Logic ---
            print(f"Processing video file: {INPUT_SOURCE}")
            # Prepare path for converted video
            base_name = os.path.splitext(os.path.basename(INPUT_SOURCE))[0]
            output_dir = os.path.dirname(INPUT_SOURCE) or '.'
            converted_video_path = os.path.join(output_dir, f"{base_name}_30fps.mp4")

            # Convert video to 30 FPS if necessary
            # Note: We might want to skip conversion if original FPS is already 30?
            # For simplicity now, we always convert to ensure exact 30 FPS timestamps.
            if not convert_video_to_30fps(INPUT_SOURCE, converted_video_path):
                 raise RuntimeError("Video conversion failed.") # Stop if conversion fails

            print(f"Processing converted 30 FPS video: {converted_video_path}")
            cap = cv2.VideoCapture(converted_video_path)
            if not cap.isOpened():
                 raise IOError(f"Cannot open video file {converted_video_path}")

            frame_index = 0
            print("Processing video frames... Press 'q' to quit.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of video.")
                    break

                # Calculate timestamp based on TARGET_FPS
                timestamp_ms = int(frame_index * (1000 / TARGET_FPS))

                # --- Process frame directly here --- 
                display_frame = frame.copy() # Work on a copy for drawing
                current_forehead_roi = None # Reset current ROIs for this frame
                current_left_cheek = None
                current_right_cheek = None
                block_features_forehead = None # Reset current block features
                block_features_left_cheek = None
                block_features_right_cheek = None
                forehead_fv = None # Reset current wavelet features
                left_cheek_fv = None
                right_cheek_fv = None

                detection_result = detect_face_and_landmarks_mediapipe(frame, timestamp_ms)

                if detection_result:
                    landmarks_coords = extract_landmarks_coordinates_mediapipe(detection_result, frame.shape)
                    if landmarks_coords is not None:
                        # --- Define, Extract & Standardize ROIs --- 
                        roi_forehead_bbox = define_roi_forehead_mediapipe(landmarks_coords)
                        roi_left_cheek_bbox, roi_right_cheek_bbox = define_roi_cheeks_mediapipe(landmarks_coords)

                        if roi_forehead_bbox:
                            extracted_forehead = extract_roi_frame(frame, roi_forehead_bbox)
                            if extracted_forehead is not None and extracted_forehead.size > 0:
                                current_forehead_roi = cv2.resize(extracted_forehead, standard_roi_size, interpolation=cv2.INTER_AREA)
                                x, y, w, h = roi_forehead_bbox
                                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
                            else:
                                current_forehead_roi = None # Ensure it's None if extraction failed
                        else:
                             current_forehead_roi = None # Ensure it's None if bbox invalid
                        
                        if roi_left_cheek_bbox:
                            extracted_left_cheek = extract_roi_frame(frame, roi_left_cheek_bbox)
                            if extracted_left_cheek is not None and extracted_left_cheek.size > 0: 
                                current_left_cheek = cv2.resize(extracted_left_cheek, standard_roi_size, interpolation=cv2.INTER_AREA)
                                x, y, w, h = roi_left_cheek_bbox
                                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                            else:
                                current_left_cheek = None
                        else:
                             current_left_cheek = None
                        
                        if roi_right_cheek_bbox:
                            extracted_right_cheek = extract_roi_frame(frame, roi_right_cheek_bbox)
                            if extracted_right_cheek is not None and extracted_right_cheek.size > 0: 
                                current_right_cheek = cv2.resize(extracted_right_cheek, standard_roi_size, interpolation=cv2.INTER_AREA)
                                x, y, w, h = roi_right_cheek_bbox
                                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
                            else:
                                current_right_cheek = None
                        else:
                            current_right_cheek = None

                        # --- PATH A: Calculate Wavelet Features (from t-1 to t) --- 
                        if prev_forehead_roi is not None and current_forehead_roi is not None:
                            diff_forehead = current_forehead_roi.astype(np.float32) - prev_forehead_roi.astype(np.float32)
                            forehead_fv = extract_wavelet_features_from_diff(diff_forehead)
                        
                        if prev_left_cheek_roi is not None and current_left_cheek is not None:
                           diff_left_cheek = current_left_cheek.astype(np.float32) - prev_left_cheek_roi.astype(np.float32)
                           left_cheek_fv = extract_wavelet_features_from_diff(diff_left_cheek)
                           
                        if prev_right_cheek_roi is not None and current_right_cheek is not None:
                           diff_right_cheek = current_right_cheek.astype(np.float32) - prev_right_cheek_roi.astype(np.float32)
                           right_cheek_fv = extract_wavelet_features_from_diff(diff_right_cheek)

                        # --- PATH B: Calculate Block Features (for current frame t) --- 
                        if current_forehead_roi is not None:
                            msr_forehead = apply_msr(current_forehead_roi)
                            if msr_forehead is not None:
                                fused_forehead = fuse_rgb_msr(current_forehead_roi, msr_forehead)
                                if fused_forehead is not None:
                                    block_features_forehead = extract_block_features(fused_forehead)
                        
                        if current_left_cheek is not None:
                            msr_left_cheek = apply_msr(current_left_cheek)
                            if msr_left_cheek is not None:
                                fused_left_cheek = fuse_rgb_msr(current_left_cheek, msr_left_cheek)
                                if fused_left_cheek is not None:
                                    block_features_left_cheek = extract_block_features(fused_left_cheek)
                        
                        if current_right_cheek is not None:
                            msr_right_cheek = apply_msr(current_right_cheek)
                            if msr_right_cheek is not None:
                                fused_right_cheek = fuse_rgb_msr(current_right_cheek, msr_right_cheek)
                                if fused_right_cheek is not None:
                                    block_features_right_cheek = extract_block_features(fused_right_cheek)

                        # --- FEATURE COMBINATION (Concatenate Path A[t-1:t] & Path B[t-1]) --- 
                        if forehead_fv is not None and prev_block_features_forehead is not None:
                            combined_fv = np.concatenate((forehead_fv, prev_block_features_forehead))
                            combined_forehead_features.append(combined_fv)
                        
                        if left_cheek_fv is not None and prev_block_features_left_cheek is not None:
                            combined_fv = np.concatenate((left_cheek_fv, prev_block_features_left_cheek))
                            combined_left_cheek_features.append(combined_fv)
                        
                        if right_cheek_fv is not None and prev_block_features_right_cheek is not None:
                            combined_fv = np.concatenate((right_cheek_fv, prev_block_features_right_cheek))
                            combined_right_cheek_features.append(combined_fv)

                        # --- Draw Landmarks --- 
                        proto_landmarks = landmark_pb2.NormalizedLandmarkList()
                        for landmark_dataclass in detection_result.face_landmarks[0]:
                            landmark_proto = landmark_pb2.NormalizedLandmark(
                                x=landmark_dataclass.x, y=landmark_dataclass.y, z=landmark_dataclass.z
                            )
                            proto_landmarks.landmark.append(landmark_proto)
                        # Use mp_solutions for drawing
                        mp_solutions.drawing_utils.draw_landmarks(
                            image=display_frame,
                            landmark_list=proto_landmarks,
                            connections=mp_solutions.face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_solutions.drawing_styles.get_default_face_mesh_tesselation_style()
                        )
                else: 
                    # If no face detected, reset previous ROIs and block features to avoid combination across gaps
                    current_forehead_roi = None # Ensure current are None if no detection
                    current_left_cheek = None
                    current_right_cheek = None
                    block_features_forehead = None 
                    block_features_left_cheek = None
                    block_features_right_cheek = None
                
                # --- Update Previous State for next iteration --- 
                prev_forehead_roi = current_forehead_roi
                prev_left_cheek_roi = current_left_cheek
                prev_right_cheek_roi = current_right_cheek
                prev_block_features_forehead = block_features_forehead
                prev_block_features_left_cheek = block_features_left_cheek
                prev_block_features_right_cheek = block_features_right_cheek
                # --- End frame processing --- 
                
                # Display the processed frame if visualize is True
                if VISUALIZE:
                    cv2.imshow("Processed Video (MediaPipe) - Press 'q' to quit", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): # Check for 'q' key press
                        print("Exiting video processing loop.")
                        break
                else: # If not visualizing, add a way to gracefully exit (e.g., check terminal input?)
                      # For now, just continue without waitKey
                      pass 

                frame_index += 1
                
            # --- After video processing loop --- 
            print(f"\nFinished processing video.")
            print(f"Collected {len(combined_forehead_features)} combined forehead feature vectors.")
            print(f"Collected {len(combined_left_cheek_features)} combined left cheek feature vectors.")
            print(f"Collected {len(combined_right_cheek_features)} combined right cheek feature vectors.")
            
            # --- Construct Final Combined STIs --- 
            forehead_sti = None
            left_cheek_sti = None
            right_cheek_sti = None
            
            if combined_forehead_features:
                 print(f"Shape of first combined forehead feature vector: {combined_forehead_features[0].shape}")
                 forehead_sti = construct_and_resize_sti(combined_forehead_features, target_size=final_sti_size)
                 if forehead_sti is not None:
                     print(f"Constructed Combined Forehead STI with shape: {forehead_sti.shape}")
                     if VISUALIZE:
                         # Cast normalized STI to uint8 for display
                         cv2.imshow("Combined Forehead STI", cv2.normalize(forehead_sti, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
                 else:
                     print("Failed to construct Combined Forehead STI.")
            
            if combined_left_cheek_features:
                 print(f"Shape of first combined left cheek feature vector: {combined_left_cheek_features[0].shape}")
                 left_cheek_sti = construct_and_resize_sti(combined_left_cheek_features, target_size=final_sti_size)
                 if left_cheek_sti is not None:
                     print(f"Constructed Combined Left Cheek STI with shape: {left_cheek_sti.shape}")
                     if VISUALIZE:
                         # Cast normalized STI to uint8 for display
                         cv2.imshow("Combined Left Cheek STI", cv2.normalize(left_cheek_sti, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
                 else:
                     print("Failed to construct Combined Left Cheek STI.")
                     
            if combined_right_cheek_features:
                 print(f"Shape of first combined right cheek feature vector: {combined_right_cheek_features[0].shape}")
                 right_cheek_sti = construct_and_resize_sti(combined_right_cheek_features, target_size=final_sti_size)
                 if right_cheek_sti is not None:
                     print(f"Constructed Combined Right Cheek STI with shape: {right_cheek_sti.shape}")
                     if VISUALIZE:
                         # Cast normalized STI to uint8 for display
                         cv2.imshow("Combined Right Cheek STI", cv2.normalize(right_cheek_sti, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
                 else:
                     print("Failed to construct Combined Right Cheek STI.")

            # The variables forehead_sti, left_cheek_sti, right_cheek_sti now hold the 
            # final 224x224 combined STIs ready for input to the CNN.

            # Add a final wait key if any STIs were displayed and visualization is enabled
            if VISUALIZE and (forehead_sti is not None or left_cheek_sti is not None or right_cheek_sti is not None):
               print("Press any key in an OpenCV window to close STI windows and continue...")
               cv2.waitKey(0)

            # Optional: Clean up the converted video file
            try:
                os.remove(converted_video_path)
                print(f"Removed temporary file: {converted_video_path}")
            except OSError as e:
                print(f"Error removing temporary file {converted_video_path}: {e}")

            # --- Stack STIs for 3-Channel Input --- 
            stacked_input_sti = None # Initialize
            if forehead_sti is not None and left_cheek_sti is not None and right_cheek_sti is not None:
                # Stack along a new first dimension (channel dimension)
                stacked_input_sti = np.stack([forehead_sti, left_cheek_sti, right_cheek_sti], axis=0)
                print(f"\nSuccessfully stacked STIs. Shape for model input: {stacked_input_sti.shape}")
                # This stacked_input_sti (NumPy array, shape (3, 224, 224)) is ready 
                # to be converted to a PyTorch tensor for the Enhanced_HR_CNN.
            else:
                print("\nWarning: One or more ROIs failed processing. Cannot create stacked 3-channel STI.")

        elif is_image:
            # --- Single Image Processing Logic --- 
            # (Feature extraction/combination doesn't apply here)
            print(f"Processing image file: {INPUT_SOURCE}")
            frame = cv2.imread(INPUT_SOURCE)
            if frame is None:
                raise FileNotFoundError(f"Image '{INPUT_SOURCE}' not found or could not be read.")

            # ROI Standardization size
            standard_roi_size = ROI_SIZE

            # Timestamp is 0 for single image detection using VIDEO mode landmarker
            timestamp_ms = 0
            display_frame = frame.copy()
            
            detection_result = detect_face_and_landmarks_mediapipe(frame, timestamp_ms)

            if detection_result:
                landmarks_coords = extract_landmarks_coordinates_mediapipe(detection_result, frame.shape)
                if landmarks_coords is not None:
                    # Define and draw ROIs
                    roi_forehead_bbox = define_roi_forehead_mediapipe(landmarks_coords)
                    if roi_forehead_bbox:
                        extracted_forehead = extract_roi_frame(frame, roi_forehead_bbox)
                        if extracted_forehead is not None and extracted_forehead.size > 0: 
                            # Resize the single image ROI as well for consistency if needed later
                            standardized_forehead_roi = cv2.resize(extracted_forehead, standard_roi_size, interpolation=cv2.INTER_AREA)
                            print(f"Extracted Standardized Forehead ROI: Shape={standardized_forehead_roi.shape}, dtype={standardized_forehead_roi.dtype}")
                            x, y, w, h = roi_forehead_bbox
                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 1) 
                        else:
                             print("Warning: Forehead ROI extraction failed for image.")

                    # Extract and Draw cheeks
                    roi_left_cheek_bbox, roi_right_cheek_bbox = define_roi_cheeks_mediapipe(landmarks_coords)
                    if roi_left_cheek_bbox:
                        extracted_left_cheek = extract_roi_frame(frame, roi_left_cheek_bbox)
                        if extracted_left_cheek is not None and extracted_left_cheek.size > 0:
                            standardized_left_cheek = cv2.resize(extracted_left_cheek, standard_roi_size, interpolation=cv2.INTER_AREA)
                            print(f"Extracted Standardized Left Cheek ROI: Shape={standardized_left_cheek.shape}, dtype={standardized_left_cheek.dtype}")
                            # Draw on display frame
                            x, y, w, h = roi_left_cheek_bbox
                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 1) 
                        else:
                            print("Warning: Left Cheek ROI extraction failed for image.")
                    if roi_right_cheek_bbox:
                        extracted_right_cheek = extract_roi_frame(frame, roi_right_cheek_bbox)
                        if extracted_right_cheek is not None and extracted_right_cheek.size > 0: 
                            standardized_right_cheek = cv2.resize(extracted_right_cheek, standard_roi_size, interpolation=cv2.INTER_AREA)
                            print(f"Extracted Standardized Right Cheek ROI: Shape={standardized_right_cheek.shape}, dtype={standardized_right_cheek.dtype}")
                            # Draw on display frame
                            x, y, w, h = roi_right_cheek_bbox
                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 1) 
                        else:
                            print("Warning: Right Cheek ROI extraction failed for image.")

                    # Draw landmarks
                    proto_landmarks = landmark_pb2.NormalizedLandmarkList()
                    for landmark_dataclass in detection_result.face_landmarks[0]:
                        landmark_proto = landmark_pb2.NormalizedLandmark(
                            x=landmark_dataclass.x, y=landmark_dataclass.y, z=landmark_dataclass.z
                        )
                        proto_landmarks.landmark.append(landmark_proto)
                    # Use mp_solutions for drawing
                    mp_solutions.drawing_utils.draw_landmarks(
                        image=display_frame,
                        landmark_list=proto_landmarks, 
                        connections=mp_solutions.face_mesh.FACEMESH_TESSELATION, 
                        landmark_drawing_spec=None, 
                        connection_drawing_spec=mp_solutions.drawing_styles.get_default_face_mesh_tesselation_style())
                else:
                    print("Could not extract landmark coordinates.")
            else:
                print("No face detected in the sample image using MediaPipe.")
            
            # Display the processed image if visualize is True
            if VISUALIZE:
                cv2.imshow("Processed Image (MediaPipe)", display_frame)
                print("Displaying image. Press any key to close.")
                cv2.waitKey(0) # Wait indefinitely until a key is pressed
            
             # Consolidated check - print only if no ROIs extracted at all (still relevant for images)
            if not roi_forehead_bbox and not roi_left_cheek_bbox and not roi_right_cheek_bbox: # Use new names
                  print("No ROIs extracted from image.")

        else:
            # Handle unsupported file types
            print(f"Error: Unsupported file type '{file_extension}' for input source: {INPUT_SOURCE}")
            print(f"Supported video: {VIDEO_EXTENSIONS}")
            print(f"Supported image: {IMAGE_EXTENSIONS}")

    except Exception as e:
        # Catch any other unexpected errors during execution
        import traceback
        print(f"\n--- An error occurred in the main execution block ---")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("Traceback:")
        traceback.print_exc()
        print("------------------------------------------------------\n")

    finally:
        # --- Resource Cleanup ---
        # Ensure resources are released regardless of success or failure
        if cap is not None:
            cap.release() # Release video capture object
        cv2.destroyAllWindows() # Close all OpenCV windows
        if face_landmarker:
            face_landmarker.close() # Close the MediaPipe landmarker
            print("MediaPipe FaceLandmarker closed.")
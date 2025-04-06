import numpy as np
import cv2 # Keep cv2 for potential future use (e.g., resizing, MSR)
import pywt # Import PyWavelets

# --- Wavelet Transform ---

def apply_wavelet_transform(diff_frame, wavelet='db4', levels=3):
    """
    Applies a multi-level 2D Discrete Wavelet Transform (DWT) to a difference frame.

    Args:
        diff_frame (np.ndarray): A single difference frame (H, W, C) or (H, W).
                                 Assumes float dtype from temporal differencing.
        wavelet (str): The name of the wavelet to use (e.g., 'db4', 'haar').
        levels (int): The number of decomposition levels.

    Returns:
        dict: A dictionary containing the HL and LH subbands for each level,
              e.g., {'HL1': ..., 'LH1': ..., 'HL2': ..., 'LH2': ..., 'HL3': ..., 'LH3': ...}.
              Returns None if the input is invalid or decomposition fails.
    """
    if not isinstance(diff_frame, np.ndarray) or diff_frame.ndim < 2:
        print("Error: Invalid input frame for wavelet transform.")
        return None

    # Process each color channel independently
    subbands = {}
    for c in range(diff_frame.shape[2]):
        channel_subbands = _process_single_channel(diff_frame[:,:,c], wavelet, levels)
        subbands[f'C{c}'] = channel_subbands
    return subbands

def _process_single_channel(channel_frame, wavelet, levels):
    """Processes a single 2D channel using wavelet decomposition."""
    # Input validation: Ensure the input is a 2D array
    if not isinstance(channel_frame, np.ndarray) or channel_frame.ndim != 2:
        print(f"Error: Invalid input frame for _process_single_channel. Expected 2D array, got shape {channel_frame.shape}")
        return None

    try:
        # Perform the multi-level 2D wavelet decomposition directly on the input channel
        coeffs = pywt.wavedec2(channel_frame, wavelet, level=levels)
        # The structure is [cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]
        # cH = Horizontal detail (HL band in some literature)
        # cV = Vertical detail (LH band in some literature)

        if len(coeffs) != levels + 1:
             print(f"Error: Wavelet decomposition did not return the expected number of levels. Got {len(coeffs)-1}, expected {levels}")
             return None

        subbands = {}
        # Extract HL (cH) and LH (cV) subbands from detail coefficients
        for i in range(1, levels + 1):
            # Detail coefficients are indexed from the highest level (n) down to 1
            # The tuple is (cH_level, cV_level, cD_level)
            level_coeffs_tuple = coeffs[i]
            if len(level_coeffs_tuple) != 3:
                print(f"Error: Unexpected coefficient structure at level {levels-i+1}")
                return None
            subbands[f'HL{levels - i + 1}'] = level_coeffs_tuple[0] # cH
            subbands[f'LH{levels - i + 1}'] = level_coeffs_tuple[1] # cV

        return subbands

    except Exception as e:
        # Use channel_frame shape in the error message
        print(f"Wavelet error: {e} | Shape: {channel_frame.shape} | "
              f"dtype: {channel_frame.dtype} | wavelet: {wavelet} | levels: {levels}")
        raise  # Preserve stack trace

def calculate_projections(subbands):
    """Calculate horizontal/vertical projections for wavelet subbands"""
    projections = {}
    for level in ['HL1', 'LH1', 'HL2', 'LH2', 'HL3', 'LH3']:
        if level.startswith('HL'):
            # Horizontal projection for HL subbands
            projections[level] = np.sum(subbands[level], axis=0) 
        elif level.startswith('LH'):
            # Vertical projection for LH subbands
            projections[level] = np.sum(subbands[level], axis=1)
    return projections

def build_feature_vector(projections):
    """Create 6-element feature vector from projections"""
    return np.concatenate([
        projections['HL1'], projections['LH1'],
        projections['HL2'], projections['LH2'],
        projections['HL3'], projections['LH3']
    ])

def extract_wavelet_features_from_diff(diff_frame, wavelet='db4', levels=3):
    """
    Calculates the combined wavelet feature vector from a single difference frame.

    Processes each color channel independently and concatenates the resulting
    feature vectors.

    Args:
        diff_frame (np.ndarray): A single difference frame (H, W, C), float dtype.
        wavelet (str): The name of the wavelet to use.
        levels (int): The number of decomposition levels.

    Returns:
        np.ndarray: A single concatenated feature vector containing features
                    from all color channels, or None if processing fails.
    """
    if diff_frame is None or diff_frame.ndim != 3:
        print("Error: Invalid difference frame for feature extraction.")
        return None

    # Apply wavelet transform to get subbands for each channel
    multi_channel_subbands = apply_wavelet_transform(diff_frame, wavelet, levels)
    if multi_channel_subbands is None:
        print("Error: Wavelet transform failed.")
        return None

    all_channel_features = []
    # Assuming channels are ordered (e.g., BGR or RGB)
    num_channels = diff_frame.shape[2]
    for c in range(num_channels):
        channel_key = f'C{c}'
        if channel_key not in multi_channel_subbands or multi_channel_subbands[channel_key] is None:
            print(f"Warning: Subband extraction failed or missing for channel {c}.")
            # Decide how to handle missing channel? Skip or return None?
            # Returning None for now to indicate failure.
            return None 

        channel_subbands = multi_channel_subbands[channel_key]
        
        # Calculate projections for this channel's subbands
        projections = calculate_projections(channel_subbands)
        if projections is None: # Should not happen if calculate_projections is robust
             print(f"Error: Projection calculation failed for channel {c}.")
             return None

        # Build feature vector for this channel
        channel_feature_vector = build_feature_vector(projections)
        if channel_feature_vector is None: # Should not happen if build_feature_vector is robust
            print(f"Error: Feature vector construction failed for channel {c}.")
            return None
            
        all_channel_features.append(channel_feature_vector)

    # Concatenate features from all channels
    if not all_channel_features:
        print("Error: No channel features were extracted.")
        return None
        
    combined_feature_vector = np.concatenate(all_channel_features)
    return combined_feature_vector

def construct_and_resize_sti(feature_vector_sequence, target_size=(224, 224)):
    """
    Constructs a Spatio-Temporal Image (STI) from a sequence of feature vectors
    and resizes it to the target dimensions.

    Args:
        feature_vector_sequence (list): A list of 1D numpy arrays (feature vectors).
                                        Assumes all vectors have the same length.
        target_size (tuple): The target size (Width, Height) for the final STI.

    Returns:
        np.ndarray: The resized STI as a 2D numpy array (Height, Width),
                    or None if the input is invalid or empty.
    """
    if not feature_vector_sequence:
        print("Warning: Feature vector sequence is empty. Cannot construct STI.")
        return None
    
    # Check if all vectors have the same length (optional but good practice)
    first_vec_len = len(feature_vector_sequence[0])
    if not all(len(fv) == first_vec_len for fv in feature_vector_sequence):
        print("Error: Feature vectors in the sequence have inconsistent lengths.")
        return None
        
    try:
        # Stack the 1D feature vectors vertically to form the initial STI
        # Resulting shape: (num_vectors, vector_length)
        sti = np.vstack(feature_vector_sequence)

        # Ensure STI is float32 for resizing interpolation quality
        sti_float = sti.astype(np.float32)

        # Resize the STI to the target size (Width, Height)
        # cv2.resize expects (Width, Height)
        resized_sti = cv2.resize(sti_float, target_size, interpolation=cv2.INTER_LINEAR) # Or INTER_AREA, INTER_CUBIC
        
        return resized_sti
        
    except Exception as e:
        print(f"Error during STI construction or resizing: {e}")
        return None

# --- Illumination-Robust Features (Path B) ---

def apply_msr(image, scales=[15, 80, 200], apply_log=True, epsilon=1e-6):
    """
    Applies Multi-Scale Retinex (MSR) to an image.

    Args:
        image (np.ndarray): Input image (H, W, C), expected to be float type (e.g., 0-255).
        scales (list): List of sigma values for Gaussian blur kernels.
        apply_log (bool): Whether to apply log transformation (standard MSR).
        epsilon (float): Small value to add before log to avoid log(0).

    Returns:
        np.ndarray: The MSR processed image (float type), same shape as input.
                    Returns None if input is invalid.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        print("Error: Invalid input image for MSR. Expected (H, W, C).")
        return None

    # Ensure float type for calculations
    image_float = image.astype(np.float32)

    msr_output = np.zeros_like(image_float)
    num_scales = len(scales)
    weight = 1.0 / num_scales # Equal weights for each scale

    # Process each channel independently
    for c in range(image.shape[2]):
        channel = image_float[:, :, c]

        # Apply log if required
        if apply_log:
            log_channel = np.log10(channel + epsilon)
        else:
            log_channel = channel # Use original values if log is disabled

        for scale in scales:
            # Gaussian blur
            # Use (0,0) kernel size so it's derived from sigma
            blurred = cv2.GaussianBlur(channel, (0, 0), sigmaX=scale, sigmaY=scale)

            # Log of blurred image
            if apply_log:
                log_blurred = np.log10(blurred + epsilon)
            else:
                log_blurred = blurred # Use original values if log is disabled

            # Difference
            diff = log_channel - log_blurred

            # Accumulate weighted difference
            msr_output[:, :, c] += weight * diff

    # Optional: Scale result back to a common range (e.g., 0-255)
    # This depends on how the subsequent fusion expects the input.
    # For now, returning the float result.
    # Example scaling:
    # msr_output = cv2.normalize(msr_output, None, 0, 255, cv2.NORM_MINMAX)

    return msr_output

def fuse_rgb_msr(rgb_roi, msr_roi, base_filter_ksize=5, epsilon=1e-12):
    """
    Fuses an RGB ROI with its MSR counterpart based on the plan.

    Args:
        rgb_roi (np.ndarray): The original standardized ROI (H, W, C), float32.
        msr_roi (np.ndarray): The MSR processed ROI (H, W, C), float32.
        base_filter_ksize (int): Kernel size for the mean filter for base layers.
        epsilon (float): Small value to avoid division by zero in weight calculation.

    Returns:
        np.ndarray: The fused image (gamma) as float32 (H, W, C),
                    or None if inputs are invalid.
    """
    if rgb_roi is None or msr_roi is None or rgb_roi.shape != msr_roi.shape or rgb_roi.ndim != 3:
        print("Error: Invalid inputs for RGB-MSR fusion.")
        return None
    if rgb_roi.dtype != np.float32:
        rgb_roi = rgb_roi.astype(np.float32)
    if msr_roi.dtype != np.float32:
        msr_roi = msr_roi.astype(np.float32)

    # 1. Extract Base Layers (using average/box filter)
    # Using cv2.blur which is a normalized box filter
    base1 = cv2.blur(rgb_roi, (base_filter_ksize, base_filter_ksize))
    base2 = cv2.blur(msr_roi, (base_filter_ksize, base_filter_ksize))

    # 2. Fuse Base Layers
    fused_base = (base1 + base2) / 2.0

    # 3. Extract Detail Layers
    detail1 = rgb_roi - base1
    detail2 = msr_roi - base2

    # 4. Calculate Saliency Maps (using absolute detail layers as proxy)
    saliency1 = np.abs(detail1)
    saliency2 = np.abs(detail2)
    # Sum saliency across channels for weighting, or weight per channel?
    # Let's try per-channel weighting first for simplicity.
    # saliency1_sum = np.sum(saliency1, axis=2, keepdims=True)
    # saliency2_sum = np.sum(saliency2, axis=2, keepdims=True)
    # total_saliency = saliency1_sum + saliency2_sum + epsilon

    # Per-channel weighting:
    total_saliency = saliency1 + saliency2 + epsilon

    # 5. Calculate Weight Maps
    weight1 = saliency1 / total_saliency
    weight2 = saliency2 / total_saliency
    # Ensure weights sum to 1 (or close to it)
    # weight1 = saliency1_sum / total_saliency
    # weight2 = saliency2_sum / total_saliency

    # 6. Fuse Detail Layers
    fused_detail = weight1 * detail1 + weight2 * detail2

    # 7. Combine Layers
    gamma = fused_base + fused_detail

    # Optional: Clip result to a valid range if needed (e.g., 0-255)
    # gamma = np.clip(gamma, 0, 255)

    return gamma

def extract_block_features(fused_roi, num_blocks_sqrt=4):
    """
    Divides the fused ROI into k=num_blocks_sqrt^2 blocks and calculates
    the average RGB value for each block.

    Args:
        fused_roi (np.ndarray): The fused ROI image (gamma) (H, W, C), float32.
        num_blocks_sqrt (int): The square root of the number of blocks (k).
                               E.g., 4 means the image will be divided into 4x4=16 blocks.

    Returns:
        np.ndarray: A 1D array containing the average RGB values for all blocks,
                    flattened. Shape: (k * C,). Returns None if input is invalid.
    """
    if fused_roi is None or fused_roi.ndim != 3:
        print("Error: Invalid input fused_roi for block feature extraction.")
        return None

    H, W, C = fused_roi.shape
    k = num_blocks_sqrt * num_blocks_sqrt
    block_features = np.zeros((k, C), dtype=np.float32)

    # Calculate block dimensions
    block_h = H // num_blocks_sqrt
    block_w = W // num_blocks_sqrt

    if block_h == 0 or block_w == 0:
        print(f"Error: ROI size ({H}x{W}) too small for {k} blocks.")
        return None

    block_idx = 0
    for r in range(num_blocks_sqrt):
        for c_block in range(num_blocks_sqrt):
            # Define block boundaries
            y1 = r * block_h
            y2 = y1 + block_h
            x1 = c_block * block_w
            x2 = x1 + block_w

            # Extract block
            block = fused_roi[y1:y2, x1:x2, :]

            # Calculate average RGB (or other channel) values for the block
            # Handle potential empty blocks if ROI size not perfectly divisible
            if block.size > 0:
                avg_vals = np.mean(block, axis=(0, 1))
                block_features[block_idx, :] = avg_vals
            else:
                 # Handle empty block case if needed (e.g., set to zero or previous)
                 block_features[block_idx, :] = 0.0

            block_idx += 1

    # Flatten the (k, C) array to (k * C,)
    return block_features.flatten()

# --- Spatio-Temporal Image (STI) Construction (Path A) ---
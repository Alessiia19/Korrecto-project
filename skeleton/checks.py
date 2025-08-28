import numpy as np
from skeleton.workout import calculate_angle, get_landmark_coords, is_lateral_orientation

def get_side(landmarks, mp_pose, image_shape):
    """
    Determine which side of the frame the body is oriented toward,
    using the nose position relative to the ankles:
    - If the nose is to the right of at least one ankle → the body faces the left side of the camera.
    - If the nose is to the left of at least one ankle → the body faces the right side of the camera.

    Args:
        landmarks: list of MediaPipe body landmarks.
        mp_pose: MediaPipe pose module to get landmark indices.
        image_shape: frame dimensions (height, width).

    Returns:
        side (str): "left" if facing left, "right" if facing right.
    """
    nose = get_landmark_coords(landmarks, mp_pose.NOSE.value, image_shape)
    left_ankle = get_landmark_coords(landmarks, mp_pose.LEFT_ANKLE.value, image_shape)
    right_ankle = get_landmark_coords(landmarks, mp_pose.RIGHT_ANKLE.value, image_shape)
    if nose[0] > left_ankle[0] or nose[0] > right_ankle[0]:
        side = "left"
    elif nose[0] < left_ankle[0] or nose[0] < right_ankle[0]:
        side = "right"
    return side

def estimate_orientation(landmarks, mp_pose, image_shape):
    """
    Estimate whether the person is seen laterally or frontally by the camera:
    - Uses Z coordinates (depth) of left and right shoulders.
    - If the left shoulder is closer to the camera (smaller Z) → person faces the right side of the screen → returns "right".
    - If the right shoulder is closer → person faces the left side of the screen → returns "left".

    Args:
        landmarks: list of MediaPipe body landmarks.
        mp_pose: MediaPipe pose module to get landmark indices.
        image_shape: frame dimensions (height, width).

    Returns:
        str: "right" if facing right side, "left" if facing left side.
    """
    left_shoulder_z = landmarks[mp_pose.LEFT_SHOULDER.value].z
    right_shoulder_z = landmarks[mp_pose.RIGHT_SHOULDER.value].z

    if left_shoulder_z < right_shoulder_z:
        return "right"   
    else:
        return "left" 
    

def check_knee_over_foot(landmarks, mp_pose, image_shape, z_threshold=0.02):
    """
    Check if the knee goes beyond the toes when the user is seen laterally.

    Args:
        landmarks: list of normalized landmarks.
        mp_pose: MediaPipe pose module.
        image_shape: frame dimensions (h, w).
        z_threshold (float): depth threshold to evaluate knee-to-foot projection.

    Returns:
        bool: True if the knee goes beyond the foot.
    """
    if estimate_orientation(landmarks, mp_pose, image_shape) == "right":
        left_knee = landmarks[mp_pose.LEFT_KNEE.value]
        left_foot = landmarks[mp_pose.LEFT_FOOT_INDEX.value]
        dz = left_knee.z - left_foot.z
        return dz < -z_threshold
    elif estimate_orientation(landmarks, mp_pose, image_shape) == "left":
        right_knee = landmarks[mp_pose.RIGHT_KNEE.value]
        right_foot = landmarks[mp_pose.RIGHT_FOOT_INDEX.value]
        dz = right_knee.z - right_foot.z
        return dz < -z_threshold
    elif not is_lateral_orientation(landmarks, mp_pose):
        left_knee = landmarks[mp_pose.LEFT_KNEE.value]
        left_foot = landmarks[mp_pose.LEFT_FOOT_INDEX.value]
        dz_left = left_knee.z - left_foot.z

        right_knee = landmarks[mp_pose.RIGHT_KNEE.value]
        right_foot = landmarks[mp_pose.RIGHT_FOOT_INDEX.value]
        dz_right = right_knee.z - right_foot.z
        return (dz_left < -z_threshold) and (dz_right < -z_threshold)



def check_knee_angle(landmarks, mp_pose, image_shape, min_angle):
    """
    Check if the knee angle is too closed during squat descent.

    Returns:
        bool: True if the knee is too bent during squat descent.
    """
    right_hip = get_landmark_coords(landmarks, mp_pose.RIGHT_HIP.value, image_shape)
    right_knee = get_landmark_coords(landmarks, mp_pose.RIGHT_KNEE.value, image_shape)
    right_ankle = get_landmark_coords(landmarks, mp_pose.RIGHT_ANKLE.value, image_shape)

    left_hip = get_landmark_coords(landmarks, mp_pose.LEFT_HIP.value, image_shape)
    left_knee = get_landmark_coords(landmarks, mp_pose.LEFT_KNEE.value, image_shape)
    left_ankle = get_landmark_coords(landmarks, mp_pose.LEFT_ANKLE.value, image_shape)

    if estimate_orientation(landmarks, mp_pose, image_shape) == "right":
        left_angle = calculate_angle(left_hip, left_knee, left_ankle)
        return left_angle < min_angle
    elif estimate_orientation(landmarks, mp_pose, image_shape) == "left":
        right_angle = calculate_angle(right_hip, right_knee, right_ankle)
        return right_angle < min_angle


def check_back_inclination(landmarks, mp_pose, image_shape, max_forward_angle=30):
    """
    Check if the back is too inclined forward (only if viewed laterally).

    Args:
        landmarks: list of MediaPipe normalized landmarks.
        mp_pose: MediaPipe PoseLandmark module.
        image_shape: (h, w).
        max_forward_angle (float): maximum acceptable forward inclination in degrees.

    Returns:
        bool: True if the back is too inclined.
    """
    
    if estimate_orientation(landmarks, mp_pose, image_shape) == "right":
        shoulder = get_landmark_coords(landmarks, mp_pose.LEFT_SHOULDER.value, image_shape)
        hip = get_landmark_coords(landmarks, mp_pose.LEFT_HIP.value, image_shape)
    elif estimate_orientation(landmarks, mp_pose, image_shape) == "left":
        shoulder = get_landmark_coords(landmarks, mp_pose.RIGHT_SHOULDER.value, image_shape)
        hip = get_landmark_coords(landmarks, mp_pose.RIGHT_HIP.value, image_shape)
 
    vector = np.array(shoulder[:2]) - np.array(hip[:2])
    vertical = np.array([0, -1])  

    cosine = np.dot(vector, vertical) / (np.linalg.norm(vector) * np.linalg.norm(vertical))
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    return angle > max_forward_angle

def check_ankle_distance(landmarks, mp_pose, image_shape, ankle_threshold=1):
    """
    Check if ankles are too close compared to shoulder width.

    Args:
        landmarks: list of MediaPipe normalized landmarks.
        mp_pose: MediaPipe PoseLandmark module.
        image_shape: (h, w).
        ankle_threshold (float): scaling factor for ankle distance vs shoulder width.

    Returns:
        bool: True if ankles are too close.
    """
    left_shoulder = get_landmark_coords(landmarks, mp_pose.LEFT_SHOULDER.value, image_shape)
    right_shoulder = get_landmark_coords(landmarks, mp_pose.RIGHT_SHOULDER.value, image_shape)
    left_ankle = get_landmark_coords(landmarks, mp_pose.LEFT_ANKLE.value, image_shape)
    right_ankle = get_landmark_coords(landmarks, mp_pose.RIGHT_ANKLE.value, image_shape)

    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    ankle_dist = np.linalg.norm(left_ankle - right_ankle)

    return ankle_dist*ankle_threshold < shoulder_width

def check_knee_valgus(landmarks, mp_pose, image_shape, ratio_threshold=0.7):
    """
    Check if knees are too close together (frontal view).

    Args:
        landmarks: list of MediaPipe landmarks.
        mp_pose: MediaPipe PoseLandmark module.
        image_shape: (h, w).
        ratio_threshold (float): minimum ratio between knee distance and hip/shoulder distance.

    Returns:
        bool: True if knees are too close.
    """
   
    left_shoulder = get_landmark_coords(landmarks, mp_pose.LEFT_SHOULDER.value, image_shape)
    right_shoulder = get_landmark_coords(landmarks, mp_pose.RIGHT_SHOULDER.value, image_shape)
    left_knee = get_landmark_coords(landmarks, mp_pose.LEFT_KNEE.value, image_shape)
    right_knee = get_landmark_coords(landmarks, mp_pose.RIGHT_KNEE.value, image_shape)
    
    shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
    knee_dist = abs(left_knee[0] - right_knee[0])

    return knee_dist < shoulder_width*ratio_threshold

def check_foot_angle(landmarks, mp_pose, image_shape, min_diff=0.0):
    """
    Check if feet are too closed inward.
    Returns (True, message) if they should be opened more, otherwise (False, None).
    """

    left_heel = get_landmark_coords(landmarks, mp_pose.LEFT_HEEL.value, image_shape)
    left_toe = get_landmark_coords(landmarks, mp_pose.LEFT_FOOT_INDEX.value, image_shape)
    right_heel = get_landmark_coords(landmarks, mp_pose.RIGHT_HEEL.value, image_shape)
    right_toe = get_landmark_coords(landmarks, mp_pose.RIGHT_FOOT_INDEX.value, image_shape)

    left_diff = left_heel[0] - left_toe[0]  
    right_diff = right_toe[0] - right_heel[0]  

    return left_diff > min_diff or right_diff > min_diff


def check_body_alignment(landmarks, mp_pose, image_shape, min_angle=150):
    """
    Check if the body is aligned during a lateral push-up.

    Returns:
        bool: True if the body is NOT aligned (needs feedback), False otherwise.
    """
    left_shoulder = get_landmark_coords(landmarks, mp_pose.LEFT_SHOULDER.value, image_shape)
    left_hip = get_landmark_coords(landmarks, mp_pose.LEFT_HIP.value, image_shape)
    right_shoulder = get_landmark_coords(landmarks, mp_pose.RIGHT_SHOULDER.value, image_shape)
    right_hip = get_landmark_coords(landmarks, mp_pose.RIGHT_HIP.value, image_shape)
    left_ankle = get_landmark_coords(landmarks, mp_pose.LEFT_ANKLE.value, image_shape)
    right_ankle = get_landmark_coords(landmarks, mp_pose.RIGHT_ANKLE.value, image_shape)
    if get_side(landmarks, mp_pose, image_shape) == "right":
        left_angle = calculate_angle(left_shoulder, left_hip, left_ankle)
        return left_angle < min_angle
    elif get_side(landmarks, mp_pose, image_shape) == "left":
        right_angle = calculate_angle(right_shoulder, right_hip, right_ankle)
        return right_angle < min_angle

def check_elbow_angle(landmarks, mp_pose, image_shape, min_angle=80, max_angle=150):
    """
    Check the elbow angle during push-ups.

    Returns:
        bool: True if elbow angle is outside the correct range, False otherwise.
    """
    right_shoulder = get_landmark_coords(landmarks, mp_pose.RIGHT_SHOULDER.value, image_shape)
    right_elbow = get_landmark_coords(landmarks, mp_pose.RIGHT_ELBOW.value, image_shape)
    right_wrist = get_landmark_coords(landmarks, mp_pose.RIGHT_WRIST.value, image_shape)
    left_shoulder = get_landmark_coords(landmarks, mp_pose.LEFT_SHOULDER.value, image_shape)
    left_elbow = get_landmark_coords(landmarks, mp_pose.LEFT_ELBOW.value, image_shape)
    left_wrist = get_landmark_coords(landmarks, mp_pose.LEFT_WRIST.value, image_shape)
    if get_side(landmarks, mp_pose, image_shape) == "right":
        left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        return left_angle > max_angle
    elif get_side(landmarks, mp_pose, image_shape) == "left":
        right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        return right_angle > max_angle

def check_arm_leg_sync(landmarks, mp_pose, image_shape, angle_threshold=30):
    """
    Check synchronization between opposite arm and leg.

    Args:
        angle_threshold (float, default=15): tolerance angle in degrees.

    Returns:
        bool: True if the difference exceeds the threshold.
    """
    left_ankle = get_landmark_coords(landmarks, mp_pose.LEFT_ANKLE.value, image_shape)
    right_ankle = get_landmark_coords(landmarks, mp_pose.RIGHT_ANKLE.value, image_shape)
    
    left_hip = get_landmark_coords(landmarks, mp_pose.LEFT_HIP.value, image_shape)
    right_hip = get_landmark_coords(landmarks, mp_pose.RIGHT_HIP.value, image_shape)

    left_wrist = get_landmark_coords(landmarks, mp_pose.LEFT_WRIST.value, image_shape)
    right_wrist = get_landmark_coords(landmarks, mp_pose.RIGHT_WRIST.value, image_shape)


    left_angle = calculate_angle(left_wrist, left_hip, left_ankle)
    right_angle = calculate_angle(right_wrist, right_hip, right_ankle)
    diff = abs(left_angle - right_angle)

    return diff > angle_threshold
        
def check_open_legs_arms(landmarks, mp_pose, image_shape, arm_angle_threshold=150, ankle_open_threshold=1.2):
    """
    Check if arms and legs are sufficiently open:
    - Determine arm openness using the angle between hip–shoulder–elbow; both must exceed "arm_angle_threshold".
    - Determine leg openness by comparing the distance between ankles with the shoulder width multiplied by "ankle_open_threshold".

    Args:
        arm_angle_threshold (float, default=150): minimum angular threshold to consider arms raised.
        ankle_open_threshold (float, default=1.2): multiplication factor of ankle distance relative to shoulder width to consider legs open.

    Returns:
        bool: True if arms and legs are correctly open, False otherwise.
    """

    left_shoulder = get_landmark_coords(landmarks, mp_pose.LEFT_SHOULDER.value, image_shape)
    right_shoulder = get_landmark_coords(landmarks, mp_pose.RIGHT_SHOULDER.value, image_shape)
    left_elbow = get_landmark_coords(landmarks, mp_pose.LEFT_ELBOW.value, image_shape)
    right_elbow = get_landmark_coords(landmarks, mp_pose.RIGHT_ELBOW.value, image_shape)
    left_hip = get_landmark_coords(landmarks, mp_pose.LEFT_HIP.value, image_shape)
    right_hip = get_landmark_coords(landmarks, mp_pose.RIGHT_HIP.value, image_shape)
    left_ankle = get_landmark_coords(landmarks, mp_pose.LEFT_ANKLE.value, image_shape)
    right_ankle = get_landmark_coords(landmarks, mp_pose.RIGHT_ANKLE.value, image_shape)
    
    # Arms
    left_arm_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
    right_arm_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
    arms_up = (left_arm_angle > arm_angle_threshold) and (right_arm_angle > arm_angle_threshold)

    # Legs
    ankle_dist = np.linalg.norm(left_ankle - right_ankle)
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    max_dist = ankle_open_threshold * shoulder_width

    return arms_up and ankle_dist < max_dist
        
def check_closed_legs_arms(landmarks, mp_pose, image_shape, arm_angle_threshold=60, ankle_close_threshold=0.6):
    """
    Check if arms and legs are closed:
    - Determine arm closure using the angle between hip–shoulder–elbow; both must be below "arm_angle_threshold".
    - Determine leg closure by comparing the distance between ankles with the shoulder width multiplied by "ankle_close_threshold".

    Args:
        arm_angle_threshold (float, default=35): maximum angular threshold to consider arms closed.
        ankle_close_threshold (float, default=0.6): multiplication factor of ankle distance relative to shoulder width to consider legs closed.

    Returns:
        bool: True if arms and legs are correctly closed, False otherwise.
    """
    left_shoulder = get_landmark_coords(landmarks, mp_pose.LEFT_SHOULDER.value, image_shape)
    right_shoulder = get_landmark_coords(landmarks, mp_pose.RIGHT_SHOULDER.value, image_shape)
    left_elbow = get_landmark_coords(landmarks, mp_pose.LEFT_ELBOW.value, image_shape)
    right_elbow = get_landmark_coords(landmarks, mp_pose.RIGHT_ELBOW.value, image_shape)
    left_hip = get_landmark_coords(landmarks, mp_pose.LEFT_HIP.value, image_shape)
    right_hip = get_landmark_coords(landmarks, mp_pose.RIGHT_HIP.value, image_shape)
    left_ankle = get_landmark_coords(landmarks, mp_pose.LEFT_ANKLE.value, image_shape)
    right_ankle = get_landmark_coords(landmarks, mp_pose.RIGHT_ANKLE.value, image_shape)
    
    left_arm_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
    right_arm_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
    arms_down = (left_arm_angle < arm_angle_threshold) and (right_arm_angle < arm_angle_threshold)

    ankle_dist = np.linalg.norm(left_ankle - right_ankle)
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    min_dist = ankle_close_threshold * shoulder_width

    return arms_down and ankle_dist > min_dist

def check_jj_knee_angle(landmarks, mp_pose, image_shape, knee_angle_threshold=125, arm_angle_threshold=150):
    """
    Check if, during a jumping jack, the knees are sufficiently bent and the arms are raised:
    - Determine the knee angle (hip–knee–ankle) and verify if at least one exceeds "knee_angle_threshold".
    - Determine arm openness (hip–shoulder–elbow) and verify if both exceed "arm_angle_threshold".

    Args:
        knee_angle_threshold (float, default=125): minimum knee angle to consider it bent.
        arm_angle_threshold (float, default=150): minimum arm angle to consider it raised.

    Returns:
        bool: True if the conditions of bent knees and raised arms are met, False otherwise.
    """
    left_hip = get_landmark_coords(landmarks, mp_pose.LEFT_HIP.value, image_shape)
    right_hip = get_landmark_coords(landmarks, mp_pose.RIGHT_HIP.value, image_shape)
    left_knee = get_landmark_coords(landmarks, mp_pose.LEFT_KNEE.value, image_shape)
    right_knee = get_landmark_coords(landmarks, mp_pose.RIGHT_KNEE.value, image_shape)
    left_ankle = get_landmark_coords(landmarks, mp_pose.LEFT_ANKLE.value, image_shape)
    right_ankle = get_landmark_coords(landmarks, mp_pose.RIGHT_ANKLE.value, image_shape)
    left_shoulder = get_landmark_coords(landmarks, mp_pose.LEFT_SHOULDER.value, image_shape)
    right_shoulder = get_landmark_coords(landmarks, mp_pose.RIGHT_SHOULDER.value, image_shape)
    left_elbow = get_landmark_coords(landmarks, mp_pose.LEFT_ELBOW.value, image_shape)
    right_elbow = get_landmark_coords(landmarks, mp_pose.RIGHT_ELBOW.value, image_shape)

    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

    left_arm_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
    right_arm_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
    braccia_su = (left_arm_angle > arm_angle_threshold) and (right_arm_angle > arm_angle_threshold)

    return braccia_su and (left_knee_angle > knee_angle_threshold or right_knee_angle > knee_angle_threshold)




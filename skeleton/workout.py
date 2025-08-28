import numpy as np
import mediapipe as mp
import cv2
from skeleton.test import predict_exercise
import threading
import speech_recognition as sr
import time
from feedback.feedback import get_star_rating, get_feedback, check_exercise, select_best_tip, text_speaker, build_tips_dict
from typing import Callable, Optional

MAX_SECONDS = 60   
SEGMENT_FRAMES = 301

COUNTDOWN_SECONDS = 5   
STOP_REQUESTED = False
VOICE_THREAD = None

SEQUENZA_SEC = 3


def calculate_angle(a, b, c):
    """
    Calculates the angle formed by three landmarks in 3D space.

    Args:
        a, b, c: NumPy arrays of real coordinates (x, y, z) of the landmarks.
                 The angle is computed at 'b' as the vertex.

    Returns:
        float: Angle in degrees between segments BA and BC.
    """
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)


def extract_angles(landmarks, image_shape):
    """
    Calculates 7 specific angles from landmarks detected by MediaPipe Pose.

    Args:
        landmarks: List of landmarks detected by MediaPipe Pose.
        image_shape: Tuple (height, width, channels) of the image, used to scale coordinates.

    Returns:
        list[float]: List of 7 angles in degrees.

    Note:
        If an error occurs during angle calculation (e.g., missing landmarks),
        the function returns a list of seven zeros.
    """

    mp_pose = mp.solutions.pose.PoseLandmark

    try:
        # 1. right_elbow_right_shoulder_right_hip
        angle1 = calculate_angle(
            get_landmark_coords(landmarks, mp_pose.RIGHT_ELBOW.value, image_shape),
            get_landmark_coords(landmarks, mp_pose.RIGHT_SHOULDER.value, image_shape),
            get_landmark_coords(landmarks, mp_pose.RIGHT_HIP.value, image_shape)
        )

        # 2. left_elbow_left_shoulder_left_hip
        angle2 = calculate_angle(
            get_landmark_coords(landmarks, mp_pose.LEFT_ELBOW.value, image_shape),
            get_landmark_coords(landmarks, mp_pose.LEFT_SHOULDER.value, image_shape),
            get_landmark_coords(landmarks, mp_pose.LEFT_HIP.value, image_shape)
        )

        # 3. right_knee_mid_hip_left_knee 
        mid_hip = (
            get_landmark_coords(landmarks, mp_pose.RIGHT_HIP.value, image_shape) +
            get_landmark_coords(landmarks, mp_pose.LEFT_HIP.value, image_shape)
        ) / 2.0
        angle3 = calculate_angle(
            get_landmark_coords(landmarks, mp_pose.RIGHT_KNEE.value, image_shape),
            mid_hip,
            get_landmark_coords(landmarks, mp_pose.LEFT_KNEE.value, image_shape)
        )

        # 4. right_hip_right_knee_right_ankle
        angle4 = calculate_angle(
            get_landmark_coords(landmarks, mp_pose.RIGHT_HIP.value, image_shape),
            get_landmark_coords(landmarks, mp_pose.RIGHT_KNEE.value, image_shape),
            get_landmark_coords(landmarks, mp_pose.RIGHT_ANKLE.value, image_shape)
        )

        # 5. left_hip_left_knee_left_ankle
        angle5 = calculate_angle(
            get_landmark_coords(landmarks, mp_pose.LEFT_HIP.value, image_shape),
            get_landmark_coords(landmarks, mp_pose.LEFT_KNEE.value, image_shape),
            get_landmark_coords(landmarks, mp_pose.LEFT_ANKLE.value, image_shape)
        )

        # 6. right_wrist_right_elbow_right_shoulder
        angle6 = calculate_angle(
            get_landmark_coords(landmarks, mp_pose.RIGHT_WRIST.value, image_shape),
            get_landmark_coords(landmarks, mp_pose.RIGHT_ELBOW.value, image_shape),
            get_landmark_coords(landmarks, mp_pose.RIGHT_SHOULDER.value, image_shape)
        )

        # 7. left_wrist_left_elbow_left_shoulder
        angle7 = calculate_angle(
            get_landmark_coords(landmarks, mp_pose.LEFT_WRIST.value, image_shape),
            get_landmark_coords(landmarks, mp_pose.LEFT_ELBOW.value, image_shape),
            get_landmark_coords(landmarks, mp_pose.LEFT_SHOULDER.value, image_shape)
        )

        return [angle1, angle2, angle3, angle4, angle5, angle6, angle7]

    except Exception as e:
        print(f"[ERROR] Angle calculation issue: {e}")
        return [0.0] * 7 






def is_motionless(prev_angles, current_angles, threshold=5.0):

    """
    Determines if the body is essentially motionless by comparing angles between two consecutive frames.

    Args:
        prev_angles: List of 7 angles (in degrees) from the previous frame.
        current_angles: List of 7 angles (in degrees) from the current frame.
        threshold: Float (default 5.0). Degree threshold; if the average change between angles is below this value,
                   the body is considered "motionless".

    Returns:
        bool: True if the average change between angles is below 'threshold', False otherwise.

    Note:
        If prev_angles is None (first frame), the function returns False because there is no previous frame to compare.
    """

    if prev_angles is None:     
        return False
    differences = [abs(c - p) for c, p in zip(current_angles, prev_angles)]  
    avg_change = sum(differences) / len(differences)
    return avg_change < threshold


def voice_command_listener():
    """
    Starts a voice listener that continuously monitors the microphone for a stop command.

    Note:
        - Sets the global variable STOP_REQUESTED to True if the user says "stop" or "basta".
        - Runs in a separate thread to allow real-time monitoring during exercise recording.
    """
    global STOP_REQUESTED
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        while not STOP_REQUESTED:
            try:
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=2)
                text = recognizer.recognize_google(audio, language="it-IT").lower()
                if "basta" in text or "stop" in text:
                    STOP_REQUESTED = True
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                pass
            except Exception as e:
                print(f"[ERROR]: {e}")


def detect_stop_gesture(hands_results, mp_hands):

    """
    Detects the "OK" hand gesture, used here as a stop signal during exercise.

    Args:
        hands_results: MediaPipe Hands result object containing detected hand landmarks.
        mp_hands: MediaPipe Hands module, used to access landmark indices.

    Returns:
        bool: True if the gesture is detected, False otherwise.
    """

    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            tip_thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            tip_index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            dist = np.linalg.norm(np.array([tip_thumb.x, tip_thumb.y]) - np.array([tip_index.x, tip_index.y]))
            if dist < 0.02:
                return True
    return False


def get_landmark_coords(landmarks, idx, image_shape):
    """
    Returns the real-world coordinates (x, y, z) of a given landmark by its index.

    Args:
        landmarks: List of landmarks detected by MediaPipe.
        idx: Index of the landmark to extract.
        image_shape: Tuple (height, width, channels) of the image, used to scale normalized coordinates.

    Returns:
        [x, y, z] -> actual coordinates of the landmark.
    """


    h, w = image_shape[:2]
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h, lm.z * w])

def coordinates_rectangle(frame, width_ratio, height_ratio):
    """
    Calculates the coordinates of a rectangle centered within a video frame.

    Args:
        frame: Numpy array representing the image.
        width_ratio: Float, percentage of the frame width occupied by the rectangle (0-1).
        height_ratio: Float, percentage of the frame height occupied by the rectangle (0-1).

    Returns:
        Tuple of rectangle coordinates: (x1, y1, x2, y2)
    """
    h, w = frame.shape[:2]
    rect_w = int(w * width_ratio)
    rect_h = int(h * height_ratio)
    rect_x1 = (w - rect_w) // 2
    rect_y1 = (h - rect_h) // 2
    rect_x2 = rect_x1 + rect_w
    rect_y2 = rect_y1 + rect_h
    return rect_x1, rect_y1, rect_x2, rect_y2

def is_lateral_orientation(landmarks, mp_pose, orientation_threshold=0.1):
    """ Checks if the user is correctly oriented for the chosen exercise """

    left_hip = landmarks[mp_pose.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.RIGHT_HIP.value]

    hip_diff_z = abs(left_hip.z - right_hip.z)

    if hip_diff_z > orientation_threshold:
        return True
    else:
        return False

def update_countdown(countdown_start_time, recording, countdown=COUNTDOWN_SECONDS):
    """

    Updates the countdown state before starting the recording.

    Args:
    countdown_start_time: float or None - the timestamp (time.time()) when the countdown started, or None to start it.
    recording: bool - indicates if recording has already started.
    countdown: int - total countdown duration in seconds (default = COUNTDOWN_SECONDS).

    Returns:
    tuple:
        - countdown_start_time: float | None, updated countdown start time.
        - recording: bool, updated recording state.
        - message: str, feedback message to show the user.

    Notes:
    - If the countdown hasn't started, it is initiated by setting countdown_start_time to current time.
    - Once the countdown completes, recording is set to True and the message indicates that recording is in progress.
    - During the countdown, the remaining seconds are displayed in the message.

    """

    now = time.time()

    if countdown_start_time is None:
        countdown_start_time = now

    elapsed = now - countdown_start_time
    if elapsed >= countdown:
        if not recording:
            print("Recording started!")
        recording = True
        message = "Recording in progress..."
    else:
        seconds_left = int(countdown - elapsed) + 1
        if seconds_left < 1:
            seconds_left = 1
        message = f"Hold still! Starting in {seconds_left}s"

    return countdown_start_time, recording, message



def all_lm_in_rect(landmarks, frame_shape, rect_coords):
    """
    Checks if all landmarks are inside a given rectangle.

    Args:
    landmarks: List of normalized landmarks (x, y, z) from MediaPipe.
    frame_shape: Tuple (height, width) of the frame, used to scale coordinates.
    rect_coords: Tuple (x1, y1, x2, y2) representing the rectangle in pixel coordinates.

    Returns:
    tuple:
        - bool: True if all landmarks are inside the rectangle, False otherwise.
        - list[int]: List of y-coordinates of the landmarks (pys), useful for repositioning feedback.

    Notes:
    - Converts normalized landmark coordinates to pixel coordinates based on the frame dimensions.
    """

    rect_x1, rect_y1, rect_x2, rect_y2 = rect_coords
    pxs, pys = [], []
    lm_inside = []
    for lm in landmarks:
        px, py = int(lm.x * frame_shape[1]), int(lm.y * frame_shape[0])
        pxs.append(px)
        pys.append(py)
        is_inside = rect_x1 <= px <= rect_x2 and rect_y1 <= py <= rect_y2
        lm_inside.append(is_inside)
    return all(lm_inside), pys

def landmarks_detected(results):
    """
    Returns True if a person is detected (i.e., pose_landmarks are available).
    """
    return results.pose_landmarks is not None


def record_video_and_angles(angles_frames, chosen_model, model_path,
                            on_frame: Optional[Callable[[np.ndarray], None]] = None,
                            stop_event: Optional[threading.Event] = None,
                            on_message: Optional[Callable[[str, str], None]] = None):
    """
    Main function for exercise recording and analysis:
    - Opens the webcam and reads frames in real time.
    - Detects pose and hands using MediaPipe Pose and Hands.
    - Computes 7 key body angles on each frame via `extract_angles`.
    - Checks if the user is centered and properly oriented within the recording rectangle.
    - Resets state if the user leaves the rectangle, is not detected, or is incorrectly oriented.
    - Handles countdown before starting recording.
    - Records the sequence of angles during the exercise.
    - Provides real-time tips via `check_exercise`.
    - Automatically stops recording if:
        * Maximum duration is reached.
        * User remains motionless for a defined period.
        * Stop gesture (for push-ups) or voice command is detected.
    - Computes and returns the list of exercise predictions.

    Args:
        angles_frames: list of angles calculated frame by frame.
        chosen_model: str -> name of the exercise to record.
        model_path: path to the ML model for exercise prediction.
        on_frame: optional callback to send processed frames to an external frontend.
        stop_event: optional threading.Event to stop recording externally.
        on_message: optional callback to send real-time messages/feedback.

    Returns:
        predictions: list of predicted labels for the recorded exercise.
    """


    # Detect body landmarks using MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_pose_landmark = mp_pose.PoseLandmark
    pose = mp_pose.Pose()

    # Detect hand landmarks using MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

    cap = cv2.VideoCapture(0)  # Initialize the webcam for capturing video frames

    predictions = []    # List to store predictions for each segment
    recording = False   # Boolean flag to track whether recording is currently active

    countdown_start_time = None      # Timestamp marking the start of the countdown
    recording_start_time = None      # Timestamp marking the start of the actual recording

    rec_accumulated = 0.0     # Actual seconds of recording accumulated so far
    rec_last_start = None     # Timestamp marking the last resume of recording

    motionless_start_time = None     # Timestamp marking the start of detected immobility
    prev_angles = None               # Used to check if the user is motionless

       
    # Parameters for automatic stopping
    stop_gesture_start_time = None          # Timestamp when stop gesture is first detected
    motionless_seconds_required = 5         # Stop if the user remains motionless for a few seconds
    stop_gesture_seconds_required = 1       # Stop if stop gesture persists for 1 second

    # Cooldowns for feedback messages to avoid repetition
    last_message_time = 0  
    message_cooldown = 5                     # Minimum seconds between general messages
    last_orientation_message_time = 0  
    orientation_message_cooldown = 3         # Minimum seconds between orientation-related messages


    # Global voice thread and stop flag
    global VOICE_THREAD, STOP_REQUESTED

    # Build the dictionary of exercise-specific tips
    tips_dict = build_tips_dict(chosen_model)
    # Current best tip to display to the user
    best_tip = ""
    
    if on_frame is None:
        cv2.namedWindow("Korrecto", cv2.WINDOW_NORMAL)

    while True:
        if stop_event is not None and stop_event.is_set():
            break

        # Capture frames from the webcam in a loop
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] webcam issue.")
            break

        # Compute the central rectangle for user positioning based on the chosen exercise
        if chosen_model == "squat_front" or chosen_model == "squat_side" or chosen_model == "jumping_jack":
            rect_x1, rect_y1, rect_x2, rect_y2 = coordinates_rectangle(frame, width_ratio=0.7, height_ratio=1)
        elif chosen_model == "push_up":
            rect_x1, rect_y1, rect_x2, rect_y2 = coordinates_rectangle(frame, width_ratio=1, height_ratio=0.7)

        # Convert frame to RGB and run pose & hand detection using MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb)
        hands_results = hands.process(rgb)

        message = ""
        curr_time = time.time()

        # Check if landmarks are detected, if all landmarks are inside the target rectangle.
        if landmarks_detected(pose_results):
            all_inside, pys = all_lm_in_rect(
                pose_results.pose_landmarks.landmark, frame.shape, (rect_x1, rect_y1, rect_x2, rect_y2)
            )

            if all_inside:
                # Check if the user's orientation matches the expected exercise position.
                is_ready = False
                if chosen_model == "squat_side":
                    is_ready = is_lateral_orientation(pose_results.pose_landmarks.landmark, mp_pose_landmark)
                    msg_if_wrong = "Please face the camera at a semi-side angle"
                elif chosen_model == "squat_front":
                    is_ready = not is_lateral_orientation(pose_results.pose_landmarks.landmark, mp_pose_landmark)
                    msg_if_wrong = "Please face forward the camera"
                elif chosen_model == "push_up":
                    is_ready = is_lateral_orientation(pose_results.pose_landmarks.landmark, mp_pose_landmark)
                    msg_if_wrong = "Please face the camera at a semi-side angle"
                else:
                    is_ready = True

                # Provides messages and resets countdown/recording state if necessary.
                if is_ready:
                    prev_recording = recording
                    countdown_start_time, recording, message = update_countdown(countdown_start_time, recording)
                    if recording and not prev_recording:
                        rec_last_start = time.time()
                        recording_start_time = rec_last_start
                        inizio_blocco = time.time()
                else:
                    # Resets the countdown and recording state if the user is incorrectly oriented
                    if recording and rec_last_start is not None:
                        rec_accumulated += time.time() - rec_last_start
                        rec_last_start = None
                    countdown_start_time = None
                    recording = False

                    message = msg_if_wrong
                    if message and (curr_time - last_orientation_message_time > orientation_message_cooldown):
                        text_speaker(message)
                        last_orientation_message_time = curr_time
            else:
                # # Resets the countdown and recording state if the user moves outside the designated rectangle.
                if recording and rec_last_start is not None:
                    rec_accumulated += time.time() - rec_last_start
                    rec_last_start = None
                countdown_start_time = None
                recording = False

                min_y, max_y = min(pys), max(pys)
                
                # Provides spoken feedback to guide repositioning.
                if max_y > rect_y2:
                    message = "Step back a little more"
                    if message and (curr_time - last_message_time > message_cooldown):
                        text_speaker(message)
                        last_message_time = curr_time
                    
                elif min_y < rect_y1:
                    message = "Move down a little more"
                    if message and (curr_time - last_message_time > message_cooldown):
                        text_speaker(message)
                        last_message_time = curr_time
                else:
                    message = "Please reposition yourself at the center of the screen"
                    if message and (curr_time - last_message_time > message_cooldown):
                        text_speaker(message)
                        last_message_time = curr_time
        else:
            # Resets countdown, recording, and tips if no person is detected in the frame.
            if recording and rec_last_start is not None:
                rec_accumulated += time.time() - rec_last_start
                rec_last_start = None
            countdown_start_time = None
            recording = False
            message = "No person detected!"
            if message and (curr_time - last_message_time > message_cooldown):
                text_speaker(message)
                last_message_time = curr_time

        
        if on_message is not None and message:
            on_message("status", message)

        # Draw landmarks
        if landmarks_detected(pose_results):
            mp.solutions.drawing_utils.draw_landmarks(
                frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
        if hands_results.multi_hand_landmarks:
            for hand in hands_results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS)
   

        # == RECORDING BLOCK ==
        if recording:
            # Starts voice thread if not active
            global VOICE_THREAD, STOP_REQUESTED
            if (VOICE_THREAD is None or not VOICE_THREAD.is_alive()):
                VOICE_THREAD = threading.Thread(target=voice_command_listener, daemon=True)
                VOICE_THREAD.start()

            # Extracts angles from current frame
            angoli = extract_angles(pose_results.pose_landmarks.landmark, frame.shape)
            tips_dict = check_exercise(pose_results.pose_landmarks.landmark, frame.shape, chosen_model, tips_dict)

            # Checks exercise correctness and updates tips
            now = time.time()
            if now - inizio_blocco >= SEQUENZA_SEC:
                best_tip = select_best_tip(tips_dict)
                tips_dict = build_tips_dict(chosen_model)
                inizio_blocco = now
                if best_tip:
                    text_speaker(best_tip)
                    
                    if on_message is not None:
                        on_message("tip", best_tip)

            angles_frames.append(angoli)

            # Check maximum recording duration: 60 seconds = MAX_SECONDS
            if recording_start_time is not None:
                elapsed_recording = rec_accumulated + (time.time() - rec_last_start if rec_last_start is not None else 0.0)
                # End of recording:
                # - Computes final prediction on accumulated angles
                # - Generates textual feedback
                # - Calculates and sends final star rating
                if elapsed_recording >= MAX_SECONDS:
                    if len(angles_frames) >= SEGMENT_FRAMES//2:
                        prob, label = predict_exercise(model_path, angles_frames, chosen_model)
                        predictions.append(label)
                        feedback = get_feedback(label)
                        if on_message is not None and feedback:
                            on_message("feedback", feedback)

                    if predictions:
                        stars = get_star_rating(predictions)
                        if on_message is not None:
                            on_message("final_rating", stars)
                    return predictions

            # Real-time prediction for the current segment of frames:
            if len(angles_frames) == SEGMENT_FRAMES:    
                prob, label = predict_exercise(model_path, angles_frames, chosen_model)
                predictions.append(label)
                angles_frames.clear()
                feedback = get_feedback(label)
                text_speaker(feedback)
                if on_message is not None and feedback:
                    on_message("feedback", feedback)
            
          
            if on_frame is not None:
                on_frame(frame)         
            else:
                cv2.imshow("Korrecto", frame)

            # == Automatic stop handling: stop gesture or motionless detection ==
            if chosen_model == "push_up":
                if detect_stop_gesture(hands_results, mp_hands):
                    if stop_gesture_start_time is None:
                        stop_gesture_start_time = time.time()
                    elif time.time() - stop_gesture_start_time >= stop_gesture_seconds_required:
                        if on_message is not None:
                            on_message("stop_reason", f"Recording stopped: stop gesture detected")
                        if len(angles_frames) >= SEGMENT_FRAMES // 2:
                            prob, label = predict_exercise(model_path, angles_frames, chosen_model)
                            predictions.append(label)
                            feedback = get_feedback(label)
                            if on_message is not None and feedback:
                                on_message("feedback", feedback)

                        if predictions:
                            stars = get_star_rating(predictions)
                            if on_message is not None:
                                on_message("final_rating", stars)
                        return predictions
                else:
                    stop_gesture_start_time = None
            else:
                if is_motionless(prev_angles, angoli):
                    if motionless_start_time is None:
                        motionless_start_time = time.time()
                    elif time.time() - motionless_start_time >= motionless_seconds_required:
                        if on_message is not None:
                            on_message("stop_reason", f"Recording stopped: no movement detected for {motionless_seconds_required} seconds.")
                        if len(angles_frames) >= SEGMENT_FRAMES // 2:
                            prob, label = predict_exercise(model_path, angles_frames, chosen_model)
                            predictions.append(label)
                            feedback = get_feedback(label)
                            if on_message is not None and feedback:
                                on_message("feedback", feedback)

                        if predictions:
                            stars = get_star_rating(predictions)
                            if on_message is not None:
                                on_message("final_rating", stars)
                        return predictions
                else:
                    motionless_start_time = None
                prev_angles = angoli
            

        else:
            if on_frame is not None:
                on_frame(frame)         
            else:
                cv2.imshow("Korrecto", frame)
        
        # Check for voice command stop
        if STOP_REQUESTED:
            if on_message is not None:
                on_message("stop_reason", f"Recording stopped: 'stop' voice command detected.")
            if len(angles_frames) >= SEGMENT_FRAMES // 2:
                prob, label = predict_exercise(model_path, angles_frames, chosen_model)
                predictions.append(label)
                feedback = get_feedback(label)
                if on_message is not None and feedback:
                    on_message("feedback", feedback)

            if predictions:
                stars = get_star_rating(predictions)
                if on_message is not None:
                    on_message("final_rating", stars)
            STOP_REQUESTED = False
            return predictions

        if on_frame is None:
            if cv2.waitKey(1) & 0xFF == 27:
                print("Early exit (ESC).")
                break
        
    hands.close()
    cap.release()
    pose.close()
    if on_frame is not None:
        cv2.destroyAllWindows()

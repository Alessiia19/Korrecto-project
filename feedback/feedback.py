import random
import mediapipe as mp
import threading
import pyttsx3

tts_lock = threading.Lock()
tts_is_speaking = False

POSITIVE_FEEDBACK = [
    "Don't give up! Keep control.",
    "I like your energy!",
    "Almost there! Stay focused."
]

NEGATIVE_FEEDBACK = [
    "Check your form for a more precise execution.",
    "Pay more attention to your exercise technique!",
    "Try again, you can do better!"
]

TIPS_SQUAT_SIDE = [
    ("Your back is leaning too far forward.", 1),
    ("Lower yourself by bending your knees.", 2),
    ("Your knees go beyond your toes.", 3),
    ("When going down, bend your knees less.", 4),
]

TIPS_SQUAT_FRONT = [
    ("Your feet should be wider than shoulder-width apart.", 1),
    ("Lower yourself by bending your knees.", 2),
    ("During the squat, your knees get too close together.", 3),
    ("Turn your feet out more.", 4)
]

TIPS_PUSH_UP = [
    ("Keep your body straighter.", 1),
    ("Bend your elbows more for a proper push-up.", 2),
    ("Don't bend your knees.", 3)
]

TIPS_JUMPING_JACK = [
    ("Open your legs more when you raise your arms!", 1),
    ("Close your legs when you lower your arms!", 2),
    ("Try to synchronize your arms and legs.", 3)
]

def text_speaker(text):
    """
    Plays a voice message.

    Args:
        text (str): The text to be spoken.
    """
    global tts_is_speaking
    def run():
        global tts_is_speaking
        with tts_lock:
            tts_is_speaking = True
            engine = pyttsx3.init()
            for voice in engine.getProperty('voices'):
                if "Zira" in voice.name and "English" in voice.name:
                    engine.setProperty('voice', voice.id)
                    break
            engine.setProperty('rate', 170)
            engine.say(text)
            engine.runAndWait()
            tts_is_speaking = False

    if not tts_is_speaking:
        t = threading.Thread(target=run, daemon=True)
        t.start()

def get_feedback(prediction):
    """
    Returns a random textual feedback based on the exercise prediction:
    - If the prediction indicates a correct exercise, selects a random message from `POSITIVE_FEEDBACK`.
    - If the prediction indicates an error, selects a random message from `NEGATIVE_FEEDBACK`.

    Args:
        prediction: String -> label predicted by the model.

    Returns:
        feedback: String -> Positive or negative feedback message.
    """
    if prediction == "correct":
        feedback = random.choice(POSITIVE_FEEDBACK)
    else:
        feedback = random.choice(NEGATIVE_FEEDBACK)
    return feedback

def get_star_rating(predictions):
    """
    Computes the final star rating (0 to 5) based on the percentage of 'correct' predictions.
    Args:
        predictions (list): List of model predictions (strings).
    Returns:
        int: Number of stars (0-5).
    """
    if not predictions:
        return 0

    correct_count = sum(1 for p in predictions if p == "correct")
    percentage = correct_count / len(predictions)

    thresholds = {
        0: 0.00,
        1: 0.01,
        2: 0.21,
        3: 0.41,
        4: 0.61,
        5: 0.81
    }
    stars = 0
    for s, threshold in thresholds.items():
        if percentage >= threshold:
            stars = s
    return stars

def build_tips_dict(chosen_model):
    """
    Builds a dictionary of tips for the selected exercise, with predefined priorities:
    - Each tip is represented as a key in the dictionary.
    - Each value is a dictionary containing:
        * "priority": the tip's priority level.
        * "count": initialized to 0, used to track how often the tip occurs.

    Args:
        chosen_model: String -> Name of the exercise for which to build the tips.

    Returns:
        dict: Dictionary -> {tip: {"priority": int, "count": 0}}.

    """

    tips_with_priority = []

    if chosen_model == "squat_front":
        tips_with_priority = TIPS_SQUAT_FRONT
    elif chosen_model == "squat_side":
        tips_with_priority = TIPS_SQUAT_SIDE
    elif chosen_model == "push_up":
        tips_with_priority = TIPS_PUSH_UP
    elif chosen_model == "jumping_jack":
        tips_with_priority = TIPS_JUMPING_JACK

    return {tip: {"priority": priority, "count": 0} for tip, priority in tips_with_priority}

def select_best_tip(tips_dict):
    """
    Selects the most relevant tip from those recorded in tips_dict:
    - Returns the tip with the highest count and highest priority.
    - If no tip has a count > 0, returns an empty string.

    Args:
        tips_dict (dict): Dictionary of tips.

    Returns:
        str: The selected tip, or "" if none.
    """

    tips_detected = []
    for tip, values in tips_dict.items():
        count = values["count"]

        if tip == "Lower yourself by bending your knees." or tip == "Bend your elbows more for a proper push-up.":
            if count < 30:              
                count = 0               
        if tip == "Close your legs when you lower your arms!" or tip == "Open your legs more when you raise your arms!":   
            if count < 15:              
                count = 0 
        if count > 0:
            tips_detected.append((tip, count, values["priority"]))

    if not tips_detected:
        return ""

    tips_detected.sort(key=lambda x: (-x[1], x[2]))
    return tips_detected[0][0]


def reset_tips_counter(tips_dict):
    """
    Resets all tip counters to 0 in the tips_dict dictionary.
    """
    for tip in tips_dict.values():
        tip["count"] = 0


def check_exercise(landmarks, image_shape, chosen_model, tips):
    """
    Checks the execution of the selected exercise for the current frame.
    
    Parameters:
        landmarks: list of MediaPipe Pose landmarks
        image_shape: tuple (height, width)

    Returns:
        list of suggestions (strings). Empty if everything is correct.
        
    """

    mp_pose = mp.solutions.pose.PoseLandmark
    
    

    import skeleton.checks as c

    if chosen_model == "squat_side":
        tips_with_priority = TIPS_SQUAT_SIDE

        # Check back angle
        if c.check_back_inclination(landmarks, mp_pose, image_shape):
            tips[tips_with_priority[0][0]]["count"] += 1

        if not c.check_knee_angle(landmarks, mp_pose, image_shape, min_angle=150):
                # Check knee are not bent 
                tips[tips_with_priority[1][0]]["count"] += 1
                

        # Check if knee is over the foot
        if c.check_knee_over_foot(landmarks, mp_pose, image_shape):
            tips[tips_with_priority[2][0]]["count"] += 1

        # Check knee angle
        if c.check_knee_angle(landmarks, mp_pose, image_shape, min_angle=30):
            tips[tips_with_priority[3][0]]["count"] += 1

    elif chosen_model == "squat_front":
        tips_with_priority = TIPS_SQUAT_FRONT

        # Check if ankles are too close
        if c.check_ankle_distance(landmarks, mp_pose, image_shape):
            tips[tips_with_priority[0][0]]["count"] += 1

        else:
            if not c.check_knee_angle(landmarks, mp_pose, image_shape, min_angle=120):
                # Check knee are not bent
                tips[tips_with_priority[1][0]]["count"] += 1

            # Check if knees are too close
            if c.check_knee_valgus(landmarks, mp_pose, image_shape):
                tips[tips_with_priority[2][0]]["count"] += 1

            # Check toe
            if c.check_foot_angle(landmarks, mp_pose, image_shape):
                tips[tips_with_priority[3][0]]["count"] += 1

    elif chosen_model == "push_up":
        tips_with_priority = TIPS_PUSH_UP

        # Check if body isn't aligned correctly
        if c.check_body_alignment(landmarks, mp_pose, image_shape):
            tips[tips_with_priority[0][0]]["count"] += 1

        # Check elbow angle
        if c.check_elbow_angle(landmarks, mp_pose, image_shape):
            tips[tips_with_priority[1][0]]["count"] += 1

        # Check Knee angle
        if c.check_knee_angle(landmarks, mp_pose, image_shape, min_angle=170):
            tips[tips_with_priority[2][0]]["count"] += 1

    
    elif chosen_model == "jumping_jack":
        tips_with_priority = TIPS_JUMPING_JACK

        # Check if legs are open together with arms
        if c.check_open_legs_arms(landmarks, mp_pose, image_shape):
            tips[tips_with_priority[0][0]]["count"] += 1

        # Check if legs are closed together with arms
        if c.check_closed_legs_arms(landmarks, mp_pose, image_shape):
            tips[tips_with_priority[1][0]]["count"] += 1

        # Check if both sides are synchronized
        if c.check_arm_leg_sync(landmarks, mp_pose, image_shape):
            tips[tips_with_priority[2][0]]["count"] += 1

    return tips
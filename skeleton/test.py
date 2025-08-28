import pandas as pd
import numpy as np
import tensorflow as tf
import os


ANGLE_NAMES = [
    "right_elbow_right_shoulder_right_hip",
    "left_elbow_left_shoulder_left_hip",
    "right_knee_mid_hip_left_knee",
    "right_hip_right_knee_right_ankle",
    "left_hip_left_knee_left_ankle",
    "right_wrist_right_elbow_right_shoulder",
    "left_wrist_left_elbow_left_shoulder"
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


MODEL_PATHS = {
    "jumping_jack": os.path.join(BASE_DIR, "..", "analysis", "checkpoints", "jumping_jack_model_0.24.keras"),
    "squat_side": os.path.join(BASE_DIR, "..", "analysis", "checkpoints", "squat_model_0.35.keras"),
    "squat_front": os.path.join(BASE_DIR, "..", "analysis", "checkpoints", "squat_model_0.35.keras"),
    "push_up": os.path.join(BASE_DIR, "..", "analysis", "checkpoints", "push_up_model_0.31.keras")
}


def predict_exercise(model_path, angles_frames, chosen_model):
    """
    Performs prediction of the exercise executed by the user.

    Workflow:
    - Loads the LSTM model from the specified path.
    - Converts the list of frame angles into a NumPy array.
    - Applies padding if the number of frames is less than 301.
    - Reshapes the array to (1, 301, 7) to match the LSTM input format.
    - Saves the calculated angles into "debug_angles.csv" for checks or debugging.
    - Feeds the data to the LSTM model to obtain the probability of exercise correctness.
    - Compares the probability with the 0.7 threshold to assign a textual label:
      "correct" or "incorrect".

    Args:
        model_path (str): Path to the LSTM model (.keras) to be used.
        angles_frames: Sequence of frames; each frame contains 7 calculated angles.
        chosen_model (str): Name of the selected exercise.

    Returns:
        prob (float): Probability that the exercise is correct according to the model.
        label (str): Predicted label.
    """
   
    model = tf.keras.models.load_model(model_path)

    angles_frames = np.array(angles_frames)    
 
    if angles_frames.shape[0] < 301:
        padding = np.zeros((301 - angles_frames.shape[0], 7))
        angles_frames = np.vstack([angles_frames, padding])

    angles_frames = angles_frames.reshape(1, 301, 7)

    angoli_2d = angles_frames[0]

    df = pd.DataFrame(angoli_2d, columns=ANGLE_NAMES)
    df.to_csv("debug_angles.csv", index=False)
  


    pred = model.predict(angles_frames)

    prob = float(pred[0][0])
    label = "correct" if prob >= 0.7 else "incorrect"

    return prob, label
   
    

    


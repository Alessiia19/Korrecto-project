import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =======================
# Parameters
# =======================
ANGLE_PATH = "analysis/angles.csv"
CLASS_PATH = "analysis/labels.csv"
MAIN_CLASS = "jumping_jack"
OTHER_CLASSES_COUNTS = {
    "pull_up": 0, 
    "push_up": 0, 
    "situp": 0, 
    "squat": 0
}

def load_data(angle_path, class_path):
    """
    Loads angle and class data from CSV files and merges them based on 'vid_id'.
    
    Args:
    angle_path (str): Path to the angles CSV file.
    class_path (str): Path to the labels CSV file.
    
    Returns:
    Merged dataframe containing angles and corresponding classes.
    """
    angles = pd.read_csv(angle_path)
    classes = pd.read_csv(class_path)
    data = angles.merge(classes, on="vid_id")
    return data

def split_videos_by_class(data, other_classes_counts, main_class=MAIN_CLASS):
    """
    Splits videos into correct and incorrect classes based on the main class and other classes counts.
    
    Args:
    data (pd.DataFrame): Merged dataframe of angles and classes.
    other_classes_counts (dict): Dictionary specifying maximum counts for other classes.
    main_class (str): Main class name.
    
    Returns:
    Two lists of dataframes: correct_videos and incorrect_videos.
    """
    videos = data.groupby("vid_id")
    correct_videos = []
    incorrect_videos = []
    class_counters = {cls: 0 for cls in other_classes_counts}
    for vid_id, video in videos:
        classes_in_video = video['class'].unique()
        cls = classes_in_video[0]
        if cls == main_class:
            correct_videos.append(video)
        elif cls in other_classes_counts and class_counters[cls] < other_classes_counts[cls]:
            incorrect_videos.append(video)
            class_counters[cls] += 1
    
    for video in incorrect_videos:
        video['class'] = 'incorrect'
    return correct_videos, incorrect_videos

def build_final_dataset(correct_videos, incorrect_videos, main_class=MAIN_CLASS):
    """
    Creates the final dataset ready for feature extraction.
    
    Args:
    correct_videos (list): List of dataframes containing correct videos.
    incorrect_videos (list): List of dataframes containing 'incorrect' videos.
    main_class (str): Main class name.
    
    Returns:
    Final dataset with classes mapped to 0 (incorrect) and 1 (main_class).
    """
    df_main = pd.concat(correct_videos)
    df_incorrect = pd.concat(incorrect_videos)
    final_dataset = pd.concat([df_main, df_incorrect]).reset_index(drop=True)
    final_dataset['class'] = final_dataset['class'].map({'incorrect': 0, main_class: 1})
    return final_dataset

def extract_features_and_labels(final_dataset, feature_slice=slice(2, -1)):
    """
    Extract features (angles) and labels for each video (video as array of frames).
    
    Args:
    final_dataset (pd.DataFrame): Final dataset with videos and classes.
    feature_slice: Slice object specifying which columns to take as features.
    
    Returns:
    Two lists: angles_videos and labels_videos.
    """
    angles_videos = []  
    labels_videos = []
    videos = final_dataset.groupby("vid_id")
    for vid_id, video in videos:
        video = video.sort_values("frame_order")
        angles_video = video.iloc[:, feature_slice].to_numpy()
        label_video = video["class"].iloc[0]
        angles_videos.append(angles_video)
        labels_videos.append(label_video)
    return angles_videos, labels_videos

def pad_video_sequences(sequences):
    """
    Pads video sequences to ensure uniform length.
    
    Args:
    sequences: List of numpy arrays.
    
    Returns:
    Padded dataset of sequences.
    """
    max_len = max([vid.shape[0] for vid in sequences])
    dataset = pad_sequences(
        sequences,
        maxlen=max_len,
        dtype='float32',
        padding='post'
    )
    return dataset

def create_dataset():
    """
    Create the dataset.
    
    Returns:
    Dataset of video sequences and corresponding labels (y).
    """
    data = load_data(ANGLE_PATH, CLASS_PATH)
    correct_videos, incorrect_videos = split_videos_by_class(data, OTHER_CLASSES_COUNTS, MAIN_CLASS)
    final_dataset = build_final_dataset(correct_videos, incorrect_videos, MAIN_CLASS)
    angles_videos, labels_videos = extract_features_and_labels(final_dataset)
    dataset = pad_video_sequences(angles_videos)
    y = np.array(labels_videos)
    return dataset, y



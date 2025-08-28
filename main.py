# Disable auto DPI scaling
import os
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
os.environ["QT_SCALE_FACTOR"] = "1"

import sys
import ctypes
from config import WINDOW_SIZE, EXERCISE_VIDEOS
from PyQt6.QtWidgets import QApplication, QStackedWidget
from PyQt6.QtGui import QIcon
from ui.home_screen import HomeScreen
from ui.workout_screen import WorkoutScreen
from skeleton.workout import record_video_and_angles
from skeleton.test import MODEL_PATHS


def main():

    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("com.korrecto.demo")
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("ui/assets/korrecto_icon.ico"))

    # Load stylesheet
    try:
        with open("ui/style.qss","r") as f:
            app.setStyleSheet(f.read())
    except FileNotFoundError:
        pass

    stack = QStackedWidget()
    stack.setWindowTitle("Korrecto")
    stack.setWindowIcon(QIcon("ui/assets/korrecto_icon.ico"))

    menu = HomeScreen() 
    workout = WorkoutScreen()

    def go_workout(exercise_key):
        """Switch to workout screen, starting the backend process."""
        
        model_path = MODEL_PATHS[exercise_key]
        demo_path = EXERCISE_VIDEOS[exercise_key]
        workout.setup_ui(exercise_key, model_path, demo_path, backend_callable=record_video_and_angles)
        stack.setCurrentWidget(workout)

    def go_home():
        """Switch to home screen."""
        stack.setCurrentWidget(menu)

    menu.exerciseConfirmed.connect(go_workout)
    workout.backRequested.connect(go_home)

    stack.addWidget(menu)
    stack.addWidget(workout)
    stack.setCurrentWidget(menu)
    stack.setFixedSize(*WINDOW_SIZE)
    stack.show()

    return app.exec()

if __name__ == "__main__":
    sys.exit(main())

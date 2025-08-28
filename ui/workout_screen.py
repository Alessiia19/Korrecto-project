from PyQt6.QtCore import Qt, pyqtSignal, QThread, QUrl, QTimer
from PyQt6.QtGui import QImage, QPixmap, QColor
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QFrame, QGraphicsDropShadowEffect, QSizePolicy, QApplication
)
from PyQt6.QtMultimedia import QMediaPlayer, QSoundEffect
from PyQt6.QtMultimediaWidgets import QVideoWidget
import numpy as np
import threading
import cv2
import re
import time
from feedback.feedback import text_speaker
from skeleton.workout import MAX_SECONDS
from config import EXERCISE_GUIDES

class CameraWorker(QThread):
    """Runs backend pipeline and streams frames and messages to the UI."""
    frameReady = pyqtSignal(np.ndarray)
    messageReady = pyqtSignal(str, str) 

    def __init__(self, backend_callable, exercise_key, model_path, stop_event):
        """Initialize the camera worker.

        Args:
            backend: Function that captures video, processes frames (angles, predictions, etc.),
                    and reports messages and results back to the UI via callbacks.
            exercise_key: Exercise identifier (e.g. "push_up").
            model_path: Path to the model to load.
            stop_event: Threading event used to request a stop from the backend.
        """
        super().__init__()
        self.backend_callable = backend_callable
        self.exercise_key = exercise_key
        self.model_path = model_path
        self.stop_event = stop_event

    def run(self):
        """Execute the backend loop."""
        angles_frames = [] 
        self.backend_callable(
            angles_frames,
            self.exercise_key,
            self.model_path,
            on_frame=self.emit_frame,
            stop_event=self.stop_event,
            on_message=self.emit_message,
        )

    def emit_frame(self, frame_bgr):
        """Forward a new frame coming from the backend to the UI."""
        self.frameReady.emit(frame_bgr)

    def emit_message(self, kind, text):
        """
        Forward a new message coming from the backend to the UI.
        
        Args:
            kind (str): type of message defined by backend.
            text (str): message text.
        """    
        self.messageReady.emit(kind, str(text))


class WorkoutScreen(QWidget):
    """Main workout UI."""
    backRequested = pyqtSignal() # go back to home screen signal

    def __init__(self, parent=None):
        """Initialize Workout Screen UI."""
        super().__init__(parent)

        self.camera_worker = None
        self.stop_event = None
        self.tips_seen = []  
        self.backend_callable = None
        self.exercise_key = None
        self.model_path = None
        self.demo_video_path = None

        # Handlers of temporary messages shown at the top of the screen.
        self.toastActive = False
        self.toastQueue = []        
        self.toastTimer = QTimer(self)
        self.toastTimer.setSingleShot(True)
        self.toastTimer.timeout.connect(self.update_toast_queue)

        # Recording timer
        self.recTimer = QTimer(self) 
        self.recTimer.timeout.connect(self.update_rec_timer)
        self.rec_last_start = None    
        self.rec_accumulated = 0.0   
        self.rec_running = False          
        self.maxSeconds = MAX_SECONDS

        main_layout = QHBoxLayout(self)

        # ------------ 
        # LEFT COLUMN  
        # ------------
        left = QVBoxLayout()

        # Back button
        self.btnBack = QPushButton("← Back")
        self.btnBack.setProperty("kind", "back")
        self.btnBack.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btnBack.clicked.connect(self.on_back)
        left.addWidget(self.btnBack, alignment=Qt.AlignmentFlag.AlignLeft)

        # Exercise guide
        self.guideFrame = QFrame()
        guideLayout = QVBoxLayout(self.guideFrame)
        guideLayout.setContentsMargins(10, 10, 10, 10)

        self.guideLabel = QLabel()
        self.guideLabel.setWordWrap(True)
        self.guideLabel.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.guideLabel.setObjectName("guideLabel")
        guideLayout.addWidget(self.guideLabel)

        shadowGuide = QGraphicsDropShadowEffect(self.guideFrame)
        shadowGuide.setBlurRadius(12)
        shadowGuide.setOffset(0, 3)
        shadowGuide.setColor(QColor(0, 0, 0, 80))
        self.guideFrame.setGraphicsEffect(shadowGuide)

        left.addWidget(self.guideFrame)
        left.addStretch(1)

        # Exercise preview
        self.demoContainer = QFrame()
        self.demoContainer.setObjectName("demoContainer")
        demoLayout = QVBoxLayout(self.demoContainer)
        demoLayout.setContentsMargins(0, 0, 0, 0)

        self.demoVideo = QVideoWidget()
        self.demoVideo.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        self.demoVideo.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        demoLayout.addWidget(self.demoVideo)

        shadowDemo = QGraphicsDropShadowEffect(self.demoContainer)
        shadowDemo.setBlurRadius(18)
        shadowDemo.setOffset(0, 4)
        shadowDemo.setColor(QColor(0, 0, 0, 100))
        self.demoContainer.setGraphicsEffect(shadowDemo)

        left.addWidget(self.demoContainer, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)

        self.player = QMediaPlayer(self)
        self.player.setVideoOutput(self.demoVideo)
        self.player.setLoops(QMediaPlayer.Loops.Infinite)
        self.completeSound = QSoundEffect(self)
        self.completeSound.setSource(QUrl.fromLocalFile("ui/assets/workout_complete.wav"))
        self.completeSound.setLoopCount(1) 

        # -------------
        # RIGHT COLUMN  
        # -------------
        right = QVBoxLayout()

        # --- Camera --- 
        self.cameraContainer = QFrame()
        self.cameraContainer.setObjectName("cameraContainer")
        self.cameraContainer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        cameraLayout = QVBoxLayout(self.cameraContainer)
        cameraLayout.setContentsMargins(0,0,0,0)
        cameraLayout.setSpacing(0)

        self.cameraView = QLabel(self.cameraContainer)
        self.cameraView.setObjectName("cameraView")
        self.cameraView.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cameraView.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)  
        cameraLayout.addWidget(self.cameraView)

        # -- Overlays --
        # - Stop reason -
        self.stopReasonLabel = QLabel("", self.cameraContainer)
        self.stopReasonLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stopReasonLabel.setWordWrap(True)
        self.stopReasonLabel.setObjectName("stopReasonLabel")
        self.stopReasonLabel.setVisible(False)

        # - Play Section -
        # Play title
        self.playTitleLabel = QLabel("Click to start the camera", self.cameraContainer)
        self.playTitleLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.playTitleLabel.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.playTitleLabel.setWordWrap(True)
        self.playTitleLabel.setObjectName("playTitle")
        self.playTitleLabel.setVisible(True)

        # Stop instructions
        self.stopInstrLabel = QLabel("", self.cameraContainer)
        self.stopInstrLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stopInstrLabel.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.stopInstrLabel.setWordWrap(True)
        self.stopInstrLabel.setObjectName("stopInstrLabel")
        self.stopInstrLabel.setVisible(True)

        # Play button
        self.playButton = QPushButton("▶", self.cameraContainer)
        self.playButton.setObjectName("playButton")
        self.playButton.setCursor(Qt.CursorShape.PointingHandCursor)
        self.playButton.clicked.connect(self.start_workout)
        self.playButton.raise_()
        self.playButton.setVisible(True)

        # Stop button
        self.stopButton = QPushButton("✕", self.cameraContainer)
        self.stopButton.setObjectName("stopButton")
        self.stopButton.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stopButton.clicked.connect(self.on_stop_clicked)
        self.stopButton.setVisible(False)
        self.stopButton.raise_()


        # - Start Overlay -
        self.startOverlay = QLabel(self.cameraContainer)
        self.startOverlay.setObjectName("startOverlay")
        self.startOverlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.startOverlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.startOverlay.setWordWrap(True)
        self.startOverlay.setVisible(True)

        # - Workout Layout -
        # Tips overlay
        self.tipsLabel = QLabel("", self.cameraContainer)
        self.tipsLabel.setObjectName("tipsOverlay")
        self.tipsLabel.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.tipsLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tipsLabel.setWordWrap(True)
        self.tipsLabel.setVisible(False)
        self.tipsLabel.raise_()
        self.startOverlay.raise_()

        # Bottom bar
        self.bottomBar = QLabel("", self.cameraContainer)
        self.bottomBar.setObjectName("bottomBar")
        self.bottomBar.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.bottomBar.setWordWrap(False)
        self.bottomBar.setVisible(False)
        self.bottomBar.raise_()

        # Feedback popup
        self.feedbackPopup = QLabel("", self.cameraContainer)
        self.feedbackPopup.setObjectName("feedbackPopup")
        self.feedbackPopup.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.feedbackPopup.setWordWrap(True)
        self.feedbackPopup.setVisible(False)
        self.feedbackPopup.raise_()

        self.feedbackTimer = QTimer(self)
        self.feedbackTimer.setSingleShot(True)
        self.feedbackTimer.timeout.connect(self.clear_feedback)

        # - Rating Overlay -
        self.ratingOverlay = RatingOverlay(self.cameraContainer)
        self.ratingOverlay.setVisible(False)
        self.ratingOverlay.closed.connect(lambda: self.set_idle_state(clear_state=True))

        shadow2 = QGraphicsDropShadowEffect(self.cameraContainer)
        shadow2.setBlurRadius(24)
        shadow2.setOffset(0, 6)
        shadow2.setColor(QColor(0, 0, 0, 90))
        self.cameraContainer.setGraphicsEffect(shadow2)

        right.addWidget(self.cameraContainer, stretch=1)

       # ------------
        # MAIN LAYOUT
        # ------------
        main_layout.addLayout(left, stretch=1)
        main_layout.addLayout(right, stretch=3)

    def start_rec_timer(self):
        """Start recording timer."""
        self.rec_accumulated = 0.0
        self.rec_last_start = time.monotonic()
        self.rec_running = True
        if not self.recTimer.isActive():
            self.recTimer.start()

    def pause_rec_timer(self):
        """Pause recording timer."""
        if self.rec_running and self.rec_last_start is not None:
            self.rec_accumulated += time.monotonic() - self.rec_last_start
            self.rec_last_start = None
            self.rec_running = False

    def resume_rec_timer(self):
        """Resume recording timer."""
        if not self.rec_running:
            self.rec_last_start = time.monotonic()
            self.rec_running = True
            if not self.recTimer.isActive():
                self.recTimer.start()

    def stop_rec_timer(self):
        """Stop recording timer."""
        self.recTimer.stop()
        self.rec_last_start = None
        self.rec_accumulated = 0.0
        self.rec_running = False
        
    def update_rec_timer(self):
        """Update the 'Recording in progress' text with the elapsed seconds."""
        elapsed = self.rec_accumulated
        if self.rec_running and self.rec_last_start is not None:
            elapsed += time.monotonic() - self.rec_last_start
        shown = int(min(elapsed, float(self.maxSeconds)))
        denom = f" / {self.maxSeconds}s"
        
        self.bottomBar.setText(f"Recording in progress... {shown}s{denom}")
        self.relayout_overlays()

    def show_toast(self, text, duration_ms=4000):
        """
        Show top toast message, queues if another toast is active.
        
        Args:
            text (str): text message.
            duration_ms (int): Time in milliseconds before the toast automatically disappears.
        """
        text = (text or "").strip()
        if not text:
            return

        if not self.toastActive:
            self.tipsLabel.setText(text)
            self.tipsLabel.setVisible(True)
            self.toastActive = True
            self.toastTimer.start(duration_ms)
            self.relayout_overlays()
        else:
            if len(self.toastQueue) < 5:
                self.toastQueue.append((text, duration_ms))

    def update_toast_queue(self):
        """
        Handle the expiration of the current toast message.
        If there are queued toasts, display the next one and restart the timer.
        Otherwise, clear the label and mark the toast system as inactive.
        """
        if self.toastQueue:
            text, ms = self.toastQueue.pop(0)
            self.tipsLabel.setText(text)
            self.toastTimer.start(ms)
            self.tipsLabel.setVisible(True)
            self.relayout_overlays()
        else:
            self.toastActive = False
            self.tipsLabel.clear()
            self.tipsLabel.setVisible(False)

    def start_countdown(self):
        """Start a 3-second on-screen countdown before recording."""
        self.countdown_value = 3
        self.startOverlay.setText(str(self.countdown_value))
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_countdown)
        self.timer.start(1000)

    def update_countdown(self):
        """Update on-screen countdown until it reaches zero, then reveal tips label."""
        self.countdown_value -= 1
        if self.countdown_value > 0:
            self.startOverlay.setText(str(self.countdown_value))
        else:
            self.timer.stop()
            self.startOverlay.setVisible(False)
            self.tipsLabel.setVisible(True)

    def relayout_overlays(self):
        """Reposition overlay widgets to keep them aligned with the camera view."""
        g = self.cameraView.geometry()
        gx, gy, gw, gh = g.x(), g.y(), g.width(), g.height()

        outer_margin = 12
        bottom_margin = 6
        hint_gap = 20

        self.startOverlay.setGeometry(g)

        # Stop button 
        sb = self.stopButton.size()
        self.stopButton.move(gx + gw - sb.width() - outer_margin, gy + outer_margin)

        # Tips label 
        self.tipsLabel.setFixedWidth(max(100, gw - 2*outer_margin))
        self.tipsLabel.adjustSize()
        self.tipsLabel.move(gx + outer_margin, gy + outer_margin)

        # Bottom bar
        self.bottomBar.setFixedWidth(gw - 2*outer_margin)
        self.bottomBar.adjustSize()
        self.bottomBar.move(gx + outer_margin, gy + gh - self.bottomBar.height() - bottom_margin)

        # Play section 
        pb = self.playButton.size()
        px = gx + (gw - pb.width()) // 2
        py = gy + (gh - pb.height()) // 2
        self.playButton.move(px, py)

        self.playTitleLabel.setFixedWidth(gw)
        title_h = self.playTitleLabel.sizeHint().height()
        self.playTitleLabel.move(gx, max(gy + 8, py - title_h - hint_gap))

        self.stopInstrLabel.setFixedWidth(gw)
        self.stopInstrLabel.move(gx, py + pb.height() + hint_gap)

        # Feedback popup
        if self.feedbackPopup.isVisible():
            fp_w = self.feedbackPopup.width()
            fp_h = self.feedbackPopup.height()
            if self.bottomBar.isVisible():
                y = self.bottomBar.y() - fp_h - 8 
            else:
                y = gy + gh - fp_h - outer_margin
            x = gx + gw - fp_w - outer_margin
            self.feedbackPopup.move(x, y)
            self.feedbackPopup.raise_()
        

        self.playTitleLabel.raise_()
        self.stopInstrLabel.raise_()
        self.tipsLabel.raise_()
        self.bottomBar.raise_()
        self.playButton.raise_()
        self.stopButton.raise_()

        if self.startOverlay.isVisible():
            self.startOverlay.raise_()
        
        if self.ratingOverlay.isVisible():
            self.ratingOverlay.setGeometry(g)
            self.ratingOverlay.raise_()
        

    def show_feedback(self, text, duration_ms=4000):
        """
        Show a bottom-right feedback popup.
        
        Args:
            text (str): feedback message.
            duration_ms (int): Time in milliseconds before the popup automatically disappears.
        """
        text = (text or "").strip()
        if not text:
            return
        
        g = self.cameraView.geometry()
        max_w = max(200, int(g.width() * 0.45)) 
        self.feedbackPopup.setFixedWidth(max_w)

        self.feedbackPopup.setText(text)
        self.feedbackPopup.adjustSize()
        self.feedbackPopup.setVisible(True)
        self.feedbackTimer.start(duration_ms)
        self.relayout_overlays()

    def clear_feedback(self):
        """Hide the feedback popup when its timer expires."""
        self.feedbackPopup.clear()
        self.feedbackPopup.setVisible(False)

    def get_stop_instructions(self):
        """
        Return hint on how to stop the workout based on the exercise chosen.

        Returns:
            str: Stop instructions.
        """
        if self.exercise_key == "push_up":
            return ('<b>To stop:</b> say <span style="color: #3B8EEA"><b>"stop"</b></span> or <span style="color: #3B8EEA"><b>"basta"</b></span>, '
                    'or show the hand <span style="color: #3B8EEA"><b>“OK”</b></span> gesture.')
        else:
            return ('<b>To stop:</b> say <b><span style="color: #3B8EEA">"stop"</b></span> or <b><span style="color: #3B8EEA">"basta"</b></span>, '
                    'or <b><span style="color: #3B8EEA">stay still</b></span> for a few seconds.')

    def update_pre_start_hint(self):
        """Refresh the pre-start title and instructions shown above/below the Play button."""
        self.playTitleLabel.setText("Click to start the camera")
        self.stopInstrLabel.setText(self.get_stop_instructions())

    def set_idle_state(self, clear_state=False):
        """
        Reset the UI to idle state.

        Args:
            clear_state (bool): If True, clears the camera as well.
        """
        if clear_state:
            self.cameraView.clear()
    
        self.playButton.setVisible(True)
        self.playTitleLabel.setVisible(True)
        self.stopInstrLabel.setVisible(True)
        self.update_pre_start_hint()
        
        # Hide workout overlays
        self.startOverlay.setVisible(False)
        self.tipsLabel.setVisible(False)
        self.tipsLabel.clear()
        self.stopButton.setVisible(False)
        self.bottomBar.setVisible(False)
        self.ratingOverlay.setVisible(False)
        self.tips_seen.clear()
        self.stop_rec_timer()

        # reset toast
        self.toastTimer.stop()
        self.toastActive = False
        self.toastQueue.clear()

        # reset feedback popup
        self.feedbackTimer.stop()
        self.feedbackPopup.clear()
        self.feedbackPopup.setVisible(False)

        self.relayout_overlays()

    def start_workout(self):
        """Start camera worker and workout session."""
        if not self.backend_callable:
            return

        # Hide pre-start overlays
        self.playButton.setVisible(False)
        self.playTitleLabel.setVisible(False)
        self.stopInstrLabel.setVisible(False)

        # Prepare workout overlays
        self.startOverlay.setVisible(True)
        self.startOverlay.setText("Starting camera...")
        self.tipsLabel.setVisible(False)
        self.bottomBar.setVisible(False)
        self.stopReasonLabel.setVisible(False)
        self.stopButton.setVisible(True)
        self.stopButton.setEnabled(True)
        self.relayout_overlays()

        # Start camera worker thread
        self.stop_event = threading.Event()
        self.camera_worker = CameraWorker(self.backend_callable, self.exercise_key, self.model_path, self.stop_event)
        self.camera_worker.frameReady.connect(self.update_frame)
        self.camera_worker.messageReady.connect(self.show_message)
        self.camera_worker.finished.connect(self.on_finished)
        self.camera_worker.start()

    def on_finished(self):
        """Restore idle state if backend finished without a final rating."""
        if not self.ratingOverlay.isVisible():
            self.stop_rec_timer()
            self.set_idle_state(clear_state=True)

    def setup_ui(self, exercise_key, model_path, demo_video_path, backend_callable):
        """
        Prepare the UI for the chosen exercise.

        Args:
            exercise_key (str): Exercise identifier.
            model_path (str): Path to the model.
            demo_video_path (str): Path to the demo video.
            backend_callable: Callable used to start the camera worker.
        """
        self.exercise_key = exercise_key
        self.model_path = model_path
        self.demo_video_path = demo_video_path
        self.backend_callable = backend_callable

        self.stopReasonLabel.setVisible(False)
        self.demoContainer.setVisible(False)

        self.set_idle_state(clear_state=True)
        self.guideLabel.setText(EXERCISE_GUIDES.get(exercise_key, "No guide available."))

        self.relayout_overlays()
        QApplication.processEvents()

        self.set_demo_size(demo_video_path, target_h=280, scale=1.0)
        self.player.setLoops(QMediaPlayer.Loops.Infinite)
        self.player.setSource(QUrl.fromLocalFile(demo_video_path))
        self.player.play()
        self.demoContainer.setVisible(True)

        QTimer.singleShot(0, self.relayout_overlays)

    def on_stop_clicked(self):
        """Stop workout."""
        self.stopButton.setEnabled(False)

        self.bottomBar.setText("Stopping...")
        self.bottomBar.setVisible(True)
        self.bottomBar.adjustSize()
        self.stop_rec_timer()

        self.relayout_overlays()

        if self.stop_event and not self.stop_event.is_set():
            self.stop_event.set()

    def show_message(self, kind, text):
        """
        Handle messages coming from the backend 

        Args:
            kind (str): type of message.
            text (str): message text.
        """
        if kind == "status":

            # countdown
            if "Starting in" in text:
                m = re.search(r'(\d+)s', text)
                if m:
                    seconds = m.group(1)
                    self.startOverlay.setText(
                        f"<span style='color:white; font-size:64px'>Hold still!</span><br>"
                    )
        
                    if int(seconds) < 4:
                        self.startOverlay.setText(
                            f"<span style='color:white; font-size:64px'>Starting in</span><br>"
                            f"<span style='font-size:64px; color:#FF6666'>{seconds}</span>"
                        )
                        text_speaker(f"{seconds}")
                else:
                    self.startOverlay.setText(text)

                if not self.startOverlay.isVisible():
                    self.startOverlay.setVisible(True)
                    self.tipsLabel.setVisible(False)
                    self.bottomBar.setVisible(False)

                self.pause_rec_timer()
                self.stopButton.setVisible(True)
                self.stopButton.setEnabled(True)
                self.relayout_overlays()
                return

            if "Recording in progress" in text:
                if self.startOverlay.isVisible():
                    self.startOverlay.setVisible(False)
                
                self.stopButton.setVisible(False)
                self.bottomBar.setText(text)
                self.bottomBar.setVisible(True)

                if not self.rec_running and self.rec_accumulated == 0.0 and self.rec_last_start is None:
                    self.start_rec_timer()  
                else:
                    self.resume_rec_timer()
                
                self.relayout_overlays()
                return

            # Other messages (e.g. instructions for positioning)
            if not self.startOverlay.isVisible():
                self.startOverlay.setVisible(True)
                self.tipsLabel.setVisible(False)
                self.bottomBar.setVisible(False)

            self.pause_rec_timer()
            self.stopButton.setVisible(True)
            self.stopButton.setEnabled(True)
            self.startOverlay.setText(text)
            self.relayout_overlays()
            return

        elif kind == "tip":
            if text:
                if text.lower() not in [t.lower() for t in self.tips_seen]:
                    self.tips_seen.append(text)
                self.show_toast(text, duration_ms=4000)
        
        elif kind == "feedback":
            if text:
                self.show_feedback(text, duration_ms=4000)

        elif kind == "stop_reason":
            self.stop_rec_timer()
            self.stopReasonLabel.setText(text)
            self.stopReasonLabel.adjustSize()
            self.stopReasonLabel.move((self.cameraView.width() - self.stopReasonLabel.width()) // 2, 20)
            self.stopReasonLabel.setVisible(True)
        
        elif kind == "final_rating":
            self.stop_rec_timer()
            stars = int(text)
            
            self.playButton.setVisible(False)
            self.playTitleLabel.setVisible(False)
            self.stopInstrLabel.setVisible(False)
            self.startOverlay.setVisible(False)
            self.tipsLabel.setVisible(False)
            self.tipsLabel.clear()
            self.bottomBar.setVisible(False)
            self.stopButton.setVisible(False)

            self.toastTimer.stop()
            self.toastActive = False
            self.toastQueue.clear()

            self.feedbackTimer.stop()
            self.feedbackPopup.clear()
            self.feedbackPopup.setVisible(False)

            # Show rating overlay
            self.ratingOverlay.setSuggestionsList(self.tips_seen) 
            self.ratingOverlay.setGeometry(self.cameraView.geometry())
            try:
                self.completeSound.play()
            except Exception as e:
                print("[sound] error:", e)
            self.show_rating_overlay(stars)
            if self.stopReasonLabel.isVisible():
                self.stopReasonLabel.raise_()
            self.relayout_overlays()

            self.tips_seen.clear()
            return

    def show_rating_overlay(self, stars):
        """Show the rating overlay with the provided number of stars."""
        self.ratingOverlay.setStars(stars)   
        self.ratingOverlay.setVisible(True)
        self.ratingOverlay.raise_()

    def stop(self, stop_demo=True):
        """
        Stop backend and, eventually, demo video.

        Args:
            stop_demo: If True, also stops the looping demo video.
        """
        if self.stop_event:
            self.stop_event.set()
        if self.camera_worker and self.camera_worker.isRunning():
            self.camera_worker.wait(1500)

        self.cameraView.clear()

        if stop_demo:
            self.player.stop()

    def update_frame(self, frame_bgr):
        """Convert a BGR numpy frame to QImage/QPixmap and paint it into the camera view."""
        rgb = np.require(frame_bgr[..., ::-1], dtype=np.uint8, requirements=["C"])
        h, w, ch = rgb.shape
        bytes_per_line = ch * w

        qimg = QImage(rgb.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888)

        pix = QPixmap.fromImage(qimg)
        self.cameraView.setPixmap(
            pix.scaled(
                self.cameraView.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.relayout_overlays()

    def on_back(self):
        """Return to the main menu: clear UI, stop backend and emit backRequested."""
        self.tipsLabel.clear()
        self.stopReasonLabel.clear()
        self.stop()
        self.backRequested.emit()

    def set_demo_size(self, path, target_h=320, scale=1.35):
        """
        Fix demo video size while preserving its aspect ratio.

        Args:
            path (str): Path to the demo video.
            target_h (int): height used to compute the final size.
            scale (float): Additional scale factor applied to the computed size.
        """
        try:
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                if vw > 0 and vh > 0:
                    ar = vw / vh
                    h = int(target_h * scale)
                    w = int(h * ar)
                    self.demoVideo.setFixedSize(w, h)
                    self.demoContainer.setFixedSize(w, h)
                    return
        except Exception as e:
            print("[demo size] error:", e)

        ar = 16 / 9
        h = int(target_h * scale)
        w = int(h * ar)
        self.demoVideo.setFixedSize(w, h)
        self.demoContainer.setFixedSize(w, h)

class RatingOverlay(QWidget):
    """Rating overlay containing star rating and tips."""
    closed = pyqtSignal()

    def __init__(self, parent=None):
        """Create the rating overlay and its internal card layout."""

        super().__init__(parent)

        # Opaque background overlay
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setObjectName("ratingOverlay")

        # Rating Card
        self.ratingCard = QFrame(self)
        self.ratingCard.setObjectName("ratingPanel")

        v = QVBoxLayout(self.ratingCard)
        v.setContentsMargins(24, 20, 24, 20)
        v.setSpacing(14)
        v.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Title
        self.titleLabel = QLabel("Workout complete", self.ratingCard)
        self.titleLabel.setObjectName("evalTitle")
        self.titleLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self.titleLabel)

        # Stars
        starsRow = QHBoxLayout()
        starsRow.setSpacing(10)
        starsRow.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.star_labels = []
        for _ in range(5):
            label = QLabel("☆", self.ratingCard)
            label.setObjectName("star")  
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.star_labels.append(label)
            starsRow.addWidget(label)
        v.addLayout(starsRow)

        # Tips
        self.tipsLabel = QLabel("", self.ratingCard)
        self.tipsLabel.setObjectName("tips")
        self.tipsLabel.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.tipsLabel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.tipsLabel.setWordWrap(True)
        self.tipsLabel.setTextFormat(Qt.TextFormat.RichText)
        self.tipsLabel.setVisible(False)
        v.addWidget(self.tipsLabel)

        # Close button
        self.closeBtn = QPushButton("✕", self.ratingCard)
        self.closeBtn.setObjectName("closeBtn")
        self.closeBtn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.closeBtn.clicked.connect(self.closed.emit)
        self.closeBtn.raise_()

        self.stars = 0

    def resizeEvent(self, e):
        """Keep the rating card centered on every resize."""
        super().resizeEvent(e)

        w = min(720, int(self.width() * 0.85))
        self.ratingCard.setFixedWidth(w)

        self.ratingCard.adjustSize()
        ph = self.ratingCard.height()
        self.ratingCard.move((self.width() - w)//2, (self.height() - ph)//2)

        self.closeBtn.adjustSize()
        m = 10
        self.closeBtn.move(self.ratingCard.width() - self.closeBtn.width() - m, m)

        content_w = w - 48
        self.tipsLabel.setFixedWidth(content_w)


    def setStars(self, stars):
        """
        Update the star rating and repaint the star labels.

        Args:
            stars: Number of stars to display.
        """
        self.stars = stars
        for i, label in enumerate(self.star_labels):
            label.setText("★" if i < stars else "☆")

    def setSuggestionsList(self, tips):
        """
        Show a list of suggestions as a bullet list, or a positive message if empty.

        Args:
            tips: list of suggestion strings.
        """
        tips = tips or []
        seen = set()
        uniq = []
        for t in tips:
            t = (t or "").strip()
            if not t:
                continue
            key = t.lower()
            if key not in seen:
                seen.add(key)
                uniq.append(t)

        if uniq:
            html = "<div style='margin-top:12px'><b>Suggestions for next time:</b><ul style='margin-top:6px'>"
            for t in uniq:
                html += f"<li style='margin-bottom:5px'>{t}</li>"
            html += "</ul></div>"
            self.tipsLabel.setText(html)
            self.tipsLabel.setVisible(True)
        else:
            self.tipsLabel.setText("<div style='margin-top:8px'><b>Great job!</b> Keep it up!</div>")
            self.tipsLabel.setVisible(True)

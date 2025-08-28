from PyQt6.QtCore import Qt, QUrl, QRectF, QSizeF, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QGraphicsView,
    QGraphicsScene, QFrame, QGraphicsDropShadowEffect
)
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QGraphicsVideoItem
from config import EXERCISE_VIDEOS

class HomeScreen(QWidget):
    """Exercise selection with video preview and description."""
    exerciseConfirmed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Korrecto")

        self.selected_button = None
        self.buttons = []

        # UI label → backend key
        self.exercise_key_map = {
            "Squat - front view": "squat_front",
            "Squat - side view": "squat_side",
            "Push-up": "push_up",
            "Jumping jack": "jumping_jack"
        }
        self.current_exercise_key = None 

        self.exercise_desc = {
            "Squat - front view": (
                "<p style=font-size:18px><b>Squat - front view</b></p>"
                "<i>The squat is a compound exercise that mainly works the quadriceps, glutes, and core. It improves posture and trunk stability.</i><br>"
                "<b><br>Benefits:</b> leg strength, better motor control.<br>"
                "<b><br>Recommended sets:</b> 3×10-12 reps with moderate load.<br>"
                "<b><br>Coach’s focus:</b> correct limb positioning."
            ),
            "Squat - side view": (
                "<p style=font-size:18px><b>Squat - side view</b></p>"
                "<i>The squat is a compound exercise that mainly works the quadriceps, glutes, and core. It improves posture and trunk stability.</i><br>"
                "<b><br>Benefits:</b> leg strength, better motor control.<br>"
                "<b><br>Recommended sets:</b> 3×10-12 reps with moderate load.<br>"
                "<b><br>Coach’s focus:</b> correct limb angle and back alignment."
            ),
            "Push-up": (
                "<p style=font-size:18px><b>Push-up</b></p>"
                "<i>A classic bodyweight exercise targeting the chest, triceps, and anterior deltoids. Improves upper body strength and muscular endurance.</i><br>"
                "<b><br>Benefits:</b> upper body and core development.<br>"
                "<b><br>Recommended sets:</b> 3×12-15 reps.<br>"
                "<b><br>Coach’s focus:</b> correct limb angle and back alignment."
            ),
            "Jumping jack": (
                "<p style=font-size:18px><b>Jumping Jack</b></p>"
                "<i>A dynamic cardio exercise that involves the whole body, ideal for warm-up and cardiovascular endurance.</i><br>"
                "<b><br>Benefits:</b> coordination, endurance, warm-up.<br>"
                "<b><br>Recommended sets:</b> 3×30 seconds.<br>"
                "<b><br>Coach’s focus:</b> correct limb positioning."
            )
        }
        self.default_desc = "<p style=font-size:26px><b>Train with me!</b></p><i>Select an exercise to continue.</i>"
        
        self.default_video = "ui/assets/hello.mp4"

        self.init_ui()
        self.play_video(self.default_video)
        QTimer.singleShot(0, self.update_video_layout)  


    def init_ui(self):
        """UI Setup."""
        main_layout = QHBoxLayout(self)
        main_layout.setObjectName("mainLayout")

        # ------------ 
        # LEFT COLUMN  
        # ------------
        left = QWidget()
        left_layout = QVBoxLayout(left)

        # Title
        self.labelTitle = QLabel("Korrecto")
        self.labelTitle.setObjectName("labelTitle")
        self.labelTitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.labelTitle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.labelTitle.mousePressEvent = self.reset_all

        self.labelChoose = QLabel("Choose an exercise")
        self.labelChoose.setObjectName("labelChoose")
        self.labelChoose.setAlignment(Qt.AlignmentFlag.AlignCenter)

        left_layout.addWidget(self.labelTitle)
        left_layout.addWidget(self.labelChoose)

        left_layout.addSpacing(5)

        # Buttons
        buttons_box = QWidget()
        buttons_layout = QVBoxLayout(buttons_box)
        buttons_layout.setSpacing(25) 
        buttons_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)  

        for name in self.exercise_key_map.keys():
            btn = self.create_button(name)
            buttons_layout.addWidget(btn)
            self.buttons.append(btn)

        left_layout.addStretch(1)     
        left_layout.addWidget(buttons_box)     
        
        left_layout.addSpacing(40)

        self.confirmSlot = QWidget()
        confirm_layout = QVBoxLayout(self.confirmSlot)
        confirm_layout.setContentsMargins(0, 0, 0, 0)
        confirm_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self.buttonConfirm = self.create_button("Confirm")
        confirm_layout.addWidget(self.buttonConfirm, alignment=Qt.AlignmentFlag.AlignHCenter)

        h = self.buttonConfirm.sizeHint().height()
        self.confirmSlot.setMinimumHeight(h)
        self.confirmSlot.setMaximumHeight(h)

        left_layout.addWidget(self.confirmSlot)

        left_layout.addStretch(1)             
        left.setMaximumWidth(350)

        # -------------
        # RIGHT COLUMN  
        # -------------
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.scene = QGraphicsScene()

        self.view = QGraphicsView()
        self.view.setFrameShape(QFrame.Shape.NoFrame)  
        self.view.setStyleSheet("background: transparent;") 
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.installEventFilter(self) 
        self.view.setScene(self.scene)

        # Exercise preview
        self.videoContainer = QFrame()
        self.videoContainer.setObjectName("videoContainer")
        shadow = QGraphicsDropShadowEffect(self.videoContainer)
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 6)
        shadow.setColor(QColor(0, 0, 0, 80))
        self.videoContainer.setGraphicsEffect(shadow)

        vc_layout = QVBoxLayout(self.videoContainer)
        vc_layout.addWidget(self.view)

        self.videoItem = QGraphicsVideoItem()
        self.scene.addItem(self.videoItem)

        self.player = QMediaPlayer()
        self.player.setVideoOutput(self.videoItem)
        self.player.setLoops(-1)

        # Overlay (exercise infos)
        self.overlayDesc = QLabel(self.default_desc)
        self.overlayDesc.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.overlayDesc.setObjectName("overlayCard")
        self.overlayDesc.setWordWrap(True)
        self.overlayDesc.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.proxyDesc = self.scene.addWidget(self.overlayDesc)

        right_layout.addWidget(self.videoContainer)

        # ------------
        # MAIN LAYOUT
        # ------------
        main_layout.addWidget(left, stretch=1)
        main_layout.addWidget(right, stretch=2)


    def update_video_layout(self):
        """Adjust video and overlays according to current viewport size."""
        rect = self.view.viewport().rect()
        self.view.setSceneRect(QRectF(rect))

        avail_w = rect.width()
        avail_h = rect.height()
        margin = 12

        # Video 
        self.videoItem.setPos(0, 0)
        self.videoItem.setSize(QSizeF(avail_w, avail_h))

        # Card description
        self.overlayDesc.setFixedWidth(int(avail_w / 3))  
        self.overlayDesc.adjustSize()
        card_w = self.overlayDesc.width()
        self.proxyDesc.setPos(avail_w - card_w, margin)

    
    def play_video(self, path):
        """Start video from a given source path."""
        self.player.setSource(QUrl.fromLocalFile(path))
        self.player.play()

    
    def create_button(self, name):
        """
        Create a QPushButton for either an exercise or the confirm action.
        
        Args: 
            name: Visible label of the button.
        """
        btn = QPushButton(name)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 3)
        shadow.setColor(QColor(0, 0, 0, 120))
        btn.setGraphicsEffect(shadow)

        if name == "Confirm":
            btn.setVisible(False)
            btn.setProperty("kind", "confirm")
            btn.clicked.connect(self.confirm_selection)
        else:
            btn.setProperty("kind", "exercise")
            btn.clicked.connect(lambda _, t=name, b=btn: self.select_button(b, t))        

        return btn
        
    def refresh_button_style(self, button):
        """Force a style refresh on the given QPushButton."""
        button.style().unpolish(button)
        button.style().polish(button)

    def clear_selection_styles(self):
        """Clear 'selected' state from all exercise buttons."""
        for b in self.buttons:
            b.setProperty("selected", False)
            self.refresh_button_style(b)

    def select_button(self, button, label_text):
        """
        Select an exercise: update style, description and video preview.
        
        Args:
            button (QPushButton): The clicked exercise button.
            label_text (str): The button's visible label.
        """

        self.clear_selection_styles()
        self.selected_button = button
        button.setProperty("selected", True)
        self.refresh_button_style(button)

        self.current_exercise_key = self.exercise_key_map.get(label_text)
        self.overlayDesc.setText(self.exercise_desc.get(label_text, ""))
        self.overlayDesc.adjustSize()

        self.buttonConfirm.setVisible(True)
        self.play_video(EXERCISE_VIDEOS.get(self.current_exercise_key, self.default_video))
        

    def reset_all(self, _event):
        """Reset screen to initial state."""
        self.selected_button = None
        self.clear_selection_styles()
        self.overlayDesc.setText(self.default_desc)
        self.overlayDesc.adjustSize()
        self.buttonConfirm.setVisible(False)
        self.play_video(self.default_video)

    def confirm_selection(self):
        """Emit signal with the selected exercise key (if any)."""
        if self.current_exercise_key:
            self.exerciseConfirmed.emit(self.current_exercise_key)


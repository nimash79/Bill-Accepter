import sys
import asyncio
import threading
from datetime import datetime

import cv2
import serial.tools.list_ports
import serial_asyncio

import serial_number
import template_matching
import fake_detection

from config import IP

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QCheckBox,
    QProgressBar,
    QFrame,
    QSizePolicy,
    QGridLayout,
)
from PyQt5.QtCore import (
    QObject,
    pyqtSignal,
    Qt,
)
from PyQt5.QtGui import (
    QPixmap,
    QImage,
    QFont,
)


# ============================================================
# CONFIG
# ============================================================

# RTSP_URL = f"rtsp://{IP}:8080/h264.sdp"
RTSP_URL = f"http://{IP}:8080/video"


# ============================================================
# LOGGER
# ============================================================

class Logger(QObject):

    log_signal = pyqtSignal(str, str)

    status_signal = pyqtSignal(str, str)

    image_signal = pyqtSignal(object, str)

    progress_signal = pyqtSignal(int)

    step_signal = pyqtSignal(str)

    # Result value can be int, float, str, None, bool, etc.
    result_signal = pyqtSignal(str, object)


logger = Logger()


def log(text, level="INFO"):
    logger.log_signal.emit(str(text), level)


# ============================================================
# SERIAL CLIENT
# ============================================================

class SerialClient(asyncio.Protocol):

    def __init__(self):

        self.transport = None
        self.buffer = asyncio.Queue()
        self.connected = asyncio.Future()

    # --------------------------------------------------------
    # CONNECTION
    # --------------------------------------------------------

    def connection_made(self, transport):

        self.transport = transport

        log(
            "Serial connection established successfully.",
            "SUCCESS"
        )

        if not self.connected.done():
            self.connected.set_result(True)

    # --------------------------------------------------------
    # DATA RECEIVED
    # --------------------------------------------------------

    def data_received(self, data):

        text = data.decode(errors="ignore").strip()

        if not text:
            log(
                f"<< [empty response, {len(data)} bytes]",
                "SERIAL"
            )
            return

        log(
            f"<< {text}",
            "SERIAL"
        )

        self.buffer.put_nowait(text)

    # --------------------------------------------------------
    # SEND
    # --------------------------------------------------------

    def send(self, text):

        if not self.transport:
            raise RuntimeError("Serial not connected")

        msg = (text + "\n").encode()

        self.transport.write(msg)

        log(
            f">> {text}",
            "SERIAL"
        )

    # --------------------------------------------------------
    # ASK
    # --------------------------------------------------------

    async def ask(self, text, timeout=30):

        await self.connected

        self.send(text)

        try:

            response = await asyncio.wait_for(
                self.buffer.get(),
                timeout
            )

            return response

        except asyncio.TimeoutError:

            log(
                f"No response received for command: {text}",
                "WARNING"
            )

            return None


# ============================================================
# CAMERA
# ============================================================

async def take_picture():

    loop = asyncio.get_event_loop()

    def capture():

        log(
            f"Connecting to camera: {RTSP_URL}",
            "INFO"
        )

        cap = cv2.VideoCapture(RTSP_URL)

        if not cap.isOpened():

            log(
                "Unable to open RTSP camera.",
                "ERROR"
            )

            return None

        frame = None
        ret = False

        log(
            "Waiting for camera frames...",
            "INFO"
        )

        for _ in range(30):

            ret, frame = cap.read()

            if ret:
                break

        if not ret or frame is None:

            log(
                "Camera did not provide a valid frame.",
                "ERROR"
            )

            cap.release()

            return None

        # ----------------------------------------------------
        # ROTATE
        # ----------------------------------------------------

        frame = cv2.rotate(
            frame,
            cv2.ROTATE_180
        )

        # ----------------------------------------------------
        # CROP
        # ----------------------------------------------------

        h, w = frame.shape[:2]

        x1 = 40
        x2 = w

        y1 = 50
        y2 = h - 230

        cropped = frame[
            y1:y2,
            x1:x2
        ]

        cap.release()

        log(
            f"Image captured successfully: {cropped.shape[1]}x{cropped.shape[0]}",
            "SUCCESS"
        )

        return cropped

    return await loop.run_in_executor(
        None,
        capture
    )


# ============================================================
# STATUS CARD
# ============================================================

class StatusCard(QFrame):

    def __init__(self, title, value="---"):

        super().__init__()

        self.setObjectName("StatusCard")

        layout = QVBoxLayout()

        layout.setContentsMargins(
            15,
            10,
            15,
            10
        )

        self.title_label = QLabel(title)

        self.title_label.setObjectName(
            "CardTitle"
        )

        self.value_label = QLabel(value)

        self.value_label.setObjectName(
            "CardValue"
        )

        self.value_label.setAlignment(
            Qt.AlignCenter
        )

        layout.addWidget(
            self.title_label
        )

        layout.addWidget(
            self.value_label
        )

        self.setLayout(layout)

    def set_value(self, value):

        self.value_label.setText(
            str(value)
        )


# ============================================================
# MAIN WINDOW
# ============================================================

class MainWindow(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "Banknote Inspection System"
        )

        self.resize(
            1400,
            850
        )

        self.serial_client = None
        self.connected = False
        self.processing = False

        self.loop = asyncio.new_event_loop()

        # ----------------------------------------------------
        # ASYNC LOOP THREAD
        # ----------------------------------------------------

        threading.Thread(
            target=self.start_loop,
            daemon=True
        ).start()

        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        self.build_ui()
        self.apply_styles()

        # ----------------------------------------------------
        # SIGNALS
        # ----------------------------------------------------

        logger.log_signal.connect(
            self.add_log
        )

        logger.status_signal.connect(
            self.set_status
        )

        logger.image_signal.connect(
            self.show_image
        )

        logger.progress_signal.connect(
            self.set_progress
        )

        logger.step_signal.connect(
            self.set_step
        )

        logger.result_signal.connect(
            self.set_result
        )

    # ========================================================
    # UI
    # ========================================================

    def build_ui(self):

        main_layout = QVBoxLayout()

        main_layout.setSpacing(12)

        # ====================================================
        # HEADER
        # ====================================================

        header = QHBoxLayout()

        title = QLabel(
            "BANKNOTE INSPECTION SYSTEM"
        )

        title.setObjectName(
            "MainTitle"
        )

        subtitle = QLabel(
            "Automated Serial Number & Authenticity Inspection"
        )

        subtitle.setObjectName(
            "Subtitle"
        )

        title_container = QVBoxLayout()

        title_container.addWidget(title)
        title_container.addWidget(subtitle)

        header.addLayout(
            title_container
        )

        header.addStretch()

        self.status_label = QLabel(
            "● Disconnected"
        )

        self.status_label.setObjectName(
            "DisconnectedStatus"
        )

        header.addWidget(
            self.status_label
        )

        main_layout.addLayout(
            header
        )

        # ====================================================
        # CONTROL BAR
        # ====================================================

        control_frame = QFrame()

        control_frame.setObjectName(
            "ControlFrame"
        )

        control_layout = QHBoxLayout()

        self.connect_btn = QPushButton(
            "🔌 Connect"
        )

        self.connect_btn.clicked.connect(
            self.connect_serial
        )

        self.start_btn = QPushButton(
            "▶ Start Inspection"
        )

        self.start_btn.setEnabled(
            False
        )

        self.start_btn.clicked.connect(
            self.start_process
        )

        self.stop_btn = QPushButton(
            "■ Stop"
        )

        self.stop_btn.setEnabled(
            False
        )

        self.stop_btn.clicked.connect(
            self.stop_process
        )

        self.clear_btn = QPushButton(
            "🗑 Clear Logs"
        )

        self.clear_btn.clicked.connect(
            self.clear_logs
        )

        control_layout.addWidget(
            self.connect_btn
        )

        control_layout.addWidget(
            self.start_btn
        )

        control_layout.addWidget(
            self.stop_btn
        )

        control_layout.addWidget(
            self.clear_btn
        )

        control_layout.addStretch()

        control_frame.setLayout(
            control_layout
        )

        main_layout.addWidget(
            control_frame
        )

        # ====================================================
        # PROGRESS
        # ====================================================

        progress_layout = QVBoxLayout()

        self.step_label = QLabel(
            "Ready"
        )

        self.step_label.setObjectName(
            "StepLabel"
        )

        self.progress_bar = QProgressBar()

        self.progress_bar.setRange(
            0,
            100
        )

        self.progress_bar.setValue(
            0
        )

        progress_layout.addWidget(
            self.step_label
        )

        progress_layout.addWidget(
            self.progress_bar
        )

        main_layout.addLayout(
            progress_layout
        )

        # ====================================================
        # RESULT CARDS
        # ====================================================

        result_layout = QGridLayout()

        self.value_card = StatusCard(
            "BANKNOTE VALUE"
        )

        self.angle_card = StatusCard(
            "ANGLE"
        )

        self.score_card = StatusCard(
            "MATCH SCORE"
        )

        self.serial_card = StatusCard(
            "SERIAL NUMBER"
        )

        self.real_card = StatusCard(
            "AUTHENTICITY"
        )

        result_layout.addWidget(
            self.value_card,
            0,
            0
        )

        result_layout.addWidget(
            self.angle_card,
            0,
            1
        )

        result_layout.addWidget(
            self.score_card,
            0,
            2
        )

        result_layout.addWidget(
            self.serial_card,
            0,
            3
        )

        result_layout.addWidget(
            self.real_card,
            0,
            4
        )

        main_layout.addLayout(
            result_layout
        )

        # ====================================================
        # IMAGE AREA
        # ====================================================

        image_layout = QHBoxLayout()

        # ----------------------------------------------------
        # FIRST IMAGE
        # ----------------------------------------------------

        image_frame_1 = QFrame()

        image_frame_1.setObjectName(
            "ImageFrame"
        )

        image_layout_1 = QVBoxLayout()

        image_title_1 = QLabel(
            "📷 Captured Image"
        )

        image_title_1.setObjectName(
            "ImageTitle"
        )

        self.image_label_1 = QLabel(
            "Waiting for image..."
        )

        self.image_label_1.setAlignment(
            Qt.AlignCenter
        )

        self.image_label_1.setMinimumSize(
            450,
            280
        )

        self.image_label_1.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

        image_layout_1.addWidget(
            image_title_1
        )

        image_layout_1.addWidget(
            self.image_label_1
        )

        image_frame_1.setLayout(
            image_layout_1
        )

        # ----------------------------------------------------
        # SECOND IMAGE
        # ----------------------------------------------------

        image_frame_2 = QFrame()

        image_frame_2.setObjectName(
            "ImageFrame"
        )

        image_layout_2 = QVBoxLayout()

        image_title_2 = QLabel(
            "📷 Authenticity Comparison"
        )

        image_title_2.setObjectName(
            "ImageTitle"
        )

        self.image_label_2 = QLabel(
            "Second image will appear here..."
        )

        self.image_label_2.setAlignment(
            Qt.AlignCenter
        )

        self.image_label_2.setMinimumSize(
            450,
            280
        )

        self.image_label_2.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

        image_layout_2.addWidget(
            image_title_2
        )

        image_layout_2.addWidget(
            self.image_label_2
        )

        image_frame_2.setLayout(
            image_layout_2
        )

        image_layout.addWidget(
            image_frame_1
        )

        image_layout.addWidget(
            image_frame_2
        )

        main_layout.addLayout(
            image_layout,
            stretch=2
        )

        # ====================================================
        # LOG
        # ====================================================

        log_title = QLabel(
            "Execution Log"
        )

        log_title.setObjectName(
            "SectionTitle"
        )

        main_layout.addWidget(
            log_title
        )

        self.log_box = QTextEdit()

        self.log_box.setReadOnly(
            True
        )

        self.log_box.setMinimumHeight(
            180
        )

        main_layout.addWidget(
            self.log_box
        )

        self.setLayout(
            main_layout
        )

    # ========================================================
    # STYLE
    # ========================================================

    def apply_styles(self):

        self.setStyleSheet(
            """
            QWidget {
                background-color: #10151c;
                color: #e8edf2;
                font-family: Segoe UI;
                font-size: 13px;
            }

            #MainTitle {
                font-size: 25px;
                font-weight: bold;
                color: #ffffff;
            }

            #Subtitle {
                color: #8c99a8;
                font-size: 12px;
            }

            #ControlFrame {
                background-color: #181f28;
                border: 1px solid #2a3542;
                border-radius: 10px;
                padding: 5px;
            }

            QPushButton {
                background-color: #273442;
                border: 1px solid #394958;
                border-radius: 7px;
                padding: 9px 16px;
                color: #ffffff;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #344454;
            }

            QPushButton:pressed {
                background-color: #202a34;
            }

            QPushButton:disabled {
                background-color: #1b222b;
                color: #66717c;
                border-color: #242c35;
            }

            #DisconnectedStatus {
                color: #ff6b6b;
                font-weight: bold;
                font-size: 14px;
            }

            #ConnectedStatus {
                color: #51cf66;
                font-weight: bold;
                font-size: 14px;
            }

            #ProcessingStatus {
                color: #339af0;
                font-weight: bold;
                font-size: 14px;
            }

            #StepLabel {
                color: #74c0fc;
                font-size: 14px;
                font-weight: bold;
            }

            QProgressBar {
                border: 1px solid #2d3945;
                border-radius: 6px;
                background-color: #171d24;
                height: 12px;
                text-align: center;
            }

            QProgressBar::chunk {
                background-color: #339af0;
                border-radius: 5px;
            }

            #StatusCard {
                background-color: #181f28;
                border: 1px solid #2a3542;
                border-radius: 10px;
                min-height: 75px;
            }

            #CardTitle {
                color: #82909f;
                font-size: 10px;
                font-weight: bold;
            }

            #CardValue {
                color: #ffffff;
                font-size: 18px;
                font-weight: bold;
            }

            #ImageFrame {
                background-color: #181f28;
                border: 1px solid #2a3542;
                border-radius: 10px;
            }

            #ImageTitle {
                color: #adb8c3;
                font-weight: bold;
                font-size: 13px;
            }

            QLabel {
                background: transparent;
            }

            QTextEdit {
                background-color: #0b0f14;
                border: 1px solid #2a3542;
                border-radius: 8px;
                padding: 8px;
                color: #cdd6df;
                font-family: Consolas;
                font-size: 12px;
            }

            #SectionTitle {
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
            }

            QCheckBox {
                color: #b8c2cc;
                spacing: 6px;
            }
            """
        )

    # ========================================================
    # ASYNC LOOP
    # ========================================================

    def start_loop(self):

        asyncio.set_event_loop(
            self.loop
        )

        self.loop.run_forever()

    # ========================================================
    # LOGGING
    # ========================================================

    def add_log(self, text, level):

        timestamp = datetime.now().strftime(
            "%H:%M:%S"
        )

        colors = {
            "INFO": "#adb5bd",
            "SUCCESS": "#69db7c",
            "WARNING": "#ffd43b",
            "ERROR": "#ff6b6b",
            "SERIAL": "#74c0fc",
        }

        color = colors.get(
            level,
            "#adb5bd"
        )

        html = (
            f'<span style="color:#6c757d;">'
            f'[{timestamp}]'
            f'</span> '
            f'<span style="color:{color};">'
            f'[{level}]'
            f'</span> '
            f'<span style="color:#dfe6ed;">'
            f'{text}'
            f'</span>'
        )

        self.log_box.append(
            html
        )

    def clear_logs(self):

        self.log_box.clear()

        log(
            "Log cleared.",
            "INFO"
        )

    # ========================================================
    # STATUS
    # ========================================================

    def set_status(self, text, status_type):

        self.status_label.setText(
            f"● {text}"
        )

        self.status_label.setObjectName(
            status_type
        )

        self.status_label.style().unpolish(
            self.status_label
        )

        self.status_label.style().polish(
            self.status_label
        )

    # ========================================================
    # PROGRESS
    # ========================================================

    def set_progress(self, value):

        self.progress_bar.setValue(
            value
        )

    # ========================================================
    # STEP
    # ========================================================

    def set_step(self, text):

        self.step_label.setText(
            text
        )

    # ========================================================
    # IMAGE
    # ========================================================

    def show_image(self, image, target):

        if image is None:
            return

        if len(image.shape) == 2:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2RGB
            )

        else:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

        h, w, ch = image.shape

        bytes_per_line = ch * w

        q_image = QImage(
            image.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888
        )

        pixmap = QPixmap.fromImage(
            q_image
        )

        if target == "first":

            pixmap = pixmap.scaled(
                self.image_label_1.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            self.image_label_1.setPixmap(
                pixmap
            )

        elif target == "second":

            pixmap = pixmap.scaled(
                self.image_label_2.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            self.image_label_2.setPixmap(
                pixmap
            )

    # ========================================================
    # RESULTS
    # ========================================================

    def set_result(self, key, value):

        if key == "value":

            if value is None:
                self.value_card.set_value("---")
            else:
                self.value_card.set_value(f"${value}")

        elif key == "angle":

            if value is None:
                self.angle_card.set_value("---")
            else:
                self.angle_card.set_value(f"{value}°")

        elif key == "score":

            if value is None:
                self.score_card.set_value("---")
            else:
                self.score_card.set_value(f"{float(value):.4f}")

        elif key == "serial":

            if value is None:
                self.serial_card.set_value("---")
            else:
                self.serial_card.set_value(str(value))

        elif key == "real":

            if value is None:
                self.real_card.set_value("---")
            else:
                self.real_card.set_value(str(value))

    # ========================================================
    # CONNECT
    # ========================================================

    def connect_serial(self):

        if self.connected:

            log(
                "Already connected to serial device.",
                "WARNING"
            )

            return

        self.connect_btn.setEnabled(
            False
        )

        self.set_status(
            "Connecting...",
            "ProcessingStatus"
        )

        log(
            "Searching for serial ports...",
            "INFO"
        )

        asyncio.run_coroutine_threadsafe(
            self.async_connect(),
            self.loop
        )

    async def async_connect(self):

        try:

            ports = serial.tools.list_ports.comports()

            if not ports:

                log(
                    "No serial ports found.",
                    "ERROR"
                )

                logger.status_signal.emit(
                    "Disconnected",
                    "DisconnectedStatus"
                )

                return

            # ------------------------------------------------
            # SHOW AVAILABLE PORTS
            # ------------------------------------------------

            for p in ports:

                log(
                    f"Available port: {p.device}",
                    "INFO"
                )

            port = ports[-1].device

            log(
                f"Connecting to {port} at 115200 baud...",
                "INFO"
            )

            transport, protocol = await serial_asyncio.create_serial_connection(
                self.loop,
                SerialClient,
                port,
                baudrate=115200
            )

            self.serial_client = protocol

            await self.serial_client.connected

            self.connected = True

            logger.status_signal.emit(
                "Connected",
                "ConnectedStatus"
            )

            logger.status_signal.emit(
                "Connected",
                "ConnectedStatus"
            )

            logger.log_signal.emit(
                f"Serial device connected on {port}.",
                "SUCCESS"
            )

            self.start_btn.setEnabled(
                True
            )

        except Exception as e:

            log(
                f"Serial connection failed: {e}",
                "ERROR"
            )

            self.connected = False

            logger.status_signal.emit(
                "Disconnected",
                "DisconnectedStatus"
            )

        finally:

            self.connect_btn.setEnabled(
                not self.connected
            )

    # ========================================================
    # START PROCESS
    # ========================================================

    def start_process(self):

        if not self.connected:

            log(
                "Serial device is not connected.",
                "ERROR"
            )

            return

        if self.processing:

            log(
                "Inspection is already running.",
                "WARNING"
            )

            return

        self.processing = True

        self.start_btn.setEnabled(
            False
        )

        self.connect_btn.setEnabled(
            False
        )

        self.stop_btn.setEnabled(
            True
        )

        self.progress_bar.setValue(
            0
        )

        self.reset_results()

        logger.status_signal.emit(
            "Inspection Running",
            "ProcessingStatus"
        )

        asyncio.run_coroutine_threadsafe(
            self.run_process(),
            self.loop
        )

    # ========================================================
    # RESET RESULTS
    # ========================================================

    def reset_results(self):

        self.value_card.set_value(
            "---"
        )

        self.angle_card.set_value(
            "---"
        )

        self.score_card.set_value(
            "---"
        )

        self.serial_card.set_value(
            "---"
        )

        self.real_card.set_value(
            "---"
        )

        self.image_label_1.clear()

        self.image_label_1.setText(
            "Waiting for image..."
        )

        self.image_label_2.clear()

        self.image_label_2.setText(
            "Second image will appear here..."
        )

    # ========================================================
    # STOP
    # ========================================================

    def stop_process(self):

        if not self.processing:
            return

        log(
            "Stop requested by user.",
            "WARNING"
        )

        self.processing = False

        self.stop_btn.setEnabled(
            False
        )

        self.start_btn.setEnabled(
            self.connected
        )

        logger.status_signal.emit(
            "Connected",
            "ConnectedStatus"
        )

        logger.step_signal.emit(
            "Process stopped by user."
        )

    # ========================================================
    # MAIN PROCESS
    # ========================================================

    async def run_process(self):


        ser = self.serial_client

        try:

            # =================================================
            # START
            # =================================================

            log(
                "==========================================",
                "INFO"
            )

            log(
                "Inspection process started.",
                "SUCCESS"
            )

            # =================================================
            # STEP 1
            # =================================================

            logger.step_signal.emit(
                "Step 1/10 — Initializing inspection..."
            )

            logger.progress_signal.emit(
                5
            )

            await asyncio.sleep(2)

            if not self.processing:
                return

            # =================================================
            # HOME
            # =================================================

            log(
                "Moving mechanism to HOME position...",
                "INFO"
            )

            await ser.ask(
                "home"
            )

            # =================================================
            # GET
            # =================================================

            log(
                "Reading device state...",
                "INFO"
            )

            await ser.ask(
                "get"
            )

            await ser.ask(
                "get"
            )

            # =================================================
            # LED
            # =================================================

            logger.step_signal.emit(
                "Step 2/10 — Turning on illumination..."
            )

            logger.progress_signal.emit(
                15
            )

            log(
                "Turning on inspection LEDs.",
                "INFO"
            )

            await ser.ask(
                "led=1,127;"
            )

            await ser.ask(
                "led=2,127;"
            )

            await ser.ask(
                "led=3,127;"
            )

            await ser.ask(
                "led=4,127;"
            )

            # =================================================
            # MOVE
            # =================================================

            logger.step_signal.emit(
                "Step 3/10 — Positioning banknote..."
            )

            logger.progress_signal.emit(
                25
            )

            log(
                "Moving mechanism to inspection position.",
                "INFO"
            )

            await ser.ask(
                "move=2000;"
            )

            # =================================================
            # FIRST IMAGE
            # =================================================

            logger.step_signal.emit(
                "Step 4/10 — Capturing first image..."
            )

            logger.progress_signal.emit(
                35
            )

            log(
                "Capturing first banknote image.",
                "INFO"
            )

            cropped = await take_picture()

            if cropped is None:

                log(
                    "Camera capture failed.",
                    "ERROR"
                )

                return

            cv2.imwrite(
                "cropped1.jpg",
                cropped
            )

            logger.image_signal.emit(
                cropped,
                "first"
            )

            log(
                "First image displayed in inspection panel.",
                "SUCCESS"
            )

            # =================================================
            # BACK SIDE DETECTION
            # =================================================

            logger.step_signal.emit(
                "Step 5/10 — Checking banknote orientation..."
            )

            logger.progress_signal.emit(
                45
            )

            log(
                "Running back-side detection...",
                "INFO"
            )

            is_back = serial_number.is_back_side(
                cropped
            )

            log(
                f"Back-side detection result: {is_back}",
                "INFO"
            )

            if is_back:

                log(
                    "Banknote is on the back side.",
                    "WARNING"
                )

                await ser.ask(
                    "back"
                )

                return

            # =================================================
            # TEMPLATE MATCHING
            # =================================================

            logger.step_signal.emit(
                "Step 6/10 — Identifying banknote..."
            )

            logger.progress_signal.emit(
                55
            )

            threshold = 0.2

            log(
                f"Running template matching (threshold={threshold}).",
                "INFO"
            )

            value, angle, score, H = template_matching.algorithm_with_homography(
                cropped,
                threshold
            )

            # -------------------------------------------------
            # RESULT
            # -------------------------------------------------

            logger.result_signal.emit(
                "value",
                value if value is not None else "---"
            )

            logger.result_signal.emit(
                "angle",
                angle if angle is not None else "---"
            )

            logger.result_signal.emit(
                "score",
                score
            )

            log(
                f"Banknote value: ${value}",
                "SUCCESS"
            )

            log(
                f"Detected angle: {angle}°",
                "INFO"
            )

            log(
                f"Template matching score: {score}",
                "INFO"
            )

            if value is None:

                log(
                    "No valid banknote match found.",
                    "ERROR"
                )

                await ser.ask(
                    "back"
                )

                return

            # =================================================
            # ROTATION
            # =================================================

            logger.step_signal.emit(
                "Step 7/10 — Correcting image orientation..."
            )

            logger.progress_signal.emit(
                65
            )

            if angle == 180:

                log(
                    "Image is rotated by 180°. Correcting orientation.",
                    "WARNING"
                )

                cropped = cv2.rotate(
                    cropped,
                    cv2.ROTATE_180
                )

                cv2.imwrite(
                    "cropped1.jpg",
                    cropped
                )

                logger.image_signal.emit(
                    cropped,
                    "first"
                )

                log(
                    "Image orientation corrected.",
                    "SUCCESS"
                )

            # =================================================
            # SERIAL NUMBER
            # =================================================

            logger.step_signal.emit(
                "Step 8/10 — Extracting serial number..."
            )

            logger.progress_signal.emit(
                75
            )

            position = (
                "left"
                if value == 100
                else "right"
            )

            log(
                f"Serial number region: {position}",
                "INFO"
            )

            extracted_serial_number = (
                serial_number.extract_serial(
                    cropped,
                    value,
                    position,
                    H
                )
            )

            logger.result_signal.emit(
                "serial",
                extracted_serial_number
            )

            log(
                f"Serial Number: {extracted_serial_number}",
                "SUCCESS"
            )

            # =================================================
            # AUTHENTICITY
            # =================================================

            # Default assumption:
            # The banknote will be rejected unless we explicitly
            # determine that it is acceptable.
            accepted = False


            # -------------------------------------------------
            # Check whether denomination was successfully detected
            # -------------------------------------------------

            if value is not None:

                if value == 100:

                    log(
                        "Turning off inspection LEDs.",
                        "INFO"
                    )

                    await ser.ask(
                        "led=1,0;"
                    )

                    if angle == 0:
                        await ser.ask(
                            "led=2,0;"
                        )
                    else:
                        await ser.ask(
                            "led=3,0;"
                        )

                    await ser.ask(
                        "led=4,0;"
                    )

                    logger.step_signal.emit(
                        "Step 9/10 — Running authenticity detection..."
                    )

                    logger.progress_signal.emit(
                        85
                    )

                    log(
                        "Capturing second image for authenticity analysis.",
                        "INFO"
                    )

                    cropped2 = await take_picture()

                    if cropped2 is not None:

                        if angle == 180:
                            cropped2 = cv2.rotate(
                                cropped2,
                                cv2.ROTATE_180
                            )

                        cv2.imwrite(
                            "cropped2.jpg",
                            cropped2
                        )

                        logger.image_signal.emit(
                            cropped2,
                            "second"
                        )

                        log(
                            "Second image captured.",
                            "SUCCESS"
                        )

                        real = fake_detection.check_color_shift(
                            cropped,
                            cropped2
                        )

                        logger.result_signal.emit(
                            "real",
                            real
                        )

                        log(
                            f"Authenticity result: {real}",
                            "SUCCESS"
                        )

                        # -----------------------------------------
                        # $100 acceptance decision
                        # -----------------------------------------

                        if real is True:
                            accepted = True

                            log(
                                "$100 banknote accepted: authenticity check passed.",
                                "SUCCESS"
                            )

                        else:
                            accepted = False

                            log(
                                "$100 banknote rejected: authenticity check failed.",
                                "WARNING"
                            )

                    else:

                        log(
                            "Second camera capture failed. $100 banknote rejected.",
                            "ERROR"
                        )

                        logger.result_signal.emit(
                            "real",
                            "UNKNOWN"
                        )

                        accepted = False

                else:

                    # ---------------------------------------------
                    # Non-$100 banknotes
                    # ---------------------------------------------

                    accepted = True

                    logger.result_signal.emit(
                        "real",
                        "N/A"
                    )

                    log(
                        f"Authenticity check skipped because banknote value is ${value}.",
                        "INFO"
                    )

            else:

                # ---------------------------------------------
                # Denomination detection failed
                # ---------------------------------------------

                accepted = False

                logger.result_signal.emit(
                    "real",
                    "UNKNOWN"
                )

                log(
                    "Banknote rejected because denomination could not be detected.",
                    "WARNING"
                )


            # =================================================
            # RETURN / ACCEPT OR REJECT
            # =================================================

            logger.step_signal.emit(
                "Step 10/10 — Processing banknote..."
            )

            logger.progress_signal.emit(
                95
            )


            if accepted:

                log(
                    "Banknote accepted. Moving mechanism forward.",
                    "SUCCESS"
                )

                await ser.ask(
                    "forward"
                )

            else:

                log(
                    "Banknote rejected. Returning mechanism backward.",
                    "WARNING"
                )

                await ser.ask(
                    "back"
                )


            # =================================================
            # FINISHED
            # =================================================

            logger.progress_signal.emit(
                100
            )

            logger.step_signal.emit(
                "Inspection completed successfully."
            )

            log(
                "Inspection process finished successfully.",
                "SUCCESS"
            )

            log(
                "==========================================",
                "INFO"
            )

        except Exception as e:

            log(
                f"Unexpected error during inspection: {e}",
                "ERROR"
            )

            logger.step_signal.emit(
                "Inspection failed."
            )

        finally:

            self.processing = False

            self.stop_btn.setEnabled(
                False
            )

            self.start_btn.setEnabled(
                self.connected
            )

            self.connect_btn.setEnabled(
                not self.connected
            )

            if self.connected:

                logger.status_signal.emit(
                    "Connected",
                    "ConnectedStatus"
                )

            else:

                logger.status_signal.emit(
                    "Disconnected",
                    "DisconnectedStatus"
                )

    # ========================================================
    # CLOSE EVENT
    # ========================================================

    def closeEvent(self, event):

        log(
            "Closing application...",
            "INFO"
        )

        try:

            if self.loop.is_running():

                self.loop.call_soon_threadsafe(
                    self.loop.stop
                )

        except Exception:
            pass

        event.accept()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    app = QApplication(
        sys.argv
    )

    window = MainWindow()

    window.show()

    sys.exit(
        app.exec_()
    )
import sys
import asyncio
import threading

import serial_number
import template_matching
import fake_detection_old

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QTextEdit,
    QVBoxLayout, QHBoxLayout, QLabel, QCheckBox
)
from PyQt5.QtCore import QObject, pyqtSignal

import serial.tools.list_ports
import serial_asyncio
import cv2


from config import IP

RTSP_URL = f"rtsp://{IP}:8080/h264.sdp"


# =========================
# LOGGER SIGNALS
# =========================
class Logger(QObject):
    log_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)


logger = Logger()


def log(text):
    logger.log_signal.emit(text)


# =========================
# SERIAL CLIENT
# =========================
class SerialClient(asyncio.Protocol):

    def __init__(self):
        self.transport = None
        self.buffer = asyncio.Queue()
        self.connected = asyncio.Future()

    def connection_made(self, transport):
        self.transport = transport
        log("✅ Connected to serial")

        if not self.connected.done():
            self.connected.set_result(True)

    def data_received(self, data):
        text = data.decode(errors="ignore").strip()
        log(f"<< {text}")
        self.buffer.put_nowait(text)

    def send(self, text):

        if not self.transport:
            raise RuntimeError("Serial not connected")

        msg = (text + "\n").encode()

        self.transport.write(msg)

        log(f">> {text}")

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
            log("⚠ No response received")
            return None


# =========================
# CAMERA
# =========================
async def take_picture():

    loop = asyncio.get_event_loop()

    def capture():
        log("RTSP_URL:", RTSP_URL)
        cap = cv2.VideoCapture(RTSP_URL)

        frame = None

        for _ in range(30):
            ret, frame = cap.read()

        if not ret:
            cap.release()
            return None

        frame = cv2.rotate(frame, cv2.ROTATE_180)

        h, w = frame.shape[:2]
        x1 = 40
        x2 = w
        y1 = 50
        y2 = h - 230
        cropped = frame[y1:y2, x1:x2]


        cap.release()

        return cropped

    return await loop.run_in_executor(None, capture)


# =========================
# MAIN WINDOW
# =========================
class MainWindow(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("Inspection System")
        self.resize(800, 500)

        self.serial_client = None
        self.connected = False

        self.loop = asyncio.new_event_loop()

        threading.Thread(
            target=self.start_loop,
            daemon=True
        ).start()

        self.build_ui()

        logger.log_signal.connect(self.add_log)
        logger.status_signal.connect(self.set_status)

    # =========================
    # UI
    # =========================
    def build_ui(self):

        layout = QVBoxLayout()

        top = QHBoxLayout()

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_serial)

        self.start_btn = QPushButton("Start")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_process)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_logs)

        self.debug_checkbox = QCheckBox("Debug mode")

        self.status_label = QLabel("Disconnected")


        top.addWidget(self.connect_btn)
        top.addWidget(self.start_btn)
        top.addWidget(self.clear_btn)
        top.addWidget(self.debug_checkbox)
        top.addStretch()
        top.addWidget(self.status_label)




        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        layout.addLayout(top)
        layout.addWidget(self.log_box)

        self.setLayout(layout)

    # =========================
    # ASYNC LOOP
    # =========================
    def start_loop(self):

        asyncio.set_event_loop(self.loop)

        self.loop.run_forever()

    # =========================
    # LOGGING
    # =========================
    def add_log(self, text):
        self.log_box.append(text)

    def clear_logs(self):
        self.log_box.clear()

    def set_status(self, text):

        self.status_label.setText(text)

    # =========================
    # CONNECT
    # =========================
    def connect_serial(self):

        if self.connected:
            log("Already connected")
            return

        asyncio.run_coroutine_threadsafe(
            self.async_connect(),
            self.loop
        )

    async def async_connect(self):

        ports = serial.tools.list_ports.comports()

        if not ports:
            log("❌ No serial ports found")
            return

        port = ports[-1].device

        log(f"Connecting to {port} ...")

        transport, protocol = await serial_asyncio.create_serial_connection(
            self.loop,
            SerialClient,
            port,
            baudrate=115200
        )

        self.serial_client = protocol

        await self.serial_client.connected

        self.connected = True

        logger.status_signal.emit("Connected")

        self.start_btn.setEnabled(True)

    # =========================
    # START PROCESS
    # =========================
    def start_process(self):

        asyncio.run_coroutine_threadsafe(
            self.run_process(),
            self.loop
        )

    async def run_process(self):

        debug = self.debug_checkbox.isChecked()

        log("================================")
        log("🚀 Process started")

        if debug:
            log("Debug mode ON")

        ser = self.serial_client

        await asyncio.sleep(2)

        await ser.ask("home")
        await ser.ask("get")
        await ser.ask("get")

        await ser.ask("led=1,127;")
        await ser.ask("led=2,127;")
        await ser.ask("led=3,127;")
        await ser.ask("led=4,127;")

        await ser.ask("move=1800;")

        log("📷 Taking first picture")

        cropped = await take_picture()

        if cropped is None:
            log("Camera failed")
            return

        cv2.imwrite("cropped1.jpg", cropped)

        if (serial_number.is_back_side(cropped) == True):
            log("The banknote is on the back side!!!")
            await ser.ask("back")
            return

        threshold = 0.3
        value, angle, score = template_matching.algorithm(cropped, threshold)

        log(f"Value: ${value}")
        log(f"Angle: {angle}°")
        log(f"Score: {score}")

        if (value == None):
            await ser.ask("back")

        # if score < threshold:
        #     log(f"NO MATCH FOUND (Best score {score:.4f} < threshold {threshold})")
        #     await ser.ask("back")
        #     return

        if angle == 180:
            cropped = cv2.rotate(cropped, cv2.ROTATE_180)
            cv2.imwrite("cropped1.jpg", cropped)


        position = "left" if value == 100 else "right"

        extracted_serial_number = serial_number.extract_serial(
            cropped,
            value,
            position
        )
        

        log(f"Serial Number: {extracted_serial_number}")

        await ser.ask("led=1,0;")
        await ser.ask("led=2,0;")
        await ser.ask("led=4,0;")

        if value == 100:
            log("📷 Taking second picture")
            cropped2 = await take_picture()
            cv2.imwrite("cropped2.jpg", cropped2)
            real = fake_detection_old.check_color_shift(
                cropped,
                cropped2
            )
            log(f"REAL? {real}")

        await ser.ask("back")

        log("✅ Process finished")


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())

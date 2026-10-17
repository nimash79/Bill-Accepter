import sys
import serial
import serial.tools.list_ports
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

class SerialUI(QWidget):

    def __init__(self):
        super().__init__()

        self.serial = None

        self.setWindowTitle("Serial Controller")
        self.resize(600,500)

        layout = QVBoxLayout()

        # PORT SELECTION
        port_layout = QHBoxLayout()
        self.port_box = QComboBox()
        self.refresh_ports()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_ports)

        connect_btn = QPushButton("Connect")
        connect_btn.clicked.connect(self.connect_serial)

        port_layout.addWidget(self.port_box)
        port_layout.addWidget(refresh_btn)
        port_layout.addWidget(connect_btn)

        layout.addLayout(port_layout)

        # BASIC COMMANDS
        cmd_layout = QHBoxLayout()

        home_btn = QPushButton("Home")
        home_btn.clicked.connect(lambda: self.send("home"))

        get_btn = QPushButton("Get")
        get_btn.clicked.connect(lambda: self.send("get"))

        back_btn = QPushButton("Back")
        back_btn.clicked.connect(lambda: self.send("back"))

        forward_btn = QPushButton("Forward")
        forward_btn.clicked.connect(lambda: self.send("forward"))

        cmd_layout.addWidget(home_btn)
        cmd_layout.addWidget(get_btn)
        cmd_layout.addWidget(back_btn)
        cmd_layout.addWidget(forward_btn)

        layout.addLayout(cmd_layout)

        # MOVE
        move_layout = QHBoxLayout()

        self.move_value = QSpinBox()
        self.move_value.setRange(200,6000)

        move_btn = QPushButton("move=")
        move_btn.clicked.connect(self.move_absolute)

        move_plus_btn = QPushButton("move+")
        move_plus_btn.clicked.connect(self.move_plus)

        move_minus_btn = QPushButton("move-")
        move_minus_btn.clicked.connect(self.move_minus)

        move_layout.addWidget(self.move_value)
        move_layout.addWidget(move_btn)
        move_layout.addWidget(move_plus_btn)
        move_layout.addWidget(move_minus_btn)

        layout.addLayout(move_layout)

        # ROLLER1
        roller1_layout = QHBoxLayout()

        self.r1_speed = QSpinBox()
        self.r1_speed.setRange(1,255)

        self.r1_time = QSpinBox()
        self.r1_time.setRange(1,10000)

        r1_btn = QPushButton("Roller1")
        r1_btn.clicked.connect(self.roller1)

        roller1_layout.addWidget(QLabel("Speed"))
        roller1_layout.addWidget(self.r1_speed)
        roller1_layout.addWidget(QLabel("Time"))
        roller1_layout.addWidget(self.r1_time)
        roller1_layout.addWidget(r1_btn)

        layout.addLayout(roller1_layout)

        # ROLLER2
        roller2_layout = QHBoxLayout()

        self.r2_state = QComboBox()
        self.r2_state.addItems(["0","1"])

        self.r2_time = QSpinBox()
        self.r2_time.setRange(1,10000)

        r2_btn = QPushButton("Roller2")
        r2_btn.clicked.connect(self.roller2)

        roller2_layout.addWidget(QLabel("State"))
        roller2_layout.addWidget(self.r2_state)
        roller2_layout.addWidget(QLabel("Time"))
        roller2_layout.addWidget(self.r2_time)
        roller2_layout.addWidget(r2_btn)

        layout.addLayout(roller2_layout)

        # LED
        led_layout = QHBoxLayout()

        self.led_num = QSpinBox()
        self.led_num.setRange(1,4)

        self.led_val = QSpinBox()
        self.led_val.setRange(0,127)

        led_btn = QPushButton("Set LED")
        led_btn.clicked.connect(self.set_led)

        led_layout.addWidget(QLabel("LED"))
        led_layout.addWidget(self.led_num)
        led_layout.addWidget(QLabel("Value"))
        led_layout.addWidget(self.led_val)
        led_layout.addWidget(led_btn)

        layout.addLayout(led_layout)

        # LOG
        self.log = QTextEdit()
        self.log.setReadOnly(True)

        layout.addWidget(self.log)

        self.setLayout(layout)

    def refresh_ports(self):
        self.port_box.clear()
        ports = serial.tools.list_ports.comports()
        for p in ports:
            self.port_box.addItem(p.device)

    def connect_serial(self):
        port = self.port_box.currentText()

        try:
            self.serial = serial.Serial(port,115200,timeout=1)
            self.log.append(f"Connected to {port}")
        except Exception as e:
            self.log.append(str(e))

    def send(self,cmd):
        if not self.serial:
            self.log.append("Serial not connected")
            return

        try:
            self.serial.write((cmd+"\n").encode())
            self.log.append(">> "+cmd)

            QTimer.singleShot(100,self.read_response)

        except Exception as e:
            self.log.append(str(e))

    def read_response(self):
        if self.serial and self.serial.in_waiting:
            data = self.serial.readline().decode(errors="ignore")
            self.log.append("<< "+data)

    def move_absolute(self):
        v = self.move_value.value()
        self.send(f"move={v};")

    def move_plus(self):
        v = self.move_value.value()
        self.send(f"move+{v};")

    def move_minus(self):
        v = self.move_value.value()
        self.send(f"move-{v};")

    def roller1(self):
        speed = self.r1_speed.value()
        time = self.r1_time.value()
        self.send(f"roler1={speed},{time};")

    def roller2(self):
        state = self.r2_state.currentText()
        time = self.r2_time.value()
        self.send(f"roler2={state},{time};")

    def set_led(self):
        num = self.led_num.value()
        val = self.led_val.value()
        self.send(f"led={num},{val};")


app = QApplication(sys.argv)
window = SerialUI()
window.show()
sys.exit(app.exec_())
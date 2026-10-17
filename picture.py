import asyncio
import serial.tools.list_ports
import serial_asyncio
import cv2
import sys
from config import IP

RTSP_URL = f"rtsp://{IP}:8080/h264.sdp"


class SerialClient(asyncio.Protocol):
    def __init__(self):
        self.transport = None
        self.buffer = asyncio.Queue()
        self.connected = asyncio.Future()

    def connection_made(self, transport):
        self.transport = transport
        print("Connected to serial")
        if not self.connected.done():
            self.connected.set_result(True)

    def data_received(self, data):
        text = data.decode(errors="ignore").strip()
        print("<<", text)
        self.buffer.put_nowait(text)

    def send(self, text):
        if not self.transport:
            raise RuntimeError("Serial not connected yet!")
        msg = (text + "\n").encode()
        self.transport.write(msg)
        print(">>", text)

    async def ask(self, text, timeout=30):
        """ارسال دستور و صبر تا دریافت جواب"""
        await self.connected
        self.send(text)
        try:
            response = await asyncio.wait_for(self.buffer.get(), timeout)
            return response
        except asyncio.TimeoutError:
            print("⚠ No response received!")
            return None

    async def send_and_wait(self, text, delay=2):
        """ارسال دستور بدون پاسخ ولی با تاخیر برای اجرای دستگاه"""
        await self.connected
        self.send(text)
        await asyncio.sleep(delay)



async def take_picture(filename):
    loop = asyncio.get_event_loop()

    def capture():
        cap = cv2.VideoCapture(RTSP_URL)
        for i in range(30):
            ret, frame = cap.read()
        if ret:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            h, w = frame.shape[:2]
            x1 = 40
            x2 = w - 10
            y1 = 50
            y2 = h - 230
            cropped = frame[y1:y2, x1:x2]
            cv2.imwrite(f"{filename}.jpg", frame)
            print("Saved frame.jpg")
        cap.release()

    await loop.run_in_executor(None, capture)


async def main():
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial ports found!")
        return

    port = ports[-1].device
    loop = asyncio.get_running_loop()

    transport, protocol = await serial_asyncio.create_serial_connection(
        loop, SerialClient, port, baudrate=115200
    )

    ser = protocol

    await ser.connected
    await asyncio.sleep(3)

    await ser.ask("home")
    await ser.ask("get")
    await ser.ask("get")
    await ser.ask("led=1,127;")
    await ser.ask("led=2,127;")
    await ser.ask("led=3,127;")
    await ser.ask("led=4,127;")
    await ser.ask("move=1800;")

    filename_arg = sys.argv[1] if len(sys.argv) > 1 else "frame"
    await take_picture(filename_arg)

    await ser.ask("back")


if __name__ == "__main__":
    asyncio.run(main())

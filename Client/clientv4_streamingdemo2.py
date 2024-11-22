import cv2
import requests
import base64
import time
import socketio
import asyncio
import threading
from aiohttp import web
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

HOST_BE = "http://localhost:5000/"


# Giữ nguyên các hàm helper
def check_result_face(result_face):
    for key in result_face:
        if len(result_face[key]) > 0:
            return True
    return False


def get_largrest_face(faces):
    if len(faces) == 0:
        return None
    return max(faces, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))


class FaceRecognitionClient:
    def __init__(self, aiapi_url="http://localhost:8000", host_be=HOST_BE, stream_port=8765):
        self.api_url = aiapi_url
        self.host_be = host_be
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

        # Các biến cũ
        self.last_api_call = 0
        self.api_cooldown = 2
        self.last_recognition_result = None
        self.threshold = 0.6
        self.current_recognition = None
        self.recognition_timeout = 3
        self.last_recognition_time = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        self.door_alive = False
        self.door_status = "CLOSE"
        self.is_calling_apibe = False
        self.result_faces = {}

        # Socket.IO cho BE (giữ nguyên)
        self.sio = socketio.Client()
        self.setup_socketio()
        self.connect_socketio()

        # Thêm các biến mới cho streaming
        self.stream_port = stream_port
        self.stream_sio = socketio.AsyncServer(cors_allowed_origins='*', async_mode='aiohttp')
        self.stream_app = web.Application()
        self.stream_sio.attach(self.stream_app)

        # Queue để xử lý frame streaming
        self.frame_queue = Queue(maxsize=10)
        self.streaming_active = False

        # Setup và start streaming server
        self.setup_stream_server()
        self.start_stream_server()

    def setup_stream_server(self):
        """Setup Socket.IO streaming server events"""

        @self.stream_sio.on('connect')
        async def handle_stream_connect(sid, environ):
            print(f"Stream client connected: {sid}")

        @self.stream_sio.on('disconnect')
        async def handle_stream_disconnect(sid):
            print(f"Stream client disconnected: {sid}")

    def start_stream_server(self):
        """Start the streaming server in a separate thread"""

        async def stream_frames():
            while True:
                try:
                    if not self.frame_queue.empty():
                        frame_data = self.frame_queue.get()
                        await self.stream_sio.emit('frame', frame_data)
                    await asyncio.sleep(0.033)  # ~30 FPS
                except Exception as e:
                    print(f"Error streaming frame: {e}")
                    await asyncio.sleep(1)

        async def run_server():
            runner = web.AppRunner(self.stream_app)
            await runner.setup()
            site = web.TCPSite(runner, '0.0.0.0', self.stream_port)
            await site.start()
            print(f"Streaming server started on port {self.stream_port}")

            # Start frame streaming task
            asyncio.create_task(stream_frames())

            # Keep the server running
            while True:
                await asyncio.sleep(1)

        def run_async_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run_server())

        # Start server in a separate thread
        self.stream_thread = threading.Thread(target=run_async_server)
        self.stream_thread.daemon = True
        self.stream_thread.start()
        time.sleep(1)  # Đợi server khởi động

    def queue_frame(self, frame_data):
        """Add frame to queue for streaming"""
        try:
            if self.frame_queue.full():
                self.frame_queue.get_nowait()  # Remove oldest frame if queue is full
            self.frame_queue.put_nowait(frame_data)
        except Exception as e:
            print(f"Error queueing frame: {e}")

    def setup_socketio(self):
        """Setup Socket.IO event handlers"""

        @self.sio.on('connect')
        def on_connect():
            print("Connected to Socket.IO server")

        @self.sio.on('disconnect')
        def on_disconnect():
            print("Disconnected from Socket.IO server")
            time.sleep(5)
            self.connect_socketio()
            print("Reconnecting to Socket.IO server...")

        @self.sio.on('door')
        def on_door(data):
            try:
                device_id, status = data.split(";")
                self.door_status = status
                print(f"Door status updated: {status}")
            except Exception as e:
                print(f"Error handling door status: {str(e)}")

        @self.sio.on('dooralive')
        def on_door_alive(data):
            try:
                is_alive = data == "True"
                self.door_alive = is_alive
                print(f"Door alive status updated: {is_alive}")
            except Exception as e:
                self.door_alive = False
                print(f"Error handling door alive status: {str(e)}")

    def connect_socketio(self):
        try:
            if not self.sio.connected:
                self.sio.connect(self.host_be)
        except Exception as e:
            self.door_alive = False
            print(f"Failed to connect to Socket.IO server: {str(e)}")
            time.sleep(5)
            self.connect_socketio()

    def encode_image_base64(self, image):
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')

    def call_recognition_api(self, image):
        try:
            base64_image = self.encode_image_base64(image)
            response = requests.post(
                f"{self.api_url}/faces/recognize",
                json={
                    "image_base64": base64_image,
                    "threshold": self.threshold,
                    "limit": 1
                },
                timeout=5
            )
            print(response.json())

            if response.status_code == 200:
                self.last_recognition_result = response.json()
                # Update current recognition if we have a match
                if (self.last_recognition_result.get("status") == "success" and
                        self.last_recognition_result.get("results")):
                    result = self.last_recognition_result["results"][0]
                    if result.get("matches"):
                        bbox = result['bbox']
                        match = result["matches"][0]
                        self.current_recognition = {
                            "bbox": bbox,
                            "name": match.get("name", "Unknown"),
                            "common_name": match.get("common_name", ""),
                            "confidence": match.get("confidence", 0),
                            "face_id": match.get("face_id", "")
                        }
                        self.last_recognition_time = time.time()
            else:
                print(f"API Error: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"API call failed: {str(e)}")

    def call_door_api(self, image_data, user_id, door_id="Main door"):
        if not self.door_alive:
            print("Door is not alive, cannot open")
            return

        if self.door_status != "LOGCLOSE":
            print("Door is not closed, cannot open")
            return

        try:
            self.is_calling_apibe = True
            response = requests.post(
                f"{self.host_be}/api/camera_door",
                json={
                    "user_id": user_id,
                    "door_id": door_id,
                    "image": image_data
                },
                timeout=5
            )
            if response.status_code == 200:
                print("Door opened successfully")
            else:
                print(f"Failed to open door: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Door API call failed: {str(e)}")
        finally:
            self.is_calling_apibe = False

    def draw_door_status(self, frame):
        status_text = f"Door: {'Online' if self.door_alive else 'Offline'} | Status: {self.door_status}"
        (text_width, text_height), _ = cv2.getTextSize(
            status_text, self.font, self.font_scale, self.font_thickness
        )

        cv2.rectangle(
            frame,
            (10, 10),
            (10 + text_width + 20, 10 + text_height + 20),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            status_text,
            (20, 30),
            self.font,
            self.font_scale,
            (0, 255, 0) if self.door_alive else (0, 0, 255),
            self.font_thickness
        )

    def draw_results(self, frame, faces):
        current_time = time.time()

        # Check if we should clear the current recognition
        if current_time - self.last_recognition_time > self.recognition_timeout:
            self.result_faces = {}
            self.current_recognition = None

        largest_face = get_largrest_face(faces)
        # Draw each detected face
        if largest_face is not None:
            x, y, w, h = largest_face
            #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # for (x, y, w, h) in faces:
            # Draw rectangle for face

            # if self.current_recognition:
            #     bbox = self.current_recognition['bbox']
            #     x, y, w, h = map(int, bbox)
            #     x, y, w, h = x, y, w - x, h - y
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # If we have a current recognition, draw the name
            if self.current_recognition:
                name = self.current_recognition["name"]
                common_name = self.current_recognition["common_name"]
                confidence = self.current_recognition["confidence"]
                face_id = self.current_recognition["face_id"]

                # Update result_faces for door control
                if common_name:
                    if common_name in self.result_faces:
                        self.result_faces[common_name].append(face_id)
                        self.result_faces[common_name] = list(set(self.result_faces[common_name]))
                    else:
                        self.result_faces[common_name] = [face_id]

                # Draw name label
                text = f"{name} ({common_name})"
                conf_text = f"Conf: {confidence:.2f}"

                (text_width, text_height), _ = cv2.getTextSize(
                    text, self.font, self.font_scale, self.font_thickness
                )

                # Background rectangle for name
                cv2.rectangle(
                    frame,
                    (x, y - text_height - 10),
                    (x + text_width + 10, y),
                    (0, 255, 0),
                    -1
                )

                # Draw name and confidence
                cv2.putText(
                    frame,
                    text,
                    (x + 5, y - 5),
                    self.font,
                    self.font_scale,
                    (0, 0, 0),
                    self.font_thickness
                )

                cv2.putText(
                    frame,
                    conf_text,
                    (x, y + h + 20),
                    self.font,
                    self.font_scale,
                    (0, 255, 0),
                    self.font_thickness
                )

    def run(self):
        print("Starting face recognition client...")
        print(f"Streaming server running on port {self.stream_port}")
        print("Press 'q' to quit")

        self.streaming_active = True
        self.result_faces = {}

        try:
            while True:
                ret, frame = self.cap.read()
                frame_send = self.encode_image_base64(frame)
                if not ret:
                    print("Failed to grab frame")
                    break

                # Convert frame to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces using OpenCV
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )

                # Xử lý nhận diện và vẽ kết quả
                current_time = time.time()
                if len(faces) > 0 and (current_time - self.last_api_call) >= self.api_cooldown:
                    self.last_api_call = current_time
                    self.call_recognition_api(frame)

                self.draw_results(frame, faces)
                self.draw_door_status(frame)

                # Xử lý mở cửa
                if check_result_face(self.result_faces) and not self.is_calling_apibe and \
                        self.door_status == "LOGCLOSE" and self.current_recognition is not None and \
                        self.current_recognition["name"] != "invalid":
                    self.result_faces = {}
                    print('Opening door...')

                    if self.current_recognition:
                        self.call_door_api(frame_send, self.current_recognition["name"])

                # Queue frame for streaming
                try:
                    frame_data = {
                        'image': frame_send,
                        'faces_detected': len(faces),
                        'door_status': self.door_status,
                        'door_alive': self.door_alive,
                        'recognition': self.current_recognition
                    }
                    self.queue_frame(frame_data)
                except Exception as e:
                    print(f"Streaming error: {str(e)}")

                # Show the frame locally
                cv2.imshow('Face Recognition Client', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            # Cleanup
            self.streaming_active = False
            self.sio.disconnect()
            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    client = FaceRecognitionClient()
    client.run()

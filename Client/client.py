import cv2
import requests
import base64
import time
from threading import Thread, Lock
import socketio

HOST_BE = "http://localhost:5000"


def check_result_face(result_face):
    for key in result_face:
        if len(result_face[key]) > 1:
            return True
    return False


class FaceRecognitionClient:
    def __init__(self, aiapi_url="http://localhost:8000", host_be=HOST_BE):
        self.api_url = aiapi_url
        self.host_be = host_be
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

        # Initialize variables for API results
        self.last_api_call = 0
        self.api_cooldown = 0.3
        self.last_recognition_result = None
        self.result_lock = Lock()
        self.threshold = 0.6

        # Font settings for OpenCV
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2

        # Door status with thread safety
        self.door_lock = Lock()
        self.door_alive = False
        self.door_status = "CLOSE"

        # Initialize Socket.IO connection
        self.sio = socketio.Client()
        self.setup_socketio()
        self.connect_socketio()
        self.is_calling_apibe = False

    def setup_socketio(self):
        """Setup Socket.IO event handlers"""

        @self.sio.on('connect')
        def on_connect():
            print("Connected to Socket.IO server")

        @self.sio.on('disconnect')
        def on_disconnect():
            print("Disconnected from Socket.IO server")
            # Attempt to reconnect after a delay
            time.sleep(5)
            self.connect_socketio()
            print("Reconnecting to Socket.IO server...")

        @self.sio.on('door')
        def on_door(data):
            """Handle door status updates"""
            try:
                device_id, status = data.split(";")
                with self.door_lock:
                    self.door_status = status
                print(f"Door status updated: {status}")
            except Exception as e:
                print(f"Error handling door status: {str(e)}")

        @self.sio.on('dooralive')
        def on_door_alive(data):
            """Handle door alive status updates"""
            try:
                is_alive = data == "True"
                with self.door_lock:
                    self.door_alive = is_alive
                print(f"Door alive status updated: {is_alive}")
            except Exception as e:
                print(f"Error handling door alive status: {str(e)}")

    def connect_socketio(self):
        """Establish Socket.IO connection"""
        try:
            if not self.sio.connected:
                self.sio.connect(self.host_be)
        except Exception as e:
            print(f"Failed to connect to Socket.IO server: {str(e)}")
            # Retry connection after delay
            time.sleep(5)
            self.connect_socketio()

    def encode_image_base64(self, image):
        """Convert image to base64 string."""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')

    def call_recognition_api(self, image):
        """Call the face recognition API."""
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
            print(f"API Response: {response.status_code} - {response.text}")

            if response.status_code == 200:
                with self.result_lock:
                    self.last_recognition_result = response.json()
            else:
                print(f"API Error: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"API call failed: {str(e)}")

    def call_door_api(self, image_data, user_id, door_id="Main door"):
        """Call the door API with current door status check."""
        with self.door_lock:
            if not self.door_alive:
                print("Door is not alive, cannot open")
                return

            if self.door_status != "LOGCLOSE":
                print("Door is not closed, cannot open")
                return

        try:
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
        self.is_calling_apibe = False

    def draw_door_status(self, frame):
        """Draw door status information on the frame."""
        with self.door_lock:
            alive = self.door_alive
            status = self.door_status

        # Status text
        status_text = f"Door: {'Online' if alive else 'Offline'} | Status: {status}"

        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(
            status_text, self.font, self.font_scale, self.font_thickness
        )

        # Draw background rectangle
        cv2.rectangle(
            frame,
            (10, 10),
            (10 + text_width + 20, 10 + text_height + 20),
            (0, 0, 0),
            -1
        )

        # Draw status text
        cv2.putText(
            frame,
            status_text,
            (20, 30),
            self.font,
            self.font_scale,
            (0, 255, 0) if alive else (0, 0, 255),
            self.font_thickness
        )

    def draw_results(self, frame, result_faces):
        """Draw recognition results on the frame."""
        with self.result_lock:
            recognition_result = self.last_recognition_result

        if recognition_result and recognition_result.get("status") == "success":
            results = recognition_result.get("results", [])

            for result in results:
                bbox = result.get("bbox")
                matches = result.get("matches", [])

                if bbox and matches:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    best_match = matches[0]
                    face_id = best_match.get("face_id", "")
                    name = best_match.get("name", "Unknown")
                    confidence = best_match.get("confidence", 0)
                    common_name = best_match.get("common_name", "")

                    if common_name in result_faces:
                        result_faces[common_name].append(face_id)
                        result_faces[common_name] = list(set(result_faces[common_name]))
                    else:
                        result_faces[common_name] = [face_id]

                    text = f"{name} ({common_name})"
                    conf_text = f"Conf: {confidence:.2f}"

                    (text_width, text_height), _ = cv2.getTextSize(
                        text, self.font, self.font_scale, self.font_thickness
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1 - text_height - 10),
                        (x1 + text_width + 10, y1),
                        (0, 255, 0),
                        -1
                    )

                    cv2.putText(
                        frame,
                        text,
                        (x1 + 5, y1 - 5),
                        self.font,
                        self.font_scale,
                        (0, 0, 0),
                        self.font_thickness
                    )

                    cv2.putText(
                        frame,
                        conf_text,
                        (x1, y2 + 20),
                        self.font,
                        self.font_scale,
                        (0, 255, 0),
                        self.font_thickness
                    )

    def run(self):
        """Main loop for capturing and processing video."""
        print("Starting face recognition client...")
        print("Press 'q' to quit")
        result_face = {}

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            # Call API if faces detected and enough time has passed
            current_time = time.time()
            if len(faces) > 0 and (current_time - self.last_api_call) >= self.api_cooldown:
                self.last_api_call = current_time
                Thread(target=self.call_recognition_api, args=(frame,)).start()
                # Draw results on frame
                self.draw_results(frame, result_face)

            if check_result_face(result_face) and not self.is_calling_apibe:
                self.is_calling_apibe = True
                result_face = {}
                print('Opening door...')
                # Convert current frame to base64 for door API
                base64_image = self.encode_image_base64(frame)
                # Call door API in separate thread
                # self.call_door_api(base64_image)
                name_user = self.last_recognition_result['results'][0]['matches'][0]['name']
                Thread(target=self.call_door_api, args=(base64_image, name_user)).start()

            # Draw door status
            self.draw_door_status(frame)

            # Show the frame
            cv2.imshow('Face Recognition Client', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        self.sio.disconnect()
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    client = FaceRecognitionClient()
    client.run()

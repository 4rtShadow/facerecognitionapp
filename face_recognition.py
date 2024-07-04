import cv2
import mediapipe as mp
import numpy as np

class FaceRecognition:
    def __init__(self, config):
        self.config = config
        self.face_detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
        self.blur_method = self.gaussian_blur if config['blur_method'] == 'gaussian' else self.pixelate

    def process_frame(self, frame, blur_enabled):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                cv2.rectangle(frame, (x, y), (x + w, y + h), self.config['bbox_color'], 2)

                if blur_enabled:
                    face_region = frame[y:y + h, x:x + w]
                    blurred_face = self.blur_method(face_region)
                    frame[y:y + h, x:x + w] = blurred_face

        return frame

    def gaussian_blur(self, face):
        if face is None or face.size == 0:
            return face
        return cv2.GaussianBlur(face, (99, 99), 30)

    def pixelate(self, face):
        h, w = face.shape[:2]
        temp = cv2.resize(face, (w // 8, h // 8), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

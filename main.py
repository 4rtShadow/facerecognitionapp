import cv2
import numpy as np
import yaml
from face_recognition import FaceRecognition

def main():
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    face_recognition = FaceRecognition(config)

    video_config = config['video_source']
    if video_config['type'] == 'webcam' or video_config['type'] == 'file' or video_config['type'] == 'ip_camera':
        cap = cv2.VideoCapture(video_config['value'])
    else:
        raise ValueError(f"Неизвестный тип видео: {video_config['type']}")

    if not cap.isOpened():
        raise Exception(f"Ошибка: {video_config}")

    cv2.namedWindow('Face', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Face', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    blur_enabled = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = face_recognition.process_frame(frame, blur_enabled)

        if config['show_fps']:
            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(processed_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Face', processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            blur_enabled = not blur_enabled

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
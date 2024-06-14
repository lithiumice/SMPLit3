import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import os
import numpy as np
from skimage.transform import estimate_transform, warp


def crop_face(frame, landmarks, scale=1.0, image_size=224):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    h, w, _ = frame.shape
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

    size = int(old_size * scale)

    # crop image
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    return tform

model_asset_path=os.path.join(os.path.dirname(__file__),'../../../model_files/smirk/face_landmarker.task')

def run_mediapipe(image):
    base_options = python.BaseOptions(model_asset_path=model_asset_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        num_faces=1,
                                        min_face_detection_confidence=0.1,
                                        min_face_presence_confidence=0.1
                                        )
    detector = vision.FaceLandmarker.create_from_options(options)

    # print(image.shape)    
    image_numpy = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    # STEP 3: Load the input image.
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_numpy)


    # STEP 4: Detect face landmarks from the input image.
    detection_result = detector.detect(image)

    if len (detection_result.face_landmarks) == 0:
        print('No face detected')
        return None
    
    face_landmarks = detection_result.face_landmarks[0]

    face_landmarks_numpy = np.zeros((478, 3))

    for i, landmark in enumerate(face_landmarks):
        face_landmarks_numpy[i] = [landmark.x*image.width, landmark.y*image.height, landmark.z]

    return face_landmarks_numpy

class Mediapipe_detector():
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1,
                                            min_face_detection_confidence=0.1,
                                            min_face_presence_confidence=0.1
                                            )
        detector = vision.FaceLandmarker.create_from_options(options)
        self.detector = detector
        
    def infer(self, image):
        # print(image.shape)    
        image_numpy = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        # STEP 3: Load the input image.
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_numpy)

        # STEP 4: Detect face landmarks from the input image.
        detection_result = self.detector.detect(image)

        if len (detection_result.face_landmarks) == 0:
            print('No face detected')
            return None
        
        face_landmarks = detection_result.face_landmarks[0]

        face_landmarks_numpy = np.zeros((478, 3))

        for i, landmark in enumerate(face_landmarks):
            face_landmarks_numpy[i] = [landmark.x*image.width, landmark.y*image.height, landmark.z]

        return face_landmarks_numpy

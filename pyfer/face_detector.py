import os
import pandas as pd
import cv2
from pathlib import Path

abs_path = os.path.dirname(__file__)

class Detector:
    def __init__(self):
        self.deploy = str(Path(abs_path) / 'models/deploy.prototxt')
        self.caffe_model = str(Path(abs_path) / 'models/res10_300x300_ssd_iter_140000.caffemodel')
        self.detector = cv2.dnn.readNetFromCaffe(self.deploy, self.caffe_model)

    def forward(self, image_path):
        # get image
        image = cv2.imread(image_path)
        base_img = image.copy()
        original_size = base_img.shape
        target_size = (300, 300)

        # resize
        image = cv2.resize(image, target_size)
        aspect_ratio_x = (original_size[1] / target_size[1])
        aspect_ratio_y = (original_size[0] / target_size[0])

        # forward step
        imageBlob = cv2.dnn.blobFromImage(image = image)
        self.detector.setInput(imageBlob)
        self.detections = self.detector.forward()

        # create points dataframes
        detections_df = pd.DataFrame(self.detections[0][0]
            , columns = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"])
        detections_df = detections_df[detections_df['is_face'] == 1] #0: background, 1: face
        detections_df = detections_df[detections_df['confidence'] >= 0.90]
        return detections_df



if __name__ == '__main__':
    print(abs_path)

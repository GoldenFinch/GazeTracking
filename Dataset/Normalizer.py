import os
import scipy.io as scio
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import pickle


def normalization(origin_dir, output_dir, is_normalized=True):
    # normalized camera parameters
    focal_norm = 1000
    distance_norm = 300
    roiSize = (448, 448)
    cam_norm = np.array([
        [focal_norm, 0, roiSize[0]/2],
        [0, focal_norm, roiSize[1]/2],
        [0, 0, 1.0]
    ])

    # dataset file content
    img_path = []
    normalized_gaze_vector = []
    index_candidate = []

    if os.path.exists(output_dir):
        raise FileExistsError('The directory exists, Please check it again!')
    else:
        os.mkdir(output_dir)
        origin_candidate_list = os.listdir(origin_dir)
        for i in range(len(origin_candidate_list)-1):
            origin_candidate_dir = os.path.join(origin_dir, origin_candidate_list[i])
            output_candidate_dir = os.path.join(output_dir, origin_candidate_list[i])
            if not os.path.exists(output_candidate_dir):
                os.mkdir(output_candidate_dir)
            origin_candidate_day_list = os.listdir(origin_candidate_dir)
            origin_candidate_calibration_dir = os.path.join(origin_candidate_dir, origin_candidate_day_list[0])
            origin_candidate_labels_path = os.path.join(origin_candidate_dir, origin_candidate_day_list[-1])
            origin_candidate_cam_Matrix = scio.loadmat(origin_candidate_calibration_dir+"/Camera.mat")["cameraMatrix"]
            origin_candidate_labels = pd.read_table(origin_candidate_labels_path, header=None)
            for j in range(1, len(origin_candidate_day_list)-1):
                # origin_candidate_day_dir = os.path.join(origin_candidate_dir,origin_candidate_day_list[j])
                output_candidate_day_dir = os.path.join(output_candidate_dir, origin_candidate_day_list[j])
                if not os.path.exists(output_candidate_day_dir):
                    os.mkdir(output_candidate_day_dir)
            for k in range(origin_candidate_labels.shape[0]):
                label = origin_candidate_labels.iloc[k][0].split()
                img = plt.imread(os.path.join(origin_candidate_dir, label[0]))

                face_center = np.array([[float(label[21])], [float(label[22])], [float(label[23])]])
                gaze_vector = np.array([[float(label[24])], [float(label[25])], [float(label[26])]]) - face_center  # gaze vector
                distance = np.linalg.norm(face_center)  # 范数，求距离
                Zc = (face_center / distance).reshape(3)
                face_center_image_coordinate_with_Zc = np.dot(origin_candidate_cam_Matrix, face_center)
                face_center_image_coordinate_without_Zc = face_center_image_coordinate_with_Zc / face_center_image_coordinate_with_Zc[2,0]

                if is_normalized:
                    left_eye_center_pixel = np.array([[int(label[3]) + (int(label[5]) - int(label[3])) / 2], [int(label[4]) + (int(label[6]) - int(label[4])) / 2], [1]])
                    left_eye_center = np.dot(np.linalg.inv(origin_candidate_cam_Matrix), left_eye_center_pixel * face_center_image_coordinate_with_Zc[2, 0])
                    right_eye_center_pixel = np.array([[int(label[7]) + (int(label[9]) - int(label[7])) / 2], [int(label[8]) + (int(label[10]) - int(label[8])) / 2], [1]])
                    right_eye_center = np.dot(np.linalg.inv(origin_candidate_cam_Matrix), right_eye_center_pixel * face_center_image_coordinate_with_Zc[2, 0])
                    Xr = (right_eye_center - left_eye_center).reshape(3)
                    Yc = np.cross(Zc, Xr)
                    Yc = Yc / np.linalg.norm(Yc)
                    Xc = np.cross(Yc, Zc)
                    Xc = Xc / np.linalg.norm(Yc)
                    R = np.c_[Xc, Yc, Zc].T
                    S = np.diag([1.0, 1.0, distance_norm / distance])
                    W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(origin_candidate_cam_Matrix)))
                    img = cv2.warpPerspective(img, W, roiSize)
                    # ---------- normalize gaze vector ----------
                    gaze_vector = np.dot(R, gaze_vector)
                    gaze_vector = gaze_vector / np.linalg.norm(gaze_vector)
                else:
                    norm = 0.9 * distance / distance_norm
                    img_scale = cv2.resize(img, (0, 0), fx=norm, fy=norm)
                    img_scale = cv2.copyMakeBorder(img_scale, 224, 224, 224, 224, cv2.BORDER_CONSTANT)
                    img = img_scale[int(face_center_image_coordinate_without_Zc[1, 0] * norm): int(face_center_image_coordinate_without_Zc[1, 0] * norm) + 448,
                                    int(face_center_image_coordinate_without_Zc[0, 0] * norm): int(face_center_image_coordinate_without_Zc[0, 0] * norm) + 448,
                                    :]

                img_path.append("/"+origin_candidate_list[i]+"/"+label[0])
                normalized_gaze_vector.append(gaze_vector)
                index_candidate.append(i)

                img = Image.fromarray(img)
                img.save(output_candidate_dir+'/'+label[0])

        data = {'img_path': img_path, 'normalized_gaze_vector': normalized_gaze_vector, 'index_candidate': index_candidate}
        if is_normalized:
            data_file = 'Dataset_normalized.data'
        else:
            data_file = 'Dataset_unnormalized.data'
        with open(data_file, 'wb') as f:
            pickle.dump(data, f)


# normalization("H:/MPIIFaceGaze", "H:/MPIIFaceGaze_normalized", True)
# normalization("H:/MPIIFaceGaze", "H:/MPIIFaceGaze_unnormalized", False)

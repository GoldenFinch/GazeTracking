import os
import scipy.io as scio
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import pickle

# normalized camera parameters
focal_norm = 1000
distance_norm = 300
roiSize = (448,448)
cam_norm = np.array([
    [focal_norm,0,roiSize[0]/2],
    [0,focal_norm,roiSize[1]/2],
    [0,0,1.0]
])

# directory path
origin_dir = 'C:/MPIIFaceGaze'
output_dir = 'C:/Normalized_MPIIFaceGaze/'

# dataset file content
img_path= []
normalized_gaze_vector= []
index_candidate = []

# each sample: (img_path, normalized_gaze-vector, index_candidate)
if __name__ == '__main__':
    if os.path.exists(output_dir):
        raise FileExistsError('The dirctory exists, Please check it again!')
    else:
        os.mkdir(output_dir)
        origin_candidate_list = os.listdir(origin_dir)
        for i in range(len(origin_candidate_list)-1):
            origin_candidate_dir = os.path.join(origin_dir, origin_candidate_list[i])
            output_candidate_dir = os.path.join(output_dir, origin_candidate_list[i])
            # print(origin_candidate_dir,output_candidate_dir)
            if not os.path.exists(output_candidate_dir):
                os.mkdir(output_candidate_dir)
            origin_candidate_day_list = os.listdir(origin_candidate_dir)
            origin_candidate_calibration_dir = os.path.join(origin_candidate_dir,origin_candidate_day_list[0])
            origin_candidate_labels_path = os.path.join(origin_candidate_dir,origin_candidate_day_list[-1])
            origin_candidate_cam_Matrix = scio.loadmat(origin_candidate_calibration_dir+"/Camera.mat")["cameraMatrix"]
            origin_candidate_labels = pd.read_table(origin_candidate_labels_path, header=None)
            for j in range(1,len(origin_candidate_day_list)-1):
                origin_candidate_day_dir = os.path.join(origin_candidate_dir,origin_candidate_day_list[j])
                output_candidate_day_dir = os.path.join(output_candidate_dir,origin_candidate_day_list[j])
                # print(output_candidate_day_dir)
                if not os.path.exists(output_candidate_day_dir):
                    os.mkdir(output_candidate_day_dir)
            for k in range(origin_candidate_labels.shape[0]):
                label = origin_candidate_labels.iloc[k][0].split( )
                img = plt.imread(os.path.join(origin_candidate_dir,label[0]))
                # print(os.path.join(origin_candidate_dir,label[0]))
                # plt.imshow(img)
                # plt.show()

                face_center = np.array([[float(label[21])], [float(label[22])], [float(label[23])]])
                distance = np.linalg.norm(face_center)  # 范数，求距离
                Zc = (face_center / distance).reshape(3)

                face_center_image_coordinate_with_Zc = np.dot(origin_candidate_cam_Matrix, face_center)

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

                img_warped = cv2.warpPerspective(img, W, roiSize)

                ## ---------- normalize gaze vector ----------
                gaze_vector = np.array([[float(label[24])], [float(label[25])], [float(label[26])]]) - face_center  # gaze vector
                gc_normalized = np.dot(R, gaze_vector)
                gc_normalized = gc_normalized / np.linalg.norm(gc_normalized)

                candidate_index = i

                img_path.append("/"+origin_candidate_list[i]+"/"+label[0])
                normalized_gaze_vector.append(gc_normalized)
                index_candidate.append(i)

                img = Image.fromarray(img_warped)
                img.save(output_candidate_dir+'/'+label[0])
                # plt.show(img_warped)
                # plt.savefig(output_candidate_dir+'/'+label[0], transparent=True)
                # cv2.imwrite(output_candidate_dir+'/'+label[0],img)

        data = {'img_path':img_path,'normalized_gaze_vector':normalized_gaze_vector,'index_candidate':index_candidate}
        with open('Dataset/Dataset.data', 'wb') as f:
            pickle.dump(data, f)
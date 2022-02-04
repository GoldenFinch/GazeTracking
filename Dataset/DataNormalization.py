import h5py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import cv2
import util as util
import pickle


# mat_data=h5py.File("C:/MPIIFaceGaze_normalized/p08.mat","r")
# print(mat_data['Data'].keys())
# data = mat_data['Data']['data']
# label = mat_data['Data']['label']
# print(data.shape,label.shape)
# img_index = 403
# img = data[img_index,:,:,:]
# label = label[img_index,:].astype(float)
# img = np.transpose(img,axes=[1,2,0]).astype(int)
# print(label)
# plt.imshow(img)
# plt.scatter([label[4],label[6],label[8],label[10],label[12],label[14]],[label[5],label[7],label[9],label[11],label[13],label[15]])
# plt.show()



dataDirectory = "C:/MPIIFaceGaze/"

index= 'p09'

data = pd.read_table(dataDirectory+index+"/"+index+".txt", header=None)
camera = scio.loadmat(dataDirectory+index+"/Calibration/Camera.mat")
cam_Matrix = camera['cameraMatrix']
print(cam_Matrix)
# print(calibration['tvecs'])
# monitorPose = scio.loadmat(dataDirectory+index+"/Calibration/monitorPose.mat")
# print(monitorPose['tvecs'])
# print(monitorPose['rvects'])
# Rt = cv2.Rodrigues(monitorPose['rvects'])[0]
# screenSize = scio.loadmat(dataDirectory+index+"/Calibration/screenSize.mat")
# print(screenSize)
print(data.shape)
detail = data.iloc[116][0].split( )
print(len(detail))
print('对应图像：'+detail[0])
print('gaze屏幕坐标：'+detail[1],detail[2]) # width height
print('landmarks:')
print(detail[3:15])
print('3D head post in camera system：')
print(detail[15:21])
print('face center in camera system：')
print(detail[21:24])
print('gaze target in camera system：')
print(detail[24:27])

img = plt.imread(dataDirectory+index+"/"+detail[0])
# print(img.shape)    # (720,1280,3)

coordinate_with_Zc = np.dot(cam_Matrix, [[float(detail[21])],[float(detail[22])],[float(detail[23])]])
coordinate_without_Zc = coordinate_with_Zc / coordinate_with_Zc[2,0]
# print(coordinate_with_Zc)
print(coordinate_without_Zc)
print(np.dot(np.linalg.inv(cam_Matrix),coordinate_with_Zc))

# target_with_Zc = np.dot(calibration['cameraMatrix'], [[float(detail[24])],[float(detail[25])],[float(detail[26])]])
# target_without_Zc = coordinate_with_Zc / coordinate_with_Zc[2,0]
# print(target_with_Zc)
# print(target_without_Zc)


plt.imshow(img)
plt.scatter([int(detail[3]),detail[5],detail[7],detail[9],detail[11],detail[13], coordinate_without_Zc[0,0]],[int(detail[4]),detail[6],detail[8],detail[10],detail[12],detail[14], coordinate_without_Zc[1,0]])
plt.show()

# normalized camera parameters
focal_norm = 1000
distance_norm = 300
roiSize = (448,448)

face_center = np.array([float(detail[21]),float(detail[22]),float(detail[23])])
distance = np.linalg.norm(face_center)  # 范数，求距离
Zc = (face_center / distance).reshape(3)
# print(Zc.shape)
left_eye_center_pixel = np.array([[int(detail[3])+(int(detail[5])-int(detail[3]))/2], [int(detail[4])+(int(detail[6])-int(detail[4]))/2], [1]])
left_eye_center = np.dot(np.linalg.inv(cam_Matrix),left_eye_center_pixel * coordinate_with_Zc[2,0])
# print(left_eye_center)
right_eye_center_pixel = np.array([[int(detail[7])+(int(detail[9])-int(detail[7]))/2], [int(detail[8])+(int(detail[10])-int(detail[8]))/2], [1]])
right_eye_center = np.dot(np.linalg.inv(cam_Matrix),right_eye_center_pixel * coordinate_with_Zc[2,0])
# print(right_eye_center)
Xr = (right_eye_center - left_eye_center).reshape(3)
Yc = np.cross(Zc, Xr)
Yc = Yc / np.linalg.norm(Yc)
Xc = np.cross(Yc,Zc)
Xc = Xc / np.linalg.norm(Yc)
R =np.c_[Xc,Yc,Zc].T
cam_norm = np.array([
    [focal_norm,0,roiSize[0]/2],
    [0,focal_norm,roiSize[1]/2],
    [0,0,1.0]
])
S = np.diag([1.0,1.0, distance_norm / distance])
W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam_Matrix)))

plt.imshow(img)
plt.show()
img_warped = cv2.warpPerspective(img,W,roiSize)
plt.imshow(img_warped)
plt.show()


## ---------- normalize gaze vector ----------
gaze_vector = np.array([float(detail[24]),float(detail[25]),float(detail[26])]) - face_center  # gaze vector
gc_normalized = np.dot(R, gaze_vector)
gc_normalized = gc_normalized / np.linalg.norm(gc_normalized)
print(gc_normalized)

# 转换前后两向量的夹角不变
# print(util.angle_calculation(gaze_vector.reshape(1,-1), np.array([[0,0,1]])))
# print(util.angle_calculation(gc_normalized.reshape(1,-1), np.dot(R, np.array([0,0,1])).reshape(1,-1)))

# data normalizationh后前后数据一致
with open('Dataset.data', 'rb') as f:
    data = pickle.load(f)
    img_path = data['img_path']
    normalized_gaze_vector = data['normalized_gaze_vector']
    index_candidate = data['index_candidate']

for i in range(len(img_path)):
    if img_path[i] == '/p08/day05/0011.jpg':
        print(i)
        print(img_path[i])
        print(normalized_gaze_vector[i]) #[-0.33117562  0.11784094 -0.93618172]
        print(index_candidate[i])
        break
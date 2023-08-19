import os.path as osp

import numpy as np
import cv2
from openvino.runtime import Core


class Facemesh:
    model_path = osp.join(osp.abspath(__file__), "../facemesh.xml")
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.compiled_model = self.load_compiled_model()

    def load_compiled_model(self):
        core = Core()
        model = core.read_model(self.model_path)
        return core.compile_model(model, "AUTO")

    def __call__(self, image):
        image = self.preprocess(image)
        image =  np.expand_dims(np.transpose(image, (2, 0, 1)), 0)
        results = self.compiled_model.infer_new_request({0: image})
        detections = list(results.values())[0][0]
        return detections

    @staticmethod
    def preprocess(x):
        """Converts the image pixels to the range [-1, 1]."""
        return x / 127.5 - 1.0


face_mesh = Facemesh()


# '''
# class facemesh(object):
#     def __init__(self,image):
#         self.core = Core()
#         self.model = self.core.read_model(r"C:\openvino-workspace\ganimation-master\ganimation-master\animations\eric_andre\ganimation_IR\face_landmark\facemesh.xml")
#         #self.image = np.transpose(image,(2,0,1))
#         self.compiled_model = self.core.compile_model(self.model, "AUTO")

#     def produce(self,image):
#         image =cv2.resize(image,(192,192))
#         if isinstance(img, np.ndarray):
#             img = torch.from_numpy(img).permute((2, 0, 1))
#         image = np.expand_dims(self.image, 0)
#         results = self.compiled_model.infer_new_request({0: self.image})
#         points = list(results.values())[0][0]
#         print(points)
#         x = []
#         y = []
#         z = []

#         for i in range(0,len(points),3):
#             x.append(points[i])
#             y.append(points[i+1])
#             z.append(points[i+2])
#         x = np.array(x)
#         y = np.array(y)
#         z = np.array(z)
#         return x,y,z
#     def preprocess(self, x):
#         """Converts the image pixels to the range [-1, 1]."""
#         return x.float() / 127.5 - 1.0
#         '''
# Inference
# model = ppp.build()

# path = "C:/openvino-workspace/ganimation-master/ganimation-master/animations/eric_andre/ganimation_IR/result/ganimation_1.jpg"
# img = cv2.imread(path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img =cv2.resize(img,(192,192))
# detections = facemesh().produce(img)
# print(type(detections))
# detections = detections.reshape(-1, 3)
# numbers = []
# # Use a for loop to add the numbers from 0 to 467 to the list
# for i in range(0, 468, 1):
#     numbers.append(i)

# removed_index_reyes = [226, 31, 228, 229, 230, 231, 232, 233, 244,143, 111, 117, 118, 119, 120, 121, 128, 245,113, 225, 224, 223, 222, 221, 189]
# silhouette= [
# 10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109]

# lipsUpperOuter= [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
# lipsLowerOuter= [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
# lipsUpperInner= [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
# lipsLowerInner= [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

# rightEyeUpper0= [246, 161, 160, 159, 158, 157, 173]
# rightEyeLower0= [33, 7, 163, 144, 145, 153, 154, 155, 133]
# rightEyeUpper1= [247, 30, 29, 27, 28, 56, 190]
# rightEyeLower1= [130, 25, 110, 24, 23, 22, 26, 112, 243]
# rightEyeUpper2= [113, 225, 224, 223, 222, 221, 189]
# rightEyeLower2= [226, 31, 228, 229, 230, 231, 232, 233, 244]
# rightEyeLower3= [143, 111, 117, 118, 119, 120, 121, 128, 245]

# rightEyebrowUpper= [156, 70, 63, 105, 66, 107, 55, 193]
# rightEyebrowLower= [35, 124, 46, 53, 52, 65]

# leftEyeUpper0= [466, 388, 387, 386, 385, 384, 398]
# leftEyeLower0= [263, 249, 390, 373, 374, 380, 381, 382, 362]
# leftEyeUpper1= [467, 260, 259, 257, 258, 286, 414]
# leftEyeLower1= [359, 255, 339, 254, 253, 252, 256, 341, 463]
# leftEyeUpper2= [342, 445, 444, 443, 442, 441, 413]
# leftEyeLower2= [446, 261, 448, 449, 450, 451, 452, 453, 464]
# leftEyeLower3= [372, 340, 346, 347, 348, 349, 350, 357, 465]

# leftEyebrowUpper= [383, 300, 293, 334, 296, 336, 285, 417]
# leftEyebrowLower= [265, 353, 276, 283, 282, 295]
# noseTip= [1]
# noseBottom= [2]
# noseRightCorner= [98]
# noseLeftCorner= [327]

# removed_index = silhouette  + lipsUpperOuter + lipsLowerOuter + lipsUpperInner + lipsLowerInner + rightEyeUpper0 + rightEyeLower0 + rightEyeUpper1 + rightEyeLower1 + rightEyeUpper2 + rightEyeLower2 + rightEyeLower3 + rightEyebrowUpper + rightEyebrowLower + leftEyeUpper0 + leftEyeLower0 + leftEyeUpper1 + leftEyeLower1 + leftEyeUpper2 + leftEyeLower2 + leftEyeLower3 + leftEyebrowUpper + leftEyebrowLower
# removed_index = removed_index + noseTip + noseBottom + noseRightCorner + noseLeftCorner
# numbers = [i for i in numbers if i not in removed_index]
# detections = np.delete(detections,numbers,axis = 0)

# print(detections)
# plt.imshow(img, zorder=1)
# x, y = detections[:,0], detections[:,1]
# print(x)
# plt.scatter(x, y, zorder=2, s=1.0)
# plt.show()

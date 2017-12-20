import cv2
import numpy as np
from os import listdir
from os.path import isfile, splitext

#Checking whether file is exist or not
#Input: link: path to file
#Output: False if file is not exist, True if otherwise
#Kiểm tra file có tồn tại hay không
#Input: link: đường dẫn tới file cần kiểm tra
#Output: False nếu không tìm thấy file, True nếu tìm thấy
def checking_file_exist(link):
    if not isfile(link):
        print("File is not avalable: ", link)
        return False
    return True

#Create folder name
#Tạo tên folder từ thứ tự của folder
#Trong tập UCSDpred1 có 34 folder với tên gọi Train<số thứ tự>
#Ví dụ folder thứ 34 có tên Train034
def get_folder_name(i):
    fname = "Train"
    if i <10:
        fname = fname + "00" + str(i)
    elif i<100:
        fname = fname + "0" + str(i)
    return fname

#Create file name
#Tạo tên fole từ thứ tự của file
#Trong tập UCSDpred1 mỗi folder có 200 frame ảnh, mỗi ảnh có tên gọi <số thứ tự>.tif
#Ví dụ frame ảnh thứ 200 có tên 200.tif
def get_file_name(i):
    if i < 10:
        fname = "00" + str(i) + ".tif"
    elif i < 100:
        fname = "0" + str(i) + ".tif"
    else:
        fname = str(i) + ".tif"
    return fname

#Checking whether file is image file or not
#Input: file_name: path to file
#Output: True if file is image, False otherwise
#Kiểm tra file có phải là ảnh hay không
#Input: file_name: đường dẫn tới file
#Output: True nếu là file ảnh, False nếu ngược lại
def is_image(file_name):
    name, ext = splitext(file_name)
    if ext==".jpg":
        return True
    return False

#Calculating Optical Flow
#Call calc(frame1, frame2) to calculate optical flow between frame1 and frame2
#Tính optical flow
#Gọi hàm calc(fram1, frame2) để tính optical flow giữa hai frame đó
class LK_Optical_Flow:
    #Init
    #Input: width, height: frame size
    #       winSize: size of slide windows used in LK optical flow
    #Input: width, height: kích thước frame
    #       winSize: kích thước cửa sổ sử dụng trong thuật toán LK optical flow
    def __init__(self, width, height, winSize=20):
        self.width = width
        self.height = height
        self.winSize = winSize
        self.max_distance = np.sqrt(2)*self.winSize
        self.prevPts = [[(int(i)%self.width, int(i/self.width))] for i in range(self.width*self.height)]
        self.prevPts = np.asarray(self.prevPts, dtype='float32')

    #Cast input values to [0, 255]
    #Input: x: values
    #Output: casted value
    #Chuyển các giá trị input về đoạn [0, 255]
    #Input: x: giá trị cần chuyển đổi
    #Output: giá trị đã chuyển đổi
    def cast(self, x):
        x[np.where(x>255)] = 255
        return x
    
    #Calculating LK optical flow between frame1 and frame2
    #Input: frame1, frame2: gray frames
    #Output: LK optical flow with HSV encoding
    #Tính toán LK optical flow giữa hai frame: frame1 và frame2
    #Input: frame1, frame2: các frame ảnh trắng đen
    #Ouput: LK optical flow được mã hóa thành ảnh HSV
    def calc(self, frame1, frame2):
        H = self.height
        W = self.width

        nextPts, status, err = cv2.calcOpticalFlowPyrLK(frame1, frame2, self.prevPts, None, winSize=(self.winSize, self.winSize))
        
        dx = nextPts[:,0,0]-self.prevPts[:,0,0]
        dy = nextPts[:,0,1]-self.prevPts[:,0,1]

        angle = np.arctan2(dy, dx)*180/np.pi
        angle[np.where(angle<0)] += 360

        distance = np.sqrt(dx*dx + dy*dy)
        
        Hue = (self.cast(angle*255/360)*status[:,0]).astype('uint8').reshape((H,W,1))
        Sar = (self.cast(distance*255/self.max_distance)*status[:,0]).astype('uint8').reshape((H,W,1))
        Val = np.full_like(Hue, 255)
        HSV = np.concatenate((Hue,Sar,Val), axis=2)
        return HSV

#Read all image files' name in folder
#Input: batch_folder: parent folder
#Output: list of image files' name in this folder
#Đọc tên tất cả các file ảnh trong thư mục
#Input: batch_folder: tên thư mục
#Output: danh sách tên các file ảnh trong thư mục
def get_list_sample(batch_folder):
    list_file = []
    for f in listdir(batch_folder):
        file_path = batch_folder + "/" + f
        if isfile(file_path) and is_image(file_path):
            list_file.append(file_path)
    return list_file

#Get training dataset
#Input: folder: parent folder
#Output: X, y
#Đọc dataset
#Input: folder: thư mục chứa dataset
#Output: X, y dùng trong huấn luyện
def get_dataset(folder):
    print ("Check files")
    list_file = get_list_sample(folder)
    print ("Loading dataset")
    X = np.empty((len(list_file), 32, 32, 3))
    Y = np.ones((len(list_file), 1))
    for i in range(len(list_file)):
        print(list_file[i])
        img = cv2.imread(list_file[i], cv2.IMREAD_ANYCOLOR)
        X[i] = img
    return X, Y


    


import cv2
import net
from keras import backend as K
import numpy as np 
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
from utils import *

################################################################
# This is main file of programe
# Đây là chương trình chính
################################################################

#Create working window
#Tạo cửa sổ làm việc
class MainProgram(QWidget):
    def __init__(self, weight_path="doc.h5", parent=None):
        super(MainProgram, self).__init__(parent)

        #Checking weight path is exist
        #Kiểm tra file trọng số có tồn tại hay không
        self.ready = True
        if checking_file_exist(weight_path)==False:
            self.ready = False
    
        #Load model
        #Load mô hình
        if self.ready==True:
            self.model = net.DOC(training=False)
            self.model.compile(loss=self.dummy_loss, optimizer='SGD')
            self.model.load_weights(weight_path)

        #Create elements of window
        #Tạo các đối tượng trên cửa sổ

        #Add vertical layout
        #Đặt layout dạng thẳng
        layout = QVBoxLayout()
        #Add button "Using real camera"
        #Thêm button "Using real camera"
        self.btnCamera = QPushButton("Using real camera")
        self.btnCamera.clicked.connect(self.camera)
        layout.addWidget(self.btnCamera)
        
        #Add button "Open test video"
        #Thêm button "Open test video"
        self.btnVideo = QPushButton("Open test video")
        self.btnVideo.clicked.connect(self.video)
        layout.addWidget(self.btnVideo)

        #Add status label
        #Thêm label trạng thái
        self.lblStatus = QLabel("Status..")
        layout.addWidget(self.lblStatus)

        if self.ready == False:
            self.lblStatus.setText("Error: could not find out weigts path")
        else:
            self.lblStatus.setText("Ready.")

        self.setLayout(layout)
        self.setWindowTitle("Abnormal detection")
        self.setGeometry(100, 100, 500, 50)

    #Handle "Using real camera" clicked event
    #Xử lý sự kiện button "Using real camera" được click
    def camera(self):
        self.use_camera = True
        self.run()

    #Handle ""Open test video"" clicked event
    #Xử lý sự kiện button ""Open test video"" được click
    def video(self):
        self.use_camera = False
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        if dlg.exec_():
            self.input_video_path = dlg.selectedFiles()[0]
            self.lblStatus.setText(self.input_video_path)
        self.run()
    
    def dummy_loss(self, y_true, y_pred):
        zero = K.variable(0.0)
        return zero

    def run(self):
        if self.ready == False:
            pass
        else:
            frame1 = None
            frame2 = None
            frame3 = None
            frame4 = None
            frame5 = None
            LKOF = LK_Optical_Flow(238, 158)

            if self.use_camera==True:
                #Read frame from camera if use camera
                #Đọc frame ảnh từ camera nếu sử dụng camera
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FPS, 10)
            else:
                #Read frame from video if use video
                #Đọc frame ảnh từ video nếu sử dụng video
                cap = cv2.VideoCapture(self.input_video_path)
                cap.set(cv2.CAP_PROP_FPS, 10)

            while(True):
                if cap.isOpened()==False:
                    break

                #Read frame
                ret, frame = cap.read()
                #Convert to gray
                frame = cv2.resize(frame,(238, 158))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if frame1 is None:
                    frame1 = gray
                elif frame2 is None:
                    frame2 = gray
                elif frame3 is None:
                    frame3 = gray
                elif frame4 is None:
                    frame4 = gray
                elif frame5 is None:
                    frame5 = gray
                else:
                    frame1 = np.copy(frame2)
                    frame2 = np.copy(frame3)
                    frame3 = np.copy(frame4)
                    frame4 = np.copy(frame5)
                    frame5 = gray
                
                if frame5 is not None:
                    #Calculating 4 optical flow of 5 frame
                    #Tính 4 optical flow của 5 frame ảnh liên tiếp
                    of1 = LKOF.calc(frame1, frame2)
                    of1 = cv2.cvtColor(of1, cv2.COLOR_HSV2BGR)
                    of2 = LKOF.calc(frame2, frame3)
                    of2 = cv2.cvtColor(of2, cv2.COLOR_HSV2BGR)
                    of3 = LKOF.calc(frame3, frame4)
                    of3 = cv2.cvtColor(of3, cv2.COLOR_HSV2BGR)
                    of4 = LKOF.calc(frame4, frame5)
                    of4 = cv2.cvtColor(of4, cv2.COLOR_HSV2BGR)

                    #Create 32x32 batch from 4 optical flow images
                    #And predict each batch
                    #If a batch is abnormal event, mark down red box
                    #Tạo các batch 32x32 từ 4 optical flow liên tiếp
                    #Dự đoán mỗi batch có phải là sự kiện bất thường không
                    #Nếu có thì đóng khung đỏ
                    for m in range(0, int(238/16)):
                        for n in range(0, int(158/16)):
                            bgr = np.zeros((32,32,3))
                            x = m*16
                            y = n*16
                            bgr[0:16, 0:16, :] = of1[y:y+16, x:x+16, :]
                            bgr[0:16, 16:36, :] = of2[y:y+16, x:x+16, :]
                            bgr[16:32, 0:16, :] = of3[y:y+16, x:x+16, :]
                            bgr[16:32, 16:32, :] = of4[y:y+16, x:x+16, :]

                            bgr = np.expand_dims(bgr, axis=0)
                            res = self.model.predict(bgr)
                            if res == -1:
                                cv2.rectangle(frame1, (x,y), (x+16, y+16), (255, 0, 0), 1)
                    
                    cv2.imshow('Video', frame1)
                
                if (cv2.waitKey(1) & 0xFF == ord('q')):
                    break
            
            cap.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainProgram()
    window.show()
    sys.exit(app.exec_())



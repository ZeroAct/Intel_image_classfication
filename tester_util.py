import cv2, os, threading
import numpy as np

from PyQt5 import QtGui

def GetMedianFrameToPixmap(fname):
    cap = cv2.VideoCapture(fname)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(int(n//2)):
        ret, frame = cap.read()
    cap.release()
    
    return Cv2QPixmap(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), w, h, fps, total_frame

def ProccessImage(fname, w, h, output_dir='temp/', name=None):
    print(fname, w, h)
    n = np.fromfile(fname, np.uint8)
    img = cv2.imdecode(n, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (int(w), int(h)), cv2.INTER_CUBIC)
    if name:
        cv2.imwrite(output_dir+"_tb.png", img)
    else:
        cv2.imwrite(output_dir+"_back.png", img)

def Cv2QPixmap(frame):
    height, width, channel = frame.shape
    bytesPerLine = 3 * width
    qImg = QtGui.QImage(frame.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
    return  QtGui.QPixmap.fromImage(qImg)

def RunThread(Section, Query, final=False):
    th = threading.Thread(target=CMDRun, args=(Section, Query, final))
    th.start()

def CMDRun(Section, Query, final):
    res = os.popen(Query).read()
    
    print(res)
    res = res.split("\n")
    
    if not final:
        if "Done" in res:
            if Section.N != 2:
                Section.MainWindow.Sections[Section.N + 1].BrowseBtn.setEnabled(True)
    else:
    	print(123145)
    	Section.Finish()
            

'''
    Diese Datei vergleicht GroundTruth Daten und detektierte Objekte miteinander
    
    classes:
        GroundTruth:    Is able the hold Data of a GroundTruth object
        BBox:               Contains data for a ground-truth-box inside a GroundTruth object
        
    methods:
        parse:                                   Reads GroundTruth-data from file and returns an Array ob all traffic lights in one picture
        loadSSDData:                       Reads Json with detected objects and returns json-object for all traffic lights in one folder
        bb_intersection_over_union:    Gets two bounding Boxes and return the Intersection over union value 

'''

import json
import numpy as np
import os
import matplotlib.pyplot as plt

#Counter
'''
    Car = 0
    Pedestrian = 1
    TrafficLight = 2
    Truck = 3
    Bus = 4

'''
gtClassCount = {
    "car": 0,
    "pedestrian": 0,
    "trafficLight": 0,
    "truck": 0,
    "bus": 0,
    "something": 0
}

sSDClassCount = {
    "car": 0,
    "pedestrian": 0,
    "trafficLight": 0,
    "truck": 0,
    "bus": 0,
    "something": 0
}

countTrafL = 0
countImgSSD = 0
countImgGT = 0


dataSet_path = '/home/sven/Dokumente/Sven/Dateien/images/'
#dataSet_path = '/home/sven/Dokumente/Sven/Dateien/Neue_Daten001_953/'
#dataSet_path = '/home/sven/Dokumente/Sven/Dateien/Neue_Daten002_977/'
#dataSet_path = '/home/sven/Dokumente/Sven/Dateien/Neue_Daten003_243/'
#dataSet_path = '/home/sven/Dokumente/Sven/Dateien/Neue_Daten004_723/'
#dataSet_path = '/home/sven/Dokumente/Sven/Dateien/Neue_Daten005_559/'
#dataSet_path = '/home/sven/Dokumente/Sven/Dateien/Neue_Daten006_746/'
#dataSet_path = '/home/sven/Dokumente/Sven/Dateien/Neue_Daten007_813/'
#dataSet_path = '/home/sven/Dokumente/Sven/Dateien/Neue_Daten008_3284/'
#dataSet_path = '/home/sven/Dokumente/Sven/Dateien/Test_Daten_001/'
json_path = dataSet_path + 'SSD_Detections/dataGMM_Height.json'
main_path = "main_img/"
gT_path = '2DGroundTruth/2dgT_'

def loadAllGT():
    count = len([name for name in os.listdir(dataSet_path+'2DGroundTruth/') if os.path.isfile(os.path.join(dataSet_path+'2DGroundTruth/', name))]) +1
    PATHS = [ os.path.join(dataSet_path+'2DGroundTruth/', '2dgT_{}.txt'.format(i))for i in range(1, count)]
    GroundTruth = []
    for data in PATHS:
        GroundTruth.append(parse(data))
    return GroundTruth

class BBox:
  x = 0
  y = 0
  width = 0
  height = 0
  
  def get(self):
    return [self.y, self.x, self.y+self.height, self.x+self.width]
  
  def getX(self):
    return self.x
  
  def getY(self):
    return self.y;
  
  def getWidth(self):
    return self.width
    
  def getHeight(self):
    return self.height
    
  def setX(self, v):
    self.x = v
  
  def setY(self, v):
    self.y = v
  
  def setWidth(self, v):
    self.width = v
    
  def setHeight(self, v):
    self.height = v
    
  def printBox(self):
    print("Ymin: " +  str(self.y) + "; Xmin: " + str(self.x) + "; Ymax: " + str(self.y + self.height) +  "; Xmax: " + str(self.x + self.width))
    
class GroundTruth:
  id = -1
  label = ""
  score = -1
  bbox = BBox()
  
  def setID(self, i):
    self.id = i
  
  def getID(self):
    return self.id
    
  def setLabel(self, i):
    self.label = i
  
  def getLabel(self):
    return self.label
  
  def setBBox(self, i):
    self.bbox = i
  
  def getBBox(self):
    return self.bbox
    
  def setScore(self, i):
    self.score = i
    
  def getScore(self):
    return self.score

def parse(p):
  global countImgGT
  f = open(p, "r")
  contents = f.read()
  f.close()
  if "TrafficLight" in contents:
    countImgGT += 1
  contents = contents.replace("[", " ")
  contents = contents.replace("]", " ")
  elements = contents.split(",")
  #List for all Boxes in one Frame
  data = []
  #print("Len: " + str(len(contents)))
  if len(contents) < 5:
    return data
  #Für jede Box
  #print(p)
  for i in range(len(elements)):
    # Zerteile den String in seine Zeilen
    lines = elements[i].splitlines()
    #Create GroundTruth object and save ID and Label
    temp = GroundTruth()
    temp.setID(int(lines[6].replace("id: ", "").strip()))
    label = lines[7].replace("label: ", "").strip()
    temp.setLabel(label)
      #Create BoundingBox Object and save x, y, width and height.
    tempBox = BBox()
    x1 = float(lines[10].replace("x: ", "").strip())
    y1 = float(lines[11].replace("y: ", "").strip())
    w = float(lines[12].replace("width: ", "").strip())
    h = float(lines[13].replace("height: ", "").strip())
    if "TrafficLight" in label :
        gtClassCount['trafficLight'] += 1
    elif "Pedestrian" in label:
        gtClassCount['pedestrian'] +=1
    elif "BoxTruck" in label:
        gtClassCount['truck'] += 1
    elif "SUV" in label:
        gtClassCount['car'] += 1
    elif "Sedan" in label:
        gtClassCount['car'] += 1
    elif "Hatchback" in label:
        gtClassCount['car'] += 1
    elif "Jeep" in label:
        gtClassCount['car'] += 1
    elif "SchoolBus" in label:
        gtClassCount['bus'] += 1
    else:
        gtClassCount['something'] += 1
        print(lines[7].replace("label: ", "").strip())
        print("test")
  
    tempBox.setX(x1-(w/2))
    tempBox.setY(y1-(h/2))
    tempBox.setWidth(w)
    tempBox.setHeight(h)
    #Give BBox to GroundTruthObject and save in List
    temp.setBBox(tempBox)
    data.append(temp)
  return data

def loadSSDData(path, images, classCount):
    SSDTrafficLight = [[] ] * (images)
    counter = 0
    with open(path) as json_file:
        data = json.load(json_file)
        for i in data['images']:
            boxes = []
            for c in i['Classes']:
                temp = GroundTruth()
                if c.get('ClassName') == 'traffic light':
                    temp.setID(c.get('Class'))
                    temp.setLabel("TrafficLight")
                    classCount['trafficLight'] += 1
                    #print("test: "+ str(counter))
                    counter +=1
                elif c.get('Class') == 3:
                    classCount['car'] += 1
                    temp.setID(c.get('Class'))
                    temp.setLabel("Car")
                elif c.get('Class') == 1:
                    classCount['pedestrian'] += 1
                    temp.setID(c.get('Class'))
                    temp.setLabel("Pedestrian")
                elif c.get('Class') == 6:
                    classCount['bus'] += 1
                    temp.setID(c.get('Class'))
                elif c.get('Class') == 8:
                    classCount['truck'] += 1
                    temp.setID(c.get('Class'))
                else:
                    classCount['something'] += 1
                    temp.setID(c.get('Class'))
                #print(str(c.get('Score')) + " ; Load SSD Score")
                temp.setScore(c.get('Score'))
                tempBox = BBox()
                breite = (c.get('Box')[3] - c.get('Box')[1]) * 720
                hoehe = (c.get('Box')[2] - c.get('Box')[0]) * 480
                tempBox.setX(c.get('Box')[1]*720)
                tempBox.setY(c.get('Box')[0] * 480)
                tempBox.setWidth(breite)
                tempBox.setHeight(hoehe)
                temp.setBBox(tempBox)
                boxes.append(temp)
                #print("index: ", (i.get('Image')-1))
            #print("länge von image " + str(i.get('Image')-1) + ": " + str(len(boxes)))
            SSDTrafficLight[i.get('Image')-1] = boxes
    return SSDTrafficLight, classCount
      
'''
Source: 
https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
'''          
def gt_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    yA = max(boxA[0], boxB[0])
    xA = max(boxA[1], boxB[1])
    yB = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

cnt_false_detection = 0
cnt_misdetection = 0
cnt_truedetection = 0
cnt_doubleboxes = 0
gt_trafLights = 0
ssd_trafLights = 0
def compareTrafLights(SSD, GT, threshold):
    cnt_true_detection_per_frame =[]
    cnt_misdetection_per_frame = []
    cnt_false_detection_per_frame = []
    
    cnt_truedetection_frame = 0
    cnt_misdetection_frame = 0
    cnt_false_detection_frame = 0
    
    for image in range(len(GT)):
        ssd_ampel = 0
        gt_ampel = 0
        try:
            for c in SSD[image].get('Classes'):
                if 'TrafficLight' in c.getLabel() and c.getScore() > threshold:
                    ssd_ampel += 1
        except:
            ssd_ampel = 0
                
        for g in range(len(GT[image])):
            if ('TrafficLight' in GT[image][g].getLabel()):
                gt_ampel += 1
                
        cnt_truedetection_frame, cnt_false_detection_frame, cnt_misdetection_frame = compareBoxes(SSD[image], GT[image], threshold, cnt_truedetection_frame, cnt_false_detection_frame, cnt_misdetection_frame)
        
        #calc values per frame
        cnt_false_detection_per_frame.append(cnt_false_detection_frame)
        cnt_false_detection_frame = 0
        cnt_misdetection_per_frame.append(cnt_misdetection_frame)
        cnt_misdetection_frame = 0
        cnt_true_detection_per_frame.append(cnt_truedetection_frame)
        cnt_truedetection_frame = 0
    
    return cnt_true_detection_per_frame, cnt_false_detection_per_frame, cnt_misdetection_per_frame
             
def compareBoxes(ssd, gt, threshold, cnt_truedetection_frame, cnt_false_detection_frame, cnt_misdetection_frame):
    boxes_used = []
    global cnt_doubleboxes
    global ssd_trafLights
    global gt_trafLights
    if not len(ssd) == 0:
        for ampel in ssd:
            if "TrafficLight" in ampel.getLabel() and ampel.getScore() > threshold:
                ampel_matched = False
                ssd_trafLights += 1
                for box in gt:
                    if "TrafficLight" in box.getLabel():
                        if gt_intersection_over_union(ampel.getBBox().get(), box.getBBox().get()) > 0.4:
                            global cnt_truedetection
                            cnt_truedetection += 1
                            cnt_truedetection_frame += 1
                            gt_trafLights += 1
                            if box in boxes_used:
                                cnt_doubleboxes += 1
                            else:
                                boxes_used.append(box)
                            ampel_matched = True
                            break
                if not ampel_matched and ampel.getBBox().getWidth() > 8:
                    global cnt_misdetection
                    cnt_misdetection += 1
                    cnt_misdetection_frame += 1
                elif ampel.getBBox().getWidth() < 8:
                    ssd_trafLights -= 1
                ampel_matched = False
    for box in gt:
        if not box in boxes_used and "TrafficLight" in box.getLabel() and box.getBBox().getWidth() > 8:
            global cnt_false_detection
            gt_trafLights += 1
            cnt_false_detection += 1
            cnt_false_detection_frame += 1
    return cnt_truedetection_frame, cnt_false_detection_frame, cnt_misdetection_frame

def do_something(GTData, j_path, number):    
    #GTData = loadAllGT()
    #Load Json File from SSD
    SSDTrafficLights, normalSSDClassCount = loadSSDData(j_path, len(GTData), sSDClassCount)

    GTAmpel = gtClassCount['car']
    
    print("\n------------GroundTruth--------------")
    print(gtClassCount)
    print("Bilder mit Ampeln: " + str(countImgGT))
    
    print("\n----------------SSD------------------")
    print(normalSSDClassCount)

    
    dataToPlot = []
    threshToPlot = []
    
    tableData = []
    global cnt_truedetection
    global ssd_trafLights
    global cnt_false_detection
    global cnt_misdetection
    
    tpr = []
    fp = []
    
    for i in range(0, 10):
        if i == 0:
            i = 0.01
        else:
            i = float(i)/10.0
        row = []
        
        cnt_true_detection_per_frame, cnt_false_detection_per_frame, cnt_misdetection_per_frame = compareTrafLights(SSDTrafficLights, GTData, i)
        #first graph
        threshToPlot.append(i)
        dataToPlot.append(float(cnt_truedetection)/float(GTAmpel))
        #table
        row.append(i)
        row.append(GTAmpel)
        row.append(ssd_trafLights)
        row.append(cnt_truedetection)
        #row.append(cnt_doubleboxes)
        row.append(cnt_false_detection)
        row.append(cnt_misdetection)
        tableData.append(row)
        #second graph
        tpr.append(np.mean(cnt_true_detection_per_frame) / (np.mean(cnt_true_detection_per_frame) + np.mean(cnt_false_detection_per_frame)))
        fp.append(np.mean(cnt_misdetection_per_frame))


        #reset the counter and do everything again with the masked image
        cnt_doubleboxes = 0
        cnt_false_detection = 0 
        cnt_misdetection = 0
        cnt_truedetection = 0
        ssd_trafLights = 0
        cnt_true_detection_per_frame = []
        cnt_false_detection_per_frame = [] 
        cnt_misdetection_per_frame = []
        ###########################################################################################################################

    '''
        Plotting first Graphs:
            true detections / GT Trafic lights vs threshold
    '''
    plt.figure(figsize=(10.4, 3.8))
    plt.subplot(121)
    plt.plot(threshToPlot, dataToPlot, 'k')
    plt.xlabel('threshold')
    plt.ylabel('True Positive / GroundTruth')
    #plt.title('Using the normal picture')
    plt.grid(True)
    
    plt.subplot(122)
    plt.plot(fp, tpr, 'k')
    plt.xlabel('fp')
    plt.ylabel('tpr')
    #plt.title('Using the normal picture')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(dataSet_path+'SSD_GMM_H_graphen_'+number+'.png')
    plt.close()
    
    '''
        Plotting Tables of the Data 
    '''
    #fig  = plt.figure(dpi=150)
    fig  = plt.figure(figsize=(10.4,3.8))
    columns = ('Threshold', 'GT Traffic Lights', 'SSD Detections', 'True Detections', 'Missed Detections', 'Misdetections')
    ax = fig.add_subplot(111)
    the_table = ax.table(cellText=tableData, colLabels=columns, loc='center', colWidths=[0.24]*6)
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(8)
    the_table.scale(1,1)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    for pos in ['right','top','bottom','left']:
        plt.gca().spines[pos].set_visible(False)
    #plt.title('Normal image')
    ax.axis('off')
    plt.subplots_adjust(left=0.16, bottom=0.1, right=0.84,  top=0.9, wspace=0.2, hspace=0.74)
    plt.savefig(dataSet_path+'SSD_GMM_H_table_'+number+'.png')
    plt.close()
    
    return tableData
    
    
if __name__ == "__main__":
  GTData = loadAllGT()
  allData = []
  count = [ "0.07", "0.08", "0.09", "0.10", "0.11", "0.12", "0.13", "0.14", "0.15", "0.16", "0.17", "0.18", "0.19", "0.20", "0.21", "0.22", "0.23"]
  json_path = dataSet_path + 'SSD_Detections/dataGMM_Height_'
  for i in count:
    allData.append(do_something(GTData, json_path + i + '.json', i))
  print("Shape allData: ", np.shape(allData))
  
  count2 = [ 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23]
  lab = ["0.01", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
  precision = []
  
  t = []
  m = []
  #gehe über alle oberen Grenzen
  for d in allData:
    p = []
    m1 = []
    t1 = []
    for k in d:
      p.append( k[3] / ( k[3] + k[5]) )
      m1.append(k[5])
      t1.append(k[3])
      
    t.append(t1)
    m.append(m1)
    precision.append(p)
  precision = np.asarray(precision)
  #print(precision)
  precision = precision.transpose()
  fig, ax = plt.subplots(1)
  ax.grid(True)
  for p in range(3, len(precision)-1):
    ax.plot(count2, precision[p], label=lab[p])
  ax.legend(loc="lower left", frameon=False)
  ax.set_title("Precision")
  plt.show()
  plt.close()
  
  
  t = np.asarray(t)
  t = t.transpose()
  m = np.asarray(m)
  m = m.transpose()
  
  fig, ax = plt.subplots(1,2, figsize=(10.4, 3.8))
  ax[0].grid(True)
  for det in range(3, len(t)-1):
    ax[0].plot(count2, t[det], label=lab[det])
  ax[0].legend(loc="upper left", frameon=False)
  ax[0].set_title('true-detections')
  
  ax[1].grid(True)
  for det in range(3, len(m)-1):
    ax[1].plot(count2, m[det], label=lab[det])
  ax[1].legend(loc="upper left", frameon=False)
  ax[1].set_title('mis-detections')
  
  plt.show()
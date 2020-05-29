import json
import numpy as np

dataSet_path = '/home/sven/Dokumente/Sven/Dateien/images/'
json_path = dataSet_path + 'SSD_Detections/data.json'
main_path = "main_img/"
gT_path = '2DGroundTruth/2dgT_'


class BBox:
  x = 0
  y = 0
  width = 0
  height = 0
  
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

def parse(p):
  f = open(p, "r")
  contents = f.read()
  f.close()
  contents = contents.replace("[", " ")
  contents = contents.replace("]", " ")
  elements = contents.split(",")
  #List for all Boxes in one Frame
  data = []
  for i in range(len(elements)):
    lines = elements[i].splitlines()
    #Create GroundTruth object and save ID and Label
    temp = GroundTruth()
    temp.setID(int(lines[6].replace("id: ", "").strip()))
    temp.setLabel(lines[7].replace("label: ", "").strip())
    #Create BoundingBox Object and save x, y, width and height.
    tempBox = BBox()
    x1 = float(lines[10].replace("x: ", "").strip())
    y1 = float(lines[11].replace("y: ", "").strip())
    w = float(lines[12].replace("width: ", "").strip())
    h = float(lines[13].replace("height: ", "").strip())
    tempBox.setX(x1-(w/2))
    tempBox.setY(y1-(h/2))
    tempBox.setWidth(w)
    tempBox.setHeight(h)
    #Give BBox to GroundTruthObject and save in List
    temp.setBBox(tempBox)
    data.append(temp)
  return data



def loadSSDData():
    SSDTrafficLight = []
    with open(json_path) as json_file:
        data = json.load(json_file)
        countTrafL = 0
        countImg = 0
        for i in data['images']:
            countImg = countImg+ 1
            #print(i)
            for c in i['Classes']:
                #print(c)
                if c.get('ClassName') == 'traffic light':
                    #print("Ampel")
                    countTrafL = countTrafL + 1
                    SSDTrafficLight.append(i)
        print("Ampeln: ", str(countTrafL))
        print("Bilder: ", str(countImg), "\n")
    return SSDTrafficLight
      
'''
Source: 
https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
'''      
def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
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

if __name__ == "__main__":
    SSDTrafficLights = loadSSDData()
    for i in SSDTrafficLights:
        img_number = i.get('Image')
        path = dataSet_path+gT_path+str(img_number)+'.txt'
        gt = parse(path)
        boxA = []
        for c in i['Classes']:
            if c.get('ClassName') == 'traffic light':
                b = c.get('Box')
                boxA.append(b[1]*1920) #Xmin
                boxA.append(b[0]*1080) #Ymin
                boxA.append(b[3]*1920) #Xmax
                boxA.append(b[2]*1080) #Ymax
        for g in range(len(gt)):
            if ('TrafficLight' in gt[g].getLabel()):
                b2 = gt[g].getBBox()
                boxB = []
                boxB.append(b2.getX())
                boxB.append(b2.getY())
                boxB.append(b2.getX() + b2.getWidth())
                boxB.append(b2.getY() + b2.getHeight())
                if bb_intersection_over_union(boxA, boxB) > 0.5:
                    print("Detected in Picture: ", img_number)
                    print("Ymin: " +  str(boxA[1]) + "; Xmin: " + str(boxA[0]) + "; Ymax: " + str(boxA[3]) +  "; Xmax: " + str(boxA[2]))
                    print("GroundTruth")
                    b2.printBox()
                    print("IoU: " +str(bb_intersection_over_union(boxA, boxB)) + "\n\n")
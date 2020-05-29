import matplotlib.pyplot as plt
import matplotlib.image as mping
import matplotlib.patches as patches
import ast
from PIL import Image
import numpy as np

image_path = "./images/"
main = "main_img/main_"

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
  

if __name__ == "__main__":
  for i in range(1,2):
    im = np.array(Image.open(image_path+main+str(i)+".jpg"), dtype=np.uint8)
    
    fig,ax = plt.subplots(1)
    ax.imshow(im)
    
    boxes = parse(image_path+"2DGroundTruth/2dgT_"+str(i)+".txt")
    for i in range(len(boxes)):
      boxes[i].getBBox().printBox()
      rect = patches.Rectangle((boxes[i].getBBox().getX(), boxes[i].getBBox().getY()), boxes[i].getBBox().getWidth(), boxes[i].getBBox().getHeight(), linewidth=1, edgecolor='b', facecolor='none')
      ax.add_patch(rect)
    
    plt.show()

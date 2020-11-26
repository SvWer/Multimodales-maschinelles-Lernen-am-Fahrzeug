# Multimodales-maschinelles-Lernen-am-Fahrzeug

In diesem Repository finden sich Skripte, Dateien und Anleitungen, die verwendet, erstellt oder geändert wurden, um die Masterarbeit mit dem Thema "Multimodales maschinelles Lernen in Fahrzeugen" zu absolvieren.

Das Vorgehen gliedert sich dabei in Folgende Schritte:
1. Installation vom LGSVL-Simulator
2. Modifikation des Simulators, damit dieser Ampeln als Ground-Truth-Daten  liefert.
3. Sammeln von Daten über die ROS-Bridge
4. Ampel-Detection mit Hilfe des pre-trained SSD-Algorithmus mit dem COCO-Datensatz


<b>Installation vom LGSVL-Simulator</b>

Damit der Simulator und die Map bearbeitet werden können, muss Unity3D als Editor installiert werden. Dazu UnityHub für Ubuntu herunterladen und von dort aus Unity 2019.3.3f oder neuer installieren. 
Damit der Simulator ausgeführt werden kann muss Cuda zusammen mit dem neusten Treiber installiert werden. Dazu den Anweisungen der Cuda-Webseite folgen.
Damit der Simulator über die PythonAPI genutzt werden kann, müssen noch ein paar weitere Dinge installiert werden:

1. $sudo apt install python3-pip
2. $pip3 install numpy
3. $pip3 install -U setuptools
4. $pip3 install tensorflow-gpu
5. $python3 -m pip install -U matplotlib
6. $sudo apt-get install -y nodejs
7. $sudo apt-get install libvulkan1
8. $sudo apt install git
9. $git clone https://github.com/lgsvl/PythonAPI
10. $cd PythonAPI
11. $pip3 install --user -e.
12. $pip3 install opencv-python
13. $pip3 install rospkg

Zudem muss ROS auf dem Rechner installiert werden, damit die Daten über die ROS-Bridge gelesen werden können.

1. $sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
2. $sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
3. $sudo apt update
4. $sudo apt install ros-melodic-desktop-full
5. $sudo apt install ros-melodic-rosbridge_*
6. $echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
7. $source ~/.bashrc

Nachdem ROS installiert ist sollte ein Workspace erzeugt und ein paar Packages hinzugefügt werden, damit ROS auch die Topics des Simulators versteht.

1. $mkdir -p ~/workspace/src
2. $cd ~/workspace/
3. $catkin_make
4. $source devel/setup.bash
5. $echo "source devel/setup.bash" >> ~/.bashrc
6. $cd src
7. $git clone https://github.com/lgsvl/rosbridge_suite.git
8. $git clone https://github.com/lgsvl/lgsvl_msgs.git
9. $cd lgsvl_msgs
10. $mkdir build
11. $cd build
12. $cmake ../
13. $make
14. $cd ../../..
15. $catkin_make




<b>Modifikation des Simulators, damit dieser Ampeln als Ground-Truth-Daten  liefert.</b>
Im Ordner Changed-simulator-files befinden sich alle Datein, die bei der Modifikation des Simulators und der Map verändert wurden.

Um die Modifikationen einbauen zu können, ist es nötig den LGSVL als Code für Unity downzuloaden und ihn zu installieren. Anschließend können die Entsprechenden Dateien ersetzt werden.

Zum Downloaden und installieren dieser Anleitung folgen: https://www.lgsvlsimulator.com/docs/build-instructions/

Die geänderten Skripte müssen wie folgt eingefügt werden:

simulator:

  |

  |--Scripts

      |

      |-- Components

            |

            |-- SignalLight.cs

      |

      |--Sensors

          |

          |--GroundTruth2DSensor.cs

          |--GroundTruth2DVisualizer.cs

          |--GroundTruth3DSensor.cs

          |--GroundTruth3DVisualizer.cs
Zudem wurden in allen Bildverarbeitenden Prefabs die Auflösungen für die Bilder angepasst, sodass nicht mehr 1920x1080 verwendet wird, sondern 720*480.

Damit die Bounding Boxen für Ampeln erkannt werden können, müssen nun die Ampeln mit Bounding Boxen versehen werden. Im Ordner SanFrancisco ist das bearbeitete Environment enthalten. Dieses muss unter "Assets/External/Environments" eingefügt werden.

Damit man ein Fahrzeug verwenden kann muss ein solches ebenfalls heruntergeladen und unter "Assets/External/Vehicles" eingebunden werden.

Nach dem builden und Starten, müssen Auto und Environment zu den Ressourcen hinzugefügt werden. Dabei ist es beim Fahrzeug wichtig, die benötigten Sensoren einzutragen, wie es in "vehicle_config.txt" dargestellt ist.

<b>Sammeln von Daten über die ROS-Bridge</b>

Damit über die ROS-Bridge Daten gesammelt werden können, wird ein Skript benötigt, dass als Subscriber die Nachrichten vom Simulator empfängt und diese Daten dann abspeichert. Dazu muss der Ordner subscriber_node aus diesem Repo in den src-Ordner im Workspace kopiert werden. Danach muss im workspace erneut 

  $catkin_make
  
aufgerufen werden. Wenn das gemacht ist, kann der Simulator gestartet werden. 
Dann müssen in unterschiedlichen Terminals folgende Befehle eingegeben werden:
1. $roscore
2. $roslaunch rosbridge_server rosbridge_websocket.launch
3. $rosrun subscriber_node sub2D.py

Damit wird zuerst der Roscore gestartet. Der zweite Befehl startet die Bridge und der dritte Befehl den Subscriber, der die Daten dann abspeichert.
Auf der Webseite muss dann die modifizierte Map und ein Fahrzeug ausgewählt werden, dessen Sensoren so definiert sind, dass die gewünschten Daten geliefert werden. 

Bevor allerdeing die Daten gesammelt werden können. muss die entsprechende Ordnerstruktur gegeben sein. Dabei sind manche Ordner davon Abhängig, welche Sensoren für das Fahrzeug verwendet werden und welche Eventhandler in Sub2D.py definiert werden.

--Dateien
     |
     |-- 2DGrountTruth
     
     |-- 3DGroundTruth
     
     |-- SSD_Detections
     
     |-- depth_img
     
     |-- lidar
     
     |-- main_img
     
     |-- seg_img



<b>Ampel-Detection mit Hilfe des pre-trained SSD-Algorithmus mit dem COCO-Datensatz</b>

Um den SSD, wie hier verwendet, nutzen zu können, muss die Tensorflow object detection API installiert werden. Zusätzlich muss der vortrainierte SSD auf dem Tensorflow Model Zoo gedownloaded werden und in die object detection API integriert werden. 

Anschließend kann der SSD verwendet werden. Dazu wurde in dieser Arbeit die Skripte in SSD_Dateien verwendet. Wichtig dabei ist zum einen der Pfad, an welcher Stelle sich die zu analysierenden Dateien befinden und wo die Ergebnisse gespeichert werden. Sind die Pfade nicht korrekt, stürzen die Programme ab.


<b> Analyse der Ergebnisse</b>

Nachdem der SSD ausgeführt wurde, steht eine json-Datei zur Verfügung, in der alle Detektionen enthalten sind. Wurde die Ordnerstruktur beim Sammeln der Daten entsprechend angelegt, können nun die Skripte im Ordner Auswertungen ausgeführt werden. Diese analysieren abhängig vom jeweiligen SSD-Algorithmus die Ergebnisse und erzeugen dabei die entsprechenden Diagramme und Tabellen.

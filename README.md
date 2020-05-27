# Multimodales-maschinelles-Lernen-am-Fahrzeug

In diesem Repository finden sich Skripte, Dateien und Anleitungen, die verwendet, erstellt oder geändert wurden, um die Masterarbeit mit dem Thema "Multimodales maschinelles Lernen in Fahrzeugen" zu absolvieren.

Das Vorgehen gliedert sich dabei in Folgende Schritte:
1. Installation vom LGSVL-Simulator
2. Modifikation des Simulators, damit dieser Ampeln als Ground-Truth-Daten  liefert.
3. Sammeln von Daten über die ROS-Bridge
4. Ampel-Detection mit Hilfe des pre-trained SSD-Algorithmus mit dem COCO-Datensatz


#Installation vom LGSVL-Simulator
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

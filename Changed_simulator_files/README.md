# Modifikation-LGSVL
In diesem Repository befinden sich alle geänderten Dateien des LGSVL Simulators, um Ampeln als Ground-Truth Objekte darstellen zu lassen. 

Um die Modifikationen einbauen zu können, ist es nötig den LGSVL als Code für Unity downzuloaden und ihn zu installieren. Anschließend können die Entsprechenden Dateien ersetzt werden.

Zum Downloaden und installieren dieser Anleitung folgen:
https://www.lgsvlsimulator.com/docs/build-instructions/

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

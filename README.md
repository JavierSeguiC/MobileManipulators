# MobileManipulators
Repository for the course "Mobile Manipulators" created by the Mobile Robots Group UC3M.

## Installation

> [!CAUTION]  
> Es necesario instalar el webots por terminal. Webots 2025 solo esta soportado para Ubuntu 22.04 y 24.04. Si tu versio nes Ubuntu 20.04 debes seguir los siguientes pasos de instalacion para isntalar el Webots 2023b.

### Installation Webots 2023b
1) Ir a: [Github Webots](https://github.com/cyberbotics/webots/releases), bajar hasta 2023b y descaragr el .tar: `webots-R2023b-x86-64.tar.bz2`.
2) Opcional: Checkear que esta todo lo necesario instalado
```bash
sudo apt update
sudo apt install build-essential python3 python3-pip git curl wget -y
sudo apt install libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev -y
```

3) Extraer:
```bash
cd ~/Downloads
tar -xjf webots-R2023b-x86-64.tar.bz2
```

4) Mover:
```bash
sudo mv webots /opt/webots
```
5) Meter en el bashrc:
```bash
echo 'export PATH=$PATH:/opt/webots' >> ~/.bashrc
source ~/.bashrc
```

### Preparar ficheros para Webots

- Checkear los .proto ya que las rutas estan hardcodeadas:
1) Esto esta en Tiago.proto: 
```bash
EXTERNPROTO "/home/nox/Escritorio/MobileManipulatorsLab/MobileManipulators/protos/Astra.proto"
EXTERNPROTO "/home/nox/Escritorio/MobileManipulatorsLab/MobileManipulators/protos/TiagoBase.proto"
```

2) Esto esta en el mundo a cargar:
```bash
EXTERNPROTO "/home/nox/Escritorio/MobileManipulatorsLab/MobileManipulators/protos/Tiago.proto"
```

En cada fichero, sustituir `/home/nox/Escritorio/MobileManipulatorsLab/MobileManipulators/` por tu ruta.

- Crear un entorno:
```bash
conda create -n MobileManipulators python==3.8
conda activate MobileManipulators
```
- Instalar las depedencias en Python mediante
```bash
pip install -r requirements.txt
```




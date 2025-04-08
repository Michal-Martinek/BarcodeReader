I'm doing this project as my graduation project for informatics.  
You can read the propositions [here](https://github.com/Michal-Martinek/BarcodeReader/blob/main/BarcodeReader-propositions.md) (in Czech)

## Features
- [x] successfully recognizes **85 %** of test images (**10 %** rate of overdetection)
- [x] UI
  - [x] single window
  - [x] displaying of intermediate detection steps
  - [x] choosing image input - file, camera, dataset
  - [x] clickable scanlines for additional info
    - lightness along scanline, detected spans, lightness histogram, detection results + checksum
  - [x] slider for dynamically changing scanline distance
  - [x] easy copying of detected code
- [ ] using detected code (displaying product info)

## Setup
- initial setup (Python 3.13.0)
```sh
pip install -r requirements.txt
```
- running (GUI version)
```sh
python main.py
```
- bundling using [PyInstaller](https://pyinstaller.org/en/stable/)
```sh
./build.bat
```

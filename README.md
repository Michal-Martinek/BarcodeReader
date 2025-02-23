I'm doing this project as my graduation project for informatics.  
You can read the propositions [here](https://github.com/Michal-Martinek/BarcodeReader/blob/main/BarcodeReader-propositions.md) (in Czech)

## Roadplan
- [x] successfully recognizes **85 %** of test images (**10 %** rate of overdetection)
- [ ] UI
  - [ ] one window
  - [ ] displaying of progress
  - [ ] choosing image input (camera, dataset, file...)
- [ ] using detected code (displaying product info)

## Setup
- initial setup (Python 3.13.0)
```sh
pip install -r requirements.txt
```
- to bundle using [PyInstaller](https://pyinstaller.org/en/stable/)
```sh
./build.bat
```

pyinstaller --onefile --specpath "build" BarcodeReader.py

@echo off
move "dist\BarcodeReader.exe" ".\"

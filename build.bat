pyinstaller --onefile --specpath "build" --windowed --name BarcodeReader --icon ../BarcodeReader.ico main.py

@echo off
move "dist\BarcodeReader.exe" ".\"

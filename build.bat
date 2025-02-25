pyinstaller --onefile --specpath "build" --windowed --name BarcodeReader main.py

@echo off
move "dist\BarcodeReader.exe" ".\"

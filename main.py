import logging
import sys, os
import numpy as np
import cv2

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPixmap, QImage

from BarcodeReader import Digits, processImg, chooseDetection
from ui import BarcodeReaderUI, numpy2Pixmap, pixmap2Numpy

os.chdir(os.path.dirname(__file__))

class BarcodeProcessor:
	def __init__(self, ui: BarcodeReaderUI):
		"""
		Initialize the processor with a reference to the UI.
		This allows the processor to update the UI components directly.
		"""
		self.ui = ui

	def process_image(self, image_path: str, pixmap: QPixmap):
		"""
		Process the image loaded from file or camera.
		This dummy implementation simply creates fake barcode text,
		dummy debug images, and fake scanlines.
		"""
		detected = self.detect_barcode(pixmap, image_path)
		debug_images = self.generate_debug_images()
		scanlines = self.get_scanlines(pixmap)

		self.ui.main_image_view.reset_zoom()
		# Update the UI:
		# 1. Set the main image in the center.
		self.ui.main_image_view.set_image(pixmap)
		# 2. Add scanlines overlay on the main image.
		self.ui.main_image_view.add_scanlines(scanlines)
		# 3. Populate the debug ribbon on the right.
		self.ui.add_debug_images(debug_images)
		# Update the detection result label.
		self.ui.display_detection_result(detected)
		# 4. Emit the detected barcode (if you wish to hook it elsewhere)
		# self.ui.barcode_detected.emit(barcode_text)
		logging.debug(f"Barcode Detected: {detected}")

	def detect_barcode(self, pixmap: QPixmap, image_path) -> Digits:
		self.images = processImg(pixmap2Numpy(pixmap), os.path.basename(image_path), pyqt=True)
		read = chooseDetection(self.images, self.images.digits)
		return read

	def generate_debug_images(self) -> dict:
		# For demonstration, we'll use the same pixmap for all debug images.
		return {
			'Original': numpy2Pixmap(self.images.inputImg),
			'Lightness': numpy2Pixmap(self.images.lightness),
			'Black & white': numpy2Pixmap(self.images.BaW),
			# 'lineReads': numpy2Pixmap(self.images.lineReads),
			'Scanlines': numpy2Pixmap(self.images.linesImg),
			'Average lightness': numpy2Pixmap(self.images.avgLightness),
		}

	def get_scanlines(self, pixmap: QPixmap) -> list:
		"""
		Create dummy scanlines as a list of tuples.
		Each tuple represents a line with coordinates (x1, y1, x2, y2).
		"""
		w = pixmap.width()
		h = pixmap.height()
		return [ # TODO
			(20, h * 0.25, w - 20, h * 0.25),
			(20, h * 0.50, w - 20, h * 0.50),
			(20, h * 0.75, w - 20, h * 0.75)
		]

# -------------------------
# Main Application Execution
# -------------------------
def main():
	logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

	app = QApplication(sys.argv)
	ui = BarcodeReaderUI()
	processor = BarcodeProcessor(ui)
	# Connect the UI's image_loaded signal to the processor's process_image method.
	# When an image is loaded, this lambda function passes the file path and QPixmap
	# to the processor for handling.
	ui.image_loaded.connect(lambda image_path, pixmap: processor.process_image(image_path, pixmap))

	ui.onload()
	ui.show()

	sys.exit(app.exec())

if __name__ == "__main__":
	main()

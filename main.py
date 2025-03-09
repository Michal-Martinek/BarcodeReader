import logging
import sys, os
import numpy as np
import cv2

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPixmap, QImage

from BarcodeReader import Digits, processImg, chooseDetection
from ui import BarcodeReaderUI, numpy2Pixmap, pixmap2Numpy

if getattr(sys, 'frozen', False):
	os.chdir(os.path.dirname(sys.executable))
else:
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

		if image_path not in ['Camera input', 'scanline-dist-resize']:
			self.ui.main_image_view.reset_zoom()
		# Update the UI:
		# 2. Add scanlines overlay on the main image.
		self.ui.main_image_view.add_scanlines(self.images.scanlineEndpoints[..., ::-1])
		# 1. Set the main image in the center.
		self.ui.main_image_view.set_image(debug_images[self.ui.debugImgName])
		# 3. Populate the debug ribbon on the right.
		self.ui.add_debug_images(debug_images)
		# Update the detection result label.
		self.ui.detection_label.setDetected(detected)
		# 4. Emit the detected barcode (if you wish to hook it elsewhere)
		# self.ui.barcode_detected.emit(barcode_text)
		logging.debug(f"Barcode Detected: {detected}")

	def detect_barcode(self, pixmap: QPixmap, image_path) -> Digits:
		self.images = processImg(pixmap2Numpy(pixmap), os.path.basename(image_path), pyqt=True)
		read = chooseDetection(self.images, self.images.digits)
		return read

	def generate_debug_images(self) -> dict:
		self.ui.main_image_view.images = self.images
		return {
			'Original': numpy2Pixmap(self.images.inputImg),
			'Lightness': numpy2Pixmap(self.images.lightness),
			'Black or white': numpy2Pixmap(self.images.BaW),
			# 'lineReads': numpy2Pixmap(self.images.lineReads),
			'Average lightness': numpy2Pixmap(self.images.avgLightness),
		}

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

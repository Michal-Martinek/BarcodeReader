import logging
import sys, os
import numpy as np
import cv2

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPixmap, QImage

from BarcodeReader import toImg, processImg, chooseDetection
from ui import BarcodeReaderUI

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
		barcode_text = self.detect_barcode(image_path)
		debug_images = self.generate_debug_images()
		scanlines = self.get_scanlines(pixmap)

		# Update the UI:
		# 1. Set the main image in the center.
		self.ui.main_image_view.set_image(pixmap)
		# 2. Add scanlines overlay on the main image.
		self.ui.main_image_view.add_scanlines(scanlines)
		# 3. Populate the debug ribbon on the right.
		self.ui.add_debug_images(debug_images)
		# Update the detection result label.
		self.ui.display_detection_result(barcode_text)
		# 4. Emit the detected barcode (if you wish to hook it elsewhere)
		self.ui.barcode_detected.emit(barcode_text)
		logging.debug("Barcode Detected: {barcode_text}")

	def detect_barcode(self, image_path):
		self.images = processImg(cv2.imread(image_path), os.path.basename(image_path), pyqt=True)
		read = chooseDetection(self.images, self.images.digits)
		return ''.join(map(str, read))
	@staticmethod
	def numpy2Pixmap(img: np.ndarray) -> QPixmap:
		"""Convert a NumPy image array (OpenCV format) to QPixmap."""
		img = toImg(img)
		height, width, channels = img.shape
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		q_image = QImage(img.data, width, height, channels * width, QImage.Format.Format_RGB888)
		return QPixmap.fromImage(q_image)
	def generate_debug_images(self) -> dict:
		# For demonstration, we'll use the same pixmap for all debug images.
		return {
			'Original': self.numpy2Pixmap(self.images.inputImg),
			'Lightness': self.numpy2Pixmap(self.images.lightness),
			'Black & white': self.numpy2Pixmap(self.images.BaW),
			# 'lineReads': self.numpy2Pixmap(self.images.lineReads),
			'Scanlines': self.numpy2Pixmap(self.images.linesImg),
			'Average lightness': self.numpy2Pixmap(self.images.avgLightness),
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

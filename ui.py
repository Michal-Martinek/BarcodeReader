import logging
import os
import random
import sys
import cv2
import numpy as np
from PyQt6 import QtCore
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QLineF
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QIcon
from PyQt6.QtWidgets import (
	QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
	QLabel, QFileDialog, QGraphicsView, QGraphicsScene, QScrollArea, QCheckBox,
	QDialog, QDialogButtonBox, QGraphicsLineItem, QSizePolicy
)

# -------------------------
# Clickable Scanline
# -------------------------
class ClickableScanline(QObject, QGraphicsLineItem):
	clicked = pyqtSignal(object)  # Emits self when clicked

	def __init__(self, line: QLineF, parent=None):
		QObject.__init__(self)
		QGraphicsLineItem.__init__(line, parent)
		self.setAcceptHoverEvents(True)
		self.default_pen = QPen(Qt.GlobalColor.red, 2)
		self.highlight_pen = QPen(Qt.GlobalColor.green, 3)
		self.setPen(self.default_pen)

	def mousePressEvent(self, event):
		self.setPen(self.highlight_pen)
		self.clicked.emit(self)
		super().mousePressEvent(event)

	def hoverEnterEvent(self, event):
		self.setCursor(Qt.CursorShape.PointingHandCursor)
		super().hoverEnterEvent(event)

	def hoverLeaveEvent(self, event):
		self.unsetCursor()
		super().hoverLeaveEvent(event)

# -------------------------
# Main Image Display Widget
# -------------------------
class MainImageView(QWidget):
	def __init__(self):
		super().__init__()
		# QGraphicsScene & QGraphicsView allow overlaying scanlines on an image.
		self.scene = QGraphicsScene(self)
		self.view = QGraphicsView(self.scene, self)
		self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
		self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
		self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

		# Control buttons
		self.save_button = QPushButton("Save Image")
		self.zoom_reset_button = QPushButton("Reset Zoom")
		self.scanline_checkbox = QCheckBox("Show Scanlines")

		controls_layout = QHBoxLayout()
		controls_layout.addWidget(self.save_button)
		controls_layout.addWidget(self.zoom_reset_button)
		controls_layout.addWidget(self.scanline_checkbox)
		controls_layout.addStretch()

		layout = QVBoxLayout(self)
		layout.addWidget(self.view)
		layout.addLayout(controls_layout)

		self.pixmap_item = None  # Holds the main image
		self.scanline_items = []  # List of overlay scanline items
		self.scanline_items_data = []  # Stores scanline coordinates
		self.current_image = None  # Currently displayed QPixmap
		self.zoom_factor = 1.0
		self.view.wheelEvent = self.handle_wheel_zoom

		self.zoom_reset_button.clicked.connect(self.reset_zoom)
		self.save_button.clicked.connect(self.save_current_image)
		self.scanline_checkbox.toggled.connect(self.toggle_scanlines)

	def show_placeholder(self):
		"""Show a default project description when no image is loaded."""
		self.scene.clear()
		placeholder = self.scene.addText("BarcodeReader")
		placeholder.setDefaultTextColor(Qt.GlobalColor.gray)
		self.scene.setSceneRect(placeholder.boundingRect())

	def set_image(self, pixmap: QPixmap):
		"""Display the provided image in the main view."""
		self.scene.clear()
		self.current_image = pixmap
		self.pixmap_item = self.scene.addPixmap(pixmap)
		self.scene.setSceneRect(*pixmap.rect().getCoords())
		# self.reset_zoom()
		if self.scanline_checkbox.isChecked() and self.scanline_items_data:
			self.add_scanlines(self.scanline_items_data)

	def add_scanlines(self, lines_data):
		"""
		Overlay scanlines on the image.
		lines_data: list of tuples (x1, y1, x2, y2)
		"""
		self.scanline_items_data = lines_data
		for item in self.scanline_items:
			self.scene.removeItem(item)
		self.scanline_items.clear()

		if not self.current_image:
			return

		# TODO
		# for coords in lines_data:
		#     line = QLineF(*coords)
		#     scanline = ClickableScanline(line)
		#     scanline.clicked.connect(self.handle_scanline_clicked)
		#     self.scene.addItem(scanline)
		#     self.scanline_items.append(scanline)

	def toggle_scanlines(self, show: bool):
		"""Show or hide scanline overlays."""
		for item in self.scanline_items:
			item.setVisible(show)

	def handle_scanline_clicked(self, scanline):
		"""Handle a scanline click by opening a details popup."""
		dialog = QDialog(self)
		dialog.setWindowTitle("Scanline Details")
		layout = QVBoxLayout(dialog)
		info_label = QLabel("Detailed scanline info goes here.\nTODO: Implement actual details.")
		layout.addWidget(info_label)
		buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
		buttons.accepted.connect(dialog.accept)
		layout.addWidget(buttons)
		dialog.exec()

	def handle_wheel_zoom(self, event):
		factor = 1.2 if event.angleDelta().y() > 0 else 0.8
		if (newFactor := self.zoom_factor * factor) >= 1.0:
			self.zoom_factor = newFactor
			self.view.scale(factor, factor)

	def reset_zoom(self):
		self.view.resetTransform()
		self.zoom_factor = 1.0

	def save_current_image(self):
		"""Save the current image to disk."""
		if self.current_image:
			file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg)")
			if file_path:
				self.current_image.save(file_path)
		else:
			logging.warning("No image to save")

# -------------------------
# Debug Ribbon Widget
# -------------------------
class DebugRibbon(QWidget):
	# Emits (name, QPixmap) when a debug image is selected.
	debug_image_selected = pyqtSignal(str, QPixmap)

	THUMBNAIL_WIDTH = 200

	def __init__(self):
		super().__init__()
		self.layout = QVBoxLayout(self)
		self.setLayout(self.layout)
		self.layout.addStretch()  # Spacer to push items upward.
		self.setMinimumWidth(self.THUMBNAIL_WIDTH + 20)

	def add_debug_image(self, name: str, pixmap: QPixmap):
		"""Add a debug image button with an icon and tooltip."""
		btn = QPushButton(name)
		# Scale the pixmap to the desired width, preserving the aspect ratio
		scaled_pixmap = pixmap.scaledToWidth(self.THUMBNAIL_WIDTH, Qt.TransformationMode.SmoothTransformation)
		btn.setIcon(QIcon(scaled_pixmap))
		btn.setIconSize(scaled_pixmap.size())
		btn.setToolTip(name)
		btn.clicked.connect(lambda: self.debug_image_selected.emit(name, pixmap))
		# Insert the button above the stretch in the layout
		self.layout.insertWidget(self.layout.count() - 1, btn)

	def clear_debug_images(self):
		"""Clear all debug image buttons from the ribbon."""
		while self.layout.count():
			item = self.layout.takeAt(0)
			if item.widget():
				item.widget().deleteLater()
		self.layout.addStretch()

# -------------------------
# Main UI Window
# -------------------------
class BarcodeReaderUI(QMainWindow):
	# Signals for hooking up to your processing code.
	image_loaded = pyqtSignal(str, QPixmap)  # (file path, image)
	barcode_detected = pyqtSignal(str)         # Detected barcode text
	debug_images_generated = pyqtSignal(dict)    # {debug image name: QPixmap}

	def __init__(self):
		super().__init__()
		self.setWindowTitle("Barcode Reader")
		self.setGeometry(50, 50, 1000, 700)
		# self.showMaximized()

		# Central widget and main layout.
		central_widget = QWidget()
		self.setCentralWidget(central_widget)
		main_layout = QVBoxLayout(central_widget)

		# Top input options with icons and hover-over tooltips.
		input_layout = QHBoxLayout()
		self.btn_load_file = QPushButton(text='Load image')
		# self.btn_load_file.setIcon(QIcon("icons/open_file.png"))
		self.btn_load_file.setToolTip("Load image from file")
		self.btn_load_file.clicked.connect(self.load_image_from_file)
		input_layout.addWidget(self.btn_load_file)

		self.btn_random_input = QPushButton(text='Random input')
		# self.btn_random_input.setIcon(QIcon("icons/camera.png"))
		self.btn_random_input.setToolTip("Choose random image from dataset")
		self.btn_random_input.clicked.connect(self.load_random_input_image)
		input_layout.addWidget(self.btn_random_input)

		self.btn_load_camera = QPushButton(text='Camera input')
		# self.btn_load_camera.setIcon(QIcon("icons/camera.png"))
		self.btn_load_camera.setToolTip("Load image from camera")
		self.btn_load_camera.clicked.connect(self.load_image_from_camera)
		input_layout.addWidget(self.btn_load_camera)

		input_layout.addStretch()
		main_layout.addLayout(input_layout)

		# Detection result label.
		self.detection_label = QLabel("Detection Result: None")
		# self.detection_label.setMinimumSize(self.detection_label.width(), self.detection_label.height())
		self.detection_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
		main_layout.addWidget(self.detection_label)

		# Main content: central image view and right-side debug ribbon.
		content_layout = QHBoxLayout()
		main_layout.addLayout(content_layout)

		self.main_image_view = MainImageView()
		self.main_image_view.show_placeholder()  # Shows a project description when no image is loaded.
		content_layout.addWidget(self.main_image_view, stretch=3)

		self.debug_ribbon = DebugRibbon()
		scroll_area = QScrollArea()
		scroll_area.setWidgetResizable(True)
		scroll_area.setWidget(self.debug_ribbon)
		content_layout.addWidget(scroll_area, stretch=1)

		# Connect the debug ribbon signal so that clicking a debug image shows it in the main view.
		self.debug_ribbon.debug_image_selected.connect(self.display_debug_image)

	def onload(self, default_image='barcode-dataset/debug-img.png'):
		self.showMaximized()
		self.load_image_from_file(default_image)

	def load_random_input_image(self, *, folder='barcode-dataset'):
		file = random.choice(os.listdir(folder))
		self.load_image_from_file(os.path.join(folder, file))
	def load_image_from_file(self, file=None):
		"""Open a file dialog and load an image from disk."""
		file_path = file or QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")[0]
		if not file_path: return
		logging.info(f"Loading image from file: {os.path.basename(file_path)}")
		pixmap = QPixmap(file_path)
		self.image_loaded.emit(file_path, pixmap)
		self.main_image_view.set_image(pixmap)

	def load_image_from_camera(self):
		"""Placeholder for camera input logic."""
		# TODO: Implement camera capture and then emit image_loaded signal.
		logging.warning("Camera input not implemented. Hook your camera logic here!")

	def display_debug_image(self, name: str, pixmap: QPixmap):
		"""
		Display a debug image (selected from the debug ribbon) in the main image view.
		"""
		self.main_image_view.set_image(pixmap)

	def add_debug_images(self, debug_images: dict):
		"""
		Clear the ribbon and add debug images.
		debug_images: dict mapping names to QPixmap objects.
		"""
		self.debug_ribbon.clear_debug_images()
		for name, pixmap in debug_images.items():
			self.debug_ribbon.add_debug_image(name, pixmap)
		self.debug_images_generated.emit(debug_images)

	def display_detection_result(self, barcode_text: str):
		"""Update the detection result label."""
		self.detection_label.setText(f"Detection Result: {barcode_text}")

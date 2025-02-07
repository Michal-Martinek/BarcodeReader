import numpy as np
import numpy.typing as npt
from typing import Optional
import cv2
import os
import math
import skimage
import logging
from dataclasses import dataclass
# parameters -----------------------------------
NUM_CHUNKS = (6, 6) # TODO rename

SCANLINE_DIST = 40
NUM_GRADIENTS = 2 + 3 # TODO count, rename
MIN_QUIETZONE_WIDTH = 5
NUM_BASEWIDTHS = 95 - 3
NUM_EDGES = 4 * 12 + 5 + 3

# type hints -----------------------------------
Point = tuple[int, int]
# TODO descriptions
ColorImage = npt.NDArray[np.uint8]
Lightness = npt.NDArray[np.float64]
'''values of lightness [0, 1]'''
BinaryImg = npt.NDArray[np.bool]
'''True - white, False - black'''
LineReads = npt.NDArray[np.bool]
'''binary reads over all scanline gradients
shape - (gradient, line, point)'''
Line = npt.NDArray[np.bool]
BAR_DTYPE = [('start', np.int64), ('len', np.int64), ('idx', np.int64)]
Bars = npt.NDArray # TODO
# TODO not class?
SPANS_DTYPE = [('start', np.int64), ('moduleWidth', np.int64), ('end', np.int64), ('gradIdx', np.int64), ('lineIdx', np.int64)]
Spans = npt.NDArray
Groups = npt.NDArray
Widths = npt.NDArray
Digits = npt.NDArray
class DetectionError(RuntimeError): # TODO name?
	pass

@dataclass # TODO rename?
class Images:
	'''class holding all intermediate steps outputs for further use'''
	inputImg: ColorImage
	lightness: Lightness = None
	localAverages: Lightness = None
	BaW: BinaryImg = None
	lineReads: LineReads = None
	linesImg: ColorImage = None

	lines: list[list[tuple[Bars, Spans]]] = None
	def initLines(self):
		self.lines = [[]] * NUM_GRADIENTS
	def addLine(self, gradIdx: int, bars: Bars, spans: Spans):
		self.lines[gradIdx].append((bars, spans))
# helpers --------------------------------------
def toImg(arr: np.ndarray) -> ColorImage:
	'''converts arbitrary ndarray to image like - 3 color channels, dtype - uint8'''
	if arr.dtype in (np.bool, np.float32, np.float64): arr = (arr * 255)
	if len(arr.shape) == 2 or arr.shape[-1] != 3: arr = np.repeat(arr[..., np.newaxis], 3, axis=-1)
	return arr.astype('uint8')

def differsByAtmost(target, args: npt.NDArray, maxDiff=2) -> bool:
	'''if all args differ from target by at most maxDiff'''
	target = target[..., np.newaxis]
	return np.all(np.abs(args - target) <= maxDiff, axis=-1)
def genColorsHUE(N):
	'''generates N colors distinct in hue'''
	hues = np.linspace(0, 179, N, endpoint=False, dtype=np.uint8)
	hsv = np.array([[h, 255, 255] for h in hues], dtype=np.uint8)
	rgbs = cv2.cvtColor(hsv[np.newaxis, :, :], cv2.COLOR_HSV2BGR)[0]
	return rgbs

def check(cond, msg):
	if cond: return
	raise DetectionError(msg)
	
# preprocessing ----------------------------------------------
def paddToShape(img: np.ndarray, shape) -> np.ndarray:
	'''padds img to shape by repeating pixels at the edges'''
	pixelsToPadd = np.array((*shape, 3)) - img.shape
	countsToPadd = np.array((pixelsToPadd // 2, pixelsToPadd - pixelsToPadd // 2)).T
	img = np.pad(img, countsToPadd, mode='edge')
	return img
def averageChunks(lightness: np.ndarray, NUM_CHUNKS, CHUNK_SIZE):
	'''averages lightness inside each chunk'''
	chunks = np.reshape(lightness, (NUM_CHUNKS[0], CHUNK_SIZE[0], NUM_CHUNKS[1], CHUNK_SIZE[1]))
	sums = chunks.sum(axis=(1,3))
	avgs = sums / (CHUNK_SIZE[0] * CHUNK_SIZE[1])
	assert avgs.shape == NUM_CHUNKS
	return avgs
def mix(ratio, a, b):
	return a * (1 - ratio) + b * ratio
def averageLightness(images: Images, NUM_CHUNKS, CHUNK_SIZE, blurSizeRatio=2.) -> Lightness:
	avgs = averageChunks(images.lightness, NUM_CHUNKS, CHUNK_SIZE)
	localAverages = np.repeat(avgs, CHUNK_SIZE[0], axis=0)
	kernelSize = np.floor_divide(CHUNK_SIZE, blurSizeRatio).astype('int')
	localAverages = np.repeat(localAverages, CHUNK_SIZE[1], axis=1)
	if blurSizeRatio: localAverages = cv2.blur(localAverages, kernelSize)
	images.localAverages = localAverages
	averages = mix(0.5, localAverages, np.average(images.lightness))
	return averages

def prepareImg(img: ColorImage) -> Images:
	'''TODO'''
	# TODO resizing
	CHUNK_SIZE = np.ceil(np.array(img.shape[:2]) / NUM_CHUNKS).astype('int')
	# print('CHUNK_SIZE', CHUNK_SIZE) # TODO logging, save?
	images = Images(inputImg=paddToShape(img, NUM_CHUNKS * CHUNK_SIZE))
	images.initLines()
	images.lightness = cv2.cvtColor(images.inputImg, cv2.COLOR_BGR2GRAY) / 255

	avgLightness = averageLightness(images, NUM_CHUNKS, CHUNK_SIZE)
	images.BaW = images.lightness > avgLightness
	return images
# scanlines ------------------------------
def genDrawLines(starts, ends, images: Images):
	images.linesImg = toImg(images.lightness)
	linePs = []
	maxLen = 0
	for color, parallelEndpoints in zip(genColorsHUE(starts.shape[0]), zip(starts, ends)):
		color = tuple(map(int, color))
		linePs.append([])
		for start, end in zip(*parallelEndpoints):
			line = np.array(skimage.draw.line(*start, *end)).T
			linePs[-1].append(line)
			maxLen = max(maxLen, len(line))
			cv2.line(images.linesImg, start[::-1], end[::-1], color)
	return linePs, maxLen

def getScanlineEndpoints(shape: tuple) -> tuple:
	angles = np.linspace(0, np.pi / 2, NUM_GRADIENTS)
	gradVecs = np.column_stack((np.sin(angles), np.cos(angles)))
	# first ^ over y axis, then > on x axis
	startPsY = np.arange(SCANLINE_DIST, shape[0], SCANLINE_DIST)[::-1]
	startPsX = np.arange(0, shape[1], SCANLINE_DIST)
	startPoints = np.zeros((len(startPsY) + len(startPsX), 2), dtype='int')
	startPoints[:len(startPsY), 0] = startPsY
	startPoints[len(startPsY):, 1] = startPsX
	# TODO from bottom

	epsilon = 1e-10
	edgeDist = (shape,) - startPoints
	scales = edgeDist / (gradVecs + epsilon)[:, np.newaxis]
	scales = np.min(scales, axis=-1)
	startPoints = np.tile(startPoints, (len(gradVecs), 1, 1))
	endPoints = startPoints + scales[..., np.newaxis] * gradVecs[:, np.newaxis]
	return startPoints, endPoints.astype('int')
def getScanLines(images: Images) -> LineReads:
	startPoints, endPoints = getScanlineEndpoints(images.BaW.shape)
	linePs, maxLen = genDrawLines(startPoints, endPoints, images)

	images.lineReads = np.zeros((len(linePs), len(linePs[0]), maxLen), dtype='bool')
	for g, grad in enumerate(linePs):
		for lineIdx, points in enumerate(grad):
			images.lineReads[g, lineIdx, np.arange(len(points))] = images.BaW[*points.T]
	return images.lineReads

def splitToBars(lineReads: Line) -> Bars:
	'''ignores first bar if it's black'''
	edges = np.logical_xor(lineReads[:-1], lineReads[1:])
	barStarts = np.where(edges)[0] + 1
	if lineReads[0] == False:
		barStarts = barStarts[1:]
	bars = np.zeros((len(barStarts) + 1,), dtype=BAR_DTYPE)
	bars[1:]['start'] = barStarts
	bars[:-1]['len'] = bars[1:]['start']
	bars[-1]['len'] = len(lineReads)
	bars['len'] -= bars['start']
	bars['idx'] = np.arange(len(barStarts) + 1)
	return bars

def findSpanStarts(bars: Bars, gradIdx, lineIdx) -> Spans:
	quietZoneIdxs = np.where(bars[:-5:2]['len'] >= MIN_QUIETZONE_WIDTH)[0] * 2
	check(quietZoneIdxs.size, 'no quietzones found')
	possibleStartPattern = bars[quietZoneIdxs[:, np.newaxis] + np.arange(4)]
	# TODO floating?
	moduleWidths = np.average(possibleStartPattern['len'][:, 1:], axis=1).astype('int64')
	# NOTE are bars in start tag same width?
	good = differsByAtmost(moduleWidths, possibleStartPattern['len'][:, 1:])
	check(good.any(), 'no start tags found')
	# NOTE is quietzone thick enough?
	good &= possibleStartPattern['len'][:, 0] >= moduleWidths * MIN_QUIETZONE_WIDTH
	check(good.any(), 'starting quietzones are too narrow')

	startPatterns = possibleStartPattern[good]
	spans = np.zeros((startPatterns.shape[0]), SPANS_DTYPE)
	spans['start'] = startPatterns['idx'][:, -1] + 1
	spans['moduleWidth'] = moduleWidths[good]
	spans[['gradIdx', 'lineIdx']] = (gradIdx, lineIdx)
	return spans

def findSpanEnds(bars: Bars, spans: Spans) -> Spans:
	spans['end'] = spans['start'] + NUM_EDGES
	fits = spans['end'] < bars.shape
	check(fits.any(), 'not enough bars after quietzone')
	return spans[fits]
def checkCodeLen(bars: Bars, spans: Spans) -> Spans:
	lens = bars[spans['end']]['start'] - bars[spans['start']]['start']
	good = differsByAtmost(spans['moduleWidth'] * NUM_BASEWIDTHS, lens, maxDiff=NUM_BASEWIDTHS)
	check(good.any(), 'incorrect code len')
	return spans[good]

# TODO use grad, lineidx
def findSpans(gradIdx: int, lineIdx: int, lineReads: np.ndarray[Line]) -> tuple[Bars, Spans]:
	'''finds spans of bars in possible barcode
	* the spans have right width and # of bars
	* they start after start tag, end before quietzone)
	'''
	bars = splitToBars(lineReads)
	check(bars.size >= NUM_EDGES, 'not enough bars on scanline')
	spans = findSpanStarts(bars, gradIdx, lineIdx)
	spans = findSpanEnds(bars, spans)
	spans = checkCodeLen(bars, spans)
	return bars, spans
def splitOnTags(span: Spans, bars: Bars) -> Groups:
	'''checks the middle and end tags
	* splits bars into digit groups (l/r half, bars)'''
	endTag = bars[span['end'] - 3:span['end']]
	check(differsByAtmost(span['moduleWidth'], endTag['len']), 'incorrect end tag width')
	center = span['start'] + 6 * 4
	centerEdges = bars[center:center + 5]
	check(differsByAtmost(span['moduleWidth'], centerEdges['len']), 'incorrect center tag width')
	digits = np.array((bars[span['start']:center], bars[center + 5:span['end'] - 3]))
	return digits
# decoding ---------------------------------------

# NOTE lens of bars in R-code
# L-code is the same - colors are flipped but this has no color information
# G-code has reverse order
DIGIT_ENCODINGS = np.array((
	(3, 2, 1, 1),
	(2, 2, 2, 1),
	(2, 1, 2, 2),
	(1, 4, 1, 1),
	(1, 1, 3, 2),
	(1, 2, 3, 1),
	(1, 1, 1, 4),
	(1, 3, 1, 2),
	(1, 2, 1, 3),
	(3, 1, 1, 2),
))
FIRST_DIGIT_ENCODING = {
	(1, 1, 1, 1, 1, 1): 0,
	(1, 1, 0, 1, 0, 0): 1,
	(1, 1, 0, 0, 1, 0): 2,
	(1, 1, 0, 0, 0, 1): 3,
	(1, 0, 1, 1, 0, 0): 4,
	(1, 0, 0, 1, 1, 0): 5,
	(1, 0, 0, 0, 1, 1): 6,
	(1, 0, 1, 0, 1, 0): 7,
	(1, 0, 1, 0, 0, 1): 8,
	(1, 0, 0, 1, 0, 1): 9,
}
def decodeDigits(span: Spans, digitGroups: Groups, *, _flipped=False) -> Digits:
	'''finds the closest digit encoding for given span (minimum squared distance)'''
	lens = digitGroups['len'].reshape((2, 6, 4))
	encodings = DIGIT_ENCODINGS * span['moduleWidth']
	encodings = np.concat((encodings, encodings[..., ::-1]))
	distances = np.sum((lens[..., np.newaxis, :] - encodings) ** 2, axis=-1)

	digits = np.argmin(distances, axis=-1)
	parity = (digits // 10 + ((1,), (0,))) % 2
	digits = digits.flatten() % 10
	if parity[0, 0] == 0 and not _flipped: # NOTE read backwards
		return decodeDigits(span, digitGroups[::-1, ::-1], _flipped=True)
	check(np.all(parity[1] == 0), 'R-encoding expected in right half')
	
	firstDigit = FIRST_DIGIT_ENCODING.get(tuple(parity[0]), 0)
	digits = np.concat(((firstDigit,), digits), axis=0)
	return digits
def detectLine(gradIdx, lineIdx, images: Images, lineReads: LineReads) -> list[Digits]:
	'''fully processes detection on single line'''
	bars, spans = findSpans(gradIdx, lineIdx, lineReads)
	images.addLine(gradIdx, bars, spans)
	detections = []
	for span in spans:
		digitGroups = splitOnTags(span, bars)
		detected = decodeDigits(span, digitGroups)
		detections.append(detected)
	return detections

def checksumDigit(digits: Digits) -> bool:
	checksum = digits.sum() + 2 * digits[1:-1:2].sum()
	return checksum % 10 == 0
def detectImage(images: Images) -> Digits:
	lineReads = getScanLines(images)
	detections = []
	for gradIdx, parallels in enumerate(lineReads):
		for lineIdx, points in enumerate(parallels):
			try:
				digits = detectLine(gradIdx, lineIdx, images, points)
				print(f'{gradIdx}:{lineIdx:<2}', digits)
				[detections.append(d) for d in digits if checksumDigit(d)]
			except DetectionError as e:
				print(f'{gradIdx}:{lineIdx:<2} ERROR:', e)
	detections = np.unique(np.array(detections), axis=0)
	return detections
# drawing ----------------------------------------------------
def drawGradLineReads(lineReadImgs: list[ColorImage], onlyInteresting=True):
	'''draws all line reads grouped by gradient with debug info'''
	for gradIdx, lineReads in enumerate(lineReadImgs):
		winname = f'Grad {gradIdx} - lines'
		# NOTE don't show (close) grads without spans
		if onlyInteresting and np.unique(lineReads.reshape((-1, 3)), axis=0).shape == (2, 3):
			if cv2.getWindowProperty(winname, cv2.WND_PROP_VISIBLE) >= 1:
				cv2.destroyWindow(winname)
			continue
		linImg = np.repeat(lineReads, SCANLINE_DIST // 2, axis=0)
		cv2.imshow(winname, linImg)
def colorcodeQuietzone(lineSlice, basewidth, quietzoneEdge, color=(0, 0, 255)):
	fourths = 2 * (color[2] != 0) + 1
	off = quietzoneEdge['start'] + (quietzoneEdge['len'] // 4) * fourths
	lineSlice[off : off + basewidth] = color
def drawSpans(img: ColorImage, images: Images):
	for grad in images.lines:
		for bars, spans in grad:
			for span in spans:
				lineSlice = img[span['gradIdx'], span['lineIdx']]
				startEdge = bars[span['start'] - 4]
				colorcodeQuietzone(lineSlice, span['moduleWidth'], startEdge)
				endEdge = bars[span['end']]
				colorcodeQuietzone(lineSlice, span['moduleWidth'], endEdge, (0, 255, 0))
def drawDebugs(images: Images, lightness=False, localAverages=False, BaW=True, grads:int=1):
	if lightness: cv2.imshow('lightness', toImg(images.lightness))
	if localAverages: cv2.imshow('localAverages', toImg(images.localAverages))
	if BaW: cv2.imshow('BaW', toImg(images.BaW))
	cv2.imshow('linesImg', images.linesImg)

	if grads:
		lineReadImgs = toImg(images.lineReads)
		drawSpans(lineReadImgs, images)
		drawGradLineReads(lineReadImgs, grads == 1)

# IO --------------------------------------------------
def processImg(img: ColorImage) -> Images:
	images = prepareImg(img)
	digits = detectImage(images)
	if digits.size: print('detected:', digits)
	drawDebugs(images)
	return images
	# TODO choose final read

def showCameraInput(camera: cv2.VideoCapture, winname: str) -> ColorImage:
	ret, img = camera.read()
	if not ret:
		raise RuntimeError('no camera input')
	cv2.imshow(winname, img)
	return img
def showLoop(winname='input', camera: Optional[cv2.VideoCapture]=None, *, delay=10) -> bool:
	'''waits in loop after showing the results
	* returns whether to continue'''
	while cv2.getWindowProperty(winname, cv2.WND_PROP_VISIBLE) >= 1:
		if camera:
			img = showCameraInput(camera, winname)
		if (key := cv2.waitKey(delay)) != -1:
			if key in [ord('q'), 27]:
				return False
			elif key == 32:
				if camera:
					cv2.imwrite('camera-input.png', img)
					return (True, img)
				return True
def showSavedCamInput(winname: str, path='camera-input.png') -> ColorImage:
	img = np.zeros((200, 200, 3), 'uint8') # default
	if os.path.exists(path):
		img = cv2.imread(path)
	cv2.imshow(winname, img)
	print('camera image shape', img.shape)
	return img
def cameraLoop(winname='Barcode reader - camera input'):
	img = showSavedCamInput(winname)
	camera = cv2.VideoCapture(0)
	while cv2.getWindowProperty(winname, cv2.WND_PROP_VISIBLE) >= 1:
		images = processImg(img)
		if start := 'start' not in vars(): # NOTE wait for input to start capture
			if not showLoop(winname, camera=None):
				break
		ret = showLoop(winname, camera)
		if not ret: break
		cv2.imshow('input', img := ret[1])
	camera.release()

def testDataset(winname='input'):
	os.chdir('barcode-dataset')
	for file in os.listdir():
		img = cv2.imread(file)
		cv2.imshow(winname, img)

		processImg(img)
		if not showLoop(winname): break
	
def main():
	# testDataset()
	cameraLoop()

	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()

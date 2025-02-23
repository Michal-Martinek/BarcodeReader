import numpy as np
import numpy.typing as npt
import cv2
import matplotlib.pyplot as plt

import os, sys
import logging
from dataclasses import dataclass
from typing import Optional

if getattr(sys, 'frozen', False):
    os.chdir(os.path.dirname(sys.executable))
else:
    os.chdir(os.path.dirname(__file__))

# parameters -----------------------------------
NUM_AVERAGING_CHUNKS = (6, 6)

SCANLINE_DIST = 40
NUM_GRADIENTS = 2 + 3 # TODO count, rename
MIN_QUIETZONE_WIDTH = 5
NUM_BASEWIDTHS = 95 - 3
NUM_EDGES = 4 * 12 + 5 + 3

# type hints -----------------------------------
Point = tuple[int, int]
ColorImage = npt.NDArray[np.uint8]
Lightness = npt.NDArray[np.float64]
'''values of lightness [0, 1]'''
BinaryImg = npt.NDArray[np.bool]
'''True - white, False - black'''
Line = npt.NDArray[np.float64]
'''lightness reads over line'''
LineReads = npt.NDArray[np.float64]
'''lightness reads over line  
shape = (gradient, line, point)'''
BAR_DTYPE = [('start', np.int64), ('len', np.float32), ('idx', np.int64)]
Bars = npt.NDArray
'''runs of pixels of same color along Line'''
SPANS_DTYPE = [('start', np.int64), ('end', np.int64), ('moduleWidth', np.float32), ('gradIdx', np.int64), ('lineIdx', np.int64)]
Spans = npt.NDArray
'''sequence of NUM_EDGES bars possibly containing a barcode'''
Groups = npt.NDArray
'''Bars grouped to individual Digits, shape = (2, 6, 4)'''
Digits = npt.NDArray
'''digits detected in Span'''

class DetectionError(RuntimeError):
	pass

@dataclass
class Images:
	'''class holding all intermediate outputs for insight'''
	inputImg: ColorImage
	lightness: Lightness = None
	avgLightness: Lightness = None
	BaW: BinaryImg = None
	lineReads: LineReads = None
	linesImg: ColorImage = None

	lines: list[list[tuple[Bars, Spans]]] = None
	digits: Digits = None
	detectionCounts: npt.NDArray[np.int64] = None

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

def differsByAtmost(target, args: npt.NDArray, maxDiff=3.5) -> bool:
	'''if all args differ from target by at most maxDiff'''
	if target.ndim < args.ndim: target = target[..., np.newaxis]
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
def logErr(e: DetectionError, gradIdx: int, lineIdx: int, spanIdx: int=None):
	lvl = logging.INFO if spanIdx is not None else logging.DEBUG
	name = 'span' if spanIdx is not None else 'scanline'
	spanLoc = f':{spanIdx}' * (spanIdx is not None)
	logging.log(lvl, f'{name} {gradIdx}:{lineIdx:<2}{spanLoc} ERROR: {e}')

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
	averages = mix(0.5, localAverages, np.average(images.lightness))
	return averages

def prepareImg(img: ColorImage) -> Images:
	# TODO resizing
	CHUNK_SIZE = np.ceil(np.array(img.shape[:2]) / NUM_AVERAGING_CHUNKS).astype('int')
	images = Images(inputImg=paddToShape(img, NUM_AVERAGING_CHUNKS * CHUNK_SIZE))
	images.initLines()
	images.lightness = cv2.cvtColor(images.inputImg, cv2.COLOR_BGR2GRAY) / 255

	images.avgLightness = averageLightness(images, NUM_AVERAGING_CHUNKS, CHUNK_SIZE)
	images.BaW = images.lightness > images.avgLightness
	return images
# scanlines ------------------------------
def genDrawLines(starts, ends, images: Images) -> tuple[list[list[Point]], int]:
	images.linesImg = toImg(images.lightness)
	linePs = []
	maxLen = 0
	for color, parallelEndpoints in zip(genColorsHUE(starts.shape[0]), zip(starts, ends)):
		linePs.append([])
		for start, end in zip(*parallelEndpoints):
			numPoints = np.abs(start - end).max() + 1
			line = np.linspace(start, end, numPoints).astype('int32')
			linePs[-1].append(line)
			maxLen = max(maxLen, len(line))
			cv2.line(images.linesImg, start[::-1], end[::-1], tuple(map(int, color)))
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
	startPoints, endPoints = getScanlineEndpoints(images.lightness.shape)
	linePs, maxLen = genDrawLines(startPoints, endPoints, images)

	images.lineReads = np.zeros((len(linePs), len(linePs[0]), maxLen))
	for g, grad in enumerate(linePs):
		for lineIdx, points in enumerate(grad):
			images.lineReads[g, lineIdx, np.arange(len(points))] = images.lightness[*points.T] - images.avgLightness[*points.T]
	return images.lineReads

def splitToBars(lineReads: Line) -> Bars:
	'''ignores first bar if it's black'''
	BaW = lineReads > 0.
	edges = np.logical_xor(BaW[:-1], BaW[1:])
	barStarts = np.where(edges)[0] + 1
	if lineReads[0] <= 0.:
		barStarts = barStarts[1:]
	edgePairs = lineReads[[barStarts - 1, barStarts]]
	xIntersect = edgePairs[1] / (edgePairs[1] - edgePairs[0])
	# NOTE x intersect of line connecting edge points
	# distance from second point: dx = y / dy

	bars = np.zeros((len(barStarts) + 1,), dtype=BAR_DTYPE)
	bars[1:]['start'] = barStarts
	bars[:-1]['len'] = bars[1:]['start'] - xIntersect
	bars[-1]['len'] = len(lineReads)
	bars[1:]['len'] += xIntersect
	bars['len'] -= bars['start']
	bars['idx'] = np.arange(len(barStarts) + 1)
	return bars

def findSpanStarts(bars: Bars, gradIdx, lineIdx) -> Spans:
	quietZoneIdxs = np.where(bars[:-5:2]['len'] >= MIN_QUIETZONE_WIDTH)[0] * 2
	check(quietZoneIdxs.size, 'no quietzones found')
	possibleStartPattern = bars[quietZoneIdxs[:, np.newaxis] + np.arange(4)]
	moduleWidths = np.average(possibleStartPattern['len'][:, 1:], axis=1)
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
	good = differsByAtmost(spans['moduleWidth'] * NUM_BASEWIDTHS, lens[..., np.newaxis], maxDiff=NUM_BASEWIDTHS)
	check(good.any(), 'incorrect code len')
	spans = spans[good]
	spans['moduleWidth'] = lens[good] / NUM_BASEWIDTHS
	return spans

def findSpans(gradIdx: int, lineIdx: int, lineReads: Line) -> tuple[Bars, Spans]:
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
	lens = lens * 7 / lens.sum(-1, keepdims=True)
	encodings = np.concat((DIGIT_ENCODINGS, DIGIT_ENCODINGS[..., ::-1])) # NOTE construct G code (reverse order)
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
def detectLine(gradIdx, lineIdx, images: Images, lineReads: Line) -> list[Digits]:
	'''fully processes detection on single line'''
	try:
		bars, spans = findSpans(gradIdx, lineIdx, lineReads)
	except DetectionError as e:
		return logErr(e, gradIdx, lineIdx)
	images.addLine(gradIdx, bars, spans)
	detections = []
	for spanIdx, span in enumerate(spans):
		try:
			digitGroups = splitOnTags(span, bars)
			detected = decodeDigits(span, digitGroups)
			detections.append(detected)
		except DetectionError as e:
			logErr(e, gradIdx, lineIdx, spanIdx)
	return detections

def checksumDigit(digits: Digits) -> bool:
	checksum = digits.sum() + 2 * digits[1:-1:2].sum()
	return checksum % 10 == 0
def chooseDetection(images: Images, detections: list[Digits]) -> Digits:
	images.digits, counts = np.unique(np.array(detections), axis=0, return_counts=True)
	if not len(detections): return np.array([])
	indices = np.argsort(-counts)
	images.digits = images.digits[indices]
	images.detectionCounts = counts[indices]
	return images.digits[0]
def detectImage(images: Images) -> Digits:
	lineReads = getScanLines(images)
	detections = []
	for gradIdx, parallels in enumerate(lineReads):
		for lineIdx, points in enumerate(parallels):
			digits = detectLine(gradIdx, lineIdx, images, points)
			if not digits: continue
			logging.info(f'scanline {gradIdx}:{lineIdx:<2} {digits}')
			[detections.append(d) for d in digits if checksumDigit(d)]
	return chooseDetection(images, detections)
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
	off = quietzoneEdge['start'] + (quietzoneEdge['len'] / 4 * fourths).astype('int')
	lineSlice[off : off + basewidth] = color
def drawSpans(img: ColorImage, images: Images):
	for grad in images.lines:
		for bars, spans in grad:
			for span in spans:
				lineSlice = img[span['gradIdx'], span['lineIdx']]
				startEdge = bars[span['start'] - 4]
				colorcodeQuietzone(lineSlice, span['moduleWidth'].astype('int'), startEdge)
				endEdge = bars[span['end']]
				colorcodeQuietzone(lineSlice, span['moduleWidth'].astype('int'), endEdge, (0, 255, 0))
def drawDebugs(images: Images, lightness=False, localAverages=False, BaW=True, grads:int=1):
	if lightness: cv2.imshow('lightness', toImg(images.lightness))
	if localAverages: cv2.imshow('localAverages', toImg(images.localAverages))
	if BaW: cv2.imshow('BaW', toImg(images.BaW))
	cv2.imshow('linesImg', images.linesImg)

	if grads:
		lineReadImgs = toImg(images.lineReads > 0.)
		drawSpans(lineReadImgs, images)
		drawGradLineReads(lineReadImgs, grads == 1)

# IO --------------------------------------------------
def processImg(img: ColorImage, num: int) -> Images:
	logging.info(f'DETECTING {num:>03} {img.shape} ----------------------------------------')
	images = prepareImg(img)
	digits = detectImage(images)
	if digits.size:
		l = [f'{d} {c}x' for d, c in zip(images.digits, images.detectionCounts)]
		logging.info(f'{num:>03} detected: {"\t".join(l)}')
	drawDebugs(images)
	return images

def showStatistics(detected: npt.NDArray[np.int64]):
	success = detected.astype('bool')
	logging.info(f'detected: {success.sum()} / {success.size} ({np.average(success * 100):.0f} %)')
	if np.any(overdetected := detected > 1):
		logging.warning(f'overdetection: {np.sum(overdetected)}x')

def showCameraInput(camera: cv2.VideoCapture, winname: str) -> ColorImage:
	ret, img = camera.read()
	if not ret:
		raise RuntimeError('no camera input')
	cv2.imshow(winname, img)
	return img

FLAG_FREERUN = False
def showLoop(winname='input', camera: Optional[cv2.VideoCapture]=None, *, delay=10) -> bool:
	'''waits in loop after showing the results
	* returns whether to continue'''
	global FLAG_FREERUN
	while cv2.getWindowProperty(winname, cv2.WND_PROP_VISIBLE) >= 1:
		if camera:
			img = showCameraInput(camera, winname)
		try:
			key = cv2.waitKey(delay)
		except KeyboardInterrupt: return False
		if key in [ord('q'), 27]: # quit
			return False
		elif key == ord('r'): # toggle run
			FLAG_FREERUN = not FLAG_FREERUN
		elif key == 32 or FLAG_FREERUN: # step
			if key == 32: FLAG_FREERUN = False
			if camera:
				cv2.imwrite('camera-input.png', img)
				return (True, img)
			return True
def showSavedCamInput(winname: str, path='camera-input.png') -> ColorImage:
	img = np.zeros((200, 200, 3), 'uint8') # default
	if os.path.exists(path):
		img = cv2.imread(path)
	cv2.imshow(winname, img)
	logging.info(f'camera image shape {img.shape}')
	return img
def cameraLoop(winname='Barcode reader - camera input'):
	img = showSavedCamInput(winname)
	camera = cv2.VideoCapture(0)
	detected = np.zeros((0,), 'int')
	while cv2.getWindowProperty(winname, cv2.WND_PROP_VISIBLE) >= 1:
		images = processImg(img, detected.shape[0])
		detected = np.append(detected, images.digits.shape[0])

		if start := 'start' not in vars(): # NOTE wait for input to start capture
			if not showLoop(winname, camera=None):
				break
		ret = showLoop(winname, camera)
		if not ret: break
		cv2.imshow('input', img := ret[1])
	camera.release()
	showStatistics(detected)

def testDataset(winname='input'):
	os.chdir('barcode-dataset')
	files = os.listdir()
	detected = np.zeros(len(files), 'int')
	for i, file in enumerate(files):
		img = cv2.imread(file)
		cv2.imshow(winname, img)

		images = processImg(img, i)
		detected[i] = images.digits.shape[0]
		# TODO check against filename tag
		if not showLoop(winname): break
	showStatistics(detected[:i+1])
def main():
	logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

	testDataset()
	# cameraLoop()

	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()

import numpy as np
import numpy.typing as npt
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
MIN_QUIETZONE_WIDTH = 8
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

	bars: Bars = None
	spans: Spans = None
	def addLine(self, bars: Bars, spans: Spans):
		return
# helpers --------------------------------------
def toImg(arr: np.ndarray) -> ColorImage:
	'''converts arbitrary ndarray to image like - 3 color channels, dtype - uint8'''
	if arr.dtype in (np.bool, np.float32, np.float64): arr = (arr * 255)
	if len(arr.shape) == 2 or arr.shape[-1] != 3: arr = np.repeat(arr[..., np.newaxis], 3, axis=-1)
	return arr.astype('uint8')

def differsByAtmost(*args, maxDiff=2):
	# TODO raise err
	# TODO what about gradual change: 1, 3, 5, 7
	res = np.ones_like(args[0], dtype='bool')
	for a, b in zip(args[:-1], args[1:]):
		res = np.logical_and(res, np.abs(a - b) <= maxDiff)
	return res
def genColorsHUE(N):
	'''generates N colors distinct in hue'''
	hues = np.linspace(0, 179, N, endpoint=False, dtype=np.uint8)
	hsv = np.array([[h, 255, 255] for h in hues], dtype=np.uint8)
	rgbs = cv2.cvtColor(hsv[np.newaxis, :, :], cv2.COLOR_HSV2BGR)[0]
	return rgbs

def check(cond, msg):
	if cond: return
	raise DetectionError(msg)
def raiseOnNone(arr):
	if arr is None or arr.size == 0:
		raise DetectionError # TODO (msg)
	
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
def averageLightness(images: Images, NUM_CHUNKS, CHUNK_SIZE, blurSizeRatio=2.) -> Lightness:
	avgs = averageChunks(images.lightness, NUM_CHUNKS, CHUNK_SIZE)
	localAverages = np.repeat(avgs, CHUNK_SIZE[0], axis=0)
	kernelSize = np.floor_divide(CHUNK_SIZE, blurSizeRatio).astype('int')
	localAverages = np.repeat(localAverages, CHUNK_SIZE[1], axis=1)
	if blurSizeRatio: localAverages = cv2.blur(localAverages, kernelSize)
	images.localAverages = localAverages
	averages = (localAverages + np.average(images.lightness)) / 2
	return averages

def prepareImg(img: ColorImage) -> Images:
	'''TODO'''
	# TODO resizing
	CHUNK_SIZE = np.ceil(np.array(img.shape[:2]) / NUM_CHUNKS).astype('int')
	# print('CHUNK_SIZE', CHUNK_SIZE) # TODO logging, save?
	images = Images(inputImg=paddToShape(img, NUM_CHUNKS * CHUNK_SIZE))
	images.lightness = cv2.cvtColor(images.inputImg, cv2.COLOR_BGR2GRAY) / 255

	avgLightness = averageLightness(images, NUM_CHUNKS, CHUNK_SIZE)
	images.BaW = images.lightness > avgLightness
	return images
# scanlines ------------------------------
def genDrawLines(starts, ends, background):
	linePs = []
	maxLen = 0
	for color, parallelEndpoints in zip(genColorsHUE(starts.shape[0]), zip(starts, ends)):
		color = tuple(map(int, color))
		linePs.append([])
		for start, end in zip(*parallelEndpoints):
			line = np.array(skimage.draw.line(*start, *end)).T
			linePs[-1].append(line)
			maxLen = max(maxLen, len(line))
			cv2.line(background, start[::-1], end[::-1], color)
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
	images.linesImg = toImg(images.lightness)
	linePs, maxLen = genDrawLines(startPoints, endPoints, background=images.linesImg)

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

def findCodeStarts(bars: Bars, gradIdx, lineIdx) -> Spans:
	quietZoneIdxs = np.where(bars[:-5:2]['len'] >= MIN_QUIETZONE_WIDTH)[0] * 2
	possibleStartPattern = bars[quietZoneIdxs[:, np.newaxis] + np.arange(4)]
	goodWidths = differsByAtmost(*possibleStartPattern['len'].T[1:], maxDiff=2)
	# the three stripes in start tag are the same width
	raiseOnNone(possibleStartPattern := possibleStartPattern[goodWidths])
	moduleWidths = np.average(possibleStartPattern['len'][:, 1:], axis=1).astype('int64')
	isQuietThickEnough = possibleStartPattern['len'][:, 0] >= moduleWidths * MIN_QUIETZONE_WIDTH
	raiseOnNone(possibleStartPattern := possibleStartPattern[isQuietThickEnough])

	spans = np.zeros((possibleStartPattern.shape[0]), SPANS_DTYPE)
	spans['start'] = possibleStartPattern['idx'][:, -1] + 1
	spans['moduleWidth'] = moduleWidths[isQuietThickEnough]
	spans[['gradIdx', 'lineIdx']] = (gradIdx, lineIdx)
	return spans


def findClosingQuietzone(bars: Bars, spans: Spans) -> Spans:
	for span in spans:
		whiteAfter = bars[span['start'] + 2::2]
		isQuietzone = whiteAfter['len'] >= MIN_QUIETZONE_WIDTH * span['moduleWidth']
		if not np.any(isQuietzone): continue
		span['end'] = whiteAfter[isQuietzone][0]['idx']
	quietFound = spans['end'].astype('bool')
	check(quietFound.any(), 'quietzone not found for any span')
	return spans[quietFound]
def filterOnLen(bars: Bars, spans: Spans):
	good = np.ones_like(spans, dtype='bool')
	lens = bars[spans['end']]['start'] - bars[spans['start']]['start']
	good &= differsByAtmost(lens / spans['moduleWidth'], NUM_BASEWIDTHS, maxDiff=2)
	edgeCounts = bars[spans['end']]['idx'] - bars[spans['start']]['idx']
	good &= edgeCounts == NUM_EDGES
	return spans[good]

# TODO use grad, lineidx
def findSpans(gradIdx: int, lineIdx: int, lineReads: np.ndarray[Line]) -> tuple[Bars, Spans]:
	'''finds spans of bars in possible barcode
	* the spans have right width and # of bars
	* they start after start tag, end before quietzone)
	'''
	bars = splitToBars(lineReads)
	check(len(bars) > 10, 'too little bars on scanline')
	spans = findCodeStarts(bars, gradIdx, lineIdx)
	spans = findClosingQuietzone(bars, spans)
	spans = filterOnLen(bars, spans)
	raiseOnNone(spans)
	return bars, spans
def splitOnTags(span: Spans, bars: Bars) -> Groups:
	'''checks the middle and end tags
	* splits bars into digit groups (l/r half, bars)'''
	endTag = bars[span['end'] - 3:span['end']]
	check(differsByAtmost(span['moduleWidth'], *endTag['len']), 'incorrect tag width')
	center = span['start'] + 6 * 4
	centerEdges = bars[center:center + 5]
	check(differsByAtmost(span['moduleWidth'], *centerEdges['len']), 'incorrect tag width')
	digits = np.array((bars[span['start']:center], bars[center + 5:span['end'] - 3]))
	return digits
def toBarWidths(span: Spans, digitGroups: Groups) -> Widths:
	assert digitGroups.size == 2 * 6 * 4
	barWidths = digitGroups['len'].reshape((2, 6, 4))
	barWidths = np.round(barWidths / span['moduleWidth']).astype('int')
	check(np.all(barWidths.sum(-1) == 7), 'barWidths don\'t sum up to 7')
	return barWidths
# decoding ---------------------------------------
# NOTE here white corresponds to 0 and black = 1
# as on wikipedia "https://en.wikipedia.org/wiki/International_Article_Number"
DIGIT_ENCODINGS = {
	'L': {
		'0001101': 0,
		'0011001': 1,
		'0010011': 2,
		'0111101': 3,
		'0100011': 4,
		'0110001': 5,
		'0101111': 6,
		'0111011': 7,
		'0110111': 8,
		'0001011': 9,
	},
	'G': {
		'0100111': 0,
		'0110011': 1,
		'0011011': 2,
		'0100001': 3,
		'0011101': 4,
		'0111001': 5,
		'0000101': 6,
		'0010001': 7,
		'0001001': 8,
		'0010111': 9,
	},
	'R': {
		'1110010': 0,
		'1100110': 1,
		'1101100': 2,
		'1000010': 3,
		'1011100': 4,
		'1001110': 5,
		'1010000': 6,
		'1000100': 7,
		'1001000': 8,
		'1110100': 9,
	}
}
FIRST_DIGIT_ENCODING = {
	'111111': 0,
	'110100': 1,
	'110010': 2,
	'110001': 3,
	'101100': 4,
	'100110': 5,
	'100011': 6,
	'101010': 7,
	'101001': 8,
	'100101': 9,
}
def accessEncodingDict(d: dict, key: np.ndarray, blackFirst: bool):
	assert key.shape == (4,) and key.dtype == np.int64
	cvt = lambda idxCommaWidth: idxCommaWidth[1] * ('1' if (idxCommaWidth[0] + blackFirst) % 2 else '0')
	s = ''.join(map(cvt, enumerate(key)))
	if s not in d:
		assert False, 'digit code not found'
	return d[s]
def getLeftParities(widths):
	'''computes digit parities, returns parities for the left side
	* turns around the read direction if necessary'''
	blackBars = np.stack((widths[0, :, 1::2], widths[1, :, 0::2]), axis=0)
	parities = np.sum(blackBars, -1) % 2
	leftParities = parities[0]
	if np.all(parities[0] == 0):
		widths = widths[::-1, ::-1, ::-1]
		leftParities = parities[1]
	return leftParities, widths
def decodeDigits(widths: Widths) -> Digits:
	leftParities, widths = getLeftParities(widths)
	firstDigitEnc = ''.join(map(str, leftParities))
	digits = np.zeros(13, 'int')
	digits[0] = FIRST_DIGIT_ENCODING.get(firstDigitEnc, 0)
	for i, bars in enumerate(widths.reshape(-1, 4)):
		encoding = 'R' if i >= 6 else 'L' if leftParities[i] else 'G'
		digits[i+1] = accessEncodingDict(DIGIT_ENCODINGS[encoding], bars, blackFirst=encoding == 'R')
	return digits
def detectLine(gradIdx, lineIdx, images: Images, lineReads: LineReads) -> Digits:
	'''fully processes detection on single line'''
	bars, spans = findSpans(gradIdx, lineIdx, lineReads)
	images.addLine(bars, spans)
	detections = []
	for span in spans:
		digitGroups = splitOnTags(span, bars)
		widths = toBarWidths(span, digitGroups)
		detected = decodeDigits(widths)
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
				[detections.append(d) for d in digits if checksumDigit(d)]
			except DetectionError: continue
	detections = np.unique(np.array(detections), axis=0)
	return detections
# drawing ----------------------------------------------------
def drawGradLineReads(lineReadImgs: list[ColorImage]):
	'''draws all line reads grouped by gradient with debug info'''
	for gradIdx, lineReads in enumerate(lineReadImgs):
		linImg = np.repeat(lineReads, SCANLINE_DIST // 2, axis=0)
		cv2.imshow(f'Grad {gradIdx} - lines', linImg)
def colorcodeQuietzone(lineSlice, basewidth, quietzoneEdge, color=(0, 0, 255)):
	fourths = 2 * (color[2] != 0) + 1
	off = quietzoneEdge['start'] + (quietzoneEdge['len'] // 4) * fourths
	lineSlice[off : off + basewidth] = color
def drawDebugs(images: Images):
	cv2.imshow('lightness', toImg(images.lightness))
	cv2.imshow('localAverages', toImg(images.localAverages))
	cv2.imshow('BaW', toImg(images.BaW))
	cv2.imshow('linesImg', images.linesImg)

	lineReadImgs = toImg(images.lineReads)
	# for span in images.spans:
	# 	lineSlice = lineReadImgs[span['gradIdx'], span['lineIdx']]
	# 	startEdge = images.bars[span['start'] - 4]
	# 	colorcodeQuietzone(lineSlice, span['moduleWidth'], startEdge)
	# 	endEdge = images.bars[span['end']]
	# 	colorcodeQuietzone(lineSlice, span['moduleWidth'], endEdge, (0, 255, 0))
	drawGradLineReads(lineReadImgs)

# IO --------------------------------------------------
def processImg(img: ColorImage) -> Images:
	images = prepareImg(img)
	digits = detectImage(images)
	print(digits)
	drawDebugs(images)
	return images
	# TODO choose final read

def getInputImg() -> np.ndarray:
	img = cv2.imread('barcodes\\barcode-crop.png')
	SHRINK_FACTOR = 2
	img = img[::SHRINK_FACTOR, ::SHRINK_FACTOR]
	return img.copy()

def main():
	img = getInputImg()
	cv2.imshow('Barcode reader - input img', img)
	images = processImg(img)

	while cv2.getWindowProperty('Barcode reader - input img', cv2.WND_PROP_VISIBLE) >= 1:
		if cv2.waitKey(50) == ord('q'):
			break

	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
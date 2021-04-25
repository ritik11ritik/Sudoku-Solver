#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 11:23:23 2021

@author: rg
"""

from Puzzle import extract_digit
from Puzzle import find_puzzle_borders
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from sudoku import Sudoku
import numpy as np
import imutils
import cv2

dbg = False

print("[INFO] loading digit classifier...")
model = load_model("model.h5")

print("[INFO] processing image...")
image = cv2.imread("Sudoku.jpg")
image = imutils.resize(image, width=600)

# find the puzzle in the image and then
(puzzleImage, warped) = find_puzzle_borders(image, debug=dbg)
# initialize our 9x9 Sudoku board
board = np.zeros((9, 9), dtype="int")
board_digit = np.zeros((9,9), dtype=int)

stepX = warped.shape[1] // 9
stepY = warped.shape[0] // 9
# list to store the (x, y)-coordinates of each cell location
cellLocs = []
# loop over the grid locations
for y in range(0, 9):
	# initialize the current list of cell locations
	row = []
	for x in range(0, 9):
		# compute the starting and ending (x, y)-coordinates of the
		# current cell
		startX = x * stepX
		startY = y * stepY
		endX = (x + 1) * stepX
		endY = (y + 1) * stepY
		# add the (x, y)-coordinates to our cell locations list
		row.append((startX, startY, endX, endY))
		
		# crop the cell from the warped transform image and then
		# extract the digit from the cell
		cell = warped[startY:endY, startX:endX]
		digit = extract_digit(cell, debug=dbg)
		# verify that the digit is not empty
		if digit is not None:
			# resize the cell to 28x28 pixels and then prepare the
			# cell for classification
			roi = cv2.resize(digit, (28, 28))
			roi = roi.astype("int")
			roi = img_to_array(roi)
			roi = np.expand_dims(roi, axis=0)
			# classify the digit and update the Sudoku board with the
			# prediction
			pred = model.predict(roi).argmax(axis=1)[0] 
			board[y, x] = pred
			board_digit[y,x] = 1
	# add the row to our cell locations
	cellLocs.append(row)
	
print("[INFO] OCR'd Sudoku board:")
puzzle = Sudoku(3, 3, board=board.tolist())
puzzle.show()
# solve the Sudoku puzzle
print("[INFO] solving Sudoku puzzle...")
solution = puzzle.solve()
solution.show_full()

# loop over the cell locations and board
for (cellRow, boardRow, prevRow) in zip(cellLocs, solution.board, board_digit):
	# loop over individual cell in the row
	for (box, digit, prevDigit) in zip(cellRow, boardRow, prevRow):
		if prevDigit == 0:
			# unpack the cell coordinates
			startX, startY, endX, endY = box
			# compute the coordinates of where the digit will be drawn
			# on the output puzzle image
			textX = int((endX - startX) * 0.33)
			textY = int((endY - startY) * -0.2)
			textX += startX
			textY += endY
			# draw the result digit on the Sudoku puzzle image
			cv2.putText(puzzleImage, str(digit), (textX, textY),
				cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
# show the output image
cv2.imshow("Sudoku Result", puzzleImage)
cv2.waitKey(0)
cv2.destroyAllWindows()


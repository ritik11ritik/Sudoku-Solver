#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 11:23:23 2021

@author: rg
"""

from PuzzleCamera import extract_digit
from PuzzleCamera import find_puzzle_borders
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from sudoku import Sudoku
import numpy as np
import imutils
import cv2

dbg = False
cap = cv2.VideoCapture(0)
print("[INFO] loading digit classifier...")
model = load_model("model.h5")

print("[INFO] processing image...")
flg = 0
solved = False
board = np.zeros((9,9), dtype="int")
board_digit = np.zeros((9,9), dtype=int)
test = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Cant open camera")
        break
    
    puzzleImage, warped, puzzle_x, puzzle_y = find_puzzle_borders(frame, dbg)
    
    if warped is not None and solved is False:
        test += 1
    
    cellLocs = []    
    if warped is None:
        cv2.putText(frame, "No Puzzle Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        # print("Can't detect puzzle")
        # detect = False
            
    elif solved is True:
        stepX = warped.shape[1] // 9
        stepY = warped.shape[0] // 9
        for y in range(9):
            row = []
            for x in range(9):
                startX = x*stepX
                startY = y*stepY
                endX = (x+1)*stepX
                endY = (y+1)*stepY
                row.append((startX, startY, endX, endY))
                
            cellLocs.append(row)
        
    
    elif warped is not None and solved is False and test >= 60:
        stepX = warped.shape[1] // 9
        stepY = warped.shape[0] // 9
        for y in range(9):
            row = []
            for x in range(9):
                startX = x*stepX
                startY = y*stepY
                endX = (x+1)*stepX
                endY = (y+1)*stepY
                row.append((startX, startY, endX, endY))
                cell = warped[startY:endY, startX:endX]
                digit = extract_digit(cell, dbg)
                
                if digit is not None:
                    roi = cv2.resize(digit, (28,28))
                    roi = roi.astype("int")
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
                    
                    pred = model.predict(roi).argmax(axis=1)[0]
                    # print(pred)
                    board[y,x] = pred
                    board_digit[y,x] = 1
                    
            cellLocs.append(row)
            
        print("[INFO] OCR'd Sudoku board:")
        puzzle = Sudoku(3, 3, board=board.tolist())
        puzzle.show()
        # solve the Sudoku puzzle
        print("[INFO] solving Sudoku puzzle...")
        solution = puzzle.solve()
        solution.show_full()
        solved = True
        # loop over the cell locations and board
    if solved is True:
        for (cellRow, boardRow, prevRow) in zip(cellLocs, solution.board, board_digit):
            # loop over individual cell in the row
            for (box, digit, prevDigit) in zip(cellRow, boardRow, prevRow):
                if prevDigit == 0:
                    startX, startY, endX, endY = box
                    # compute the coordinates of where the digit will be drawn
                    # on the output puzzle image
                    textX = int((endX - startX) * 0.5)
                    textY = int((endY - startY) * -0.2)
                    textX += startX + puzzle_x
                    textY += endY + puzzle_y
                    # draw the result digit on the Sudoku puzzle image
                    cv2.putText(frame, str(digit), (textX, textY),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        # show the output image
    cv2.imshow("Sudoku Result", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
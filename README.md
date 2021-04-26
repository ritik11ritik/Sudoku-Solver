# Sudoku-Solver
## Introduction
Sudoku is one of the most popular puzzle games of all time. The goal of Sudoku is to fill a 9×9 grid with numbers so that each row, column and 3×3 section contain all of the digits between 1 and 9. As a logic puzzle, Sudoku is also an excellent brain game. If you play Sudoku daily, you will soon start to see improvements in your concentration and overall brain power.

## Requirements
The project uses python 3.8, tensorflow, keras, pandas, opencv-python, numpy and py-sudoku.

## Run
There are two ways to run the sudoku solver - 
1. Solve via input image
2. Solve via camera

To run the solver via input image, execute the following command
```
python solve.py
```
To run the solver via camera, execute the following command
```
python solveCamera.py
```
The solver will detect the puzzle and write the solved digits in their corresponding cells.


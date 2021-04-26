# Sudoku-Solver
## Introduction
Sudoku is one of the most popular puzzle games of all time. The goal of Sudoku is to fill a 9×9 grid with numbers so that each row, column and 3×3 section contain all of the digits between 1 and 9. As a logic puzzle, Sudoku is also an excellent brain game. If you play Sudoku daily, you will soon start to see improvements in your concentration and overall brain power.

## Requirements
The project uses python 3.8, tensorflow, keras, pandas, opencv-python, numpy and py-sudoku.

## Digit Recognition
The model for digit recognition is trained using the dataset.zip file. First it is converted to train.csv using image_to_csv.py. The csv file is then read using pandas and a CNN model is trained using tensorflow and keras. The accuracy of the trained model was found to be 99.61 %.

Since the MNIST dataset contains hand written digits and in this project, there is a need to predict printed digits, the MNIST dataset doesn't give sufficient accuracy. If there is even a single wrong prediction, whole board may be affected. Hence, the model is trained on printed digits dataset.

## Run
There are two ways to run the sudoku solver
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

## Results
The program was tested on several sudoku puzzles. One example is shown below
### Puzzle
<img src="https://github.com/ritik11ritik/Sudoku-Solver/blob/main/Sudoku.jpg" width="400">

### Solved Puzzle
<img src="https://github.com/ritik11ritik/Sudoku-Solver/blob/main/Sudoku_Solved.jpg" width="400">


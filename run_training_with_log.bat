@echo off
echo Starting YOLO training...
echo Output will be saved to: training_output.log
echo.
python train_yolo.py > training_output.log 2>&1
echo.
echo Training completed or stopped.
echo Check training_output.log for details.
pause


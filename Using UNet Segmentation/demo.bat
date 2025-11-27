@echo off
REM Demo: run inference on sample image (update paths if needed)
set ROOT=%~dp0
python "%ROOT%src\infer.py" "%ROOT%sample_outputs\input.jpg" "%ROOT%checkpoints\unet_idcard.pth" "%ROOT%out"
pause

@echo off
REM Quick setup script for L1 GPU training on Windows
echo 🎮 L1 Quick Setup for GPU Training
echo ================================

REM Check if we're in the right directory
if not exist "train_gpu_compatible.py" (
    echo ❌ Please run this from the L1 project directory
    pause
    exit /b 1
)

echo.
echo 🔧 Step 1: Installing Python packages...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

echo.
echo 🔍 Step 2: Checking GPU...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo.
echo 📚 Step 3: Setting up dataset...
echo Choose your dataset:
echo   1. Wikipedia Simple English (recommended - fast setup)
echo   2. Advanced preset (Wikipedia + Books + News)
echo   3. Beginner preset (quick training)
echo.
set /p dataset_choice="Enter choice (1, 2, or 3): "

if "%dataset_choice%"=="1" (
    echo Setting up Wikipedia Simple English dataset...
    python prepare_large_dataset.py
) else if "%dataset_choice%"=="2" (
    echo Setting up advanced dataset preset...
    python add_dataset.py --preset advanced
) else if "%dataset_choice%"=="3" (
    echo Setting up beginner dataset preset...
    python add_dataset.py --preset beginner
) else (
    echo Invalid choice, using default Wikipedia Simple English...
    python prepare_large_dataset.py
)

echo.
echo ✅ Setup complete! 
echo.
echo 🚀 Ready to train! Options:
echo   1. Start GPU training now (RTX 40/50 series optimized)
echo   2. Exit and train later
echo.
set /p choice="Enter choice (1 or 2): "

if "%choice%"=="1" (
    echo.
    echo 🎮 Starting GPU-optimized training...
    echo Features: Mixed precision, auto-checkpointing every 100 steps, automatic resume
    echo.
    python train_gpu_compatible.py
) else (
    echo.
    echo 💡 To start training later, run:
    echo    python train_gpu_compatible.py
    echo.
    echo 📁 Your trained model will be saved to: models/l1-gpu-compatible/
    echo 🔄 Training automatically resumes from checkpoints
)

pause

@echo off
REM Quick setup script for L1 GPU training on Windows
echo 🎮 L1 Quick Setup for GPU Training
echo ================================

REM Check if we're in the right directory
if not exist "train_gpu.py" (
    echo ❌ Please run this from the L1 project directory
    pause
    exit /b 1
)

echo.
echo 🔧 Step 1: Installing Python packages...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install kaggle pandas numpy tqdm pyyaml

echo.
echo 🔍 Step 2: Checking GPU...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo.
echo 📚 Step 3: Downloading sample dataset...
echo Setting up Kaggle (you may need to enter credentials)
mkdir datasets 2>nul
kaggle datasets download -d snapcrack/all-the-news -p ./datasets/ --unzip

echo.
echo ⚙️ Step 4: Processing dataset...
python prepare_large_dataset.py "datasets/articles1.csv" --text-column "content" --max-samples 50000 --vocab-size 15000

echo.
echo ✅ Setup complete! 
echo.
echo 🚀 Ready to train! Options:
echo   1. Start training now
echo   2. Exit and train later
echo.
set /p choice="Enter choice (1 or 2): "

if "%choice%"=="1" (
    echo Starting GPU training...
    python train_gpu.py
) else (
    echo.
    echo 💡 To start training later, run:
    echo    python train_gpu.py
    echo.
    echo 📁 Your trained model will be saved to: models/l1-gpu-v1/
)

pause

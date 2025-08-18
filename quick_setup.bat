@echo off
REM Quick setup script for L1 GPU training on Windows
echo 🎮 L1 Quick Setup for GPU Training
echo ================================
echo Features: BPE tokenization, intelligent text generation, automatic tokenizer fixes
echo.

REM Check if we're in the right directory
if not exist "tools\train.py" (
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
echo 📚 Step 3: Setting up dataset with BPE tokenization...
echo Choose your dataset:
echo   1. Advanced preset (Wikipedia + ArXiv - recommended for intelligence)
echo   2. Beginner preset (quick training)
echo   3. Intermediate preset (balanced training)
echo.
set /p dataset_choice="Enter choice (1, 2, or 3): "

if "%dataset_choice%"=="1" (
    echo Setting up advanced dataset preset...
    python data_tools/add_dataset.py --preset advanced
    echo Preparing dataset with BPE tokenization (32k vocab for intelligence)...
    python data_tools/prepare_dataset.py data/raw/combined_dataset.txt --vocab-size 32000
) else if "%dataset_choice%"=="2" (
    echo Setting up beginner dataset preset...
    python data_tools/add_dataset.py --preset beginner
    echo Preparing dataset with BPE tokenization...
    python data_tools/prepare_dataset.py data/raw/combined_dataset.txt --vocab-size 32000
) else if "%dataset_choice%"=="3" (
    echo Setting up intermediate dataset preset...
    python data_tools/add_dataset.py --preset intermediate
    echo Preparing dataset with BPE tokenization...
    python data_tools/prepare_dataset.py data/raw/combined_dataset.txt --vocab-size 32000
) else (
    echo Invalid choice, using advanced preset...
    python data_tools/add_dataset.py --preset advanced
    python data_tools/prepare_dataset.py data/raw/combined_dataset.txt --vocab-size 32000
)

echo.
echo 🔧 Step 4: Fixing tokenizer (ensures proper text generation)...
python data_tools/fix_tokenizer.py

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
    echo Features: Mixed precision, auto-checkpointing every 1000 steps, automatic resume
    echo Expected: ~9.5 steps/second, 134M parameters, BPE tokenization
    echo.
    python tools/train.py
) else (
    echo.
    echo 💡 To start training later, run:
    echo    python tools/train.py
    echo.
    echo 📁 Your trained model will be saved to: models/l1-gpu-compatible/
    echo 🔄 Training automatically resumes from checkpoints
    echo 🎯 Test generation with: python tools/generate.py
)

pause

@echo off
REM =========================================================================
REM  Yard Hazard Detection - Training Pipeline Launcher
REM =========================================================================
setlocal enabledelayedexpansion

set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"
call .venv\Scripts\activate.bat 2>nul
set PYTHONPATH=.;src

:MAIN_MENU
REM Clear all choice variables each time we return to menu
set "CHOICE="
set "FOLDER_CHOICE="
set "DATASET_DIR="
set "AUGMENT_TARGET="
set "SIZE_CHOICE="
set "TARGET_PER_CLASS="
set "COPIES="
set "TRAIN_DATA="
set "TRAIN_FOLDER="
set "INF_CHOICE="
set "ADD_HAZARD="
set "INSERT_HAZARD="

echo.
echo  ============================================================
echo   Yard Hazard Detection - Training Pipeline
echo  ============================================================
echo.
echo   1) Check folder dataset
echo   2) Augment a dataset
echo   3) Train new model
echo   4) Evaluate current best
echo   5) Run inference engine test
echo   6) Exit
echo.
set /p CHOICE="  Select an option (1-6): "

if "%CHOICE%"=="1" goto CHECK_DATASET
if "%CHOICE%"=="2" goto AUGMENT_DATASET
if "%CHOICE%"=="3" goto TRAIN_MODEL
if "%CHOICE%"=="4" goto EVALUATE
if "%CHOICE%"=="5" goto INFERENCE_TEST
if "%CHOICE%"=="6" goto END
echo  Invalid option. Try again.
goto MAIN_MENU

REM =========================================================================
:CHECK_DATASET
REM =========================================================================
echo.
echo  --- Check Folder Dataset ---
echo.
echo   Select dataset folder:
echo     1) image_data_normal
echo     2) image_data_with_synth
echo     3) roboflow data
echo.
set /p FOLDER_CHOICE="  Select folder (1-3): "

if "!FOLDER_CHOICE!"=="1" set "DATASET_DIR=image_data_normal"
if "!FOLDER_CHOICE!"=="2" set "DATASET_DIR=image_data_with_synth"
if "!FOLDER_CHOICE!"=="3" set "DATASET_DIR=roboflow data"

if "!DATASET_DIR!"=="" (
    echo  Invalid selection.
    goto MAIN_MENU
)

echo.
echo  Checking dataset: !DATASET_DIR!
echo.
python scripts/check_dataset.py --source "!DATASET_DIR!"
echo.

set /p ADD_HAZARD="  Do you need to add hazard data? (y/n): "
if /i "!ADD_HAZARD!"=="y" (
    set "AUGMENT_TARGET=!DATASET_DIR!"
    goto AUGMENT_SIZE_PROMPT
)

echo.
echo  Running inference viewer on: !DATASET_DIR!
echo  (Annotated images will be saved to runs/detect/dataset_check/)
echo.
python scripts/inference_test.py --save
echo.
echo  Done. Check runs/inference_test/ for saved annotated images.
echo.
goto MAIN_MENU

REM =========================================================================
:AUGMENT_DATASET
REM =========================================================================
echo.
echo  --- Augment a Dataset ---
echo.
echo   Select a dataset to augment:
echo     1) image_data_normal
echo     2) image_data_with_synth
echo     3) roboflow data
echo.
set /p FOLDER_CHOICE="  Select folder (1-3): "

if "!FOLDER_CHOICE!"=="1" set "AUGMENT_TARGET=image_data_normal"
if "!FOLDER_CHOICE!"=="2" set "AUGMENT_TARGET=image_data_with_synth"
if "!FOLDER_CHOICE!"=="3" set "AUGMENT_TARGET=roboflow data"

if "!AUGMENT_TARGET!"=="" (
    echo  Invalid selection.
    goto MAIN_MENU
)

:AUGMENT_SIZE_PROMPT
echo.
echo   Select target size per class:
echo     1) 100 per class (300 total)
echo     2) 300 per class (900 total)
echo     3) 500 per class (1500 total)
echo.
set /p SIZE_CHOICE="  Select size (1-3): "

if "!SIZE_CHOICE!"=="1" set "TARGET_PER_CLASS=100"
if "!SIZE_CHOICE!"=="2" set "TARGET_PER_CLASS=300"
if "!SIZE_CHOICE!"=="3" set "TARGET_PER_CLASS=500"

if "!TARGET_PER_CLASS!"=="" (
    echo  Invalid selection.
    goto MAIN_MENU
)

echo.
echo  Augmenting !AUGMENT_TARGET! to ~!TARGET_PER_CLASS! per class...
echo  (Calculating copies needed based on current dataset size)
echo.

REM Calculate copies needed using Python
python -c "import os, math; d='!AUGMENT_TARGET!'; imgs=len([f for f in os.listdir(d+'/train/images') if f.lower().endswith(('.jpg','.jpeg','.png'))]) if os.path.exists(d+'/train/images') else 30; copies=max(1, math.ceil(!TARGET_PER_CLASS!/max(1,imgs))-1); print(copies)" > _copies_tmp.txt 2>nul
set /p COPIES=<_copies_tmp.txt
del _copies_tmp.txt 2>nul

if "!COPIES!"=="" set "COPIES=3"

echo  Using --copies !COPIES!
echo.
python scripts/augment_dataset.py --copies !COPIES!
echo.

set /p INSERT_HAZARD="  Insert hazard data? (y/n): "
if /i "!INSERT_HAZARD!"=="y" (
    echo.
    echo  Running synthetic hazard injection...
    echo  Normal dir: !AUGMENT_TARGET!
    echo.
    python scripts/generate_hazard_augmentations.py --normal_dir "!AUGMENT_TARGET!" --roboflow_dir "roboflow data" --output_dir "!AUGMENT_TARGET!_augmented_hazards" --max_images 500
    echo.
    echo  Hazard injection complete.
    echo  Output saved to: !AUGMENT_TARGET!_augmented_hazards/
)

echo.
echo  Augmentation complete.
echo.
goto MAIN_MENU

REM =========================================================================
:TRAIN_MODEL
REM =========================================================================
echo.
echo  --- Train New Model ---
echo.
echo   Select dataset folder for training:
echo     1) image_data_with_synth
echo     2) roboflow data
echo.
set /p TRAIN_FOLDER="  Select folder (1-2): "

if "!TRAIN_FOLDER!"=="1" (
    echo.
    echo  Re-packaging image_data_with_synth into 12-class trainable format...
    echo.
    python scripts/package_image_data_with_synth.py --reduced_classes --output_dir "image_data_with_synth_split" --val_fraction 0.15
    echo.
    set "TRAIN_DATA=image_data_with_synth_split/data.yaml"
)
if "!TRAIN_FOLDER!"=="2" set "TRAIN_DATA=roboflow data/data.yaml"

if "!TRAIN_DATA!"=="" (
    echo  Invalid selection.
    goto MAIN_MENU
)

echo.
echo  Starting training with: !TRAIN_DATA!
echo.
python scripts/train_yolo.py --data "!TRAIN_DATA!"
echo.
echo  Training complete.
echo.
goto MAIN_MENU

REM =========================================================================
:EVALUATE
REM =========================================================================
echo.
echo  --- Evaluate Current Best ---
echo.
echo  Running evaluation...
echo.
python scripts/evaluate_yolo.py
echo.
echo  Results saved to evaluation_results/
start "" "evaluation_results"
echo.
goto MAIN_MENU

REM =========================================================================
:INFERENCE_TEST
REM =========================================================================
echo.
echo  --- Run Inference Engine Test ---
echo.
echo   Select image source:
echo     1) Random image from dataset
echo     2) Live camera snapshot
echo.
set /p INF_CHOICE="  Select source (1-2): "

if "!INF_CHOICE!"=="2" (
    echo.
    echo  ============================================================
    echo   Auto-photo system still in progress.
    echo   Camera snapshot pipeline not yet configured.
    echo   This feature will be available once live camera
    echo   integration is complete.
    echo  ============================================================
    echo.
    goto MAIN_MENU
)

if "!INF_CHOICE!"=="1" (
    echo.
    echo  Running inference on a random dataset image...
    echo.
    python scripts/inference_test.py --save
    echo.
)

goto MAIN_MENU

REM =========================================================================
:END
REM =========================================================================
echo.
echo  Goodbye.
endlocal
exit /b 0

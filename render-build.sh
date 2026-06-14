#!/bin/bash
# render-build.sh

# 1. Ensure virtual environment is created
python -m venv venv

# 2. Activate virtual environment
# On Linux/macOS
source venv/bin/activate

# On Windows Git Bash (Render uses Linux environment, so above works)

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install requirements
pip install -r requirements.txt

# 5. Download dlib model files if not present
if [ ! -f shape_predictor_68_face_landmarks.dat ]; then
    echo "Downloading shape_predictor_68_face_landmarks.dat..."
    wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
fi

if [ ! -f dlib_face_recognition_resnet_model_v1.dat ]; then
    echo "Downloading dlib_face_recognition_resnet_model_v1.dat..."
    wget http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
    bzip2 -d dlib_face_recognition_resnet_model_v1.dat.bz2
fi

echo "Build complete!"

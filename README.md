# FlexMotion

FlexMotion is a real-time hand gesture recognition system built with OpenCV, MediaPipe, and PyTorch. It detects hands from a webcam, extracts landmark features, and classifies gestures such as thumbs up, open palm, fist, and OK.

## Overview

This project is designed to collect labeled hand landmark data, train a lightweight neural network on top of those features, and run live predictions in real time. The goal is to provide a simple but functional pipeline for gesture-based interaction using a standard webcam.

## Features

- Webcam hand detection using MediaPipe
- Real-time gesture inference with a PyTorch classifier
- Data collection utility for building a custom gesture dataset
- Training pipeline with validation metrics and saved checkpoints
- Exported scaler and model artifacts for inference


## Requirements

The project depends on Python 3 and the packages listed in `requirements.txt`.

Install them with:

```bash
pip install -r requirements.txt
```

## Data collection

To collect new gesture samples, run the data collector and record a label from the webcam:

```bash
python src/data_collector.py --gesture thumbs_up --samples 300
```

Available gesture labels are defined in `src/data_collector.py`:

- `thumbs_up`
- `open_palm`
- `fist`
- `ok`

Use the space bar to start or pause recording, and press `Q` to exit.

## Training

Once your dataset is ready, train the model:

```bash
python src/train.py
```

This script:

- loads the dataset from `data/raw/gestures.csv`
- scales the input features
- trains a small neural network
- saves the best model checkpoint to `models/gesture_model.pth`
- saves the scaler to `models/scaler.joblib`
- stores training metrics and visualization plots in `models/`

## Live inference

To run the webcam recognition demo:

```bash
python src/main.py
```

The app:

- opens the default camera
- detects a hand
- extracts 3D landmark coordinates
- normalizes them with the trained scaler
- predicts the gesture label with confidence
- displays the result on the live video feed

Press `Q` in the video window to quit.

## Model details

The classifier is a lightweight feedforward neural network with:

- an input size of 63 features (21 hand landmarks × 3 coordinates)
- two hidden layers
- a final output layer matching the number of gesture classes

The model is trained using cross-entropy loss and monitored with validation accuracy and loss.

## Example workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Collect training data
python src/data_collector.py --gesture open_palm --samples 300
python src/data_collector.py --gesture fist --samples 300
python src/data_collector.py --gesture thumbs_up --samples 300
python src/data_collector.py --gesture ok --samples 300

# 3. Train the model
python src/train.py

# 4. Run real-time recognition
python src/main.py
```

## Notes

- The project is intended for local experimentation and learning purposes.
- For best results, collect a balanced dataset with consistent lighting and camera framing.
- Model performance can improve by adding more gesture classes and more diverse training samples.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

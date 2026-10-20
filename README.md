# Real-Time ASL Sign Recognition Using MediaPipe and GRU

A real-time American Sign Language (ASL) recognition system that uses webcam input, MediaPipe Holistic landmark extraction, and a Gated Recurrent Unit (GRU) deep learning model to predict signs and display their corresponding labels.

This project adapts an existing pose-based ASL recognition pipeline for a college project, with a focus on real-time inference, prediction display, and evaluation of sign recognition performance.

> **Disclaimer:** This is an educational prototype, not a professional sign language translation system. Recognition accuracy can vary depending on the signer, lighting, camera positioning, and similarity between signs. The system is designed to recognize individual signs, not reliably translate complete sentences.

## Features

- **Real-time webcam input** using OpenCV.
- **Pose and landmark extraction** using MediaPipe Holistic.
- **Sequence-based sign recognition** using a GRU neural network.
- **Preprocessing of landmark sequences** before model inference.
- **Prediction display** for recognized sign labels.
- **Local inference** using a pretrained model, without requiring an OpenAI API key.
- **Experimental evaluation** using repeated trials of selected ASL signs.

## How It Works

The system follows a pose-based recognition pipeline:

1. **Video capture:** Captures frames from a webcam.
2. **Landmark extraction:** MediaPipe Holistic extracts relevant body, hand, and facial landmarks.
3. **Preprocessing:** Landmark data is processed into a sequence compatible with the trained model.
4. **Sign classification:** A GRU-based neural network processes the sequence and predicts a class.
5. **Label decoding:** The predicted class index is mapped to its corresponding sign label.
6. **Display:** The prediction is shown in the live interface.

## Model and Dataset

### Dataset

The project is based on the [ASL Citizen Dataset](https://www.microsoft.com/en-us/research/project/asl-citizen/), a community-sourced dataset developed for isolated sign language recognition.

**Dataset reference:**

Desai, A., et al. (2023). *ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition*. NeurIPS 2023 Datasets and Benchmarks Track.

[Research paper](https://arxiv.org/abs/2304.05934)

### Recognition model

The project uses a pretrained GRU-based sequence classification model.

The current model and its associated label mapping are expected to be available in the `models/` directory:

- `best_model200_new.keras` — GRU model with 209 output classes.
- `index_to_gloss_200.json` — mapping from model output indices to sign labels.

The model processes sequences of pose-derived features and produces a probability distribution over its output classes. The highest-scoring class is used to obtain the predicted label.

**Important:** The model's output classes and the number of unique signs in the dataset are not necessarily the same. The mapping file must match the specific model checkpoint being loaded.

The model's training provenance and validation performance should be verified against the original training records before reporting them as results for this project.

## Technology Stack

- **Python 3.11**
- **TensorFlow / Keras** — model inference
- **MediaPipe Holistic** — landmark extraction
- **OpenCV** — webcam access and image processing
- **NumPy** — numerical processing
- **Jupyter Notebook** — interactive development and execution

Refer to `requirements.txt` for the project's dependency versions.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/aryanthu/asl-translation.git
cd asl-translation
```

### 2. Create a virtual environment

Using Python 3.11:

```bash
py -3.11 -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify the model files

Ensure the following files exist:

```text
models/
├── best_model200_new.keras
└── index_to_gloss_200.json
```

The model and label mapping must be compatible. If either file is missing or belongs to a different checkpoint, inference may fail or produce incorrect labels.

### 5. Run the live recognition system

Open `main_livefeed.ipynb` from the repository root using Jupyter Notebook or VS Code, select the project's `.venv` Python interpreter, and run the live-inference section.

Use the model and mapping paths configured for the current project. If the notebook still contains the original API-based sentence refinement code, use the local inference path with the API integration disabled.

The pretrained model allows live inference without downloading the complete training dataset. Reproducing the training pipeline, however, requires the relevant dataset, preprocessing outputs, and training configuration.

**Controls:** Press `q` to quit the webcam window. Press `a` to remove the most recent displayed prediction, if supported by the current implementation.

## Evaluation

The model is evaluated through repeated webcam trials using selected sign labels.

For each trial:

1. Select the intended sign.
2. Perform the sign in front of the webcam.
3. Record the predicted label.
4. Mark the trial as correct only when the prediction matches the intended label.
5. Repeat the procedure across multiple attempts and signs.

Recognition accuracy is calculated as:

\[
\text{Accuracy} =
\frac{\text{Correct Predictions}}{\text{Total Trials}}
\times 100
\]

Evaluation results should be reported using the actual recorded trials. Performance may differ across signs and signing conditions; overall accuracy alone does not capture every limitation.

## Limitations

- The model recognizes a fixed set of trained classes rather than arbitrary ASL signs.
- Similar-looking signs may be confused.
- Performance can vary with lighting, camera angle, signing speed, and landmark detection quality.
- A predicted label does not guarantee that the intended sign was performed correctly.
- Isolated sign recognition is different from continuous sign language translation, which also requires temporal segmentation, grammatical interpretation, and sentence-level context.
- The current system should not be treated as a substitute for a qualified sign language interpreter.

## References

1. Desai, A., et al. (2023). *ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition*. [Paper](https://arxiv.org/abs/2304.05934)
2. Roh, K., Lee, H., Hwang, E. J., Cho, S., & Park, J. C. (2024). *Preprocessing Mediapipe Keypoints with Keypoint Reconstruction and Anchors for Isolated Sign Language Recognition*. LREC-COLING 2024, Workshop on the Representation and Processing of Sign Languages. [Paper](https://aclanthology.org/2024.signlang-1.36.pdf)
3. Original project repository: [tobp03/live-asl-translation](https://github.com/tobp03/live-asl-translation)
4. Original project's [Kaggle training notebook](https://www.kaggle.com/code/tobypu/aslcitizen-top200-training)

## Acknowledgements

This project builds upon the work of the original repository's author and the creators and contributors of the ASL Citizen Dataset.

Please consult the original project's licence and the dataset's terms of use before redistributing code, model files, or dataset-derived material.

---

**Project status:** Educational prototype under evaluation.

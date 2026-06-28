# skin-disease-classifier — Skin Disease Classification with Grad-CAM Explainability

A deep learning system that classifies images of skin lesions into one of nine disease categories, served through a Streamlit web application. The model is a fine-tuned convolutional neural network selected from six candidate architectures, paired with Grad-CAM visual explanations and an out-of-distribution (OOD) confidence threshold to flag uncertain predictions.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Models and Methodology](#models-and-methodology)
- [Evaluation](#evaluation)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Running the Application](#running-the-application)
- [Model Artefact](#model-artefact)
- [Tech Stack](#tech-stack)


## Overview

This application takes an uploaded skin image and returns a predicted disease class with a confidence score, a Grad-CAM heatmap highlighting the regions that most influenced the prediction, and the top-3 most likely classes. If the model's confidence falls below a calibrated threshold, the image is flagged as "Unknown / OOD" instead of being forced into one of the nine known categories.

## System Architecture

```
model/
  skin_disease_model.keras    Trained Keras model (DenseNet121 backbone)
  metadata.json                Class names, image size, OOD threshold

skin_disease_classifier.ipynb  Data exploration, model training, evaluation
app.py                         Streamlit application
```

At runtime, `app.py` loads `skin_disease_model.keras` and `metadata.json` from the local `model/` directory via `@st.cache_resource`, so the model is loaded once per session rather than on every prediction. Each uploaded image is preprocessed using the backbone-specific preprocessing function (resolved from `metadata.json`), passed through the model for prediction, and through a Grad-CAM routine for visual explanation.

---

## Models and Methodology

### Transfer Learning with Six Candidate Backbones

Six ImageNet-pretrained backbones were evaluated under an identical training pipeline: **MobileNetV2, VGG16, ResNet50V2, InceptionV3, DenseNet121, EfficientNetB3**. Each backbone is paired with the same custom classification head:

```
GlobalAveragePooling2D → BatchNormalization → Dense(512, relu) → Dropout(0.5)
                       → Dense(256, relu) → Dropout(0.3) → Dense(9, softmax)
```

### Two-Phase Training

- **Phase 1 — Head training:** backbone frozen, only the classification head is trained (up to 20 epochs, Adam, lr = 3e-4).
- **Phase 2 — Fine-tuning:** the top 12 backbone layers are unfrozen (BatchNorm layers excluded) and trained at a lower learning rate (up to 20 epochs, Adam, lr = 1e-5).

`EarlyStopping` and `ReduceLROnPlateau` are applied in both phases, and class weights (computed from training set class frequencies) are used to counteract class imbalance across the nine disease categories.

### Model Selection

All six backbones are compared on validation accuracy, weighted F1, AUC (one-vs-rest), and training time. **DenseNet121** was selected as the final model based on the best validation accuracy and weighted F1 score.

### Ensemble Check

The top-2 performing backbones were additionally combined via accuracy-weighted soft voting, to check whether ensembling could outperform the single best model. For this run, the single best model (DenseNet121) was selected as the final model. Ensembling remains a natural direction for further improvement, since it tends to help most when the combined models make sufficiently diverse errors.

### Out-of-Distribution (OOD) Detection

Rather than always committing to one of the nine trained classes, predictions with a maximum softmax confidence below a calibrated threshold are labelled **"Unknown / OOD"**. The threshold (0.6) was chosen by analysing the trade-off between accuracy on accepted predictions and the percentage of samples rejected across a range of candidate thresholds.

### Explainability — Grad-CAM

For every prediction, a Grad-CAM heatmap is generated from the last convolutional layer of the model and overlaid on the original image, visually indicating which regions of the skin image most influenced the predicted class.


## Evaluation

Final model: **DenseNet121**, evaluated on the held-out validation set (181 images across 9 classes).

| Metric        | Score  |
|---------------|--------|
| Accuracy      | 0.8453 |
| Weighted F1   | 0.8415 |
| Macro F1      | 0.8406 |

**OOD threshold analysis** (threshold = 0.6): accuracy on accepted samples = 89.47%, rejection rate = 16.0% of valid skin images.

**Limitation:** recall is lowest precisely on the three cancerous classes — Melanoma (0.70), Squamous Cell Carcinoma (0.65), and Actinic Keratosis (0.55) — compared to 0.85+ on every benign/non-cancerous class. Since a false negative on these classes means a missed cancer, this is the main limitation to address before any real-world clinical use.


## Data Sources

| Dataset | Source | Description |
|---|---|---|
| Skin Disease Classification Image Dataset | [Kaggle — riyaelizashaju](https://www.kaggle.com/datasets/riyaelizashaju/skin-disease-classification-image-dataset) | Labelled skin lesion images across 9 disease classes (Actinic Keratosis, Atopic Dermatitis, Benign Keratosis, Dermatofibroma, Melanocytic Nevus, Melanoma, Squamous Cell Carcinoma, Tinea Ringworm Candidiasis, Vascular Lesion) |


## Project Structure

```
skin-disease-classifier/
├── model/
│   ├── skin_disease_model.keras
│   └── metadata.json
├── skin_disease_classifier.ipynb
├── app.py
├── requirements.txt
└── README.md
```


## Setup and Installation

**Prerequisites:** Python 3.10 or higher.

```bash
git clone https://github.com/Yuvraj428/skin-disease-classifier.git
cd skin-disease-classifier

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```


## Running the Application

```bash
streamlit run app.py
```


## Model Artefact

`skin_disease_model.keras` (~31 MB) and `metadata.json` are included directly in this repository under `model/`, well within GitHub's 100 MB file size limit. No separate download step is required — once you clone the repo, the model is already in place.

To regenerate these files from scratch instead:

1. Download the dataset from the [Data Sources](#data-sources) section (or run the Kaggle download cell in the notebook).
2. Open `skin_disease_classifier.ipynb` in Jupyter, Colab, or VS Code.
3. Set your Kaggle username and API key in the designated cell near the start of the notebook.
4. Run all cells. The model-saving cell writes `skin_disease_model.keras` and `metadata.json` to your chosen output directory.


## Tech Stack

- **Python 3.10+**
- **Streamlit** — web application framework
- **TensorFlow / Keras** — model training and inference (transfer learning on DenseNet121, ResNet50V2, MobileNetV2, EfficientNetB3, InceptionV3, VGG16)
- **OpenCV** — Grad-CAM heatmap generation and image processing
- **scikit-learn** — classification metrics, class weighting
- **NumPy / Pandas** — data processing
- **Matplotlib / Seaborn** — exploratory data analysis and training visualisations
- **Pillow** — image I/O within the Streamlit app

Developed by Yuvraj Aarsh

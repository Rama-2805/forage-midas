# Handwritten Digit/Character Recognition with a Custom MLP

## 1) Problem Statement
Postal services and document digitisation pipelines often depend on expensive OCR vendors. This project builds an end-to-end, low-cost handwritten recognition system using a Multi-Layer Perceptron (MLP) trained from scratch (NumPy) and in Keras, then deployed to a browser drawing canvas for real-time inference.

## 2) Success Criteria
- Accuracy target: greater than 99% test accuracy on MNIST.
- Generalisation target: competitive performance on EMNIST (letters+digits) and IAM subsets.
- User experience target: browser canvas with top-3 predictions plus softmax probabilities in real time.
- Operations target: misclassified samples automatically logged for retraining.

## 3) Dataset Strategy

### A. MNIST (primary benchmark)
- 60k train / 10k test grayscale 28x28 digits.
- Use as the baseline for architecture and optimisation sweeps.

### B. EMNIST (letters and digits)
- Add labels for broader postal OCR use-cases.
- Start with EMNIST Balanced to avoid severe class imbalance.

### C. IAM Handwriting Database
- Use only single-character crops (or segmented patches) for MLP compatibility.
- Keep IAM as a domain-shift evaluation set and future fine-tuning pool.

### Unified preprocessing
1. Convert to grayscale.
2. Resize to 28x28 (or 32x32 if architecture is upgraded).
3. Center by center-of-mass and normalize stroke thickness.
4. Scale pixels to [0, 1].
5. Flatten image to vector for MLP input.
6. Optional augmentations: random shift (+/-2 px), slight rotation (+/-10 deg), light elastic distortions, random contrast jitter.

## 4) Model Design and Experiment Plan

### Baseline MLP
- Input: 784
- Hidden: [512, 256]
- Activation: ReLU
- Output: 10-way softmax (MNIST), then 47/62-way softmax for EMNIST variants
- Loss: Categorical cross-entropy
- Optimiser: Adam (lr=1e-3)
- Epochs: 20
- Batch size: 128

### Controlled experiment matrix
Perform one-variable-at-a-time and then factorial sweeps:

1. Depth / width
   - [256]
   - [512, 256]
   - [1024, 512, 256]
2. Activation
   - ReLU
   - tanh
   - sigmoid
3. Regularisation
   - Dropout: 0.1 / 0.2 / 0.3
   - L2 weight decay: 1e-5 / 1e-4 / 1e-3
   - BatchNorm on/off after each dense layer
4. Learning-rate schedules
   - Constant lr
   - ReduceLROnPlateau
   - Cosine annealing or OneCycle

### Recommended high-performing config (to reach >99% MNIST)
- Layers: Dense(1024) -> BN -> ReLU -> Dropout(0.2) -> Dense(512) -> BN -> ReLU -> Dropout(0.2) -> Dense(256) -> BN -> ReLU -> Dense(10, softmax)
- Optimiser: AdamW, lr=1e-3, weight_decay=1e-4
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=2)
- Early stopping: patience=5, restore best weights
- Label smoothing: 0.05 (optional)

## 5) From-Scratch NumPy Implementation (Learning Goal)
Implement a pedagogical MLP with:
- Xavier or He initialisation
- Forward pass with configurable activations
- Backpropagation (dW, db per layer)
- Mini-batch SGD/Adam
- Dropout mask in train mode only
- BatchNorm forward/backward (optional advanced task)
- Gradient checking on a tiny synthetic batch

## 6) Keras Production Training Pipeline

### Folder structure
```text
mlp-ocr/
  data/
  notebooks/
  src/
    data.py
    model.py
    train.py
    evaluate.py
    export.py
    web/
      app.py
      static/
        index.html
        app.js
        styles.css
  artifacts/
    best_model.keras
    class_map.json
    misclassified/
```

### Training loop requirements
- Stratified train/val split.
- TensorBoard logging: loss, accuracy, lr.
- Save best checkpoint by validation accuracy.
- Confusion matrix plus per-class precision/recall report.
- Persist misclassified images with predicted-vs-true metadata.

### Suggested metrics
- Top-1 accuracy (primary)
- Top-3 accuracy (for UI relevance)
- Expected Calibration Error (optional)

## 7) Deployment: Real-Time Browser Canvas

### Inference architecture options
1. TensorFlow.js in-browser (best latency, no server inference cost).
2. Flask/FastAPI backend serving predictions via REST.

For rapid delivery, backend inference is simplest:
- Frontend canvas captures 280x280 drawing.
- Downsample to 28x28 and normalize.
- Send JSON array (784 values) to /predict.
- Backend returns top-3 labels and probabilities.

### Response format
```json
{
  "top3": [
    {"label": "8", "prob": 0.9731},
    {"label": "3", "prob": 0.0208},
    {"label": "9", "prob": 0.0034}
  ],
  "entropy": 0.17,
  "needs_review": false
}
```

### Misclassification logging flow
- UI asks for correction when confidence is below threshold (e.g., 0.75).
- Corrected sample stored with timestamp and source.
- Nightly job appends approved corrections to retraining dataset.

## 8) Retraining and MLOps Loop
1. Collect hard examples (low confidence, user corrected, misclassified).
2. Data quality checks (duplicates, malformed strokes, label drift).
3. Scheduled retraining (weekly or monthly based on volume).
4. Compare challenger vs champion model on fixed holdout.
5. Promote only if there is no regression on MNIST and there is improvement on production hard examples.

## 9) Risks and Mitigations
- Class imbalance (EMNIST/IAM): weighted loss or class-balanced sampling.
- Domain shift from clean MNIST to real postal scribbles: stronger augmentation and user correction ingestion.
- Overfitting with large MLPs: early stopping, dropout, L2, batch norm.
- Ambiguous glyphs (1/l/I, 0/O): expose top-3 predictions and confidence.

## 10) Milestone Plan (4 Weeks)
- Week 1: data loaders, baseline MLP, MNIST above 98.5%.
- Week 2: architecture/activation/regularisation sweep, reach above 99% MNIST.
- Week 3: EMNIST + IAM adaptation, evaluation dashboard, misclassification logger.
- Week 4: browser canvas + API, top-3 UX, packaging and demo.

## 11) Expected Deliverables
- Reproducible training scripts (NumPy + Keras).
- Best model artifact plus class mapping.
- Experiment report with ablations and confusion matrices.
- Live browser demo for handwritten input with top-3 predictions.
- Misclassification capture pipeline for continual learning.

## Data
~8,000 raw images from [BreakHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/).
The dataset contains 4 magnification levels (40X, 100X, 200X, 400X) of the same tissue slides.
Filenames encode the slide ID: in `SOB_B_TA-14-4659-40-001.png`, `14-4659` is the slide and `40` is the magnification.
Since the same slide appears across all magnifications, splits must be done at the **slide level** to prevent leakage.
1. Group all images by slide ID
2. Split slides into test / train-val at a **1:7 ratio** (stratified by benign/malignant)
3. ~1,000 test images are held out entirely and never used during training
4. Apply augmentation to the remaining ~7,000 training images: rotations (90°, 180°, 270°) + horizontal and vertical flips → **~42,000 images**
5. Apply **5-fold cross-validation** at the slide level on the augmented training set

## Models
Each model trains on one magnification level. The 5th model trains on all magnifications combined,
acting as a generalist and tiebreaker in the ensemble vote.

| Model | Magnification | Architecture     | Notes                                      |
|-------|---------------|------------------|--------------------------------------------|
| M1    | 40X           | ResNet50         | Most spatial context; standard baseline    |
| M2    | 100X          | EfficientNet-B3  | Balanced zoom; efficient CNN               |
| M3    | 200X          | EfficientNet-B4  | Higher cell detail; slightly larger model  |
| M4    | 400X          | Swin Transformer | Finest detail; global attention useful here|
| M5    | All (40–400X) | ResNet50         | Generalist; trained with additional blur and compression augmentation. Tiebreaker in ensemble vote. |

All models use pretrained ImageNet weights and are fine-tuned on their respective split.

## Inference

At inference time, all 5 models receive the input image and output a malignancy probability. These are combined via **soft voting** (simple average of probabilities) to produce a final prediction.

If the input magnification is known, the corresponding specialist model's output can be up-weighted in the average.

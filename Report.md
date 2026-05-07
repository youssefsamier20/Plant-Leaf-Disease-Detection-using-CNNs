# Plant Disease Classification Project Report  
  
---  
  
## 1. Data Preparation & Engineering (Phase 1)  
  
### 1.1 Dataset Verification & Splitting  
  
All dataset splits were verified and validated before training.  
  
* **Training Set:** 70,295 images — 38 classes  
* **Validation Set:** 14,069 images — 38 classes  
* **Test Set:** 3,803 images — 38 classes (path issue fixed)  
  
Folder structures were checked to ensure one folder per class for all splits.  
  
---  
  
### 1.2 Preprocessing Pipeline  
  
A unified preprocessing pipeline was applied across all datasets:  
  
* Resize images to **128 × 128**  
* Convert images to RGB format  
* Normalize pixel values to **[0, 1]**  
* Apply **CLAHE** (Contrast Limited Adaptive Histogram Equalization) to enhance leaf details  
  
This step improved visual clarity and helped the model focus on disease patterns.  
  
---  
  
### 1.3 Data Augmentation  
  
To improve generalization and reduce overfitting, augmentation was applied to the training set:  
  
* Rotation  
* Zoom  
* Horizontal flip  
* Width & height shifts  
* Brightness adjustment  
  
Validation and test sets were kept **augmentation-free** to ensure fair evaluation.  
  
---  
  
### 1.4 Data Generators  
  
Three generators were prepared using `ImageDataGenerator`:  
  
* `train_generator`  
* `val_generator`  
* `test_generator`  
  
Each generator loads images in batches and automatically assigns labels.  
  
**Verified batch shapes:**  
  
* X batch: `(32, 128, 128, 3)`  
* y batch: `(32, 38)`  
  
---  
  
### 1.5 Class Weights Handling  
  
Class weights were computed for all 38 classes to address class imbalance.  
  
This ensures that underrepresented classes contribute fairly during training.  
  
---  
  
### 1.6 Data Visualization  
  
Visualizations were generated to analyze dataset distribution:  
  
* Pie charts showing class balance  
  
These visuals helped confirm that the dataset is reasonably balanced.  
  
---  
  
## 2. Model Architecture  
  
A Convolutional Neural Network (CNN) was designed using a sequential architecture.  
  
### 2.1 Feature Extraction Layers  
  
* Conv2D (32 filters) + MaxPooling  
* Conv2D (64 filters) + MaxPooling  
* Conv2D (128 filters) + MaxPooling  
  
### 2.2 Classification Layers  
  
* Flatten  
* Dense (256 units, ReLU)  
* Dropout (0.5)  
* Dense (38 units, Softmax)  
  
---  
  
## 3. Model Training  
  
### 3.1 Training Configuration  
  
* Optimizer: **Adam**  
* Loss Function: **Categorical Crossentropy**  
* Metrics: **Accuracy**  
* Epochs: up to **40**  
  
### 3.2 Callbacks  
  
* **EarlyStopping:** monitors validation loss with patience = 5  
* **ModelCheckpoint:** saves best model based on validation loss  
  
Class weights were applied during training.  
  
---  
  
## 4. Model Evaluation  
  
### 4.1 Training Curves Analysis  
  
* Training and validation accuracy increased steadily  
* No significant overfitting observed  
  
### 4.2 Test Set Performance  
  
| Metric        | Value |     |
| ------------- | ----- | --- |
| Test Accuracy | ~91%  |     |
| Test Loss     | ~0.32 |     |
  
### 4.3 Confusion Matrix  
  
The confusion matrix shows strong diagonal dominance, indicating correct predictions for most classes.  
  
Minor confusion appears mainly among **Tomato disease classes**, which is expected due to visual similarity.  
  
---  
  
## 5. Conclusion  
  
The project successfully completed data preparation, model design, training, and evaluation phases.  
  
The CNN achieved strong generalization performance with **91% accuracy** on unseen test data.  
  
The system is now ready for deployment and inference on new plant leaf images.  
  
---  
  
## Appendix – Technical Notes  
  
* All code verified and running correctly  
* Generators, preprocessing, augmentation, and class weights fully implemented  
* Test set must contain 38 class folders; a single-folder test set will naturally detect only one class

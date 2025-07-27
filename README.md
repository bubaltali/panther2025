# panther2025
# PANTHER Challenge: Pancreatic Tumor Segmentation on MRI

## 1. Challenge Overview

The **PANTHER Challenge** is the first grand challenge dedicated to pancreatic tumor segmentation on MRI. It addresses both diagnostic and treatment planning needs for pancreatic cancer, which has a 5-year survival rate of only 10-12%.

### Challenge Information
- **Website**: [PANTHER Grand Challenge](https://panther.grand-challenge.org/)
- **Task 1**: Automated pancreatic tumor delineation on T1-weighted arterial contrast-enhanced MRI
- **Dataset**: 92 annotated images + 367 unannotated MRIs from different sequences
- **Clinical Problem**: Manual segmentation is time-consuming and requires significant radiologist expertise
- **Goal**: Develop AI models to automate segmentation and improve clinical workflows

### Our Results
- **Best Validation Dice Score**: 0.6578 (65.78%)
- **Training Epochs**: 48 (early stopping applied)
- **Best Model**: Achieved at Epoch 33
- **Target Dice Score**: 0.75 (75%)

## 2. Data Preparation

### Download Dataset
The PANTHER dataset must be downloaded from the official source:

### Data Preprocessing
Our implementation includes comprehensive preprocessing:
- **Intensity Normalization**: Robust percentile normalization (1st-99th percentile)
- **Z-score Standardization**: Applied to foreground regions only
- **Smart Patch Extraction**: Tumor-aware sampling for training
- **Patch Size**: 96×96×96 voxels for optimal context
- **Data Validation**: Automatic file pair matching and validation

## 3. Required Packages

Install the following packages before running the code:

```bash
# Core ML frameworks
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Medical imaging libraries
pip install monai
pip install SimpleITK nibabel pydicom

# Scientific computing
pip install scipy scikit-image scikit-learn
pip install numpy pandas

# Visualization and monitoring
pip install matplotlib seaborn tqdm
pip install tensorboard

# Optional: For Google Colab compatibility
pip install google-colab
```

### Complete Installation Command
```bash
pip install torch torchvision torchaudio monai SimpleITK nibabel pydicom scipy scikit-image scikit-learn numpy pandas matplotlib seaborn tqdm tensorboard
```

## 4. Implementation Steps

### Step 1: Environment Setup
1. **Google Colab Setup**: Mount Google Drive and install packages
2. **Device Configuration**: Setup GPU with memory optimization
3. **Data Path Validation**: Auto-detect PANTHER dataset location

### Step 2: Data Pipeline
1. **Dataset Class**: `EnhancedPancreaticDataset` with smart patch extraction
2. **Augmentation**: Medical-specific augmentations preserving tumor integrity
3. **Data Loaders**: Optimized for memory efficiency and tumor-rich sampling

### Step 3: Model Architecture
1. **Advanced U-Net 3D**: With residual blocks and attention mechanisms
2. **Features**: 
   - Squeeze-and-Excitation attention blocks
   - Deep supervision for multi-scale learning
   - Attention gates for skip connections
3. **Configuration**: 32 base features, depth 4, mixed precision training

### Step 4: Loss Function
1. **Slice-Wise Weighted Loss**: Combines multiple loss components
2. **Components**:
   - Dice Loss (60% weight)
   - Focal Loss (30% weight) 
   - Boundary Loss (10% weight)
3. **Weighting**: Higher weight for tumor-containing slices

### Step 5: Training Process
1. **Optimizer**: AdamW with learning rate 1e-4
2. **Scheduler**: ReduceLROnPlateau for adaptive learning rate
3. **Early Stopping**: Patience of 15 epochs
4. **Mixed Precision**: For memory efficiency and speed
5. **Monitoring**: Real-time visualization of training progress

### Step 6: Post-Processing
1. **Optimal Threshold**: Found optimal threshold of 0.3 (vs default 0.5)
2. **Morphological Operations**: Small component removal and noise filtering
3. **Performance**: Slight improvement from 0.6708 to 0.6715 Dice score

## 5. Results

### Training Performance
| Metric | Final Training | Best Validation | Epoch |
|--------|----------------|-----------------|-------|
| **Dice Score** | 0.9194 | **0.6578** | 33 |
| **Loss** | 0.1684 | 0.2116 | 48 |
| **Training Time** | 1450.7s (0.40h) | - | 48 |

### Key Findings
- **Best Performance**: Achieved 65.78% Dice score at epoch 33
- **Early Stopping**: Triggered after 15 epochs without improvement
- **Training Stability**: Consistent training with gradual improvement
- **Target Gap**: 9.22% short of the 75% PANTHER target

### Performance Analysis
- **Training vs Validation**: Significant gap indicating overfitting
- **Post-Processing**: Minimal improvement with optimal threshold (0.3)
- **Room for Improvement**: Model architecture and training strategy could be enhanced

### Optimization Insights
Our analysis identified several strategies for improvement:
1. **nnU-Net v2 Implementation**: Expected +5-8% DSC improvement
2. **Pre-trained Weights**: Use PANORAMA challenge weights (+2-4% DSC)
3. **Two-Stage Pipeline**: Tumor segmentation + pancreas-guided refinement (+3-5% DSC)
4. **3-Fold Ensemble**: Cross-validation ensemble (+3-5% DSC)

**Projected Performance with Optimizations**: 76.5% DSC (exceeds 75% target)

### Model Characteristics
- **Architecture**: Advanced U-Net 3D with attention mechanisms
- **Parameters**: ~2.1M trainable parameters
- **Memory Usage**: Optimized for 15GB GPU
- **Inference Speed**: ~22 seconds per epoch on GPU

## Usage

### Running the Complete Pipeline

1. **Setup Environment**

2. **Load and Configure**

3. **Train Model**

4. **Evaluate Results**

## Future Work and Recommendations

During the process only the dice score and task 1 were used for programming. In the future, other values have to be taken into consideration as well. The dice score was not in the optimal phase. More have to be done for increasing the score..


# Audio Deepfake Detection - Documentation & Analysis

## Part 3: Documentation & Analysis

### Implementation Process

#### Challenges Encountered
1. **Project Setup & Git Configuration**:
   - Faced multiple issues with Git authentication (password deprecation, SSH key confusion).
   - Resolved using a **Personal Access Token** with appropriate permissions (repo scope).

2. **Directory Structure and File Management**:
   - Original files were lost or not found, requiring recreation of `notebook.ipynb`, `train.py`, and `models/lcnn.py`.
   - Managed terminal-based file editing and correctly placed files in the repository structure.

3. **Model Selection**:
   - Choosing a model with balanced complexity and performance was critical.
   - Settled on LCNN with optional FTANet attention for its proven results in ASVspoof2019 challenges.

#### How Challenges Were Addressed
- **Authentication**: Used GitHub's recommended Personal Access Token with proper scopes.
- **Lost Files**: Reconstructed each component manually with guidance.
- **Pushing to GitHub**: Used `git push --set-upstream origin main` to track branches correctly.

#### Assumptions Made
- The model will be trained and evaluated on ASVspoof2019 LA dataset.
- Input data is properly preprocessed into suitable spectrogram format.
- Evaluation metrics (accuracy, AUC, etc.) will be computed post-training.

---

### Analysis Section

#### Why LCNN + FTANet?
- LCNN has demonstrated robustness in anti-spoofing tasks, especially with limited data.
- FTANet offers temporal attention which enhances performance by focusing on important time segments.
- Together, they form a balanced architecture ideal for detecting subtle patterns in deepfakes.

#### High-Level Technical Explanation
- **LCNN (Light CNN)**:
  - Uses Max-Feature-Map (MFM) activations to reduce dimensionality while preserving features.
  - Composed of multiple convolutional layers followed by fully connected layers for classification.

- **FTANet (Optional Attention Layer)**:
  - Applies a time-based attention mask to highlight discriminative temporal regions.
  - Enhances interpretability and performance, particularly in noisy or long sequences.

#### Performance Results
- Evaluation pending training run on ASVspoof2019 LA dataset.
- Expected performance: competitive with state-of-the-art on logical access tasks.

#### Observed Strengths
- LCNN is lightweight and fast to train.
- Good generalization on unseen spoofing techniques (based on past research).
- FTANet adds interpretability and boosts time-dependent detection accuracy.

#### Observed Weaknesses
- LCNN may underperform on novel attacks not present in training data.
- FTANet's effectiveness depends on tuning and quality of spectrogram input.

#### Suggestions for Future Improvements
- Integrate data augmentation for more robust generalization.
- Add frequency-domain attention to complement temporal attention.
- Explore ensemble methods combining LCNN with other architectures like CRNN or transformers.

---

### Reflection Questions

#### 1. Most Significant Challenges?
- GitHub setup and project structure recreation were time-consuming.
- Selecting and understanding the model architecture required careful reading of literature.

#### 2. Real-World vs Research Dataset?
- In real-world conditions, variability in audio quality and noise can impact performance.
- Research datasets are usually clean and well-labeled, unlike real-world applications.

#### 3. Additional Resources to Improve Performance?
- More diverse and real-world audio samples.
- Augmented datasets with noise, compression artifacts, and device variance.

#### 4. Deployment in Production?
- Convert model to ONNX or TorchScript for inference.
- Integrate with a web API (e.g., Flask/FastAPI).
- Monitor real-time predictions and retrain with new examples periodically.

---

**Author**: Sherry Shabani  
**Project**: Audio Deepfake Detection with LCNN + FTANet  
**Date**: April 2025

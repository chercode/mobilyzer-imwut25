# MobiLyzer: Fine-grained Mobile Liquid Analyzer

This repository provides the official implementation of the paper:  

**MobiLyzer: Fine-grained Mobile Liquid Analyzer**  
*[Shahrzad Mirzaei]()*<sup>1</sup>, *[Mariam Bebawy]()*<sup>1</sup>, *[Amr Mohamed Sharafeldin]()*<sup>1</sup>, and *[Mohamed Hefeeda]()*<sup>1,2</sup>  

<sup>1</sup> School of Computing Science, Simon Fraser University, BC, Canada

<sup>2</sup>  Qatar Computing Research Institute, Hamad Bin Khalifa University, Doha, Qatar

📄 [Paper – pending]() | 📑 [Supplementary – pending]() | 🎥 [Demo Video – pending]()

![mobilyzer](figures/Picture1.png)  

---

## Overview and Components

MobiLyzer is a novel system for fine-grained analysis of liquids using commodity smartphones.  
It integrates physics-based modeling, machine learning, and mobile deployment to support tasks such as:  

- **Fraud detection** (e.g., adulterated olive oil)  
- **Quality assessment** (e.g., Extra virgin olive oil vs Refined olive oil)  
- **Origin labeling** (e.g., Italian olive oil vs American olive oil)
- **composition labeling** (Fat, Protein, and sugar composition analysis)
- **Medical diagnostics** (e.g., urine analysis) 

![System Overview](figures/Picture1.png)

The repository is organized into four main modules:

1. **Intrinsic Decomposition**  
   Separation of reflectance and illumination components to recover material properties.

2. **Hyperspectral Reconstruction**  
   Truthful reconstruction of spectral signatures from RGB smartphone images.  

3. **Liquid Analysis**  
    classification for fraud detection, composition analysis, and diagnosis.  

4. **Mobile Application**  
   A lightweight Android app for real-time liquid analysis on smartphones. [Link – pending]()

---

## Citation

If you use this code or dataset in your research, please cite:  

```bibtex


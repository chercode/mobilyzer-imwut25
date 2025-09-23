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

Most current methods for liquid analysis and fraud detection rely on expensive tools and controlled lab environments, making them inaccessible to lay users. We present MobiLyzer, a mobile system that enables fine-grained liquid analysis on unmodified commodity smartphones in realistic environments such as homes and grocery stores. MobiLyzer conducts spectral analysis of liquids based on how their chemical components reflect different wavelengths. Conducting spectral analysis of liquids on smartphones, however, is challenging due to the limited sensing capabilities of smartphones and the heterogeneity in their camera designs. This is further complicated by the uncontrolled nature of ambient illumination and the diversity in liquid containers. The ambient illumination, for example, introduces distortions in measured spectra, and liquid containers cause specular reflections that degrade accuracy. To address these challenges, MobiLyzer utilizes RGB images captured by regular smartphone cameras. It then introduces intrinsic decomposition ideas to mitigate the effects of illumination and interference from liquid containers. It further leverages the near-infrared (NIR) sensors on smartphones to collect complementary signals in the NIR spectral range, partially mitigating the limited sensing capabilities of smartphones. It finally presents a new machine-learning model that reconstructs the entire spectrum in the visible and NIR ranges using the captured RGB and NIR images, which enables fine-grained spectral analysis of liquids on smartphones without the need for expensive equipment. Unlike prior models, the presented spectral reconstruction model preserves the original RGB colors during reconstruction, which is critical for liquid analysis since many liquids differ only in subtle spectral cues. We demonstrate the accuracy and robustness of MobiLyzer through extensive experiments with multiple liquids, four different smartphones, and seven illumination sources. MobiLyzer supports tasks such as:  

- **Fraud detection** (e.g., adulterated olive oil)  
- **Quality assessment** (e.g., Extra virgin olive oil vs Refined olive oil)  
- **Origin labeling** (e.g., Italian olive oil vs American olive oil)
- **composition labeling** (Fat, Protein, and sugar composition analysis)
- **Medical diagnostics** (e.g., urine analysis) 

![System Overview](figures/overview.png)

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


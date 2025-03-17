# Hybrid Credit Card Fraud Detection  
### Neural Network + Rule-Based System  

**Published in:** [Journal of Information and Optimization Sciences (JIOS), Taru Publications](http://dx.doi.org/10.47974/JIOS-1932)  
**DOI:** [10.47974/JIOS-1932](http://dx.doi.org/10.47974/JIOS-1932)  

## Overview  
This project implements a **hybrid fraud detection system** that integrates:  
- **Deep Neural Network** for learning fraud patterns  
- **Rule-Based System** for improved interpretability & accuracy  

The system **reduces false positives** while ensuring robust fraud detection, achieving **98.41% accuracy**.  

## Features  
- **Neural Network Training**: Uses **log transformation, focal loss, and Nadam optimizer**  
- **Rule-Based System**: Applies expert-defined fraud detection rules  
- **Hybrid Approach**: Improves model performance & interpretability  
- **Optimized for Real-World Fraud Detection**  

## Technologies Used  
- **Python 3.12.9**  
- **TensorFlow/Keras**  
- **Scikit-Learn**  
- **Pandas & NumPy**  
- **Matplotlib & Seaborn**  


## Installation Guide  

1. Clone the Repository
   ```sh
    git clone https://github.com/Suryatejaeasari/hybrid-credit-card-fraud-detection.git
    cd hybrid-credit-card-fraud-detection
    ```
2. Create a Virtual Environment (Optional but Recommended)
   ```sh
   python -m venv venv
   source venv/bin/activate  # For macOS/Linux
   venv\Scripts\activate     # For Windows
   ```
3. Install Dependencies
   ```sh
    pip install -r requirements.txt
   ```
4. Run the Project
   ```sh
   python training.py
   python hybrid.py
   ```


## License  
This project is licensed under the [MIT License](LICENSE).  


---

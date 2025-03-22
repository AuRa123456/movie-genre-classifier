
# 🎬 Movie Genre Classification using XGBoost 🎯

This project uses **TF-IDF vectorization** and **XGBoost** to classify movies into genres based on their descriptions. It efficiently handles **large datasets** and **imbalanced classes** while maintaining high accuracy.

---

## 🚀 Features
✅ **TF-IDF for Text Representation**  
✅ **XGBoost (CPU-Optimized) for Multi-Class Classification**  
✅ **Handles Imbalanced Classes with Class Weighting**  
✅ **Supports Large Datasets (54K+ Movies)**  
✅ **Exports Predictions to CSV**

---

## 📂 Dataset Format
- **Training Data (`train_data.txt`)**
  ```
  ID ::: TITLE ::: GENRE ::: DESCRIPTION
  ID ::: TITLE ::: GENRE ::: DESCRIPTION
  ```
- **Test Data (`test_data.txt`)**
  ```
  ID ::: TITLE ::: DESCRIPTION
  ID ::: TITLE ::: DESCRIPTION
  ```

---

## 🛠️ Installation
### **1️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```
If using Jupyter Notebook, run:
```sh
!pip install -r requirements.txt
```

### **2️⃣ Required Libraries**
Manually install if needed:
```sh
pip install xgboost scikit-learn pandas numpy
```

---

## 🏗️ Implementation
### **1️⃣ Load Data**
- Reads **train_data.txt** and **test_data.txt**.
- Handles **missing values** automatically.

### **2️⃣ Convert Text to TF-IDF Vectors**
- Uses **n-grams (1,2)** and **7,000 features** for better context.

### **3️⃣ Train XGBoost Model**
- Optimized with `tree_method="hist"` (fastest for CPU).
- Uses **100 trees** for balance between speed and accuracy.

### **4️⃣ Predict and Evaluate**
- Generates **accuracy, precision, recall, and F1-score**.
- Saves predictions to **`predicted_test_data.csv`**.

---

## ⏳ Time Complexity
| **Step**                | **Complexity**  | **Time (for 54K samples)** |
|------------------------|--------------|------------------|
| **TF-IDF Vectorization** | `O(N * M)`  | ~30-60 sec  |
| **XGBoost Training**    | `O(N log N)` | ~3-5 min  |
| **Prediction**          | `O(N log N)` | ~30-60 sec  |
| **Total Time**          | -            | ~5-10 min  |

🔹 `N = Number of samples`, `M = Number of words per movie description`.

**P.S. The time mentioned can differ based on the system.**

---

## 🎯 Usage
### **Run the Script**
```sh
python classifier.py
```

### **Output File:**
- `predicted_test_data.csv` contains (example):
  ```
  ID,TITLE,DESCRIPTION,PREDICTED_GENRE
  123,Movie Name,This is a sci-fi adventure,Sci-Fi
  ```

---

## 📈 Future Improvements
- **Hyperparameter tuning** with `GridSearchCV`.
- **Deep Learning approach** using Transformers (BERT).
- **Feature selection** for reducing computation time.

---

## 🤝 Contributing
Feel free to **open issues** and **submit pull requests** to improve this project!

---

## 📝 License
This project is **open-source** under the **MIT License**.

---

🎬 **Movie Genre Classification using XGBoost - Fast, Accurate, and Scalable!** 🚀

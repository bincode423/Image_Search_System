---

## Image Search System

Welcome!
This project aims to **build and deploy a basic AI-powered image retrieval system**. It provides fundamental tools to search for similar images within a large dataset. The system uses a deep learning model to extract image features and returns search results based on them.

---

## 🌟 Key Features

* **Basic Image Search Implementation:** Uses the **ResNet-18 model** to learn image features and implements a simple image retrieval function.
* **Streamlit-based Web Interface:** Through `app.py`, the project provides a **lightweight web application** that can be easily accessed and used in a browser.
* **Custom Data Training & Search:** Includes steps to train the model on your own collected images and use it to build a custom search system.

---

### 🚀 Getting Started

Follow these steps to run the project locally.

#### 📝 Prerequisites

* Python 3.8 or higher
* `pip` (Python package manager)
* Git

#### 📦 Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/bindobi/imageretrievalsystem.git
   cd imageretrievalsystem
   ```

2. **Install dependencies:**

   Install all required packages listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

#### 🏃‍♂️ Running the System

1. **Prepare the dataset:**
   Use the `data_download_python` script to download the necessary image dataset and store it in the `my_dataset` directory.
   Alternatively, you can prepare your own dataset following the `my_dataset` folder structure.

   ```bash
   python data_download_python/download_script.py  # Example (check actual script name)
   ```

2. **Train the model (optional):**
   If you already have a pre-trained `trained_model.pth` file, you can skip this step.
   Open `ai_train.ipynb` in Jupyter Notebook to train the model from scratch or fine-tune an existing one.

   ```bash
   jupyter notebook ai_train.ipynb
   ```

   After training, the `trained_model.pth` file will be created or updated.

3. **Run the application:**

   ```bash
   streamlit run app.py
   ```

   This will automatically open your web browser with the image search system interface.

---

### 📁 Project Structure

```
imageretrievalsystem/
├── data_download_python/   # Scripts for downloading datasets
├── my_dataset/             # Directory for storing image datasets
├── ai_train.ipynb          # Jupyter Notebook for model training and evaluation
├── app.py                  # Streamlit web application code
├── public_function.py      # Reusable utility functions
├── trained_model.pth       # Trained deep learning model weights
├── README.md               # Project documentation (this file)
└── requirements.txt        # List of required Python packages
```

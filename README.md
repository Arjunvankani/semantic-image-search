# ❄️ Snowflake: Semantic Image Search with CLIP and Streamlit

Welcome to **Snowflake**, a beautiful and intuitive Streamlit app for performing **semantic image search** using OpenAI's CLIP model and FAISS vector similarity search. Just type a natural-language prompt like "sunset", "road", or "forest", and get the most relevant images from your dataset instantly!

---

## 🚀 Features

- 🔍 **Semantic Text-to-Image Search** using CLIP embeddings.
- ⚡ **Efficient image similarity search** using FAISS.
- 🖼️ Displays top-matching images based on your search query.
- 💡 Built-in suggested keywords for quick exploration.
- 🎨 Snow-themed, elegant UI with hover effects.
- 🧑‍💻 Created by [Arjun Vankani](https://github.com/Arjunvankani)

---


## 📂 Project Structure


- project-root/
- ├── sem1.py # Main Streamlit app
- ├── images/ # Your searchable image folder
- ├── README.md # Project documentation
- └── requirements.txt # Python dependencies

---

## ⚙️ Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/semantic-image-search.git
cd semantic-image-search

```


### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add Your Images
```bash
Place all the .jpg, .png, .jpeg, etc. files inside the images/ folder. The app will automatically index them.
```

### 4. Run the Streamlit App
```bash
streamlit run sem1.py
```

---
✨ Demo Preview
```bash
https://semantic-image-search-tjr6zniss5thkxw2sr9kgx.streamlit.app/

```
---

🧠 How It Works
Uses CLIP (ViT-B/32) from OpenAI to encode both image and text into a shared embedding space.

FAISS is used to create a similarity index of all image embeddings.

Your search query is encoded as a text vector, and matched against image vectors using cosine similarity.

The app returns the top-N most relevant images based on similarity.
---

📦 Requirements
Python 3.7+

torch

streamlit

faiss-cpu

numpy

Pillow (PIL)

tqdm

OpenAI CLIP

All required packages are listed in the requirements.txt.

---

💡 Example Search Prompts
Try typing the following into the search bar:

traffic light

sunset road

people walking

snowy mountain

helmet

forest

---

🔗 Author & Links
Created by Arjun Vankani

🌐 Portfolio(https://arjunvankani.github.io/arjun/)

🐙 GitHub(https://github.com/Arjunvankani)

🔗 LinkedIn(https://www.linkedin.com/in/arjun-vankani/)



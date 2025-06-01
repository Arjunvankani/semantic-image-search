
# Run with: streamlit run app.py --server.fileWatcherType=none
import streamlit as st
import os
import torch
import clip
import numpy as np
import faiss
import glob
from PIL import Image
import streamlit as st
import os
import json

def load_counter():
    if not os.path.exists("counter.json"):
        with open("counter.json", "w") as f:
            json.dump({"views": 0}, f)

    with open("counter.json", "r") as f:
        data = json.load(f)
    return data

def increment_counter():
    data = load_counter()
    data["views"] += 1
    with open("counter.json", "w") as f:
        json.dump(data, f)
    return data["views"]

views = increment_counter()
st.sidebar.markdown(f"👁️ Total Page Views: **{views}**")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class SemanticImageSearch:
    def __init__(self, image_dir="images", device=None):
        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        self.image_dir = image_dir
        os.makedirs(self.image_dir, exist_ok=True)

        self.index = None
        self.image_paths = []
        self.features = None
        self.build_index()

    def build_index(self):
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp']
        self.image_paths = []
        for ext in image_extensions:
            self.image_paths.extend(glob.glob(os.path.join(self.image_dir, '**', ext), recursive=True))

        if not self.image_paths:
            st.warning(f"No images found in {self.image_dir}")
            return

        features = []
        with torch.no_grad():
            for img_path in self.image_paths:
                try:
                    image = self.preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(self.device)
                    feature = self.model.encode_image(image)
                    features.append(feature.cpu().numpy())
                except Exception as e:
                    st.error(f"Error processing {img_path}: {e}")

        if not features:
            st.error("No valid images processed")
            return

        self.features = np.vstack(features)
        self.features = self.features / np.linalg.norm(self.features, axis=1, keepdims=True)

        self.index = faiss.IndexFlatIP(self.features.shape[1])
        self.index.add(self.features.astype('float32'))

    def search(self, query, top_k=10):
        if self.index is None or not self.image_paths:
            st.error("Index not built yet.")
            return []

        with torch.no_grad():
            text_features = self.model.encode_text(clip.tokenize([query]).to(self.device))
            text_features = text_features.cpu().numpy()

        text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
        scores, indices = self.index.search(text_features.astype('float32'), min(top_k, len(self.image_paths)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.image_paths):
                results.append({
                    "path": self.image_paths[idx],
                    "similarity": float(score)
                })
        return results

def app():
    
    st.set_page_config(page_title="❄️ Snowflake Image Search", page_icon="❄️", layout="wide")
    st.snow()
    st.markdown("""
        <style>
        body {
            background: linear-gradient(to right, #e0f7fa, #ffffff);
        }
        .main {
            background: #f8fbfc;
            border-radius: 15px;
            padding: 2rem;
            margin-top: 20px;
        }
        h1 {
            font-family: 'Segoe UI', sans-serif;
            font-weight: bold;
            color: #0277bd;
            text-align: center;
        }
        .image-card {
            background: white;
            border-radius: 15px;
            padding: 1rem;
            box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.2s ease-in-out;
        }
        .image-card:hover {
            transform: scale(1.03);
        }
        footer {
            margin-top: 30px;
            padding: 10px;
            text-align: center;
            color: #0288d1;
        }
        footer a {
            margin: 0 10px;
            color: #01579b;
            font-weight: bold;
            text-decoration: none;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1>❄️ Snowflake Semantic Image Search</h1>', unsafe_allow_html=True)

    @st.cache_resource
    def get_search_engine():
        return SemanticImageSearch()

    search_engine = get_search_engine()

    if "search_query" not in st.session_state:
        st.session_state.search_query = ""
                
    query = st.text_input("🔍 Enter your image query", value=st.session_state.search_query, key="search_input", placeholder="Type e.g., road, traffic, forest")

    if st.button("Search"):
        st.session_state.search_query = query
            
        
                
        results = search_engine.search(query)
        results = results[:6]

        if results:
            st.markdown(f"#### Top results for: `{query}`")
            cols = st.columns(3)
            for i, result in enumerate(results):
                with cols[i % 3]:
                    #st.markdown('<div class="image-card">', unsafe_allow_html=True)
                    st.image(Image.open(result["path"]).resize((380, 280)))
                    st.caption(f"{os.path.basename(result['path'])} | Score: {result['similarity']:.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("No similar images found. Try another keyword.")


    st.markdown("""
        <footer>
            Created by <strong>Arjun Vankani</strong> |
            <a href="https://github.com/Arjunvankani" target="_blank">GitHub</a> |
            <a href="https://arjunvankani.github.io/arjun/" target="_blank">Portfolio</a> |
            <a href="https://www.linkedin.com/in/arjun-vankani/" target="_blank">LinkedIn</a>
        </footer>
    """, unsafe_allow_html=True)

if __name__ == "__main__":

    app()

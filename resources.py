import os
import sys
from csv import DictReader
from io import BytesIO, TextIOWrapper
from pathlib import Path
from time import sleep
from zipfile import ZipFile

import cv2
import joblib
import numpy as np
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from streamlit_drawable_canvas import st_canvas

os.chdir(os.path.dirname(__file__))

mnist = fetch_openml("mnist_784", version=1, cache=True, as_frame=False)
X = mnist["data"]
y = mnist["target"].astype(np.uint8)

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=10000, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=10000, random_state=42)

bg_img_url = "https://i.imgur.com/ctUckTP.png"
css = f"""
        <style>
        .stApp {{
            background-image: url("{bg_img_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        body {{
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
        }}
        </style>
        """
radio = """
        <style>
            div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {font-size: 19px;}
        </style>
        """
no_label_error = """
                <div style="
                background-color: rgba(200, 0, 0, 0.4);
                padding: 10px;
                border-radius: 10px;
                margin-bottom: 20px;
                display: flex;
                justify-content: center;
                align-items: center;
                text-align: center;">
                <p style="color: white; margin: 0;">
                    Kunde inte hitta en etiket för bilden med filnamnet [{}].
                </p>
                </div>"""


def display_confusion_matrix(true_labels, all_predictions, model_name=None):
    cm = confusion_matrix(true_labels, all_predictions, labels=np.arange(10))
    fig, _ = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis")
    plt.title(f"Confusion Matrix {'för ' + model_name or ''}")
    plt.ylabel("Sanna värden")
    plt.xlabel("Predikterade värden")
    plt.yticks(rotation=0)
    st.pyplot(fig)

import streamlit as st
import numpy as np
import joblib as jb

model = jb.load("notebooks/final_model.pkl")
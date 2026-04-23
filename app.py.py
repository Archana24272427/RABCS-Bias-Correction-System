import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.title("🧠 Unbiased AI Decision System")

uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview", df.head())

    target = st.selectbox("Select Target Column", df.columns)
    sensitive = st.selectbox("Select Sensitive Feature", df.columns)

    if st.button("Run Bias Check"):
        X = df.drop(columns=[target])
        y = df[target]

        X_num = X.select_dtypes(include='number')

        model = LogisticRegression()
        model.fit(X_num, y)

        preds = model.predict(X_num)
        df["Prediction"] = preds

        bias = df.groupby(sensitive)["Prediction"].mean()

        st.subheader("📊 Bias Analysis")
        st.write(bias)

        score = 1 - abs(bias.max() - bias.min())

        st.subheader("⚖️ Fairness Score")
        st.write(f"{round(score*100,2)}%")

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import time

def model():
    st.write("> Model Type:")
    value = 42
    st.markdown(f'<span style="color:blue;">&nbsp;&nbsp;&nbsp;{model_type}</span>', unsafe_allow_html=True)
    st.write("> Will predict:")
    st.markdown(f'<span style="color:blue;">&nbsp;&nbsp;&nbsp;{target}</span>', unsafe_allow_html=True)
    st.write("> Fill Null Value With:")
    st.markdown(f'<span style="color:blue;">&nbsp;&nbsp;&nbsp;{fill_value}</span>', unsafe_allow_html=True)
    st.write("> Scaling Method:")
    st.markdown(f'<span style="color:blue;">&nbsp;&nbsp;&nbsp;{scale_type}</span>', unsafe_allow_html=True)

    

    try:
       
        X = df[feature]
        y = df[target]

        if fill_value == "Mean":
            X = X.fillna(X.mean())
        elif fill_value == "Median":
            X = X.fillna(X.median())
        elif fill_value == "Drop":
            X = X.dropna()
            y = y[X.index]

        if scale_type == "Standard scaler":
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        elif scale_type == "Minmax scaler":
            scaler = MinMaxScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_type == "Random Forest":
            model = RandomForestClassifier()
        elif model_type == "Logistic Regression":
            model = LogisticRegression()
        elif model_type == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_type == "KNN":
            model = KNeighborsClassifier()
        elif model_type == "SVM":
            model = SVC()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)*100
        if score < 60:
            st.warning(f"Model Accuracy: {score:.2f}% - The model accuracy is below 60%. Consider trying different parameters or data.")
        elif 60 <= score < 80:
            st.info(f"Model Accuracy: {score:.2f}% - The model accuracy is decent but could be improved.")
        else:
            st.success("Model Accuracy: {score:.2f}% - the accuracy is excillent")    



        model_bytes = pickle.dumps(model)
        download = st.download_button(
            label="Download Model",
            data=model_bytes,
            file_name="trained_model.pkl",
            mime="application/octet-stream"
                  ) 
        st.empty()
        st.empty()
        with st.form(key="my_form"):
            name = st.text_input("Your Name")
            submit = st.form_submit_button("Submit")
 
    except Exception:
        st.error("An error occurred: ")




st.set_page_config(
    page_title="No code ML",
    page_icon="my_icon.png"   # ← your image file
)

st.title("NO CODE ML MODEL BUILDER")
st.write("by Sakib Hossain Tahmid")
st.warning("*Errors may occur(ignore them). More features will be added soon*")
home, mod = st.tabs(["Home", "Model Details"])

with home:
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df)
    with col2:
        st.write("Instructions:")
        feature = st.multiselect("Select Training Columns(must be numerical):", df.columns.tolist() if uploaded_file is not None else ["no option"])
        target = st.selectbox("Select Target Column(must be 0 or 1):", df.columns.tolist() if uploaded_file is not None else ["none"])
        fill_value = st.selectbox("Fill Null Value With:", ["Mean","Median","Drop"] if uploaded_file is not None else ["no value"])
        scale_type = st.selectbox("Select Scaling Method:", ["Standard scaler","Minmax scaler"] if uploaded_file is not None else ["no scaler"])
        model_type = st.selectbox("Select Model:", ["Random Forest","Logistic Regression","Decision Tree","KNN","SVM"] if uploaded_file is not None else ["no model"])
        start = st.button("Train Model")
        if start:
            if uploaded_file is not None and feature and target and fill_value and scale_type and model_type:
                with st.spinner("Training the model..."):
                    time.sleep(3)
                st.success("Model trained successfully!")
                st.write("* Go to the 'Model Details' tab to see the model details.")

            else:
                st.error("Please upload a file and select all options.")
with mod:
    if start is True and uploaded_file is not None and feature and target and fill_value and scale_type and model_type:
        st.subheader("Detailes of the trained model:")
        model()
    else:
        st.write("> No model trained yet. Please go to the 'Home' tab to upload data and train a model.")   

st.markdown(
            f'<div style="text-align: center; color: grey;">&copy; 2025 Sakib Hossain Tahmid. All Rights Reserved.</div>',
            unsafe_allow_html=True
           ) 








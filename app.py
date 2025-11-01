import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import time

def model():
    st.write("> Model Type:")
    value = 42
    st.markdown(f'<span style="color:orange;">&nbsp;&nbsp;&nbsp;{model_type}</span>', unsafe_allow_html=True)
    st.write("> Will predict:")
    st.markdown(f'<span style="color:orange;">&nbsp;&nbsp;&nbsp;{target}</span>', unsafe_allow_html=True)
    st.write("> Fill Null Value With:")
    st.markdown(f'<span style="color:orange;">&nbsp;&nbsp;&nbsp;{fill_value}</span>', unsafe_allow_html=True)
    st.write("> Scaling Method:")
    st.markdown(f'<span style="color:orange;">&nbsp;&nbsp;&nbsp;{scale_type}</span>', unsafe_allow_html=True)

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
    elif model_type == "Linear Regression":
        model = LinearRegression()
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

    if model_type == "Linear Regression":
        score = r2_score(y_test, y_pred)*100
        if score < 0:
            st.warning(f"Model R² Score: {score:.2f}% - The model is performing poorly. Consider trying different parameters or data.")
        elif 0 <= score < 50:
            st.info(f"Model R² Score: {score:.2f}% - The model performance is decent but could be improved.")
        else:
            st.success(f"Model R² Score: {score:.2f}% - The model is performing excellently.")
    else:        
        score = accuracy_score(y_test, y_pred)*100
        if score < 60:
            st.warning(f"Model Accuracy: {score:.2f}% - The model accuracy is below 60%. Consider trying different parameters or data.")
        elif 60 <= score < 80:
            st.info(f"Model Accuracy: {score:.2f}% - The model accuracy is decent but could be improved.")
        elif 80 <= score <= 100 :
            st.success(f"Model Accuracy: {score:.2f}% - the accuracy is excillent")    
      
    with st.expander("Download your Model"):
        st.write("please ensure:")
        st.markdown('* You must have <span style="color:pink;">scikit-learn 1.7.2</span>',unsafe_allow_html=True)
        st.markdown(f'* Input features order must be: <span style="color:orange;">{feature}</span>',unsafe_allow_html=True)
        model_bytes = pickle.dumps(model)
        download = st.download_button(
            label="Download Model",
            data=model_bytes,
            file_name="trained_model.pkl",
            mime="application/octet-stream"
            ) 
        
 
    



def graphs():
    x_axis = st.selectbox("Select X-axis:", df.columns.tolist())
    y_axis = st.selectbox("Select Y-axis:", df.columns.tolist())
    plot_type = st.selectbox("Select Plot Type:", ["Line","Bar","Scatter"])
    plot = st.button("Deploy")
    fig, ax = plt.subplots(facecolor="none")  # whole figure bg
    ax.set_facecolor("none")
    
    if plot_type == "Line" and plot:
        ax.plot(df[x_axis], df[y_axis], color="orange")

    elif plot_type == "Bar" and plot:

        ax.bar(df[x_axis], df[y_axis], color="orange")

    elif plot_type == "Scatter" and plot:
    
        ax.scatter(df[x_axis], df[y_axis], color="orange")
    elif not plot:
        st.info("Select options and click 'Deploy' to generate the plot.")
    else:
        st.error("An error occurred while generating the plot.")
    ax.set_xlabel(x_axis, color="red")
    ax.set_ylabel(y_axis, color="red")
    ax.set_title(f"{plot_type} Plot of {y_axis} vs {x_axis}", color="white")

    ax.tick_params(colors="white")

    st.pyplot(fig)



st.set_page_config(
    page_title="No code ML",
    page_icon="my_icon.png"
)



page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background:  linear-gradient(to bottom,#000000,#2F2F2F,#808080,#D3D3D3); 
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stToolbar"] {
    right: 2rem;
}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)



st.title("NO CODE ML MODEL BUILDER")
st.write("by Sakib Hossain Tahmid")
st.warning("⚠*Early development . More features will be added soon . Ignore errors*")
home, mod, gph = st.tabs(["Home", "Model Details"," Graphs"])
confirm = False
with home:
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df)
            except Exception:
                df = pd.DataFrame()
                st.error("This file is not loading. Try other files.")    
    with col2:
        st.write("Instructions:")
        feature = st.multiselect("Select Training Columns(must be numerical):", df.columns.tolist() if uploaded_file is not None else ["no option"])
        target = st.selectbox("Select Target Column(must be 0 or 1):", df.columns.tolist() if uploaded_file is not None else ["none"])
        scale_type = st.selectbox("Select Scaling Method:", ["Standard scaler","Minmax scaler"] if uploaded_file is not None else ["no scaler"])
        model_type = st.selectbox("Select Model:", ["Random Forest","Linear Regression","Logistic Regression","Decision Tree","KNN","SVM"] if uploaded_file is not None else ["no model"])
        
        if uploaded_file is not None and df.isnull().sum().sum() > 0:
            fill_value = st.selectbox("Fill Null Value With:", ["Mean","Median","Drop"] if uploaded_file is not None else ["no value"])
            miss_val = df.isnull().sum().sum()
            st.markdown(f'Total <span style="color:red;">&nbsp;{miss_val}&nbsp;</span>missing values in dataset', unsafe_allow_html=True)
        else:
            if uploaded_file is not None:
                st.success("No missing values detected in the dataset.")
            else:
                st.info("After uploading a file, missing values will be checked autometically.")    
            fill_value = True    
      
        if model_type == "Linear Regression":
            st.info("Note: Linear Regression is for regression tasks. Ensure your target variable is continuous.")
            check = st.checkbox("I aggree that my target variable is continuous(For Valid Accuracy)", key="linreg_check")
            if check:
                start = st.button("Train Model")
            else:
                st.markdown('<span style="color:red;">Please agree to the condition to proceed.</span>', unsafe_allow_html=True)
                
                start = False
        else:
            start = st.button("Train Model")
        if start:
            if uploaded_file is not None and feature and target and fill_value and scale_type and model_type:
                with st.spinner("Training the model..."):
                    time.sleep(3)
    
                st.markdown(
                """
                <div style='background:#2F2F2F; padding:10px; border-radius:10px; text-align: center'>
                Model trained!
                </div>
                """,
                unsafe_allow_html=True
                )
                st.markdown("""<ul>
    <li style="color:#800080;">Go to the "Model Details" tab to see the model details.</li></ul>""", unsafe_allow_html=True)
            else:    
                st.error("Please ensure all options are selected and a file is uploaded before training the model.")
with mod:
    try:
        if start is True and uploaded_file is not None and feature and target and fill_value and scale_type and model_type:
            st.subheader("Detailes of the trained model:")
            model()
            with st.form(key="my_form"):
                name = st.text_input("Your Name")
                submit = st.form_submit_button("Submit")
        else:
            st.write(">  No model trained yet. Please go to the 'Home' tab to upload data and train a model.")   
    except Exception :
        st.error("An error occurred. Please ensure you have numeriacal features and a valid target column.")


with gph:
    if uploaded_file is not None:
        try:
            st.subheader("Data Visualization")
            graphs()
        except Exception as e:
            st.error(f"An error occurred: {e}")    
    else:
        st.write("> Please upload a CSV file in the 'Home' tab to enable data visualization.") 



st.markdown(
    """
    <style>
        .social-icons {
            text-align: center;
            margin-top: 60px;
        }

        .social-icons a {
            text-decoration: none !important;
            margin: 0 20px;
            font-size: 28px;
            display: inline-block;
            color: inherit !important; /* force child i to use its color */
        }

        

        /* Hover glitch animation */
        .social-icons a:hover {
            animation: glitch 0.3s infinite;
        }

        
        /* Contact us heading */
        .contact-heading {
            text-align: center;
            font-size: 25px;
            font-weight: bold;
            margin-bottom: 15px;
        }
        @keyframes glitch {
            0% { transform: translate(0px, 0px); text-shadow: 2px 2px #0ff, -2px -2px #f0f; }
            20% { transform: translate(-2px, 1px); text-shadow: -2px 2px #0ff, 2px -2px #f0f; }
            40% { transform: translate(2px, -1px); text-shadow: 2px -2px #0ff, -2px 2px #f0f; }
            60% { transform: translate(-1px, 2px); text-shadow: -2px 2px #0ff, 2px -2px #f0f; }
            80% { transform: translate(1px, -2px); text-shadow: 2px -2px #0ff, -2px 2px #f0f; }
            100% { transform: translate(0px, 0px); text-shadow: 2px 2px #0ff, -2px -2px #f0f; }
        }
    </style>
    <div class="social-icons">
    <div class="contact-heading">Contact Us:</div>
        <a class='fb' href='https://www.facebook.com/sakibhossain.tahmid' target='_blank'>
            <i class='fab fa-facebook'></i> 
        </a> 
        <a class='insta' href='https://www.instagram.com/_sakib_000001' target='_blank'>
            <i class='fab fa-instagram'></i> 
        </a> 
        <a class='github' href='https://github.com/sakib-12345' target='_blank'>
            <i class='fab fa-github'></i> 
        </a> 
        <a class='email' href='mailto:sakibhossaintahmid@gmail.com'>
            <i class='fas fa-envelope'></i> 
        </a>
    </div>
    
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    """,
    unsafe_allow_html=True
)

st.markdown(
            f'<div style="text-align: center; color: grey;">&copy; 2025 Sakib Hossain Tahmid. All Rights Reserved.</div>',
            unsafe_allow_html=True
           ) 


















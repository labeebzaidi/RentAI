# ==========================================================
# RentAI - app.py (PART 1)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# Page Config
# ----------------------------

st.set_page_config(
    page_title="RentAI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------
# Load Model
# ----------------------------

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    columns = joblib.load("encoder_columns.pkl")
    return model, columns

model, encoder_columns = load_model()

# ----------------------------
# CSS
# ----------------------------
st.markdown("""
<style>

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

/* Background */

.stApp{

background:linear-gradient(135deg,#050816,#0f172a,#111827);

color:white;

}

/* Hero */

.hero{

background:linear-gradient(135deg,#2563eb,#7c3aed);

padding:45px;

border-radius:25px;

text-align:center;

color:white;

box-shadow:0px 10px 35px rgba(0,0,0,.5);

margin-bottom:25px;

}

.hero h1{

font-size:65px;

font-weight:800;

margin-bottom:5px;

}

.hero h3{

font-size:24px;

font-weight:300;

}

/* Cards */

.card{

background:#1e293b;

padding:22px;

border-radius:18px;

box-shadow:0px 8px 20px rgba(0,0,0,.35);

border:1px solid #334155;

transition:0.3s;

}

.card:hover{

transform:translateY(-5px);

border:1px solid #38bdf8;

}

/* Metrics */

.metric{

font-size:32px;

font-weight:bold;

color:#38bdf8;

}

.subtitle{

color:#cbd5e1;

font-size:18px;

}

/* Result */

.result{

background:linear-gradient(135deg,#059669,#10b981);

padding:30px;

border-radius:20px;

text-align:center;

box-shadow:0px 10px 25px rgba(0,0,0,.4);

color:white;

}

/* Buttons */

div.stButton>button{

background:linear-gradient(90deg,#2563eb,#7c3aed);

color:white;

height:60px;

font-size:22px;

font-weight:bold;

border-radius:15px;

border:none;

box-shadow:0px 5px 20px rgba(37,99,235,.4);

transition:0.3s;

}

div.stButton>button:hover{

transform:scale(1.03);

background:linear-gradient(90deg,#3b82f6,#8b5cf6);

}

/* Inputs */

div[data-baseweb="select"]{

color:white;

}

.stNumberInput input{

background:#1e293b !important;

color:white !important;

border-radius:10px;

}

.stSelectbox div{

background:#1e293b;

color:white;

}

/* Divider */

hr{

border:1px solid #334155;

}

</style>
""",unsafe_allow_html=True) 

# ----------------------------
# Hero
# ----------------------------

st.markdown("""

<div class="hero">

<h1>🏠 RentAI</h1>

<h3>
AI Powered House Rent Prediction
</h3>

<p style="font-size:18px;">
Predict rental prices instantly using Machine Learning 🚀
</p>

</div>

""",unsafe_allow_html=True)

# ----------------------------
# Dashboard
# ----------------------------

c1,c2,c3,c4=st.columns(4)

with c1:

    st.markdown("""

    <div class="card">

    <div class="metric">

    14,000+

    </div>

    <div class="subtitle">

    Houses

    </div>

    </div>

    """,unsafe_allow_html=True)

with c2:

    st.markdown("""

    <div class="card">

    <div class="metric">

    Random Forest

    </div>

    <div class="subtitle">

    Model

    </div>

    </div>

    """,unsafe_allow_html=True)

with c3:

    st.markdown("""

    <div class="card">

    <div class="metric">

    87.75%

    </div>

    <div class="subtitle">

    Accuracy

    </div>

    </div>

    """,unsafe_allow_html=True)

with c4:

    st.markdown("""

    <div class="card">

    <div class="metric">

    0.02 sec

    </div>

    <div class="subtitle">

    Prediction Time

    </div>

    </div>

    """,unsafe_allow_html=True)

st.write("")
st.write("")

st.subheader("🏡 Property Details")

left,right=st.columns(2)

with left:

    bhk=st.number_input(

        "🛏️ BHK",

        1,

        10,

        2

    )

    area=st.number_input(

        "📐 Area (sqft)",

        100,

        20000,

        1200

    )

    bathroom=st.number_input(

        "🛁 Bathrooms",

        1,

        10,

        2

    )

with right:

    property_type=st.selectbox(

        "🏢 Property Type",

        [

            "Apartment",

            "Independent Floor",

            "Independent House",

            "Villa",

            "Penthouse",

            "Studio Apartment"

        ]

    )

    seller_type=st.selectbox(

        "👤 Seller Type",

        [

            "Agent",

            "Owner",

            "Verified Owner",

            "Builder"

        ]

    )

    status=st.selectbox(

        "🛋️ Status",

        [

            "Furnished",

            "Semi-Furnished",

            "Unfurnished"

        ]

    )



# ==========================================================
# RentAI - app.py (PART 2)
# ==========================================================

# ----------------------------
# Load Dataset (for locations)
# ----------------------------

# ----------------------------
# Load Dataset
# ----------------------------

df = pd.read_csv("Makaan_data_700pages.csv")

# ----------------------------
# Location Dropdown
# ----------------------------

if "Location" in df.columns:

    locations = sorted(
        df["Location"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

else:

    locations = ["Unknown"]

location = st.selectbox(

    "📍 Location",

    locations

)

st.write("")
st.write("")

predict = st.button(

    "🔮 Predict Rent",

    use_container_width=True

)

# ==========================================================
# Prediction
# ==========================================================

if predict:

    # ----------------------------
    # Create input dataframe
    # ----------------------------

    input_df = pd.DataFrame({

        "Size":[bhk],

        "Area_sqft":[area],

        "Bathroom":[bathroom],

        "Property_type":[property_type],

        "Location":[location],

        "Seller_type":[seller_type],

        "Status":[status]

    })

    # ------------------------------------------------
    # Size_unit was used during training
    # ------------------------------------------------

    input_df["Size_unit"] = "BHK"

    # ------------------------------------------------
    # One Hot Encoding
    # ------------------------------------------------

    input_encoded = pd.get_dummies(

        input_df

    )

    # ------------------------------------------------
    # Match training columns
    # ------------------------------------------------

    input_encoded = input_encoded.reindex(

        columns=encoder_columns,

        fill_value=0

    )

    # ------------------------------------------------
    # Prediction
    # ------------------------------------------------

    prediction = model.predict(

        input_encoded

    )[0]

    if prediction < 0:

        prediction = 0

    prediction = round(prediction)

    st.write("")
    st.write("")

    st.markdown("""

    <div class="result">

    <div class="small">

    💰 Estimated Monthly Rent

    </div>

    <br>

    <div class="big">

    ₹ {:,}

    </div>

    <br>

    <div class="small">

    AI Prediction using Random Forest

    </div>

    </div>

    """.format(prediction),

    unsafe_allow_html=True)

    st.write("")
    st.write("")

    a,b,c,d = st.columns(4)

    with a:
        st.metric(
            "🤖 Model",
            "Random Forest"
        )

    with b:
        st.metric(
            "🎯 Accuracy",
            "87.75%"
        )

    with c:
        st.metric(
            "🏠 Dataset",
            "14,000+"
        )

    with d:
        st.metric(
            "⚡ Speed",
            "0.02 sec"
        )

        # ==========================================================
# RentAI - app.py (PART 3)
# ==========================================================

st.write("")
st.write("")
st.divider()

# ==========================================================
# Information Section
# ==========================================================

c1, c2, c3 = st.columns(3)

with c1:

    st.info("""
### 📊 Dataset

- 14,000+ Properties
- Multiple Property Types
- Multiple Locations
- Cleaned & Processed
""")

with c2:

    st.success("""
### 🤖 Model

- Random Forest Regressor
- Feature Engineering
- One Hot Encoding
- ML Based Prediction
""")

with c3:

    st.warning("""
### 💡 Tips

✔ Correct Area

✔ Correct Location

✔ Correct Property Type

for better predictions.
""")

st.write("")
st.divider()

# ==========================================================
# About
# ==========================================================

st.markdown(
"""
## 👨‍💻 About RentAI

RentAI is an AI-powered House Rent Prediction System developed using
Machine Learning to estimate rental prices based on various property
characteristics.

### 🔧 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit

### 📈 Model Performance

- R² Score : **0.8775**
- Dataset : **14,000+ Houses**

""")

st.write("")

# ==========================================================
# Expanders
# ==========================================================

with st.expander("📌 Features Used"):

    st.write("""

- Size

- Area (sqft)

- Bathroom

- Property Type

- Location

- Seller Type

- Status

- Size Unit

""")

with st.expander("📌 Model Workflow"):

    st.write("""

Dataset

↓

Cleaning

↓

Feature Engineering

↓

One Hot Encoding

↓

Random Forest

↓

Prediction

""")

with st.expander("📌 Future Improvements"):

    st.write("""

- Better Hyperparameter Tuning

- Streamlit Cloud Deployment

- Interactive Dashboard

- Price Trend Analysis

- Map Integration

""")

st.write("")
st.divider()

# ==========================================================
# Footer
# ==========================================================

st.markdown(
"""
<center>

## 🏠 RentAI

AI Powered House Rent Prediction System

Developed using ❤️ with Python & Machine Learning

**Developer:** Mohd Labeeb Zaidi

</center>
""",
unsafe_allow_html=True
)
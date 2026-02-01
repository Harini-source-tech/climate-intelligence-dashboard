import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import plotly.express as px

st.set_page_config(layout="wide")

st.markdown("""
<style>

.stApp {
background:#ffd6e8;
}

[data-testid="stSidebar"] {
background:#ffb703;
}

.stSelectbox div,
.stSlider {
background:white !important;
color:black !important;
border-radius:12px;
}

h1.main-title {
text-align:center;
font-size:48px;
font-weight:900;
line-height:1.2;
background:linear-gradient(90deg,#00ffcc,#ff006e,#ffee00,#7cff00);
-webkit-background-clip:text;
color:transparent;
margin-bottom:40px;
}

.section {
text-align:center;
font-size:30px;
font-weight:800;
color:#7209b7;
margin-top:50px;
}

.sidebar-label {
color:#6a040f;
font-weight:900;
font-size:20px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 class='main-title'>
Global Climate Intelligence<br>
Platform
</h1>
""", unsafe_allow_html=True)

df = pd.read_csv("GlobalLandTemperaturesByCountry.csv")
df = df.dropna()
df["Year"] = pd.to_datetime(df["dt"]).dt.year

with st.sidebar:
    st.markdown("<span class='sidebar-label'>Select Country</span>", unsafe_allow_html=True)
    country = st.selectbox("", sorted(df["Country"].unique()))
    st.markdown("<span class='sidebar-label'>Year Range</span>", unsafe_allow_html=True)
    year_range = st.slider("", int(df["Year"].min()), int(df["Year"].max()), (1980, 2013))

filtered = df[(df["Country"] == country) & (df["Year"].between(year_range[0], year_range[1]))]

st.markdown("<div class='section'>Historical Temperature Trend</div>", unsafe_allow_html=True)

fig1, ax1 = plt.subplots(figsize=(13,6))
fig1.patch.set_facecolor("#ffd6e8")
ax1.set_facecolor("#ffe5f1")
ax1.plot(filtered["Year"], filtered["AverageTemperature"], color="#ff006e")
ax1.set_xlabel("Year")
ax1.set_ylabel("°C")
st.pyplot(fig1)

st.markdown("<div class='section'>Temperature Heatmap</div>", unsafe_allow_html=True)

heat_df = filtered.copy()
heat_df["Month"] = pd.to_datetime(heat_df["dt"]).dt.month
pivot = heat_df.pivot_table(index="Month", columns="Year", values="AverageTemperature")

fig2, ax2 = plt.subplots(figsize=(18,6))
fig2.patch.set_facecolor("#ffd6e8")
ax2.set_facecolor("#ffe5f1")
sns.heatmap(pivot, cmap="plasma", ax=ax2)
st.pyplot(fig2)

st.markdown("<div class='section'>AI Forecast - Next 20 Years</div>", unsafe_allow_html=True)

yearly = df[df["Country"] == country].groupby("Year")["AverageTemperature"].mean().reset_index()

X = yearly[["Year"]]
y = yearly["AverageTemperature"]

model = LinearRegression()
model.fit(X, y)

future_years = np.arange(yearly["Year"].max()+1, yearly["Year"].max()+21).reshape(-1,1)
preds = model.predict(future_years)

forecast_df = pd.DataFrame({"Year":future_years.flatten(),"PredictedTemp":preds})

fig3, ax3 = plt.subplots(figsize=(13,6))
fig3.patch.set_facecolor("#ffd6e8")
ax3.set_facecolor("#ffe5f1")
ax3.plot(yearly["Year"], y, color="#06d6a0", label="Past")
ax3.plot(forecast_df["Year"], preds, color="#ffbe0b", linestyle="--", label="AI Prediction")
ax3.legend()
st.pyplot(fig3)

st.dataframe(forecast_df)

st.markdown("<div class='section'>Global Temperature Ranking</div>", unsafe_allow_html=True)

rank_df = df[df["Year"] == year_range[1]].groupby("Country")["AverageTemperature"].mean().sort_values(ascending=False).head(20).reset_index()

fig4, ax4 = plt.subplots(figsize=(14,8))
fig4.patch.set_facecolor("#ffd6e8")
ax4.set_facecolor("#ffe5f1")
ax4.barh(rank_df["Country"], rank_df["AverageTemperature"], color="#ff7b00")
ax4.invert_yaxis()
st.pyplot(fig4)

st.markdown("<div class='section'>Global Temperature Map</div>", unsafe_allow_html=True)

map_df = df[df["Year"] == year_range[1]].groupby("Country")["AverageTemperature"].mean().reset_index()

fig_map = px.choropleth(
    map_df,
    locations="Country",
    locationmode="country names",
    color="AverageTemperature",
    color_continuous_scale=["yellow","orange","red","#00ffcc"]
)

fig_map.update_layout(
    height=350,
    margin=dict(l=10,r=10,t=10,b=10),
    geo=dict(
        bgcolor="#ffd6e8",
        showcoastlines=True,
        coastlinecolor="#7209b7",
        showframe=False
    ),
    paper_bgcolor="#ffd6e8",
    plot_bgcolor="#ffd6e8"
)

fig_map.update_coloraxes(
    colorbar=dict(
        title="Avg Temp °C",
        tickcolor="#7209b7",
        title_font=dict(color="#7209b7")
    )
)

st.plotly_chart(fig_map, use_container_width=True)
from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import plotly.express as px
import plotly
import json
import joblib
from flask import send_file
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)

# =========================
# Load Dataset
# =========================

df = pd.read_csv("dataset/customers.csv")
print(df.columns)
print(df.head())

# =========================
# Load Models
# =========================

purchase_model = joblib.load("saved_models/purchase_model.pkl")
churn_model = joblib.load("saved_models/churn_model.pkl")
kmeans = joblib.load("saved_models/kmeans.pkl")
scaler = joblib.load("saved_models/scaler.pkl")

# =========================
# Segment Mapping
# =========================

SEGMENT_NAMES = {
    0: "Premium Customers",
    1: "Regular Customers",
    2: "Budget Customers",
    3: "New Customers"
}

RECOMMENDATIONS = {
    0: [
        "Premium Smartphones",
        "Gaming Laptops",
        "Smart Watches",
        "Luxury Accessories"
    ],

    1: [
        "Fashion Products",
        "Footwear",
        "Daily Essentials",
        "Accessories"
    ],

    2: [
        "Budget Products",
        "Discount Offers",
        "Combo Deals",
        "Seasonal Sales"
    ],

    3: [
        "Starter Packages",
        "Welcome Offers",
        "First Purchase Discounts",
        "Membership Plans"
    ]
}


# =========================
# Dashboard
# =========================
@app.route("/")
def dashboard():

    total_customers = len(df)

    avg_income = round(
        df["AnnualIncome"].mean(),
        2
    )

    avg_spending = round(
        df["SpendingScore"].mean(),
        2
    )

    purchase_rate = round(
        df["Purchase"].mean() * 100,
        2
    )

    churn_rate = round(
        df["Churn"].mean() * 100,
        2
    )

    highest_income = int(
        df["AnnualIncome"].max()
    )

    # Plotly Age Distribution

    age_fig = px.histogram(
        df,
        x="Age",
        nbins=20,
        title="Customer Age Distribution"
    )

    age_graph = json.dumps(
        age_fig,
        cls=plotly.utils.PlotlyJSONEncoder
    )

    # Plotly Scatter Chart

    scatter_fig = px.scatter(
        df,
        x="AnnualIncome",
        y="SpendingScore",
        title="Income vs Spending Score"
    )

    scatter_graph = json.dumps(
        scatter_fig,
        cls=plotly.utils.PlotlyJSONEncoder
    )

    ai_insights = [

        f"Average customer income is ₹{avg_income}",

        f"Purchase success rate is {purchase_rate}%",

        f"Churn risk affects {churn_rate}% of customers",

        f"Highest customer income recorded is ₹{highest_income}"

    ]

    return render_template(
        "dashboard.html",

        total_customers=total_customers,

        avg_income=avg_income,

        avg_spending=avg_spending,

        purchase_rate=purchase_rate,

        churn_rate=churn_rate,

        insights=ai_insights,

        age_graph=age_graph,

        scatter_graph=scatter_graph
    )

# =========================
# Prediction Page
# =========================

@app.route("/prediction")
def prediction():
    return render_template("prediction.html")


# =========================
# Predict Purchase + Churn
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    age = int(request.form["age"])
    income = int(request.form["income"])
    score = int(request.form["score"])
    frequency = int(request.form["frequency"])
    spent = int(request.form["spent"])

    data = np.array([
        [age, income, score, frequency, spent]
    ])

    purchase_prediction = purchase_model.predict(data)[0]

    churn_prediction = churn_model.predict(data)[0]

    return render_template(

        "prediction.html",

        purchase_result=
        "Likely To Purchase ✅"
        if purchase_prediction == 1
        else
        "Unlikely To Purchase ❌",

        churn_result=
        "High Churn Risk ⚠️"
        if churn_prediction == 1
        else
        "Low Churn Risk ✅"
    )


# =========================
# Segmentation Page
# =========================

@app.route("/segmentation")
def segmentation():

    features = df[
        ["AnnualIncome", "SpendingScore"]
    ]

    scaled_data = scaler.transform(features)

    clusters = kmeans.predict(
        scaled_data
    )

    df["Cluster"] = clusters

    # =========================
    # Customer Cluster Scatter Plot
    # =========================

    scatter_fig = px.scatter(
        df,
        x="AnnualIncome",
        y="SpendingScore",
        color=df["Cluster"].astype(str),
        title="Customer Cluster Visualization",
        labels={
            "AnnualIncome": "Annual Income",
            "SpendingScore": "Spending Score",
            "color": "Customer Segment"
        }
    )

    cluster_graph = json.dumps(
        scatter_fig,
        cls=plotly.utils.PlotlyJSONEncoder
    )

    # =========================
    # Segment Counts
    # =========================

    cluster_counts = (
        df["Cluster"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    segment_data = []

    for cluster_id, count in cluster_counts.items():

        segment_data.append({

            "id": cluster_id,

            "name": SEGMENT_NAMES.get(
                cluster_id,
                "Unknown"
            ),

            "count": count

        })

    # =========================
    # Pie Chart Data
    # =========================

    cluster_labels = []
    cluster_values = []

    for item in segment_data:

        cluster_labels.append(
            item["name"]
        )

        cluster_values.append(
            item["count"]
        )

    return render_template(
        "segmentation.html",

        segment_data=segment_data,

        cluster_labels=cluster_labels,

        cluster_values=cluster_values,

        cluster_graph=cluster_graph
    )
# =========================
# Recommendation Page
# =========================

@app.route("/recommendation")
def recommendation():

    return render_template(
        "recommendation.html"
    )


# =========================
# Recommendation Engine
# =========================

@app.route(
    "/recommend_customer",
    methods=["POST"]
)
def recommend_customer():

    income = int(
        request.form["income"]
    )

    score = int(
        request.form["score"]
    )

    scaled = scaler.transform(
        [[income, score]]
    )

    segment = int(
        kmeans.predict(scaled)[0]
    )

    segment_name = SEGMENT_NAMES.get(
        segment
    )

    products = RECOMMENDATIONS.get(
        segment
    )

    return render_template(

        "recommendation.html",

        segment=segment,

        segment_name=segment_name,

        recommendations=products

    )
@app.route("/download_report")
def download_report():

    pdf_file = "customer_report.pdf"

    doc = SimpleDocTemplate(pdf_file)

    styles = getSampleStyleSheet()

    content = []

    content.append(
        Paragraph(
            "AI Customer Analytics Report",
            styles["Title"]
        )
    )

    content.append(Spacer(1, 20))

    content.append(
        Paragraph(
            f"Total Customers: {len(df)}",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"Average Income: ₹{round(df['AnnualIncome'].mean(),2)}",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"Average Spending Score: {round(df['SpendingScore'].mean(),2)}",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"Purchase Rate: {round(df['Purchase'].mean()*100,2)}%",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"Churn Rate: {round(df['Churn'].mean()*100,2)}%",
            styles["BodyText"]
        )
    )

    doc.build(content)

    return send_file(
        pdf_file,
        as_attachment=True
    )
@app.route("/upload", methods=["POST"])
def upload():

    global df

    file = request.files["file"]

    if file:

        temp_df = pd.read_csv(file)

        required_columns = [
            "CustomerID",
            "Age",
            "AnnualIncome",
            "SpendingScore",
            "PurchaseFrequency",
            "TotalSpent",
            "Purchase",
            "Churn"
        ]

        if all(col in temp_df.columns for col in required_columns):

            df = temp_df

        else:

            return "Invalid CSV Format"

    return dashboard()
@app.route("/customer/<int:customer_id>")
def customer_details(customer_id):

    customer = df[
        df["CustomerID"] == customer_id
    ]

    if customer.empty:
        return "Customer Not Found"

    customer = customer.iloc[0]

    return render_template(
        "customer_details.html",
        customer=customer
    )


# =========================
# Run App
# =========================

if __name__ == "__main__":
    app.run(
        debug=True
    )
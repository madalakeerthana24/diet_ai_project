from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

app = Flask(__name__)

print("🔥 App is starting...")

# --------- TRAIN MODEL ---------
data = {
    "bmi": [16, 18, 22, 25, 28, 32],
    "goal": ["gain", "gain", "maintain", "loss", "loss", "loss"],
    "diet": [
        "High calorie diet",
        "Protein rich diet",
        "Balanced diet",
        "Low carb diet",
        "Low calorie diet",
        "Strict fat loss diet"
    ]
}

df = pd.DataFrame(data)
df["goal"] = df["goal"].map({"gain": 0, "maintain": 1, "loss": 2})

X = df[["bmi", "goal"]]
y = df["diet"]

model = DecisionTreeClassifier()
model.fit(X, y)

# --------- ROUTES ---------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    weight = float(request.form["weight"])
    height = float(request.form["height"])
    gender = request.form.get("gender")

    if not gender:
        return "⚠️ Please select gender"

    bmi = weight / (height ** 2)

    # -------- GOAL LOGIC --------
    if bmi < 18.5:
        goal = "Weight Gain"
        goal_key = 0
    elif bmi < 25:
        goal = "Maintain"
        goal_key = 1
    elif bmi < 30:
        goal = "Weight Loss"
        goal_key = 2
    else:
        goal = "Strict Fat Loss"
        goal_key = 2

    # -------- GENDER-BASED MACROS --------
    if gender == "male":
        if goal == "Weight Gain":
            macros = [55, 25, 20]
        elif goal == "Maintain":
            macros = [45, 30, 25]
        else:
            macros = [35, 40, 25]
    else:  # female
        if goal == "Weight Gain":
            macros = [50, 30, 20]
        elif goal == "Maintain":
            macros = [40, 35, 25]
        else:
            macros = [30, 40, 30]

    # -------- PREDICTION --------
    prediction = model.predict([[bmi, goal_key]])[0]

    # -------- PIE CHART --------
    labels = ['Carbs', 'Protein', 'Fats']

    plt.figure(figsize=(8, 8))  # BIG CHART

    plt.pie(
        macros,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 14}
    )

    plt.title(f"{gender.capitalize()} Diet ({goal})", fontsize=18)

    plt.savefig("static/chart.png", bbox_inches='tight')
    plt.close()

    return render_template(
        "result.html",
        bmi=round(bmi, 2),
        goal=goal,
        gender=gender,
        diet=prediction,
        chart="chart.png"
    )

# --------- RUN ---------
if __name__ == "__main__":
    app.run(debug=True)
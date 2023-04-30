from flask import Flask, render_template, request, redirect, url_for, flash
from book_recommendation import run_recommendation_system

app = Flask(__name__)
app.secret_key = "your_secret_key"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_id = request.form.get("user_id")
        if user_id:
            try:
                recommendations, rmse, accuracy = run_recommendation_system(user_id)
                return render_template("recommendations.html", recommendations=recommendations, user_id=user_id, rmse=rmse, accuracy=accuracy)
            except Exception as e:
                flash(f"Error: {str(e)}")
                return redirect(url_for("index"))
        else:
            flash("Please enter a user ID.")
            return redirect(url_for("index"))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
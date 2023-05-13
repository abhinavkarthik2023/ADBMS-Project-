#app.py
from flask import Flask, render_template, request, redirect, url_for, flash
from book_recommendation import main

app = Flask(__name__)
app.secret_key = "your_secret_key"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_id = request.form.get("user_id")
        if user_id:
            try:
                recommendations = main(user_id)
                precomputed_rmse = 0.77  # Replace with your precomputed RMSE value
                precomputed_accuracy = 0.9456  # Replace with your precomputed accuracy value
                return render_template("recommendations.html", recommendations=recommendations, user_id=user_id, rmse=precomputed_rmse, accuracy=precomputed_accuracy)
            except Exception as e:
                flash(f"Error: {str(e)}")
                return redirect(url_for("index"))
        else:
            flash("Please enter a user ID.")
            return redirect(url_for("index"))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)



    
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import ttkthemes as ttkthemes
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def run_recommendation_system(user_id_entry):
    user_id = user_id_entry.get()

    if not user_id:
        messagebox.showerror("Error", "Please enter a user ID.")
        return

    spark = create_spark_session()

    books_df, ratings_df, users_df = read_dataframes(spark)

    data_df = join_dataframes(ratings_df, books_df, users_df)

    transformed_df = transform_dataframe(data_df)

    training, test = split_dataframe(transformed_df)

    model, rmse, accuracy = train_and_evaluate_als(training, test)

    messagebox.showinfo("Results", f"RMSE: {rmse}\nAccuracy: {accuracy}")

    user_recommendations = get_recommendations(model, user_id)

    if not user_recommendations:
        messagebox.showerror("Error", f"No recommendations found for user ID {user_id}.")
        return

    recommended_books = [item.item_index for item in user_recommendations[0][0]]

    show_recommendations(transformed_df, books_df, user_id, recommended_books)

def create_spark_session():
    return (
        SparkSession.builder.appName("BookRecommendationSystem")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .getOrCreate()
    )

def read_dataframes(spark):
    return (
        spark.read.csv(books_path, header=True),
        spark.read.csv(ratings_path, header=True),
        spark.read.csv(users_path, header=True),
    )

def join_dataframes(ratings_df, books_df, users_df):
    return (
        ratings_df.join(books_df, "ISBN", "inner")
        .join(users_df, "User-ID", "inner")
        .select(
            ratings_df["User-ID"].alias("user"),
            ratings_df["ISBN"].alias("item"),
            ratings_df["Book-Rating"].alias("rating"),
        )
    )

def transform_dataframe(data_df):
    indexers = [
        StringIndexer(inputCol=column, outputCol=column + "_index", handleInvalid="skip")
        for column in ["user", "item", "rating"]
    ]

    pipeline = Pipeline(stages=indexers)

    return pipeline.fit(data_df).transform(data_df)

def split_dataframe(transformed_df):
    return transformed_df.randomSplit([0.8, 0.2])

def train_and_evaluate_als(training, test):
    als = ALS(
        maxIter=5,
        regParam=0.09,
        rank=25,
        userCol="user_index",
        itemCol="item_index",
        ratingCol="rating_index",
        coldStartStrategy="drop",
        nonnegative=True,
    )
    model = als.fit(training)

    predictions = model.transform(test)
    evaluator = RegressionEvaluator(
        metricName="rmse", labelCol="rating_index", predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    accuracy = 1 - rmse

    return model, rmse, accuracy

def get_recommendations(model, user_id):
    return (
        model.recommendForAllUsers(20)
        .filter(col("user_index") == user_id)
        .select("recommendations")
        .collect()
    )

def show_recommendations(transformed_df, books_df, user_id, recommended_books):
    results_window = tk.Toplevel(root)
    results_window.title("Top 20 Recommended Books")

    tree = ttk.Treeview(
        results_window, columns=("User ID", "Book-Title"), show="headings"
    )
    tree.heading("User ID", text="User ID")
    tree.heading("Book-Title", text="Book Title")
    tree.grid(row=0, column=0, padx=10, pady=10)

    scrollbar = ttk.Scrollbar(results_window, orient="vertical", command=tree.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    tree.configure(yscrollcommand=scrollbar.set)

    recommended_books_df = (
        transformed_df.filter(transformed_df["item_index"].isin(recommended_books))
        .select("item", "item_index")
        .distinct()
        .join(books_df, transformed_df["item"] == books_df["ISBN"], "inner")
        .select("item", "Book-Title")
    )

    for row in recommended_books_df.collect():
        tree.insert("", "end", values=(user_id, row["Book-Title"]))

    results_window.mainloop()

def open_file(entry):
    file_path = filedialog.askopenfilename()
    entry.delete(0, tk.END)
    entry.insert(0, file_path)

def create_file_entries(frame, labels):
    entries = []

    for i, label_text in enumerate(labels):
        label = ttk.Label(frame, text=label_text, font=("Helvetica", 12))
        label.grid(row=i, column=0, pady=10)
        entry = ttk.Entry(frame, width=50, font=("Helvetica", 12))
        entry.grid(row=i, column=1, pady=10)
        entries.append(entry)

        button = ttk.Button(
            frame, text="Browse", command=lambda e=entry: open_file(e), style="Accent.TButton"
        )
        button.grid(row=i, column=2, padx=(10, 0), pady=10)

    return entries

def create_user_id_entry(frame):
    user_id_label = ttk.Label(frame, text="User ID:", font=("Helvetica", 12))
    user_id_label.grid(row=3, column=0, pady=10)
    user_id_entry = ttk.Entry(frame, width=50, font=("Helvetica", 12))
    user_id_entry.grid(row=3, column=1, pady=10)
    return user_id_entry

def run_button_action(entries, user_id_entry):
    global books_path, ratings_path, users_path
    books_path, ratings_path, users_path = (e.get() for e in entries)

    try:
        run_recommendation_system(user_id_entry)
    except Exception as e:
        messagebox.showerror("Error", str(e))

def create_run_button(frame, entries, user_id_entry):
    run_button = ttk.Button(
        frame,
        text="Run",
        command=lambda: run_button_action(entries, user_id_entry),
        style="Accent.TButton",
    )
    run_button.grid(row=4, columnspan=2, pady=(20, 0))

root = tk.Tk()
root.title("Book Recommendation System")

style = ttkthemes.ThemedStyle(root)
style.theme_use("equilux")

style.configure("Accent.TButton", foreground="#00897b", background="#00897b", font=("Helvetica", 12))
style.map("Accent.TButton", background=[("active", "#00796b")])

frame = ttk.Frame(root, padding="20 20 20 20")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
frame.columnconfigure(0, weight=1)

root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

labels = ["Books CSV:", "Ratings CSV:", "Users CSV:"]
file_entries = create_file_entries(frame, labels)

user_id_entry = create_user_id_entry(frame)

create_run_button(frame, file_entries, user_id_entry)

root.mainloop()
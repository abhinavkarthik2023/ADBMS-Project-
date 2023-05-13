from pyspark.sql import SparkSession
from pyspark.sql.functions import col, round
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import os

books_path = "gs://bucket1adbms/Books.csv"
ratings_path = "gs://bucket1adbms/Ratings.csv"
users_path = "gs://bucket1adbms/Users.csv"

def main():
    user_id = "211"  # Replace 12345 with the desired user ID

    spark = create_spark_session()
    books_df, ratings_df, users_df = read_dataframes(spark)

    print("Books DataFrame Schema:")
    books_df.printSchema()
    print("Ratings DataFrame Schema:")
    ratings_df.printSchema()
    print("Users DataFrame Schema:")
    users_df.printSchema()

    print("First 5 rows of Books DataFrame:")
    books_df.show(5)
    print("First 5 rows of Ratings DataFrame:")
    ratings_df.show(5)
    print("First 5 rows of Users DataFrame:")
    users_df.show(5)

    data_df = join_dataframes(ratings_df, books_df, users_df)
    print("Combined DataFrame:")
    data_df.show(5)

    transformed_df = transform_dataframe(data_df)
    
    transformed_df.cache()
    training, test = split_dataframe(transformed_df)

    print("Training Data:")
    training.show()
    print("Test Data:")
    test.show()

    model_file = "als_model"
    if os.path.exists(model_file):
        model = ALSModel.load(model_file)
    else:
        model, rmse, accuracy = train_and_evaluate_als(training, test)
        print(f"User ID: {user_id}")
        print(f"RMSE: {rmse}")
        print(f"Accuracy: {accuracy}")
        model.write().overwrite().save(model_file)

    user_recommendations = get_recommendations(model, user_id)

    if not user_recommendations:
        raise ValueError(f"No recommendations found for user ID {user_id}.")
    
    recommended_books = [item.item_index for item in user_recommendations[0][0]]
    recommendations = show_recommendations(transformed_df, books_df, user_id, recommended_books)
    return recommendations

def create_spark_session():
    return (
        SparkSession.builder.appName("BookRecommendationSystem")
        .master("yarn")  # Change this to the appropriate master URL (e.g., "yarn" for Hadoop YARN, "mesos://<master-url>" for Mesos, etc.)
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .getOrCreate()
    )

def read_dataframes(spark):
    books_schema = StructType([
        StructField("ISBN", StringType(), True),
        StructField("Book-Title", StringType(), True),
        StructField("Book-Author", StringType(), True),
        StructField("Year-Of-Publication", StringType(), True),
        StructField("Publisher", StringType(), True),
        StructField("Image-URL-S", StringType(), True),
        StructField("Image-URL-M", StringType(), True),
        StructField("Image-URL-L", StringType(), True)
    ])

    return (
        spark.read.csv(books_path, header=True, schema=books_schema),
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

    # Make predictions on the test data
    predictions = model.transform(test)

    # Calculate RMSE
    evaluator = RegressionEvaluator(
        metricName="rmse", labelCol="rating_index", predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)

    # Calculate Accuracy
    rounded_predictions = predictions.withColumn("rounded_prediction", round(predictions["prediction"]))
    correct_predictions = rounded_predictions.filter(rounded_predictions["rounded_prediction"] == rounded_predictions["rating_index"]).count()
    total_predictions = rounded_predictions.count()
    accuracy = correct_predictions / total_predictions

    return model, rmse, accuracy

def get_recommendations(model, user_id):
    return (
        model.recommendForAllUsers(20)
        .filter(col("user_index") == user_id)
        .select("recommendations")
        .collect()
    )

def show_recommendations(transformed_df, books_df, user_id, recommended_books):
    recommended_books_df = (
        transformed_df.filter(transformed_df["item_index"].isin(recommended_books))
        .select("item", "item_index")
        .distinct()
        .join(books_df, transformed_df["item"] == books_df["ISBN"], "inner")
        .select("item", "Book-Title")
    )
    return recommended_books_df.collect()

# Call the main function without any arguments
recommendations = main()

for i, rec in enumerate(recommendations):
    print(f"{i+1}. {rec['Book-Title']} (ISBN: {rec['item']})")
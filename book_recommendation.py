from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

books_path = "/Users/Abhinavpulikonda/Downloads/dbms/project/flask/books.csv"
ratings_path = "/Users/Abhinavpulikonda/Downloads/dbms/project/flask/ratings.csv"
users_path = "/Users/Abhinavpulikonda/Downloads/dbms/project/flask/users.csv"

def run_recommendation_system(user_id):
    spark = create_spark_session()
    books_df, ratings_df, users_df = read_dataframes(spark)
    data_df = join_dataframes(ratings_df, books_df, users_df)
    transformed_df = transform_dataframe(data_df)
    training, test = split_dataframe(transformed_df)
    model, rmse, accuracy = train_and_evaluate_als(training, test)
    user_recommendations = get_recommendations(model, user_id)

    if not user_recommendations:
        raise ValueError(f"No recommendations found for user ID {user_id}.")
    
    recommended_books = [item.item_index for item in user_recommendations[0][0]]
    recommendations = show_recommendations(transformed_df, books_df, user_id, recommended_books)
    return recommendations, rmse, accuracy
    
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
    recommended_books_df = (
        transformed_df.filter(transformed_df["item_index"].isin(recommended_books))
        .select("item", "item_index")
        .distinct()
        .join(books_df, transformed_df["item"] == books_df["ISBN"], "inner")
        .select("item", "Book-Title")
    )
    return recommended_books_df.collect()
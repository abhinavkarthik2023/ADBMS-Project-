# ADBMS-Project-
Book Recommendation System using Pyspark 

dataset Link - https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset

# Book Recommendation System

The Book Recommendation System is a machine learning project that provides personalized book recommendations to users based on their historical data of book ratings. The system is implemented using Python, Flask, Spark ML, and the Alternating Least Squares (ALS) algorithm for collaborative filtering.

## Table of Contents

- [Project Structure](#Project Structure)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

##Project Structure

- `app.py`: Flask application entry point. Handles user input and displays book recommendations.
- `book_recommendation.py`: Contains the PySpark code for training the ALS model and generating recommendations.
- templates/
- `index.html`: Home page for the Flask app where users enter their user ID.
- `recommendations.html`: Page showing the top 20 recommended books for the given user ID.
- books.csv (not included): Dataset containing book metadata.
- ratings.csv (not included): Dataset containing user ratings for books.
- users.csv (not included): Dataset containing user information.

## Features

- Data preprocessing: Cleansing and transforming raw dataset into the desired format.
- Model training: Training a collaborative filtering model using Alternating Least Squares (ALS) algorithm on the preprocessed data.
- Model evaluation: Evaluating the trained model using Root Mean Squared Error (RMSE) metric.
- Recommendations: Generating top recommended books for a given user ID.
- User interface: A simple Flask web application to input user ID and display recommendations.

## System Architecture

The Book Recommendation System is designed with the following modules:

1. Importing required libraries and initializing Flask app and Spark session.
2. Reading data from CSV files into DataFrames.
3. Joining dataframes and performing data transformations.
4. Splitting data into training and testing sets.
5. Training the ALS model and evaluating its performance.
6. Generating recommendations for a given user ID.
7. Displaying the recommended books in the Flask web application.
8.The recommended books are displayed along with their ISBN, RMSE, and accuracy.

## Requirements

- Python 3.7 or higher
- Apache Spark 3.1.2 or higher
- Flask
- Required Python packages (see requirements.txt)

## Installation

1. Install Python 3.7 or higher and Apache Spark 3.1.2 or higher on your machine.

2. Clone the repository from GitHub:
   ```
   git clone https://github.com/yourusername/BookRecommendationSystem.git
   ```

3. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Flask app:
   ```
   python app.py
   ```

2. Open a web browser and go to `http://127.0.0.1:5000/`.

3. Enter a user ID in the "User ID" field.

4. Click the "Recommend Books" button to generate recommendations for the entered user ID.

5. The top 20 recommended books for the given user ID will be displayed in the browser.

## Testing

The Book Recommendation System was tested on the Book-Crossing dataset containing over 270,000 books, 1 million ratings, and 278,000 users. The model achieved an RMSE of 1.609 and an accuracy of 0.9391. The top 20 recommended books for a given user ID were displayed in the Flask web application.

## Contributing

1. Fork the repository on GitHub.
2. Clone the forked repository to your machine.
3. Create a new branch for your feature or bugfix.
4. Make your changes, add tests, and ensure that all tests pass.
5. Commit your changes and push your branch to GitHub.
6. Open a pull request, and provide a detailed description of your changes.

## License

This project is released under the MIT License.




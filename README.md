# ADBMS-Project-
Book Recommendation System using Pyspark 

dataset Link - https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset


# Book Recommendation System

The Book Recommendation System is a machine learning project that provides personalized book recommendations to users based on their historical data of book ratings. The system is implemented using Python, Spark ML, and the Alternating Least Squares (ALS) algorithm for collaborative filtering.

## Table of Contents

1. [Features](#features)
2. [System Architecture](#system-architecture)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Testing](#testing)
7. [Contributing](#contributing)
8. [License](#license)

## Features

- Data preprocessing: Cleansing and transforming raw dataset into the desired format.
- Model training: Training a collaborative filtering model using Alternating Least Squares (ALS) algorithm on the preprocessed data.
- Model evaluation: Evaluating the trained model using Root Mean Squared Error (RMSE) metric.
- Recommendations: Generating top recommended books for a given user ID.

## System Architecture

The Book Recommendation System is designed with the following modules:

1. Importing required libraries and initializing Spark session.
2. Reading data from CSV files into DataFrames.
3. Joining dataframes and performing data transformations.
4. Splitting data into training and testing sets.
5. Training the ALS model and evaluating its performance.
6. Generating recommendations for a given user ID.
7. Displaying the recommended books.

## Requirements

- Python 3.7 or higher
- Apache Spark 3.1.2 or higher
- Required Python packages (see `requirements.txt`)

## Installation

1. Install Python 3.7 or higher and Apache Spark 3.1.2 or higher on your machine.
2. Clone the repository from GitHub:

3. Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Open the `book_recommendation_system.py` script in your favorite Python IDE.
2. Run the script, which will open the main application window.
3. Browse and select the Books, Ratings, and Users CSV files.
4. Enter a user ID in the "User ID" field.
5. Click the "Run" button to generate recommendations for the entered user ID.

## Testing

The Book Recommendation System was tested on the Book-Crossing dataset containing over 270,000 books, 1 million ratings, and 278,000 users. The model achieved an RMSE of 1.609 and an accuracy of 0.9391. The top 20 recommended books for a given user ID were displayed in a separate window.

## Contributing

1. Fork the repository on GitHub.
2. Clone the forked repository to your machine.
3. Create a new branch for your feature or bugfix.
4. Make your changes, add tests, and ensure that all tests pass.
5. Commit your changes and push your branch to GitHub.
6. Open a pull request, and provide a detailed description of your changes.

## License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).

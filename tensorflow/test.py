import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf

import importer

def main(argv):
  """Builds, trains, and evaluates the model."""
  assert len(argv) == 1
  (train, test) = importer.dataset()

  # Switch the labels to units of thousands for better convergence.
  def to_thousands(features, labels):
    return features, labels / PRICE_NORM_FACTOR

  train = train.map(to_thousands)
  test = test.map(to_thousands)

  # Build the training input_fn.
  def input_train():
    return (
        # Shuffling with a buffer larger than the data set ensures
        # that the examples are well mixed.
        train.shuffle(1000).batch(128)
        # Repeat forever
        .repeat().make_one_shot_iterator().get_next())

  # Build the validation input_fn.
  def input_test():
    return (test.shuffle(1000).batch(128)
            .make_one_shot_iterator().get_next())

  feature_columns = [
      # "curb-weight" and "highway-mpg" are numeric columns.
      tf.feature_column.numeric_column(key="ITIN_YIELD"),
      tf.feature_column.numeric_column(key="DISTANCE"),
  ]

  # Build the Estimator.
  model = tf.estimator.LinearRegressor(feature_columns=feature_columns)

  # Train the model.
  # By default, the Estimators log output every 100 steps.
  model.train(input_fn=input_train, steps=STEPS)

  # Evaluate how the model performs on data it has not yet seen.
  eval_result = model.evaluate(input_fn=input_test)

  # The evaluation returns a Python dictionary. The "average_loss" key holds the
  # Mean Squared Error (MSE).
  average_loss = eval_result["average_loss"]

  # Convert MSE to Root Mean Square Error (RMSE).
  print("\n" + 80 * "*")
  print("\nRMS error for the test set: ${:.0f}"
        .format(PRICE_NORM_FACTOR * average_loss**0.5))

  # Run the model in prediction mode.
  input_dict = {
      "ITIN_YIELD": np.array([0.9135, 0.2306]),
      "DISTANCE": np.array([425, 850])
  }
  predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      input_dict, shuffle=False)
  predict_results = model.predict(input_fn=predict_input_fn)

  # Print the prediction results.
  print("\nPrediction results:")
  for i, prediction in enumerate(predict_results):
    msg = ("ITIN_YIELD: {: 4d}lbs, "
           "DISTANCE: {: 0d}mpg, "
           "Price Prediction: ${: 9.2f}")
    msg = msg.format(input_dict["ITIN_YIELD"][i], input_dict["DISTANCE"][i],
                     PRICE_NORM_FACTOR * prediction["predictions"][0])

    print("    " + msg)
  print()


if __name__ == "__main__":
  # The Estimator periodically generates "INFO" logs; make these logs visible.
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)

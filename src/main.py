# main.py
import datetime

import numpy as np


def final_model_predict(use_cached_model=True):
    from pickle import load
    from pandas import read_csv, DataFrame
    from src.final_model import generate_final_model_and_print_error_estimates

    new_data = read_csv('./test_data.csv')  # read test holdout data

    if use_cached_model:  # use cached model to avoid longer runtime
        with open('./final_model.pkl', 'rb') as file:
            final_model = load(file)

        predictions = final_model.predict(new_data)
        DataFrame(predictions).to_csv('./predictions.csv')
        return predictions
    else:
        final_model = generate_final_model_and_print_error_estimates()
        predictions = final_model.predict(new_data)
        DataFrame(predictions).to_csv(f'./predictions{datetime.datetime.now()}.csv')
        return predictions


# Run main function
final_predictions = final_model_predict(use_cached_model=False)
print('Final Predictions Summary:')
print(f'Len: {len(final_predictions)}, avg: {np.mean(final_predictions)}')


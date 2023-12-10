# generate_final_data.py
import src.utils as utils
import pandas as pd

raw_data_with_encoded_y = utils.load_data()
final_pipeline = utils.initialize_model_pipeline(None)
final_pipeline.fit(X=raw_data_with_encoded_y, y=raw_data_with_encoded_y['increase_stock'])
# rebind new col names and save as dataframe
final_data = pd.DataFrame(final_pipeline.transform(X=raw_data_with_encoded_y),
                          columns=final_pipeline[:-1].get_feature_names_out())
final_data['increase_stock'] = raw_data_with_encoded_y['increase_stock']
final_data.to_csv('./final.csv', index=False)

# main.py
def final_model_predict(new_data_x):
    from pickle import load

    # new_data = pd.read_csv('./new_data_that_we_dont_have_yet.csv'):
    with open('./final_model.pkl', 'rb') as file:
        final_model = load(file)

    return final_model.predict(new_data_x)
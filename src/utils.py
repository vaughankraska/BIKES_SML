def create_weather_score(df_row):
    import numpy as np

    b0 = -0.8138192466580598
    betas = np.array([-5.56042822e-04 , 6.31851025e-02, -5.21309866e-02 , 6.49687899e-03,
                      -1.10383706e-02 , 6.86816015e-03, -5.95565495e-05 , 7.78956846e-04,
                      -1.72291318e-03])
    exes = np.matrix([df_row['summertime'],df_row['temp'],df_row['dew'],
                      df_row['humidity'],df_row['precip'],df_row['snowdepth'],
                      df_row['windspeed'],df_row['cloudcover'],df_row['visibility']])
    if exes.shape[0] != 9:
        exes = exes.T
    return b0 + np.dot(betas, exes.tolist())

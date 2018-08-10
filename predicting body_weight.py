import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# reading the data
data_frame = pd.read_fwf('brain_body.txt')
x_values = data_frame[['Brain']]
y_values = data_frame[['Body']] 

#training the model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

#visualising the results
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()
from Utils.DataGenerator import generate_data
from Utils.DataGenerator import restricted_three_body

# Define the parameters for data generation
t_span = (0, 50)

# Generate the data
t, data = generate_data(restricted_three_body, t_span, [0.5, 0.5, 0.0, 0.0])

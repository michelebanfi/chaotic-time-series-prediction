# Chaotic time series prediction using Reservoir Computing

> The project report is available at `Report - Chaotic Time series prediction.pdf`

## Hands on
The main file is `main.py`. From there the prediction of three
differen problem is possible, by choosing the `problem` variable, which can be:
- `MackeyGlass`
- `lorenz`
- `R3BP`

## Reservoir
The reservoir clas is defined in the `Reservoir` folder. The `ESNReservoir` class is defined in the `ESN.py` file. 
The class allows to fit the data using the `fit` method
and then generate data using the `forward` method.

## Data generation
The data generation is done in the `SyntheticData` folder. The data for `lorenz` and 
`R3BP` is generated with `solve_ivp` from `scipy.integrate`. The data for `MackeyGlass` is 
generated using `jitcdde` from `jitcdde` package. 

## Data loading
The data is loaded in the `DataLoader` file. The data is loaded from the `Data` folder.
The `loadData` method splits the data to fit and warmup the reservoir.`

## Optimization
Optimization can be done with `OptimizaionESN.py` file. The optimization is done using
either a random search or with a grid search.

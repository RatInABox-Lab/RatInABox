## Add your own trajectory data into this folder. 

### How to format your data: 
Name it like: "my_trajectory_data.npz"
The file itself must be a list of timestamps (key = "t") and a list of positions (key = "pos"), i.e. if run from this directory the following code should work 
```python
    data = np.load("./my_trajectory_data.npz")
    times = data["t"] #--> array of timestamps, size = (N_times)
    pos = data["pos"] #--> np.array_of_position_data, size = (N_times,N_dim)
```

### How to import your data:
With an `Agent` initialised, import your trajectory by calling
```python
    Agent.import_trajectory(dataset="my_trajectory_data") #or whatever the name is
```
Now, when you update `Agent` (`Agent.update()`) its position and velocity will take steps along this trajectory, the random motion policy will not be used. Note, `Agent.dt` can *still* be arbitrary since the provided trajectory data is "interpolated" using a cubic spline technique. Note if you simulate the `Agent` for longer than the time available at the end it just loops becak to the start (t = t % t_max).

Note it is also possible to load data *without* first saving it as a `.npz`, using 
```python
Agent.import_trajectory(times=numpy_array_of_times,
                        positions=numpy_array_of_positions))
```

### Sargolini dataset

The provided dataset "sargolini.npz" was downloaded from public available data here https://www.ntnu.edu/kavli/research/grid-cell-data  and republished here (with minor preprocessing, to extract only the trajaectory relevant data and save as `.npz`, with permission from Profs. Edvard Moser and Francesca Sorgolini) . If you use this dataset remember to reference this website in your methods and cite the paper where it came from (Sargolini et al. (2006) DOI:10.1126/science.1125572). Import the sargolini dataset using
```python
    Agent.import_trajectory(dataset="sargolini") 
```


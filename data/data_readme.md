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
With an Agent initialised, import your trajectory by calling
```python
    Agent.import_trajectory(dataset="my_trajectory_data") #or whatever the name is
```
Now, when you update Agent (Agent.update()) it's position and velocity will take steps along this trajectory. Note, Agent.dt can sSTILL be anything you like. This trajectory is "interpolated" using a cubic spline technique. Note if you simulate the Agent for longer than the time available at teh end it just loops becak to the start (t = t % t_max)

### Sargolini data

The provided dataset "sargolini.npz" was taken and reformatted from public available data here https://www.ntnu.edu/kavli/research/grid-cell-data. If you use this dataset remember to reference this website in your methods and cite the paper where it came from (Sargolini et al. (2006) DOI:10.1126/science.1125572). Import the sargolini dataset using
```python
    Agent.import_trajectory(dataset="sargolini") 
```


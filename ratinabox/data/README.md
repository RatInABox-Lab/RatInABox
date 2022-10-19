## Import trajectory data
With an `Agent` initialised, import your trajectory by calling
```python
Agent.import_trajectory(times=numpy_array_of_times,
                        positions=numpy_array_of_positions))
```
Now, when you update `Agent` (`Agent.update()`) its position and velocity will take steps along this trajectory, the random motion policy will not be used. Note, `Agent.dt` can *still* be arbitrary since the provided trajectory data is "interpolated" using a cubic spline technique. Note if you simulate the `Agent` for longer than the time available at the end it just loops becak to the start (t = t % t_max).

Note we also provide premade datasets, each a `.npz` file within this directory listed and described below, which can be imported as follows:
```python
    Agent.import_trajectory(dataset="dataset_name")
```
You are welcome to contribute your own data if you would like. 


## Contribute a dataset 
If you would lie to contribute a dataset to this directory, open a pull request by saving a dtafile `my_trajectory_dataset.npz` into this folder.
The file itself must be a list of timestamps (key = "t") and a list of positions (key = "pos") which must be in metres, i.e. save your data as follows
python
```
    t = ... #array of timestamps, size = (N_times)
    pos = ... #np.array_of_position_data, size = (N_times,N_dim)
    np.savez("<ratinabox-dir>/data/my_trajectory_data.npz", #or whatever you want to call it
              t=t, 
              pos=pos)
```

## List of datasets: 
### **Sargolini**
The provided dataset "sargolini.npz" was downloaded from public available data here https://www.ntnu.edu/kavli/research/grid-cell-data  and republished here with minor preprocessing, to extract only the trajaectory relevant data, with permission from Profs. Edvard Moser and Francesca Sorgolini. If you use this dataset remember to reference this website in your methods and cite the paper where it came from (Sargolini et al. (2006) DOI:10.1126/science.1125572). The room size is 1 m x 1 m and the total amount of data is 600 s.


### **Tanni**
The provided dataset "tanni.npz" was downloaded from public available data here https://rdr.ucl.ac.uk/articles/dataset/Supporting_data_for_State_transitions_in_the_statistically_stable_place_cell_population_are_determined_by_rate_of_perceptual_change/18128891/1 and republished here (with minor preprocessing, to extract only the trajaectory relevant data and save as `.npz`, with permission from Prof. Caswell Barry) . If you use this dataset remember to reference this website in your methods and cite the paper where it came from (Tanni et al. (2022) DOI: 10.1016/j.cub.2022.06.046). The room size is 2.5 m x 3.5 m and the total amount of data is 7323 s.
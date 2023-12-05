# Africa 

This animation was made by students on the [TReND in Africa computational neuroscience summer school](https://trendinafrica.org/computational-neuroscience-basics/). Code is provided below. 

<img src="https://raw.githubusercontent.com/RatInABox-Lab/RatInABox/main/.images/readme/riab_africa.gif" width=1000>


```python
# Step 1 get the coordinates of the boundary of Africa (
# These are stored as a massive list 
 africa_boundary = np.array([
[-0.83935318,  0.30607098],
[-0.83532515,  0.18512068],
[-0.84741901,  0.06820623],
[-0.75063883, -0.06080795],
[-0.61756729, -0.16160638],
[-0.48852379, -0.1414467 ],
[-0.43207275, -0.13338087],
[-0.29093539, -0.10919315],
[-0.27076592, -0.16160638],
[-0.17398574, -0.16160638],
[-0.16591991, -0.25836702],
[-0.20221126, -0.31480827],
[-0.09736524, -0.41559693],
[-0.0651117 , -0.58493048],
[-0.12560053, -0.70991663],
[-0.07720555, -0.82682718],
[-0.01671671, -1.02034844],
[ 0.05183795, -1.1251749 ],
[ 0.03536411, -1.19568492],
[ 0.08340712, -1.21970643],
[ 0.18979787, -1.19225328],
[ 0.26873546, -1.18195835],
[ 0.40944265, -1.0344172 ],
[ 0.40944265, -0.95893081],
[ 0.47464389, -0.91776086],
[ 0.47464389, -0.7907998 ],
[ 0.62222414, -0.67071181],
[ 0.57760299, -0.40650454],
[ 0.69428866, -0.25210011],
[ 0.76293132, -0.21779345],
[ 0.88991193, -0.00505106],
[ 0.87618536,  0.03955053],
[ 0.71144688, -0.00848271],
[ 0.67026715,  0.0292556 ],
[ 0.59132956,  0.13905648],
[ 0.56387641,  0.20768056],
[ 0.52955997,  0.22483683],
[ 0.53642326,  0.30375487],
[ 0.40944265,  0.51992107],
[ 0.46434895,  0.47531556],
[ 0.47464389,  0.52678436],
[ 0.44719073,  0.57138987],
[ 0.34423163,  0.57482151],
[ 0.33050506,  0.5542336 ],
[ 0.13488179,  0.61599635],
[ 0.08683877,  0.59884009],
[ 0.09027041,  0.54737129],
[ 0.05938562,  0.53021503],
[-0.02641526,  0.58511449],
[-0.15682752,  0.65030791],
[-0.12937436,  0.68118879],
[-0.14996423,  0.75324452],
[-0.21517524,  0.74295057],
[-0.42108367,  0.72579528],
[-0.46570482,  0.68118879],
[-0.54120099,  0.68462044],
[-0.56866392,  0.70863901],
[-0.59611707,  0.64687724],
[-0.66474996,  0.61599635],
[-0.67162302,  0.54394063],
[-0.7402559 ,  0.46502259],
[-0.79174034,  0.40669148],
[-0.82948843,  0.32777344]])
accra = np.array([-0.4,-0.1])

# Step 2: create an Environment object with the boundary of Africa as the boundary
Africa = Environment(params={
    'dx': 0.05,
    'boundary': africa_boundary})


# Step 3: create many Agents (or students) within the Environment you just created
# put them all in a list called Students
Ag0 = Agent(Africa)
Ag1 = Agent(Africa)
Ag2 = Agent(Africa)
Ag3 = Agent(Africa)
Ag4 = Agent(Africa)
Ag5 = Agent(Africa)
Ag6 = Agent(Africa)
Ag7 = Agent(Africa)
Ag8 = Agent(Africa)
Students = [Ag0,Ag1,Ag2,Ag3,Ag4,Ag5,Ag6,Ag7,Ag8]

# Step 4: Simulate!
for i in range(int(60/dt)):
    for Student in Students:
        if Student.t < 30: # initially students just randomly drift around Africa
            Student.update(dt=dt)
        else: # after 30 seconds students drift towards Accra
            vector_to_accra = accra - Student.pos
            vector_to_accra = vector_to_accra / np.linalg.norm(vector_to_accra)
            drift_velocity = 0.1 * vector_to_accra
            Student.update(dt=dt,drift_velocity=drift_velocity)

# Step 5: Plot the trajectories of all the students
Students[0].plot_trajectory(plot_all_agents=True)


# or animate but...this may be VERY slow to run on Google Colab.
Students[0].animate_trajectory(speed_up=5,fps=10,plot_all_agents=True,autosave=True)
```

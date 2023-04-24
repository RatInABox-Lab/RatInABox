import ratinabox

import copy
import pprint
import numpy as np
import os
import matplotlib
from matplotlib import pyplot as plt


from ratinabox import utils

"""AGENT"""


class Agent:
    """This class defines an Agent which moves around the Environment.
    Specifically this class handles the movement policy and communicates with the Environment class to ensure the Agent's movement obeys boundaries and walls etc.

    Must be initialised with the Environment in which it lives and a params dictionary containing key parameters required for the motion model.
    The most important function "update(dt)" updates the position/velocity of the agent along in time by dt.

    A default parameters dictionary (with descriptions) can be fount in __init__()

    List of functions:
        • update()
        • import_trajectory()
        • plot_trajectory()
        • animate_trajectory()
        • plot_position_heatmap()
        • plot_histogram_of_speeds()
        • plot_histogram_of_rotational_velocities()
        • save_to_history()
        • reset_history()

    The default params for this agent are:
        default_params = {
            "dt": 0.01,
            "speed_coherence_time": 0.7,
            "speed_mean": 0.08,
            "speed_std": 0.08,
            "rotational_velocity_coherence_time": 0.08,
            "rotational_velocity_std": 120 * (np.pi / 180),
            "thigmotaxis": 0.5,
            "wall_repel_distance": 0.1,
            "walls_repel": True,
            "save_history":True,


        }
    """

    default_params = {
        "dt": 0.01,
        # Speed params (leave empty if you are importing trajectory data)
        # These defaults are fit to match data from Sargolini et al. (2016)
        # also given are the parameter names as refered to in the methods section of the paper
        "speed_coherence_time": 0.7,  # time over which speed decoheres, τ_v1 & τ_v2
        "speed_mean": 0.08,  # mean of speed, σ_v2 μ_v1
        "speed_std": 0.08,  # std of speed (meaningless in 2D where speed ~rayleigh), σ_v1
        "rotational_velocity_coherence_time": 0.08,  # time over which speed decoheres, τ_w
        "rotational_velocity_std": (
            120 * (np.pi / 180)
        ),  # std of rotational speed, σ_w wall following parameter
        "thigmotaxis": 0.5,  # tendency for agents to linger near walls [0 = not at all, 1 = max]
        "wall_repel_distance": 0.1,
        "walls_repel": True,  # whether or not the walls repel
        "save_history": True,  # whether to save position and velocity history as you go
    }

    def __init__(self, Environment, params={}):
        """Initialise Agent, takes as input a parameter dictionary.
        Any values not provided by the params dictionary are taken from a default dictionary below.

        Args:
            params (dict, optional). Defaults to {}.
        """
        self.Environment = Environment
        self.Environment.Agents.append(self)

        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        utils.update_class_params(self, self.params, get_all_defaults=True)
        utils.check_params(self, params.keys())

        # initialise history dataframes
        self.history = {}
        self.history["t"] = []
        self.history["pos"] = []
        self.history["vel"] = []
        self.history["rot_vel"] = []

        # time and runID
        self.t = 0
        self.distance_travelled = 0
        self.average_measured_speed = max(self.speed_mean, self.speed_std)
        self.use_imported_trajectory = False

        # motion model stufff
        self.walls_repel = True  # over ride to switch of wall repulsion

        # initialise starting positions and velocity
        if self.Environment.dimensionality == "2D":
            self.pos = self.Environment.sample_positions(n=1, method="random")[0]
            direction = np.random.uniform(0, 2 * np.pi)
            self.velocity = self.speed_mean * np.array(
                [np.cos(direction), np.sin(direction)]
            )
            self.rotational_velocity = 0

        if self.Environment.dimensionality == "1D":
            self.pos = self.Environment.sample_positions(n=1, method="random")[0]
            self.velocity = np.array([self.speed_mean])
            if self.Environment.boundary_conditions == "solid":
                if self.speed_mean != 0:
                    print(
                        "Warning: You have solid 1D boundary conditions and non-zero speed mean."
                    )

        if ratinabox.verbose is True:
            print(
                f"""An Agent has been successfully initialised with the following parameters {self.params}.
                Use Ag.update() to move the Agent.
                Positions and velocities are saved into the Agent.history dictionary.
                Import external trajectory data using Ag.import_trajectory(). Plot trajectory using Ag.plot_trajectory().
                Other plotting functions are available."""
            )
        return

    @classmethod
    def get_all_default_params(cls, verbose=False):
        """Returns a dictionary of all the default parameters of the class, including those inherited from its parents."""
        all_default_params = utils.collect_all_default_params(cls)
        if verbose:
            pprint.pprint(all_default_params)
        return all_default_params

    def update(self, dt=None, drift_velocity=None, drift_to_random_strength_ratio=1):
        """Movement policy update.
        In principle this does a very simple thing:
        • updates time by dt
        • updates velocity (speed and direction) according to a movement policy
        • updates position along the velocity direction
        In reality it's a complex function as the policy requires checking for immediate or upcoming collisions with all walls at each step as well as
        handling boundary conditions.
        Specifically the full loop looks like this:
        1) Update time by dt
        2) Update velocity for the next time step.
           In 2D this is done by varying the agents heading direction and speed according to ornstein-uhlenbeck processes.
           In 1D, simply the velocity is varied according to ornstein-uhlenbeck. This includes, if turned on, being repelled by the walls.
        3) Propose a new position (x_new =? x_old + velocity.dt)
        3.1) Check if this step collides with any walls (and act accordingly)
        3.2) Check you distance and direction from walls and be repelled by them is necessary
        4) Check position is still within maze and handle boundary conditions appropriately
        6) Store new position and time in history data frame
        """
        if dt == None:
            dt = self.dt
        self.dt = dt
        self.t += dt
        self.velocity = self.velocity.astype(float)
        self.pos = np.array(
            self.pos
        )  # check pos is an array (may have external been set as a list)

        if self.use_imported_trajectory == False:  # use random motion model
            if self.Environment.dimensionality == "2D":
                # UPDATE VELOCITY there are a number of contributing factors
                # 1 Stochastically update the direction
                self.rotational_velocity += utils.ornstein_uhlenbeck(
                    dt=dt,
                    x=self.rotational_velocity,
                    drift=0,
                    noise_scale=self.rotational_velocity_std,
                    coherence_time=self.rotational_velocity_coherence_time,
                )
                dtheta = self.rotational_velocity * dt
                self.velocity = utils.rotate(self.velocity, dtheta)

                # 2 Stochastically update the speed
                speed = np.linalg.norm(self.velocity)
                if speed == 0:  # add tiny velocity in [1,0] direction to avoid nans
                    self.velocity, speed = 1e-8 * np.array([1, 0]), 1e-8

                normal_variable = utils.rayleigh_to_normal(speed, sigma=self.speed_mean)
                new_normal_variable = normal_variable + utils.ornstein_uhlenbeck(
                    dt=dt,
                    x=normal_variable,
                    drift=0,
                    noise_scale=1,
                    coherence_time=self.speed_coherence_time,
                )
                speed_new = utils.normal_to_rayleigh(
                    new_normal_variable, sigma=self.speed_mean
                )
                self.velocity = (speed_new / speed) * self.velocity

                # Deterministically drift velocity towards the drift_velocity which has been passed into the update function
                if drift_velocity is not None:
                    self.velocity += utils.ornstein_uhlenbeck(
                        dt=dt,
                        x=self.velocity,
                        drift=drift_velocity,
                        noise_scale=0,
                        coherence_time=self.speed_coherence_time
                        / drift_to_random_strength_ratio,  # <--- this controls how "powerful" this signal is
                    )

                # Deterministically drift the velocity away from any nearby walls
                if (self.walls_repel == True) and (len(self.Environment.walls > 0)):
                    vectors_from_walls = self.Environment.vectors_from_walls(
                        self.pos
                    )  # shape=(N_walls,2)
                    if len(self.Environment.walls) > 0:
                        distance_to_walls = np.linalg.norm(vectors_from_walls, axis=-1)
                        normalised_vectors_from_walls = (
                            vectors_from_walls
                            / np.expand_dims(distance_to_walls, axis=-1)
                        )
                        x, d, v = (
                            distance_to_walls,
                            self.wall_repel_distance,
                            self.speed_mean,
                        )

                        """Wall repulsion and wall following works as follows:
                        When an agent is near the wall, the acceleration and velocity of a hypothetical spring mass tied to a line self.wall_repel_distance away from the wall is calculated.
                        The spring constant is calibrated so that if if starts with the Agent.speed_mean it will ~~just~~ not hit the wall.
                        Now, either the acceleration can be used to update the velocity and guide the agent away from the wall OR the counteracting velocity can be used to update the agents position and shift it away from the wall. Both result in repulsive motion away from the wall.
                        The difference is that the latter (and not the former) does not update the agents velocity vector to reflect this, in which case it continues to walk (unsuccessfully) in the same direction barging into the wall and 'following' it.
                        The thigmotaxis parameter allows us to divvy up which of these two dominate.
                        If thigmotaxis is low the acceleration-gives-velocity-update is most dominant and the agent will not linger near the wall.
                        If thigmotaxis is high the velocity-gives-position-update is most dominant and the agent will linger near the wall."""

                        """Spring acceletation model:
                        In this case this is done by applying an acceleration whenever the agent is near to a wall.
                        This acceleration matches that of a spring with spring constant 3x that of a spring which would, if the agent arrived head on at v = self.speed_mean, turn around exactly at the wall.
                        This is solved by letting d2x/dt2 = -k.x where k = v**2/d**2 (v=seld.speed_mean, d = self.wall_repel_distance)

                        See paper for full details"""

                        spring_constant = v**2 / d**2
                        wall_accelerations = np.piecewise(
                            x=x,
                            condlist=[
                                (x <= d),
                                (x > d),
                            ],
                            funclist=[
                                lambda x: spring_constant * (d - x),
                                lambda x: 0,
                            ],
                        )
                        wall_acceleration_vecs = (
                            np.expand_dims(wall_accelerations, axis=-1)
                            * normalised_vectors_from_walls
                        )
                        wall_acceleration = wall_acceleration_vecs.sum(axis=0)
                        dv = wall_acceleration * dt
                        self.velocity += 3 * ((1 - self.thigmotaxis) ** 2) * dv

                        """Conveyor belt drift model.
                        Instead of a spring model this is like a converyor belt model.
                        When the agent is < wall_repel_distance from the wall the agents position is updated as though it were on a conveyor belt which moves at the speed of spring mass attached to the wall with starting velocity 5*self.speed_mean.
                        This has a similar effect effect  as the spring model above in that the agent moves away from the wall BUT, crucially the update is made directly to the agents position, not it's speed, so the next time step will not reflect this update.
                        As a result the agent which is walking into the wall will continue to barge hopelessly into the wall causing it the "hug" close to the wall."""
                        wall_speeds = np.piecewise(
                            x=x,
                            condlist=[
                                (x <= d),
                                (x > d),
                            ],
                            funclist=[
                                lambda x: v * (1 - np.sqrt(1 - (d - x) ** 2 / d**2)),
                                lambda x: 0,
                            ],
                        )
                        wall_speed_vecs = (
                            np.expand_dims(wall_speeds, axis=-1)
                            * normalised_vectors_from_walls
                        )
                        wall_speed = wall_speed_vecs.sum(axis=0)
                        dx = wall_speed * dt
                        self.pos += 6 * (self.thigmotaxis**2) * dx

                # proposed position update
                proposed_new_pos = self.pos + self.velocity * dt
                proposed_step = np.array([self.pos, proposed_new_pos])
                wall_check = self.Environment.check_wall_collisions(proposed_step)
                walls = wall_check[0]  # shape=(N_walls,2,2)
                wall_collisions = wall_check[1]  # shape=(N_walls,)

                if (wall_collisions is None) or (True not in wall_collisions):
                    # it is safe to move to the new position
                    self.pos = self.pos + self.velocity * dt

                # Bounce off walls you collide with
                elif True in wall_collisions:
                    colliding_wall = walls[np.argwhere(wall_collisions == True)[0][0]]
                    self.velocity = utils.wall_bounce(self.velocity, colliding_wall)
                    self.velocity = (
                        0.5 * self.speed_mean / (np.linalg.norm(self.velocity))
                    ) * self.velocity
                    self.pos += self.velocity * dt

                # handles instances when agent leaves environmnet
                if (
                    self.Environment.check_if_position_is_in_environment(self.pos)
                    is False
                ):
                    self.pos = self.Environment.apply_boundary_conditions(self.pos)

                # calculate the velocity of the step that, after all that, was taken.
                if len(self.history["vel"]) >= 1:
                    last_pos = np.array(self.history["pos"][-1])
                    shift = self.Environment.get_vectors_between___accounting_for_environment(
                        pos1=self.pos, pos2=last_pos
                    )
                    self.save_velocity = (
                        shift.reshape(-1) / self.dt
                    )  # accounts for periodic
                else:
                    self.save_velocity = self.velocity

            elif self.Environment.dimensionality == "1D":
                self.pos = self.pos + dt * self.velocity
                if (
                    self.Environment.check_if_position_is_in_environment(self.pos)
                    is False
                ):
                    if self.Environment.boundary_conditions == "solid":
                        self.velocity *= -1
                    self.pos = self.Environment.apply_boundary_conditions(self.pos)

                self.velocity += utils.ornstein_uhlenbeck(
                    dt=dt,
                    x=self.velocity,
                    drift=self.speed_mean,
                    noise_scale=self.speed_std,
                    coherence_time=self.speed_coherence_time,
                )
                self.save_velocity = self.velocity

        elif self.use_imported_trajectory == True:
            # use an imported trajectory to
            if (
                self.interpolate is True
            ):  # interpolate along the trajectory by an amount dt
                if self.Environment.dimensionality == "2D":
                    interp_time = self.t % max(self.t_interp)
                    pos = self.pos_interp(interp_time)
                    ex = self.Environment.extent
                    self.pos = np.array(
                        [min(max(pos[0], ex[0]), ex[1]), min(max(pos[1], ex[2]), ex[3])]
                    )

                    # calculate velocity and rotational velocity
                    if len(self.history["vel"]) >= 1:
                        last_pos = np.array(self.history["pos"][-1])
                        shift = self.Environment.get_vectors_between___accounting_for_environment(
                            pos1=self.pos, pos2=last_pos
                        )
                        self.velocity = (
                            shift.reshape(-1) / self.dt
                        )  # accounts for periodic
                    else:
                        self.velocity = np.array([0, 0])
                    self.save_velocity = self.velocity

                    angle_now = utils.get_angle(self.velocity)
                    if len(self.history["vel"]) >= 1:
                        angle_before = utils.get_angle(self.history["vel"][-1])
                    else:
                        angle_before = angle_now
                    if abs(angle_now - angle_before) > np.pi:
                        if angle_now > angle_before:
                            angle_now -= 2 * np.pi
                        elif angle_now < angle_before:
                            angle_before -= 2 * np.pi
                    self.rotational_velocity = (angle_now - angle_before) / self.dt

                if self.Environment.dimensionality == "1D":
                    interp_time = self.t % max(self.t_interp)
                    pos = self.pos_interp(interp_time)
                    ex = self.Environment.extent
                    self.pos = np.array([min(max(pos, ex[0]), ex[1])])
                    if len(self.history["vel"]) >= 1:
                        self.velocity = (self.pos - self.history["pos"][-1]) / self.dt
                    else:
                        self.velocity = np.array([0])
                    self.save_velocity = self.velocity
            else:  # just jump one count along the trajectory
                self.t = self.times[self.imported_trajectory_id]
                pos = self.positions[self.imported_trajectory_id]
                ex = self.Environment.extent
                if self.Environment.dimensionality == "1D":
                    self.pos = np.array([min(max(pos, ex[0]), ex[1])])
                    if len(self.history["vel"]) >= 1:
                        self.velocity = (self.pos - self.history["pos"][-1]) / self.dt
                    else:
                        self.velocity = np.array([0])
                if self.Environment.dimensionality == "2D":
                    self.pos = np.array(
                        [min(max(pos[0], ex[0]), ex[1]), min(max(pos[1], ex[2]), ex[3])]
                    )
                    if len(self.history["vel"]) >= 1:
                        self.velocity = (self.pos - self.history["pos"][-1]) / self.dt
                    else:
                        self.velocity = np.array([0, 0])
                self.save_velocity = self.velocity
                self.imported_trajectory_id = (self.imported_trajectory_id + 1) % len(
                    self.times
                )

        if len(self.history["pos"]) >= 1:
            self.distance_travelled += np.linalg.norm(
                self.Environment.get_vectors_between___accounting_for_environment(
                    self.pos, np.array(self.history["pos"][-1])
                )
            )
            tau_speed = 10
            self.average_measured_speed = (
                1 - dt / tau_speed
            ) * self.average_measured_speed + (dt / tau_speed) * np.linalg.norm(
                self.save_velocity
            )

        # write to history
        if self.save_history is True:
            self.save_to_history()

        return

    def save_to_history(self):
        self.history["t"].append(self.t)
        self.history["pos"].append(list(self.pos))
        self.history["vel"].append(list(self.save_velocity))
        if self.Environment.dimensionality == "2D":
            self.history["rot_vel"].append(self.rotational_velocity)
        return

    def reset_history(self):
        for key in self.history.keys():
            self.history[key] = []
        return

    def import_trajectory(
        self, times=None, positions=None, dataset=None, interpolate=True
    ):
        """Import trajectory data into the agent by passing a list or array of timestamps and a list or array of positions.
        These will used for moting rather than the random motion model. The data is interpolated using cubic splines.
        This means imported data can be low resolution and smoothly upsampled (aka "augmented" with artificial data). Interpolation can be turned off, in which case each time Ag.update() is called the Agent just moves one count along the imported trajectory (no matter how coarse this is), this may be a lot quicker in cases when your imported behaviour data is high resolution.

        Note after importing trajectory data you still need to run a simulation using the Agent.update(dt=dt) function.
        Each update moves the agent by a time dt along its imported trajectory.
        If the simulation is run for longer than the time availble in the imported trajectory, it loops back to the start.
        Imported times are shifted so that time[0] = 0.

        Args:
            times (array-like): list or array of time stamps
            positions (_type_): list or array of positions
            dataset: if `sargolini' will load `sargolini' trajectory data from './data/sargolini.npz' (Sargolini et al. 2006).
               Else you can pass a path to a .npz file which must contain time and trajectory data under keys 't' and 'pos'
            interpolate (bool, True): Whether to smoothyl interpolate this trajectory or not.
        """
        from scipy.interpolate import interp1d

        self.interpolate = interpolate
        assert (
            self.Environment.boundary_conditions == "solid"
        ), "Only solid boundary conditions are supported"

        if dataset is not None:
            import ratinabox
            import os

            if dataset == "sargolini":
                print(
                    """Attempting to import Sargolini locomotion dataset.
                    Please cite Sargolini et al. (2006) DOI:10.1126/science.1125572 if you use this in your work.
                    The full dataset (along with many more) can be found here https://www.ntnu.edu/kavli/research/grid-cell-data
                    The exact datafile being used is 8F6BE356-3277-475C-87B1-C7A977632DA7_1/11084-03020501_t2c1.mat"""
                )
            dataset = os.path.join(
                os.path.join(
                    os.path.abspath(os.path.join(ratinabox.__file__, os.pardir)),
                    "data",
                ),
                dataset + ".npz",
            )
            print(dataset)
            try:
                data = np.load(dataset)
            except FileNotFoundError:
                print(
                    f"IMPORT FAILED. No datafile found at {dataset}. Please try a different one. For now the default inbuilt random policy will be used."
                )
                return
            times = data["t"]
            positions = data["pos"]
            print(f"Successfully imported dataset from {dataset}")
        else:
            if (times is not None) and (positions is not None):
                times, positions = np.array(times), np.array(positions)
                print("Successfully imported dataset from arrays passed")
            else:
                print("No data passed, provided arguments 'times' and 'positions'")

        assert len(positions) == len(
            times
        ), "time and position arrays must have same length"

        times = times - min(times)
        print(f"Total of {times[-1]:.1f} s of data available")

        self.use_imported_trajectory = True

        ex = self.Environment.extent

        if self.Environment.dimensionality == "2D":
            positions = positions.reshape(-1, 2)
            if (
                (max(positions[:, 0]) > ex[1])
                or (min(positions[:, 0]) < ex[0])
                or (max(positions[:, 1]) > ex[3])
                or (min(positions[:, 1]) < ex[2])
            ):
                print(
                    f"""WARNING: the size of the trajectory is significantly larger than the environment you are using.
                    The Environment extent is [minx,maxx,miny,maxy]=[{ex[0]:.1f},{ex[1]:.1f},{ex[2]:.1f},{ex[3]:.1f}], whereas extreme coords are [{min(positions[:,0]):.1f},{max(positions[:,0]):.1f},{min(positions[:,1]):.1f},{max(positions[:,1]):.1f}].
                    Recommended to use larger environment."""
                )
            self.t_interp = times

            if interpolate is True:
                self.pos_interp = interp1d(
                    times, positions, axis=0, kind="cubic", fill_value="extrapolate"
                )
            else:
                self.positions = positions
                self.times = times
                self.imported_trajectory_id = 0

        if self.Environment.dimensionality == "1D":
            positions = positions.reshape(-1, 1)
            if (max(positions) > ex[1]) or (min(positions) < ex[0]):
                print(
                    f"""WARNING: the size of the trajectory is significantly larger than the environment you are using.
                    The Environment extent is [minx,maxx]=[{ex[0]:.1f},{ex[1]:.1f}], whereas extreme coords are [{min(positions[:,0]):.1f},{max(positions[:,0]):.1f}].
                    Recommended to use larger environment."""
                )
            self.t_interp = times
            if interpolate is True:
                self.pos_interp = interp1d(
                    times, positions, axis=0, kind="cubic", fill_value="extrapolate"
                )
            else:
                self.positions = positions
                self.times = times
                self.imported_trajectory_id = 0

        return

    def plot_trajectory(
        self,
        t_start=0,
        t_end=None,
        framerate=10,
        fig=None,
        ax=None,
        plot_all_agents=False,
        point_size=15,
        decay_point_size=False,
        decay_point_timescale=10,
        plot_agent=True,
        color=None,
        alpha=0.7,
        xlim=None,
        background_color=None,
        axis_labels=True,
        autosave=None,
        **kwargs,
    ):
        """Plots the trajectory between t_start (seconds) and t_end (defaulting to the last time available)
        Args:
            • t_start: start time in seconds
            • t_end: end time in seconds (default = self.history["t"][-1])
            • framerate: how many scatter points / per second of motion to display
            • fig, ax: the fig, ax to plot on top of, optional, if not provided used self.Environment.plot_Environment().
              This can be used to plot trajectory on top of receptive fields etc.
            • plot_all_agents: if True, this will plot the trajectory of all agents in the list Environment.Agents
            • point_size: size of scatter points
            • decay_point_size: decay trajectory point size over time (recent times = largest)
            • decay_point_timescale: if decay_point_size is True, this is the timescale over which sizes decay
            • plot_agent: dedicated point show agent current position
            • color: plot point color, if color == 'changing' will smoothly change trajectory color from start to finish
            • alpha: plot point opaqness
            • xlim: In 1D, forces the xlim to be a certain time (minutes) (useful if animating this function)
            • background_color: color of the background if not matplotlib default, only for 1D (probably white)
            • axis_labels: whether to show axes labels
            • autosave: if True, will try to save the figure to the figure directory `ratinabox.figure_directory`. Defaults to None in which case looks for global constant ratinabox.autosave_plots

        Returns:
            fig, ax
        """
        # loop over all agents in the Environment if plot_all_agents is True
        if plot_all_agents == False:
            agent_list = [self]
            if color is None:
                color = "#7b699a"
        else:
            agent_list = self.Environment.Agents
        replot_env = True
        for i, self_ in enumerate(agent_list):
            dt = self_.dt
            t, pos = np.array(self_.history["t"]), np.array(self_.history["pos"])
            if t_end == None:
                t_end = t[-1]
            startid = np.nanargmin(np.abs(t - (t_start)))
            endid = np.nanargmin(np.abs(t - (t_end)))
            if self_.Environment.dimensionality == "2D":
                skiprate = max(1, int((1 / framerate) / dt))
                trajectory = pos[startid:endid, :][::skiprate]
            if self_.Environment.dimensionality == "1D":
                skiprate = max(1, int((1 / framerate) / dt))
                trajectory = pos[startid:endid][::skiprate]
            time = t[startid:endid][::skiprate]
            if color is None:
                color_list = [f"C{i}"] * len(time)
            elif color == "changing":
                trajectory_cmap = matplotlib.colormaps["viridis_r"]
                color_list = [trajectory_cmap(t / len(time)) for t in range(len(time))]
                decay_point_size = (
                    False  # if changing colour, may as well show WHOLE trajectory
                )
            else:
                color_list = [color] * len(time)

            if self_.Environment.dimensionality == "2D":
                if replot_env == True:
                    fig, ax = self_.Environment.plot_environment(
                        fig=fig, ax=ax, autosave=False
                    )
                replot_env = False
                s = point_size * np.ones_like(time)
                if decay_point_size == True:
                    s = point_size * np.exp((time - time[-1]) / decay_point_timescale)
                    s[(time[-1] - time) > (1.5 * decay_point_timescale)] *= 0

                if plot_agent == True:
                    s[-1] = 40
                    color_list[-1] = "r"

                ax.scatter(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    s=s,
                    alpha=alpha,
                    zorder=1.1,
                    c=color_list,
                    linewidth=0,
                )
                # #plot the rat? TODO haha probably never gonna do this
                # ratpath = os.path.join(
                #     os.path.abspath(os.path.join(ratinabox.__file__, os.pardir)),
                #         "data/rat.png",
                #     )
                # rat = plt.imread(ratpath)
                # rect = 0.5, 0.4, 0.4, 0.4 # What should these values be?
                # newax = fig.add_axes(rect, anchor='NE', zorder=1)
                # newax.axis('off')
                # newax.imshow(rat)

            if self_.Environment.dimensionality == "1D":
                if fig is None and ax is None:
                    fig, ax = plt.subplots(figsize=(3, 1.5))
                ax.scatter(
                    time / 60, trajectory, alpha=alpha, linewidth=0, c=color_list, s=5
                )
                ax.spines["left"].set_position(("data", t_start / 60))
                if axis_labels == True:
                    ax.set_xlabel("Time / min")
                    ax.set_ylabel("Position / m")
                ax.set_xlim([t_start / 60, t_end / 60])
                if xlim is not None:
                    ax.set_xlim(right=xlim)

                ax.set_ylim(bottom=0, top=self_.Environment.extent[1])
                ax.spines["right"].set_color(None)
                ax.spines["top"].set_color(None)
                ax.set_xticks([t_start / 60, t_end / 60])
                ex = self_.Environment.extent
                ax.set_yticks([ex[1]])
                if background_color is not None:
                    ax.set_facecolor(background_color)
                    fig.patch.set_facecolor(background_color)

        ratinabox.utils.save_figure(fig, "trajectory", save=autosave)

        return fig, ax

    def animate_trajectory(
        self, t_start=None, t_end=None, fps=15, speed_up=1, autosave=None, **kwargs
    ):
        """Returns an animation (anim) of the trajectory, 25fps.
        Args:
            t_start: Agent time at which to start animation
            t_end (_type_, optional): _description_. Defaults to None.
            fps: frames per second of end video
            speed_up: #times real speed animation should come out at
            autosave (bool): whether to automatical try and save this. Defaults to None in which case looks for global constant ratinabox.autosave_plots
            kwargs: passed to trajectory plotting function (chuck anything you wish in here). A particularly useful kwarg is 'additional_plot_func': any function which takes a fig, ax and t as input. The animation wll be passed through this each time after plotting the trajectory, use it to modify your animations however you like

        Returns:
            animation
        """
        plt.rcParams["animation.html"] = "jshtml"  # for animation rendering in juypter

        dt = 1 / fps
        if t_start == None:
            t_start = self.history["t"][0]
        if t_end == None:
            t_end = self.history["t"][-1]

        def animate_(i, fig, ax, t_start, t_max, speed_up, dt, kwargs):
            t_end = t_start + (i + 1) * speed_up * dt
            ax.clear()
            if self.Environment.dimensionality == "2D":
                fig, ax = self.Environment.plot_environment(
                    fig=fig, ax=ax, autosave=False
                )
            fig, ax = self.plot_trajectory(
                t_start=t_start,
                t_end=t_end,
                fig=fig,
                ax=ax,
                decay_point_size=True,
                xlim=t_max / 60,
                autosave=False,
                **kwargs,
            )
            if "additional_plot_func" in kwargs.keys():
                fig, ax = kwargs["additional_plot_func"](
                    fig=fig, ax=ax, t=t_end, **kwargs  # the current time
                )

            plt.close()
            return

        fig, ax = self.plot_trajectory(
            t_start=0, t_end=10 * self.dt, xlim=t_end / 60, autosave=False, **kwargs
        )

        from matplotlib import animation

        anim = matplotlib.animation.FuncAnimation(
            fig,
            animate_,
            interval=1000 * dt,
            frames=int((t_end - t_start) / (dt * speed_up)),
            blit=False,
            fargs=(fig, ax, t_start, t_end, speed_up, dt, kwargs),
        )

        ratinabox.utils.save_animation(anim, "trajectory", save=autosave)

        return anim

    def plot_position_heatmap(
        self,
        dx=None,
        fig=None,
        ax=None,
        autosave=None,
    ):
        """Plots a heatmap of postions the agent has been in.
        vmin is always set to zero, so the darkest colormap color (if seen) represents locations which have never been visited
        Args:
            dx (float, optional): The heatmap bin size. Defaults to 5cm in 2D or 1cm in 1D.
            fig, ax: if provided, will plot onto this
            autosave (bool, optional): If True, will try to save the figure into `ratinabox.figure_directory`. Defaults to None in which case looks for global constant ratinabox.autosave_plots

        """
        if self.Environment.dimensionality == "1D":
            if dx is None:
                dx = 0.01
            pos = np.array(self.history["pos"])
            ex = self.Environment.extent
            if fig is None and ax is None:
                fig, ax = self.Environment.plot_environment(autosave=False)
            heatmap, centres = utils.bin_data_for_histogramming(
                data=pos, extent=ex, dx=dx
            )
            # maybe do smoothing?
            ax.plot(centres, heatmap)
            ax.fill_between(centres, 0, heatmap, alpha=0.3)
            ax.set_ylim(top=np.max(heatmap) * 1.2)
            return fig, ax

        elif self.Environment.dimensionality == "2D":
            if dx is None:
                dx = 0.05
            pos = np.array(self.history["pos"])
            ex = self.Environment.extent
            heatmap = utils.bin_data_for_histogramming(data=pos, extent=ex, dx=dx)
            if fig == None and ax == None:
                fig, ax = self.Environment.plot_environment()
            else:
                _, _ = self.Environment.plot_environment(fig=fig, ax=ax)
            vmin = 0
            vmax = np.max(heatmap)
            ax.imshow(
                heatmap,
                extent=ex,
                interpolation="bicubic",
                vmin=vmin,
                vmax=vmax,
                zorder=0,
            )
        ratinabox.utils.save_figure(fig, "position_heatmap", save=autosave)

        return fig, ax

    def plot_histogram_of_speeds(
        self,
        fig=None,
        ax=None,
        color="C1",
        return_data=False,
        autosave=None,
    ):
        """Plots a histogram of the observed speeds of the agent.
        args:
            fig, ax: not required. the ax object to be drawn onto.
            color: optional. the color.
            return_data: if True, will return the histogram data (bins and patches)
            autosave: if True, will try to save the figure into `ratinabox.figure_directory`. Defaults to None in which case looks for global constant ratinabox.autosave_plots
        Returns:
            fig, ax: the figure
        """
        velocities = np.array(self.history["vel"])
        speeds = np.linalg.norm(velocities, axis=1)
        # exclude speeds above 3sigma
        mu, std = np.mean(speeds), np.std(speeds)
        speeds = speeds[speeds < mu + 3 * std]
        if (fig is None) and (ax is None):
            fig, ax = plt.subplots()
        n, bins, patches = ax.hist(
            speeds, bins=np.linspace(0, 1.2, 100), color=color, alpha=0.8, density=True
        )
        ax.set_xlabel(r"Speed  / $ms^{-1}$")
        ax.set_yticks([])
        ax.set_xlim(left=0, right=8 * std)
        ax.spines["left"].set_color(None)
        ax.spines["right"].set_color(None)
        ax.spines["top"].set_color(None)

        ratinabox.utils.save_figure(fig, "speed_histogram", save=autosave)

        if return_data == True:
            return fig, ax, n, bins, patches
        else:
            return fig, ax

    def plot_histogram_of_rotational_velocities(
        self,
        fig=None,
        ax=None,
        color="C1",
        return_data=False,
        autosave=None,
    ):
        """Plots a histogram of the observed speeds of the agent.
        args:
            fig, ax: not required. the ax object to be drawn onto.
            color: optional. the color.
            return_data: if True, will return the histogram data (bins and patches)
            auto_save: if True, will try to save the figure into `ratinabox.figure_directory`. Defaults to None in which case looks for global constant ratinabox.autosave_plots
        Returns:
            fig, ax: the figure
        """
        rot_vels = np.array(self.history["rot_vel"]) * 180 / np.pi
        # exclude rotational velocities above/below 3sigma
        mu, std = np.mean(rot_vels), np.std(rot_vels)
        rot_vels = rot_vels[rot_vels < mu + 3 * std]
        rot_vels = rot_vels[rot_vels > mu - 3 * std]
        if (fig is None) and (ax is None):
            fig, ax = plt.subplots()
        n, bins, patches = ax.hist(
            rot_vels,
            bins=np.linspace(-2000, 2000, 100),
            color=color,
            alpha=0.8,
            density=False,
        )
        ax.set_yticks([])
        ax.set_xlim(-5 * std, 5 * std)
        ax.spines["left"].set_color(None)
        ax.spines["right"].set_color(None)
        ax.spines["top"].set_color(None)
        ax.set_xlabel(r"Rotational velocity / $^{\circ} s^{-1}$")

        ratinabox.utils.save_figure(fig, "rotational_velocity_histogram", save=autosave)

        if return_data == True:
            return fig, ax, n, bins, patches
        return fig, ax

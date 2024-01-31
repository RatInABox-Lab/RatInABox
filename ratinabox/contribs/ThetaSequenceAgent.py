import ratinabox
from ratinabox.Agent import Agent
from scipy.interpolate import interp1d
import copy
import numpy as np


class ThetaSequenceAgent(Agent):
    """ThetaSequneceAgent is a type of Agent who's position is NOT the true position but instead a "theta sequence" over the position. This starts from behind the "true" position and rapidly moves to infront of the true position (default sequence speed = 5ms-1) once every "theta cycle" (default 10Hz). Each theta sequence is split into the following phases (marked as fraction of the theta cycle):

    |.......A.........|................B..............|.................C.............|........A'.......|
    0              1/2-β/2                           1/2                           1/2+β/2              1

    • A and A': within these segments the position is [nan], the sequence hasn't started yet or has finished.
    • B, "Look behind": The sequence starts behind the agents current position and moves along the historic trajectory until it meets the agent half way through the theta cycle.
    • C, "Look ahead": A new "random" trajectory into the future is sampled starting from the agents current position and velocity.

    The velocity of the sequence, v_sequence, is constant. This is the velocity of the sequence in the reference frame of the TrueAgent (i.e. ground truth see below) so the "apparent" velocity of the sequence will be v_sequence + the speed of the TrueAgent.

    ThetaSequenceAgent has within it two other Agent classes:
        • self.TrueAgent (riab.Agent) is the real Agent moving in the Environment
        • self.ForwardSequenceAgent (riab.Agent) is a sham Agent only used to access riab's stochastic motion model and generate  the forward sequences.

    The default params (beyond the standard Agent params) are:
    default_params = {
        "dt"          : 0.001,  #this MUST be at least 10x smaller than the theta time period
        "theta_freq"  : 10.0, #theta frequency
        "v_sequence"     : 5.0, #sequence speed in reference frame of Agent, ms-1
        "theta_frac"  : 0.5, #fraction of theta cycle over which}
    """

    default_params = {
        "v_sequence": 5.0,  # sequence speed in reference frame of Agent, ms-1
        "theta_freq": 10.0,  # theta frequency
        "theta_frac": 0.5,  # fraction of theta cycle over which
        "dt": 0.001,
    }

    def __init__(self, Environment, params={}):

        self.Environment = Environment

        self.params = copy.deepcopy(__class__.default_params)        
        self.params.update(params)

        super().__init__(Environment, self.params)

        # ground truth Agent
        self.TrueAgent = Agent(self.Environment, self.params)
        self.TrueAgent.history[
            "distance_travelled"
        ] = []  # history of distance travelled
        # a sham Agent we're initialising just in order to do a forward sequence
        self.ForwardSequenceAgent = Agent(self.Environment, self.params)

        # some variables/constants
        self.T_theta = 1 / self.theta_freq
        self.d_half = (
            (self.theta_frac / 2) * self.T_theta * self.v_sequence
        )  # how far agent will travel in half a sequence
        self.last_theta_phase = 0
        self.d_half = (
            (self.theta_frac / 2) * self.T_theta * self.v_sequence
        )  # how far agent will travel in half a sequence

        # its very time consuming to continually convert position data into arrays so we preallocate a memory location
        self.n_half = int(
            2 * self.d_half / (self.TrueAgent.speed_mean * self.dt)
        )  # approx how many steps for the agent to travel d_half in real time
        self.keep_count = (
            20 * self.n_half
        )  # how many data points to save in preallocated memory
        self.recent_data_stash = {}  # its time consuming
        self.recent_data_stash["distance"] = np.zeros(
            (self.keep_count)
        )  # its time consuming
        self.recent_data_stash["position"] = np.zeros(
            (self.keep_count, self.Environment.D)
        )  # its time consuming
        self.recent_data_stash["distance"][0] = self.TrueAgent.distance_travelled
        self.recent_data_stash["position"][0, :] = self.TrueAgent.pos
        self.counter = 1

        assert (
            self.dt < self.T_theta / 10
        ), f"params['dt'] is too large. It must be < 10% of theta time period., i.e. smaller than {self.T_theta/10:.5f}"

    def update(self, dt=None, drift_velocity=None, drift_to_random_strength_ratio=1):
        """
        Updates and saves the position of the Agent along the theta sequence.

        None that this is quite a complicated function! Some complexities which may help you to understand this code include:

        • Achilles and the tortoise: When behind the Agent we can interpolate along historic data but on each step the true agent moves forwards a little, so we must recollect this new data. The ThetaSequenceAgent position is Achilles, the TrueAgent is the tortoise.
        • Interpolation expense: We must interpolate smoothly over historic data but this is expensive since it requires converting the list of past positions into an array then running scipy.interpolate.interp1d. So we want to take the least possible historic data which guarantees we'll have enough to do the behind sequence.
        • Reference frame: In the current model the speed of the sequence is constant (in the reference frame of the TrueAgent) but the speed of the TrueAgent may not be. Therefore it is not enough to just interpolate over the past trajectory (indexed by time), we must transform coordinates to "distance travelled" (which is linear wrt the sequence).
        • Boundary conditions
        """

        # update True position of Agent (ground truth) in normal fashion
        self.TrueAgent.update(
            dt=None, drift_velocity=None, drift_to_random_strength_ratio=1
        )
        self.TrueAgent.history["distance_travelled"].append(
            self.TrueAgent.distance_travelled
        )

        # append TrueAgent position and distance data into our preallocated arrays:
        if self.counter == self.keep_count:
            self.counter = 10 * self.n_half
            self.recent_data_stash["distance"][: self.counter] = self.recent_data_stash[
                "distance"
            ][-self.counter :]
            self.recent_data_stash["position"][
                : self.counter, :
            ] = self.recent_data_stash["position"][-self.counter :, :]
        self.recent_data_stash["distance"][
            self.counter
        ] = self.TrueAgent.distance_travelled
        self.recent_data_stash["position"][self.counter, :] = self.TrueAgent.pos

        self.t = self.TrueAgent.t
        theta_phase = (self.t % (1 / self.theta_freq)) / ((1 / self.theta_freq))
        self.d_half = (
            (self.theta_frac / 2) * self.T_theta * self.v_sequence
        )  # how far agent will travel in half a sequence

        # PRE SWEEP (returns nan's)
        if theta_phase < (0.5 - (self.theta_frac / 2)):
            # No position
            pos = np.full(shape=(self.Environment.D,), fill_value=np.nan)

        # LOOK BEHIND (EARLY SWEEP, from behind to current position, taken from historical data)
        if (theta_phase >= (0.5 - self.theta_frac / 2)) and (theta_phase < 0.5):
            true_distances = self.TrueAgent.history["distance_travelled"]
            # Backwards sequence
            if true_distances[-1] < self.d_half:
                # handle case where not enough data has been collected yet
                # just dont do a backwards sequence and take current positions
                pos = self.TrueAgent.pos
            else:
                # get just enough past data
                lookback = int(
                    5 * self.d_half / (self.dt * self.TrueAgent.average_measured_speed)
                )  # so argmin will never grow arbitrarily large, 3 to be safe
                lookback = min(lookback, self.counter)
                true_positions = self.recent_data_stash["position"][
                    self.counter - lookback + 1 : self.counter + 1, :
                ]
                true_distances = self.recent_data_stash["distance"][
                    self.counter - lookback + 1 : self.counter + 1
                ]
                # interpolate it
                a = np.argmin(true_positions)
                # calculate how far back the current sequence should be look (sequence closing in on Agent at speed v_sequence so net speed of sequence = v_sequence + v_agent)
                # converts current theta phase to how far back along the current trajectory to take position from
                c = self.d_half / self.theta_frac
                m = -2 * c
                distance_back = (
                    m * theta_phase + c
                )  # how far behind the agents current position the sequence should be at
                interp_distance = (
                    true_distances[-1] - distance_back
                )  # and the TrueAgent's actual distance travelled at this point
                idx = np.argmin(np.abs(true_distances - interp_distance))
                self.pos_interp = interp1d(
                    true_distances[idx - 3 : idx + 3],
                    true_positions[idx - 3 : idx + 3],
                    axis=0,
                )
                pos = self.pos_interp(interp_distance)

        # LOOK AHEAD (LATE SWEEP, from current position to infront, stochastically generated)
        if (theta_phase >= 0.5) and (theta_phase < 0.5 + self.theta_frac / 2):
            # Forward sequence
            if (
                theta_phase >= 0.5 and self.last_theta_phase < 0.5
            ):  # catch on first time each loop
                self.ForwardSequenceAgent.pos = self.TrueAgent.pos
                self.ForwardSequenceAgent.history["pos"].append(self.TrueAgent.pos)
                self.ForwardSequenceAgent.velocity = self.TrueAgent.velocity
                self.ForwardSequenceAgent.history["vel"].append(self.TrueAgent.velocity)
                if self.Environment.dimensionality == "2D":
                    self.ForwardSequenceAgent.rotational_velocity = (
                        self.TrueAgent.rotational_velocity
                    )
                    self.ForwardSequenceAgent.history["rot_vel"].append(
                        self.TrueAgent.rotational_velocity
                    )
                self.ForwardSequenceAgent.distance_travelled = (
                    self.TrueAgent.distance_travelled
                )
                recent_speed = self.TrueAgent.average_measured_speed
                forward_distance_to_simulate = (
                    self.d_half
                    + 100 * recent_speed * (self.theta_frac / 2) * self.T_theta
                )
                future_positions = [self.ForwardSequenceAgent.pos]
                future_distances = [self.ForwardSequenceAgent.distance_travelled]
                while (
                    self.ForwardSequenceAgent.distance_travelled
                    < self.TrueAgent.distance_travelled + forward_distance_to_simulate
                ):
                    self.ForwardSequenceAgent.update(
                        dt=self.dt
                        * self.v_sequence
                        / self.TrueAgent.average_measured_speed
                    )
                    future_positions.append(self.ForwardSequenceAgent.pos)
                    future_distances.append(
                        self.ForwardSequenceAgent.distance_travelled
                    )
                future_positions, future_distances = np.array(
                    future_positions
                ), np.array(future_distances)
                self.pos_interp = interp1d(future_distances, future_positions, axis=0)
            # calculate how far forward the current sequence should be look (sequence moving away fromAgent at speed v_sequence so net speed of sequence = v_sequence + v_agent)
            # converts current theta phase to how far forward along the current trajectory to take position from
            c = -self.d_half / self.theta_frac
            m = -2 * c
            distance_ahead = (
                m * theta_phase + c
            )  # how far ahead of the agents current position the sequence should be at
            interp_distance = (
                self.TrueAgent.distance_travelled + distance_ahead
            )  # and the ForwardSequenceAgent's actual distance travelled at this point
            pos = self.pos_interp(interp_distance)

        # POST SWEEP (returns nan's)
        if theta_phase >= (0.5 + (self.theta_frac / 2)):
            # No position
            pos = np.full(shape=(self.Environment.D,), fill_value=np.nan)

        # handle periodic boundaries by just testing if the distance between current and true position of the Agent is over d_half this can only be because the interpolation has crossed a boundary, in which case just set the position to nan (minimally damaging for small dt)
        dist = self.Environment.get_distances_between___accounting_for_environment(
            pos.reshape(1, -1), self.TrueAgent.pos.reshape(1, -1)
        )
        if np.isnan(dist):
            pass
        elif dist > self.d_half:
            # pos = np.array(self.history['pos'][-1])
            pos = np.full(shape=(self.Environment.D,), fill_value=np.nan)

        self.last_theta_phase = theta_phase
        self.counter += 1
        self.pos = np.array(pos)
        self.history["t"].append(self.t)
        self.history["pos"].append(list(pos))

        return

    def plot_trajectory(self, sequences_ontop=False, **kwargs):
        """A bespoke plotting function taking the same arguments as Agent.plot_trajectory() except now it will jointly plot the True trajectory and the the ThetaSequenceTrajectory() below that.

        • sequences_ontop (bool, default False): determines whether sequences get plotted ontop of or below the true trajectory.
        """

        kwargs_ = kwargs.copy()
        kwargs_["decay_point_timescale"] = (
            self.T_theta / 2
        )  # decays sequences fast if animated
        kwargs_["framerate"] = (
            self.v_sequence / 0.02
        )  # 2cm point seperation for sequences
        kwargs_["color"] = "C1"
        kwargs_["alpha"] = 0.4
        kwargs_["show_agent"] = False
        kwargs_["autosave"] = False

        if sequences_ontop == False:
            fig, ax = super(ThetaSequenceAgent, self).plot_trajectory(**kwargs_)
            kwargs["fig"] = fig
            kwargs["ax"] = ax
            kwargs["alpha"] = 0.4
            fig, ax = self.TrueAgent.plot_trajectory(**kwargs)
        else:
            fig, ax = self.TrueAgent.plot_trajectory(**kwargs)
            kwargs_["fig"] = fig
            kwargs_["ax"] = ax
            fig, ax = super(ThetaSequenceAgent, self).plot_trajectory(**kwargs_)

        return fig, ax
    
    


import ratinabox
from ratinabox.Agent import Agent 
import numpy as np
import copy
from scipy.interpolate import interp1d
import warnings



class SubAgent(Agent):
    """SubAgents take as input, and are "subservient" to, another Agent  (the LeadAgent). The key thing idea is that the SubAgent may have an update() function which relies heavily on the state of the LeadAgent. 

    List of functions: 
    • update() : updates the SubAgent, likely requiring data from LeadAgent. This is the key function that needs to be implemented by subclasses.
    • plot_trajectory() : plots the trajectory of the SubAgent. By default this is plotted alongside the LeadAgent's trajectory. 
    """ 
    
    default_params = {}
    def __init__(self, LeadAgent : Agent, params = {}):
        self.params =  copy.deepcopy(__class__.default_params)
        self.params.update(params)
        self.LeadAgent = LeadAgent
        if "dt" in self.params: 
            warnings.warn("You have passed 'dt as a parameter but this will be overwritten to match dt of the LeadAgent")
        self.params["dt"] = self.LeadAgent.dt
        self.Environment = self.LeadAgent.Environment
        super().__init__(self.Environment, self.params)

        #Initialises the position and velocity of the SubAgent to be the same as the LeadAgent
        self.pos = self.LeadAgent.pos.copy()
        self.velocity = self.LeadAgent.velocity.copy()

    def update(self, **kwargs):
        """Updates the position of the SubAgent. This is the key function that needs to be implemented by subclasses. By default this doesn't take kwargs like dt or drift velocity (you could add these) because the LeadAgent must have been updated _before_ the SubAgent."""
        self.t = self.LeadAgent.t
        super().update(**kwargs)

    def plot_trajectory(self, 
                        # standard kwargs for Agent.plot_trajectory()
                        t_start=0,
                        t_end=None,
                        framerate=10,
                        fig=None,
                        ax=None,
                        color=None, 
                        autosave=None,

                        # special kwargs
                        ontop=False, 
                        plot_error=False,
                        show_lead_agent=True, 
                        lead_agent_plot_kwargs={}, # defaults outlined below
                        
                        # other kwargs for SubAgent.plot_trajectory()
                        **kwargs):
        """A bespoke plotting function taking the same arguments as Agent.plot_trajectory() except now it will jointly plot the True SubAgent and LeadAgent trajectories. By default all kwargs refer to how the SubAgent trajectory is plots and LeadAgent trajectory is plotted in a dimmer colour and smaller point size (although this can be controlled with lead_agent_plot_kwargs).
        
        Args:
            • t_start --> autosave: see Agent.plot_trajectory
            specific args for SubAgent classes
            • ontop: if True, plot the SubAgent trajectory on top of the LeadAgent trajectory
            • plot_error: if True, will plot an arrow showing the error between the SubAgent and LeadAgent at the end of the trajectory
            • show_lead_agent: if True, will plot the trajectory of the LeadAgent
            • lead_agent_plot_kwargs: kwargs for plotting the LeadAgent trajectory e.g. it's color, alpha, point size etc. (any kwarg you would pass to Agent.plot_trajectory() can be passed here)    
            • kwargs: any other kwargs you would pass to Agent.plot_trajectory() can be passed here

        """
        fig, ax = super().plot_trajectory(
            t_start=t_start, 
            t_end=t_end, 
            framerate=framerate,
            fig=fig,
            ax=ax,
            color=color,
            autosave=False, # don't save this intermediate plot
            **kwargs,
            )
        
        lead_agent_plot_kwargs_ = copy.deepcopy(kwargs)
        default_lead_agent_plot_kwargs = {
                            'color':'k',
                            'point_size': 15, 
                            'alpha': 0.2,
                            'show_agent':False, # don't show a big point at the end of the trajectory
                            }
        lead_agent_plot_kwargs_.update(default_lead_agent_plot_kwargs)
        lead_agent_plot_kwargs_.update(lead_agent_plot_kwargs)
        if show_lead_agent == True: 
            fig, ax = self.LeadAgent.plot_trajectory(
                t_start=t_start,
                t_end=t_end,
                framerate=framerate,
                fig=fig,
                ax=ax,
                zorder=1.1 - 1e-3*ontop,
                autosave=autosave,
                **lead_agent_plot_kwargs_
        )

        if plot_error == True:
            if t_end == None: t = self.history['t'][-1]
            if (self._last_history_array_cache_time != self.t): 
                self.cache_history_as_arrays()
                self.LeadAgent.cache_history_as_arrays()
            slice = self.get_history_slice(t_start=t_end-1, t_end=t_end)
            self_pos = self._history_arrays["pos"][slice][-1]
            lead_pos = self.LeadAgent._history_arrays["pos"][slice][-1]
            [x,y] = list(lead_pos)
            [dx, dy] = list(self_pos - lead_pos)
            # add an arrow 
            ax.arrow(x, y, dx, dy, head_width=0.015, fc='k', ec=None, linewidth=0.5, length_includes_head=True, zorder=1.2)
            # add text saying "δ" half way along the arrow 
            # ax.text(x + dx/2, y + dy/2, "δ", fontsize=6, color='k')
            pass

        return fig, ax
    
class DumbAgent(SubAgent):
    """The DumbAgent moves around like a normal RatInABox Agent but its position is wrong. It works as follows: 
    as the LeadAgent moves around the Environment it is as if there is a SubAgent which is attached to it on a spring. In the frame of reference of the LeadAgent the dynamics of the SubAgent as as follows: 
    • The velocity of the SubAgents relative position follows a smooth stochatic process of variance σ and timescale τ_v
    • The spring exerts a restoring force, F_s=-kx, pulling the SubAgent back towars the LeadAgent so it doesn't get too far. 
    The stochastic motion can be viewed as a brownian force on the SubAgent with scale of order F_b = mσ (m is a hypothetical mass which will drop out). We want that the likely maximum extension of the SubAgent from the Agent is a parameter (drift_distance), d, found by solving F_s = F_b = kd = mσ --> k = mσ/d, or, as an acceleration dividing through by m, a_s = σ/d. Finally to remove the last free parameter we cponstain that the spring recoil time is determined by a parameter (drift_timescale) τ_s, which, from oscillatory motion, is order π√(m/k) = π√(d/σ) --> σ = π^2 d / τ_s^2. Lastly let the spring recoil time ot be about twice the stochastic velocity coherence time (so the motion of the subagent is a bit but not too wiggly around the LeadAgent). Thus we're left with: 

    Define: 
        • d = drift_distance - like expected deviation of the SubAgent from the LeadAgent
        • τ_s = drift_timescale - like the timescale over which the SubAgent drifts away from and back to the LeadAgent
    Gives: 
        • σ = π^2 d / τ_s^2 - the variance of the stochastic velocity of the SubAgent
        • τ_v = τ_s / 2 - the timescale over which the stochastic velocity of the SubAgent varies
        • σ / d - how acceleration due to the spring dynamics scales with distance i.e. dv/dx = -σx/d
    """
    
    default_params = {'drift_distance' : 0.05,  # ~Average distance between the two agents
                      'drift_timescale' : 3.0, # ~Timescale over which the wrong position journeys around the true position
                      } 

    def __init__(self, LeadAgent, params = {}):

        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)
        super().__init__(LeadAgent, self.params)

        # Some variables 
        self.displacement = np.zeros(self.Environment.D)
        self.displacement_velocity = np.zeros(self.Environment.D) 
        self.tau_v = self.drift_timescale / 2 #the velocity of the noise varies about 5 times faster than the position of the SubAgent (relative to LeadAgent)
        self.sigma = np.pi**2 * self.drift_distance / (self.drift_timescale**2)
        self.acceleration_scale = self.sigma / self.drift_distance

    def update(self):
        """Updates the position of the DumbAgent"""
        # update the displacement velocity under stochastic + spring dynamics
        ou_update_to_displacement_velocity = ratinabox.utils.ornstein_uhlenbeck(
                        dt=self.LeadAgent.dt,
                        x=self.displacement_velocity,
                        drift=0.0,
                        noise_scale=self.sigma,
                        coherence_time=self.tau_v)
        spring_update_to_displacement_velocity = -self.acceleration_scale * self.displacement * self.LeadAgent.dt
        self.displacement_velocity += ou_update_to_displacement_velocity + spring_update_to_displacement_velocity 
        # use the displacement velocity to update the displacement
        self.displacement += self.displacement_velocity * self.LeadAgent.dt
        
        if self.Environment.dimensionality == '2D':
            # Check for wall crossings
            displacement_step = np.array([self.LeadAgent.pos,self.LeadAgent.pos + self.displacement])
            collision_coords, collision_bools = ratinabox.utils.vector_intercepts(self.Environment.walls, displacement_step,return_collisions="as_well")
            if True in collision_bools:
                collisions = collision_coords[collision_bools]
                # self.displacement = collision_coords[collision_bools][0] - self.LeadAgent.pos
                closest_collision_along_displacement = np.min(collisions[:,1])
                self.displacement *= 0.95*closest_collision_along_displacement 

        pos = self.LeadAgent.pos + self.displacement
        #apply strict boundary conditions
        pos = self.Environment.apply_boundary_conditions(pos)
        self.displacement = self.Environment.get_vectors_between___accounting_for_environment(pos, self.LeadAgent.pos)[0,0,:]
        super().update(forced_next_position=pos)
    

class ThetaSequenceAgent(SubAgent):
    """ThetaSequneceAgent is a type of Agent who's position is NOT the true position but instead a "theta sequence" over the position. This starts from behind the "true" position and rapidly moves to infront of the true position (default sequence speed = 5ms-1) once every "theta cycle" (default 10Hz). Each theta sequence is split into the following phases (marked as fraction of the theta cycle):

    |.......A.........|................B..............|.................C.............|........A'.......|
    0              1/2-β/2                           1/2                           1/2+β/2              1

    • A and A': within these segments the position is [nan], the sequence hasn't started yet or has finished.
    • B, "Look behind": The sequence starts behind the agents current position and moves along the historic trajectory until it meets the agent half way through the theta cycle.
    • C, "Look ahead": A new "random" trajectory into the future is sampled starting from the agents current position and velocity.

    The velocity of the sequence, v_sequence, is constant. This is the velocity of the sequence in the reference frame of the LeadAgent (i.e. ground truth see below) so the "apparent" velocity of the sequence will be v_sequence + the speed of the LeadAgent.

    ThetaSequenceAgent has within it two other Agent classes:
        • self.LeadAgent (riab.Agent) is the real Agent moving in the Environment
        • self.ForwardSequenceAgent (riab.Agent) is a sham Agent only used to access riab's stochastic motion model and generate  the forward sequences.

    The default params (beyond the standard Agent params) are:
    default_params = {
        "theta_freq"  : 10.0, #theta frequency
        "v_sequence"     : 5.0, #sequence speed in reference frame of Agent, ms-1
        "theta_frac"  : 0.5, #fraction of theta cycle over which}
    """

    default_params = {
        "v_sequence": 5.0,  # speed of the theta sequence in the reference frame of the LeadAgent, ms^-1
        "theta_freq": 10.0,  # theta frequency
        "theta_frac": 0.5,  # fraction of theta cycle over which
        }
    

    def __init__(self, LeadAgent, params={}):


        self.params = copy.deepcopy(__class__.default_params)   
        self.params.update(params)
        super().__init__(LeadAgent, self.params)

        # ground truth Agent
        self.LeadAgent.distance_travelled = 0  # distance travelled
        self.LeadAgent.history["distance_travelled"] = []  # history of distance travelled
        # a sham Agent we're initialising just in order to do a forward sequence
        ForwardSequenceAgent_params = copy.deepcopy(self.params)
        for param in __class__.default_params.keys(): ForwardSequenceAgent_params.pop(param)
        self.ForwardSequenceAgent = Agent(self.Environment, ForwardSequenceAgent_params); 

        # some variables/constants
        self.T_theta = 1 / self.theta_freq
        self.d_half = ((self.theta_frac / 2) * self.T_theta * self.v_sequence)  # how far agent will travel in half a sequence
        self.last_theta_phase = 0

        # its very time consuming to continually convert position data into arrays so we preallocate a memory location
        self.n_half = int(2 * self.d_half / (self.LeadAgent.speed_mean * self.LeadAgent.dt))  # approx how many steps for the agent to travel d_half in real time
        self.keep_count = max(1,(20 * self.n_half))  # how many data points to save in preallocated memory
        self.recent_data_stash = {}  # its time consuming
        self.recent_data_stash["distance"] = np.zeros((self.keep_count))  # its time consuming
        self.recent_data_stash["position"] = np.zeros((self.keep_count, self.Environment.D))  # its time consuming
        self.recent_data_stash["distance"][0] = self.LeadAgent.distance_travelled
        self.recent_data_stash["position"][0, :] = self.LeadAgent.pos
        self.counter = 1

        assert (self.LeadAgent.dt <= self.T_theta / 10), f"params['dt'] for the LeadAgent is too large. It must be < 10% of theta time period., i.e. smaller than {self.T_theta/10:.5f}"
        assert (self.v_sequence >= 4*self.LeadAgent.speed_mean), f"params['v_sequence'] is too small. It must be > 4*LeadAgent.speed_mean, i.e. larger than {4*self.LeadAgent.speed_mean:.2f}"

    def update(self, dt=None, drift_velocity=None, drift_to_random_strength_ratio=1, forward_agent_update_kwargs={}):
        """
        Updates and saves the position of the Agent along the theta sequence.

        None that this is quite a complicated function! Some complexities which may help you to understand this code include:

        • Achilles and the tortoise: When behind the Agent we can interpolate along historic data but on each step the true agent moves forwards a little, so we must recollect this new data. The ThetaSequenceAgent position is Achilles, the LeadAgent is the tortoise.
        • Interpolation expense: We must interpolate smoothly over historic data but this is expensive since it requires converting the list of past positions into an array then running scipy.interpolate.interp1d. So we want to take the least possible historic data which guarantees we'll have enough to do the behind sequence.
        • Reference frame: In the current model the speed of the sequence is constant (in the reference frame of the LeadAgent) but the speed of the LeadAgent may not be. Therefore it is not enough to just interpolate over the past trajectory (indexed by time), we must transform coordinates to "distance travelled" (which is linear wrt the sequence).
        • Boundary conditions
        """

        
        # append LeadAgent position and distance data into our preallocated arrays:
        if self.counter == self.keep_count:
            self.counter = 10 * self.n_half
            self.recent_data_stash["distance"][: self.counter] = self.recent_data_stash["distance"][-self.counter :]
            self.recent_data_stash["position"][: self.counter, :] = self.recent_data_stash["position"][-self.counter :, :]
        self.recent_data_stash["distance"][self.counter] = self.LeadAgent.distance_travelled
        self.recent_data_stash["position"][self.counter, :] = self.LeadAgent.pos

        self.t = self.LeadAgent.t
        theta_phase = (self.t % (1 / self.theta_freq)) / ((1 / self.theta_freq))

        # PRE SWEEP (returns nan's)
        if theta_phase < (0.5 - (self.theta_frac / 2)):
            pos = np.full(shape=(self.Environment.D,), fill_value=np.nan)   # No position

        # LOOK BEHIND (EARLY SWEEP, from behind to current position, taken from historical data)
        if (theta_phase >= (0.5 - self.theta_frac / 2)) and (theta_phase < 0.5):
            true_distances = self.LeadAgent.history["distance_travelled"]
            # Backwards sequence
            if true_distances[-1] < self.d_half:
                # handle case where not enough data has been collected yet
                # just dont do a backwards sequence and take current positions
                pos = self.LeadAgent.pos
            else:
                # get just enough past data
                lookback = int(5 * self.d_half / (self.LeadAgent.dt * self.LeadAgent.average_measured_speed))  # so argmin will never grow arbitrarily large, 3 to be safe
                lookback = min(lookback, self.counter)
                true_positions = self.recent_data_stash["position"][self.counter - lookback + 1 : self.counter + 1, :]
                true_distances = self.recent_data_stash["distance"][self.counter - lookback + 1 : self.counter + 1]
                # interpolate it
                a = np.argmin(true_positions)
                # calculate how far back the current sequence should be look (sequence closing in on Agent at speed v_sequence so net speed of sequence = v_sequence + v_agent)
                # converts current theta phase to how far back along the current trajectory to take position from
                c = self.d_half / self.theta_frac
                m = -2 * c
                distance_back = (m * theta_phase + c)  # how far behind the agents current position the sequence should be at
                interp_distance = (true_distances[-1] - distance_back)  # and the LeadAgent's actual distance travelled at this point
                idx = np.argmin(np.abs(true_distances - interp_distance))
                self.pos_interp = interp1d(
                    true_distances[idx - 3 : idx + 3],
                    true_positions[idx - 3 : idx + 3],
                    axis=0,)
                pos = self.pos_interp(interp_distance)
        
        # LOOK AHEAD (LATE SWEEP, from current position to infront, stochastically generated)
        if (theta_phase >= 0.5) and (theta_phase < 0.5 + self.theta_frac / 2):
            # Forward sequence
            if (theta_phase >= 0.5 and self.last_theta_phase < 0.5):  # catch on first time each loop
                self.ForwardSequenceAgent.pos = self.LeadAgent.pos
                self.ForwardSequenceAgent.history["pos"].append(self.LeadAgent.pos)
                self.ForwardSequenceAgent.velocity = self.LeadAgent.velocity
                self.ForwardSequenceAgent.history["vel"].append(self.LeadAgent.velocity)
                if self.Environment.dimensionality == "2D":
                    self.ForwardSequenceAgent.rotational_velocity = (self.LeadAgent.rotational_velocity)
                    self.ForwardSequenceAgent.history["rot_vel"].append(self.LeadAgent.rotational_velocity)
                self.ForwardSequenceAgent.distance_travelled = self.LeadAgent.distance_travelled
                recent_speed = self.LeadAgent.average_measured_speed
                forward_distance_to_simulate = (self.d_half + 100 * recent_speed * (self.theta_frac / 2) * self.T_theta)
                future_positions = [self.ForwardSequenceAgent.pos]
                future_distances = [self.ForwardSequenceAgent.distance_travelled]
                while ( self.ForwardSequenceAgent.distance_travelled < self.LeadAgent.distance_travelled + forward_distance_to_simulate):
                    self.ForwardSequenceAgent.update(
                        dt=self.LeadAgent.dt
                        * self.v_sequence
                        / self.LeadAgent.average_measured_speed,
                        **forward_agent_update_kwargs)
                    future_positions.append(self.ForwardSequenceAgent.pos)
                    future_distances.append(self.ForwardSequenceAgent.distance_travelled)
                future_positions, future_distances = np.array(future_positions), np.array(future_distances)
                self.pos_interp = interp1d(future_distances, future_positions, axis=0)
            # calculate how far forward the current sequence should be look (sequence moving away fromAgent at speed v_sequence so net speed of sequence = v_sequence + v_agent)
            # converts current theta phase to how far forward along the current trajectory to take position from
            c = -self.d_half / self.theta_frac
            m = -2 * c
            distance_ahead = (m * theta_phase + c)  # how far ahead of the agents current position the sequence should be at
            interp_distance = (self.LeadAgent.distance_travelled + distance_ahead)  # and the ForwardSequenceAgent's actual distance travelled at this point
            pos = self.pos_interp(interp_distance)

        # POST SWEEP (returns nan's)
        if theta_phase >= (0.5 + (self.theta_frac / 2)):
            pos = np.full(shape=(self.Environment.D,), fill_value=np.nan)

        # handle periodic boundaries by just testing if the distance between current and true position of the Agent is over d_half this can only be because the interpolation has crossed a boundary, in which case just set the position to nan (minimally damaging for small dt)
        dist = self.Environment.get_distances_between___accounting_for_environment(pos.reshape(1, -1), self.LeadAgent.pos.reshape(1, -1))
        if np.isnan(dist): pass
        elif dist > self.d_half: pos = np.full(shape=(self.Environment.D,), fill_value=np.nan)

        self.last_theta_phase = theta_phase
        self.counter += 1

        super().update(forced_next_position=pos)

        return
    
    # def plot_trajectory(self, framerate=10, **kwargs):
    #     subagent_framerate = framerate * 0.75 * self.v_sequence / self.LeadAgent.speed_mean
    #     kwargs['subagent_framerate'] = subagent_framerate
    #     fig, ax = super().plot_trajectory(**kwargs)
    #     return fig, ax  

class ReplayAgent(SubAgent):
    """This agents position usually equals the position of the LeagAgent but it can, at times, initiate a "replay" event where the Agent disconnects, moves to another region of the environment and explores their for a short time."""
    default_params = {
        "replay_freq" : 0.3, #frequency of replay events
        "replay_duration" : 0.1, #duration of replay events
        "replay_speed" : 1.0, #speed of replay events
    }
    def __init__(self, LeadAgent, params={}):
        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)
        super().__init__(LeadAgent, self.params)

        self.mean_replay_speed = self.replay_speed
        self.mean_replay_duration = self.replay_duration
        self.is_undergoing_replay = False
        self.history["replay"] = []

        # a sham Agent we're initialising just in order to do access the RiaB random motion model and do a replay
        ReplayAgent_params = copy.deepcopy(self.params)
        for param in __class__.default_params.keys(): ReplayAgent_params.pop(param)
        self.ReplayAgent = Agent(self.Environment, ReplayAgent_params); 

    def update(self):
        """Either: 
        1) You're not currently in a replay - 
            a) Update position to match LeadAgent
            b) Start a replay by moving to a random new position and velocity

        2) You are currently in a replay -
            a) Update position stochastically with speed replay_speed  
            b) If replay_duration has elapsed then stop the replay and reset position to that of LeadAgent

        """
        if self.is_undergoing_replay is False:
            if np.random.uniform() > self.replay_freq * self.dt: #Don't initialise a replay
                # Track the LeadAgent
                pos = self.LeadAgent.pos
            else: # Initialise a replay 
                self.is_undergoing_replay = True
                self.replay_speed = np.random.rayleigh(scale=self.mean_replay_speed)
                self.replay_duration = max(np.random.rayleigh(scale=self.mean_replay_duration), self.mean_replay_duration/2)
                self.replay_start_time = self.t
                self.replay_end_time = self.t + self.replay_duration
                
                #Generate a replay using ReplayAgent and extract the trajectory. Given the replay speed is faster we must simulate at least replay_speed * replay_duration of trajectory
                self.ReplayAgent.initialise_position_and_velocity()
                self.ReplayAgent.save_to_history()
                start_distance_travelled = self.ReplayAgent.distance_travelled 
                count = 0
                pos = self.ReplayAgent.pos #set the pos to the start of the replay
                while self.ReplayAgent.distance_travelled < start_distance_travelled + 1.1 * self.replay_speed * self.replay_duration:
                    self.ReplayAgent.update()
                    count += 1
                replay_distances_travelled = np.array(self.ReplayAgent.history["distance_travelled"][-count-1:])
                replay_distances_travelled -= start_distance_travelled #zero the distance travelled
                replay_positions = np.array(self.ReplayAgent.history["pos"][-count-1:])
                self.replay_pos_interp = interp1d(replay_distances_travelled, replay_positions, axis=0)
        else: 
            if self.t < self.replay_end_time:
                distance_along_replay = self.replay_speed * (self.t - self.replay_start_time)
                pos = self.replay_pos_interp(distance_along_replay)
            else:
                #reset position to that of the LeadAgent
                self.is_undergoing_replay = False
                pos = self.LeadAgent.pos
                self.velocity = self.LeadAgent.velocity
        
        super().update(forced_next_position=pos)
        return 

    def save_to_history(self, **kwargs):
        """Saves a flag of whether replay was happening at this time."""
        self.history["replay"].append(self.is_undergoing_replay)
        super().save_to_history(**kwargs)
        return

    def plot_trajectory(self, t_start=0, t_end=None, framerate=10, **kwargs):
        """Plots the trajectory with a higher framerate during replay events so they can be more easily resolved. This repeats some of the logic already performed in Agent but it's not too bad."""
        if framerate is None: skiprate = 1
        else: skiprate = max(1, int((1 / framerate) / self.dt))
        replay_skiprate = max(1,int(skiprate * (self.LeadAgent.speed_mean / self.replay_speed))) #smaller dt during replay
        # all the available data 
        time = np.array(self.history["t"])
        trajectory = np.array(self.history["pos"])
        head_direction = np.array(self.history["head_direction"])
        was_replay = np.array(self.history["replay"])

        t_start = t_start or time[0]
        startid = np.nanargmin(np.abs(time - (t_start)))
        t_end = t_end or time[-1]
        endid = np.nanargmin(np.abs(time - (t_end)))
        time_slice = slice(startid, endid, 1)
        time, trajectory, head_direction, was_replay = time[time_slice], trajectory[time_slice], head_direction[time_slice], was_replay[time_slice]
        keep_nonreplay = np.zeros_like(time, dtype=bool)
        keep_nonreplay[slice(0, -1, skiprate)] = True
        keep_replay = np.zeros_like(time, dtype=bool)
        keep_replay[slice(0, -1, replay_skiprate)] = True
        keep_replay = np.logical_and(keep_replay, was_replay)
        keep = np.logical_or(keep_nonreplay, keep_replay)
        time = time[keep]
        trajectory = trajectory[keep]
        head_direction = head_direction[keep]

        fig, ax = super().plot_trajectory(t_start=t_start, t_end=t_end, time=time, trajectory=trajectory, head_direction=head_direction, framerate=framerate, **kwargs)

        return fig, ax


class ShiftAgent(SubAgent):
    """
    ShiftAgent reports the position of the LeadAgent but with a fixed shift in position shift_m ahead of it along its current heading direction. Can be positive or negative. 
    """
    
    default_params = {
        "shift_m" : 0.01, #distance ahead of the LeadAgent (can be negative)
    }
    
    def update(self,):
        pos = self.LeadAgent.pos + self.LeadAgent.head_direction * self.shift_m
        super().update(forced_next_position=pos)
        return

class UnrelatedAgent(SubAgent):
    """The SubAgent is totally indepdendent from the LeadAgent. This is just to exploit the plotting functionality."""

    default_params = {}
    def __init__(self, LeadAgent: Agent, params={}):
        super().__init__(LeadAgent, params)
    
    def update(self):
        super().update()
        return
    

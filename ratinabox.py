import scipy
from scipy import stats
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

class Environment:
    """Environment class: defines the environment the agent lives in. 
    This class needs no other classes to initialise it. It exists independently of any Agents or Neurons. 

    A default parameters dictionary (with descriptions) can be fount in __init__()

    List of functions...
        ...that you might use:
            • add_wall()
            • sample_positions()
            • plot_environment()
        ...that you probably won't directly use:
            • discretise_environment()
            • get_vectors_between___accounting_for_environment()
            • get_distances_between___accounting_for_environment()
            • check_if_posision_is_in_environment()
            • check_walls()
            • vectors_from_walls()
            • apply_boundary_conditions()
    Key variables:
            • walls
            • extent
            • dimensionality
            • boundary_conditions
    """

    def __init__(self,
                 params={}):
        """Initialise environment, takes as input a parameter dictionary who's values supercede a default dictionary.

        Args:
            params (dict, optional). Defaults to {}.
        """
        default_params = {
            "dimensionality":'2D', #1D or 2D environment 
            "boundary_conditions":"solid", #solid vs periodic
            "scale": 1, #scale of environment
            "aspect":1, #x/y aspect ratio 2D only
        }
        update_class_params(self, default_params)
        update_class_params(self, params)

        self.dx=0.01 #superficially discretises the environment for plotting purposes only 

        if self.dimensionality == "1D":
            self.extent = np.array([0, self.scale])
            self.centre = np.array([self.scale / 2, self.scale / 2])

        self.walls = np.array([])
        if self.dimensionality == "2D":
            if self.boundary_conditions != 'periodic': 
                self.walls = np.array(
                    [   [[0, 0], [0, self.scale]],
                        [[0, self.scale], [self.aspect*self.scale, self.scale]],
                        [[self.aspect*self.scale, self.scale], [self.aspect*self.scale, 0]],
                        [[self.aspect*self.scale, 0], [0, 0]]])
            self.centre = np.array([self.aspect*self.scale / 2, self.scale / 2])
            self.extent = np.array([0, self.aspect*self.scale, 0, self.scale])

        # save some prediscretised coords
        self.discrete_coords = self.discretise_environment(dx=self.dx)
        self.flattened_discrete_coords = self.discrete_coords.reshape(
            -1, self.discrete_coords.shape[-1]
        )

    def add_wall(self, 
                 wall):
        """Add a wall to the (2D) environment.
        Extends self.walls array to include one new wall. 
        Args:
            wall (np.array): 2x2 array [[x1,y1],[x2,y2]]
        """  
        assert self.dimensionality == "2D", "can only add walls into a 2D environment"      
        wall = np.expand_dims(np.array(wall),axis=0)
        if len(self.walls) == 0: 
            self.walls = wall
        else:
            self.walls = np.concatenate((self.walls,wall),axis=0)
        return 

    def sample_positions(self, 
                         n=10,
                        method="uniform_jitter"):
        """Scatters n locations across the environment which can act as, for example, the centres of gaussian place fields, or as a random starting position. 
        If method == "uniform" an evenly spaced grid of locations is returns.  If "uniform_jitter" these locations are jittered slightly. Note is n doesn't uniformly divide the size (i.e. n !/ m^2 in a square environment) then the largest number that can be scattered uniformly are found, the remaining are randomly placed. 
        Args:
            n (int): number of features 
            method: "uniform", "uniform_jittered" or "random" for how points are distributed
            true_random: if True, just randomly scatters point
        Returns:
            array: (n x dimensionality) of positions 
        """
        if self.dimensionality == "1D":
            if method == "random":
                positions = np.random.uniform(self.extent[0],self.extent[1],size=(n,1))
            elif method[:7] == "uniform":
                dx = self.scale / n
                positions = np.arange(0 + dx/2, self.scale, dx).reshape(-1, 1)
                if method[7:] == "_jitter":
                    positions += np.random.uniform(
                        -0.45 * dx, 0.45 * dx, positions.shape)
            return positions

        elif self.dimensionality == "2D":
            if method == "random":
                positions = np.random.uniform(size=(n, 2))
                positions[:, 0] *= self.extent[1] - self.extent[0]
                positions[:, 1] *= self.extent[3] - self.extent[2]
            elif method[:7] == "uniform":
                ex = self.extent
                area = (ex[1] - ex[0]) * (ex[3] - ex[2])
                delta = np.sqrt(area / n)
                x = np.arange(ex[0] + delta / 2, ex[1] - delta / 2 + 1e-6, delta)
                y = np.arange(ex[2] + delta / 2, ex[3] - delta / 2 + 1e-6, delta)
                positions = np.array(np.meshgrid(x, y)).reshape(2, -1).T
                n_uniformly_distributed = positions.shape[0]
                if method[7:] == "_jitter":
                    positions += np.random.uniform(-0.45 * delta, 0.45 * delta, positions.shape)
                n_remaining = n - n_uniformly_distributed
                if n_remaining > 0: 
                    positions_remaining = self.sample_positions(n=n_remaining,method="random")
                    positions = np.vstack((positions,positions_remaining))
            return positions
                                
    def plot_environment(self,
                         fig=None,
                         ax=None,
                         height=1):
        """Plots the environment, dark grey lines show the walls
        Args:        
            fig,ax: the fig and ax to plot on (can be None)
            height: if 1D, how many line plots will be stacked (5.5mm per line)
        Returns:
            fig, ax: the environment figures, can be used for further downstream plotting.
        """        

        if self.dimensionality == '1D':
            extent = self.extent
            fig, ax = plt.subplots(
                    figsize=(3*(extent[1] - extent[0]),height*(5.5/25))
                )
            ax.set_xlim(left=extent[0],right=extent [1])
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')        
            ax.spines['bottom'].set_position('zero')
            ax.spines['top'].set_color('none')
            ax.set_yticks([])
            ax.set_xticks([extent[0],extent[1]])
            ax.set_xlabel("Position / m")

        if self.dimensionality == '2D':
            extent, walls = self.extent, self.walls
            if fig is None and ax is None: 
                fig, ax = plt.subplots(
                    figsize=(3*(extent[1] - extent[0]), 3*(extent[3] - extent[2]))
                )
            background = matplotlib.patches.Rectangle((extent[0],extent[2]),extent[1],extent[3],facecolor='lightgrey',zorder=-1)
            ax.add_patch(background)
            for wall in walls:
                ax.plot(
                    [wall[0][0], wall[1][0]],
                    [wall[0][1], wall[1][1]],
                    color="grey",
                    linewidth=4,
                )
            ax.set_aspect("equal")
            ax.grid(False)
            ax.axis("off")
            ax.set_xlim(left=extent[0]-0.03,right=extent[1]+0.03)
            ax.set_ylim(bottom=extent[2]-0.03,top=extent[3]+0.03)
        return fig, ax

    def discretise_environment(self,
                               dx=None):
        """Discretises the environment, for plotting purposes.
        Returns an array of positions spanning the environment 
        Important: this discretisation is not used for geometry or firing rate calculations which are precise. Its typically used if you want to, say, display the receptive field of a neuron so you want to calculate its firing rate at all points across the environment and plot those. 
        Args:
            dx (float): discretisation distance
        Returns:
            array: and Ny x Mx x 2 array of position coordinates or Nx x 1 for 1D
        """  # returns a 2D array of locations discretised by dx
        if dx is None: dx = self.dx
        [minx, maxx] = list(self.extent[:2])
        self.x_array = np.arange(minx + dx / 2, maxx, dx)
        discrete_coords = self.x_array.reshape(-1,1)
        if self.dimensionality == "2D":
            [miny,maxy] = list(self.extent[2:])
            self.y_array = np.arange(miny + dx / 2, maxy, dx)[::-1]
            x_mesh, y_mesh = np.meshgrid(self.x_array, self.y_array)
            coordinate_mesh = np.array([x_mesh, y_mesh])
            discrete_coords = np.swapaxes(np.swapaxes(coordinate_mesh, 0, 1), 1, 2)
        return discrete_coords

    def get_vectors_between___accounting_for_environment(self, 
     pos1=None,
     pos2=None,
     line_segments=None):
        """Takes two position arrays and returns the array of pair-wise vectors from pos1's to pos2's, taking into account boundary conditions. Unlike the global function "get_vectors_between()' (which this calls) this additionally accounts for the boundaries such that if two positions fall on either sides of the boundary AND boundary cons are periodic then the returned shortest vector actually goes around the loop, not across the environment)... 
            pos1 (array): N x dimensionality array of poisitions
            pos2 (array): M x dimensionality array of positions
            wall_geometry: how the distance calculation handles walls in the env
        Returns:
            N x M x dimensionality array of pairwise vectors 
        """
        vectors = get_vectors_between(pos1=pos1,pos2=pos2,line_segments=line_segments)
        if self.boundary_conditions == "periodic":
                flip = np.abs(vectors) > (self.scale / 2)
                vectors[flip] = -np.sign(
                    vectors[flip])*(self.scale-np.abs(vectors[flip]))
        return vectors

    def get_distances_between___accounting_for_environment(self,
     pos1,
     pos2,
     wall_geometry='euclidean'):
        """Takes two position arrays and returns the array of pair-wise distances between points, taking into account walls and boundary conditions. Unlike the global function get_distances_between() (which this one, at times, calls) this additionally accounts for the boundaries and walls in the environment)... 
        ...For example, geodesic geometry estimates distance by shortest walk...line_of_sight geometry distance is euclidean but if there is a wall in between two positions (i.e. no line of sight) then the returned distance is "very high"...if boundary conditions are periodic distance is via the shortest possible route, which may or may not go around the back. euclidean geometry essentially ignores walls when calculating distances between two points.
        Allowed geometries, typically passed from the neurons class, are "euclidean", "geodesic" or "line_of_sight"
        Args:
            pos1 (array): N x dimensionality array of poisitions
            pos2 (array): M x dimensionality array of positions
            wall_geometry: how the distance calculation handles walls in the env
        Returns:
            N x M array of pairwise distances 
        """

        line_segments = get_line_segments_between(pos1=pos1,pos2=pos2) 
        vectors = self.get_vectors_between___accounting_for_environment(pos1=None,pos2=None,line_segments=line_segments)

        #shorthand
        walls = self.walls
        dimensionality = self.dimensionality
        boundary_conditions = self.boundary_conditions

        if dimensionality == "1D":
            distances = get_distances_between(vectors=vectors)


        if dimensionality == "2D":
            if wall_geometry == 'euclidean':
                distances = get_distances_between(vectors=vectors)
            
            if wall_geometry == 'line_of_sight': 
                assert boundary_conditions == 'solid', "line of sight geometry not available for periodic boundary conditions"
                #if a wall obstructs line-of-sight between two positions, distance is set to 1000
                internal_walls = walls[4:] #only the internal walls (not room walls) are worth checking
                line_segments_ = line_segments.reshape(-1, *line_segments.shape[-2:])
                wall_obstructs_view_of_cell = vector_intercepts(line_segments_,
                                                               internal_walls,
                                                               return_collisions=True)
                wall_obstructs_view_of_cell = wall_obstructs_view_of_cell.sum(axis=-1) #sum over walls axis as we don't care which wall it collided with
                wall_obstructs_view_of_cell = (wall_obstructs_view_of_cell != 0)
                wall_obstructs_view_of_cell = wall_obstructs_view_of_cell.reshape(line_segments.shape[:2])
                distances = get_distances_between(vectors=vectors)
                distances[wall_obstructs_view_of_cell==True] = 1000
            
            if wall_geometry == 'geodesic':
                assert boundary_conditions == 'solid', "line of sight geometry not available for periodic boundary conditions"
                assert len(walls) <= 5, "unfortunately geodesic geomtry is only defined in closed rooms with one additional wall"
                distances = get_distances_between(vectors=vectors)
                if len(walls) == 4: 
                    pass
                else:
                    wall = walls[4]
                    via_wall_distances = []
                    for part_of_wall in wall: 
                        wall_edge = np.expand_dims(part_of_wall,axis=0)
                        if self.check_if_position_is_in_environment(part_of_wall):
                            distances_via_part_of_wall = (
                            get_distances_between(pos1,wall_edge)
                            +
                            get_distances_between(wall_edge,pos2))
                            via_wall_distances.append(distances_via_part_of_wall)
                    via_wall_distances = np.array(via_wall_distances)
                    line_segments_ = line_segments.reshape(-1, *line_segments.shape[-2:])
                    wall_obstructs_view_of_cell = vector_intercepts(line_segments_,
                                                               np.expand_dims(wall,axis=0),
                                                               return_collisions=True)
                    wall_obstructs_view_of_cell = wall_obstructs_view_of_cell.reshape(line_segments.shape[:2])
                    flattened_distances = distances.reshape(-1)
                    flattened_wall_obstructs_view_of_cell = wall_obstructs_view_of_cell.reshape(-1)
                    flattened_distances[flattened_wall_obstructs_view_of_cell] = np.amin(via_wall_distances,axis=0).reshape(-1)[flattened_wall_obstructs_view_of_cell]
                    distances = flattened_distances.reshape(distances.shape)
                
        return distances

    def check_if_position_is_in_environment(self, 
                                            pos):
        """Returns True if np.array(pos) is INside the environment
        Points EXACTLY on the edge of the environment are NOT classed as being inside the environment. This is relevant in geodesic geometry calculations since routes past the edge of a wall connection with the edge of an environmnet are not valid routes.
        Args:
            pos (array): np.array([x,y])
        Returns:
            bool: True if pos is inside environment.
        """        
        pos = np.array(pos).reshape(-1)
        if self.dimensionality == '2D':
            if ((pos[0] > self.extent[0]) and
                (pos[0] < self.extent[1]) and
                (pos[1] > self.extent[2]) and 
                (pos[1] < self.extent[3])):
                return True
            else: 
                return False
        elif self.dimensionality == '1D':
            if ((pos[0] > self.extent[0]) and
                (pos[0] < self.extent[1])):
                return True
            else: 
                return False
    
    def check_wall_collisions(self,
                    proposed_step):
        """Given proposed step [current_pos, next_pos] it returns two lists 
        1. a list of all the walls in the environment #shape=(N_walls,2,2)
        2. a boolean list of whether the step directly crosses (collides with) any of these walls  #shape=(N_walls,)
        Args:
            proposed_step (array): The proposed step. np.array( [ [x_current, y_current] , [x_next, y_next] ] )
        Returns:
            tuple: (1,2)
        """
        if self.dimensionality == '1D': 
            #no walls in 1D to collide with 
            return (None, None, None)
        elif self.dimensionality == '2D':
            if (self.walls is None) or (len(self.walls) == 0): 
                #no walls to collide with 
                return (None, None, None)
            elif self.walls is not None: 
                walls = self.walls
                wall_collisions = vector_intercepts(walls,proposed_step,return_collisions=True).reshape(-1)
                return (walls,wall_collisions)

    def vectors_from_walls(self,pos):
        """Given a position, pos,it returns a list of the vectors of shortest distance from all the walls to current_pos #shape=(N_walls,2)
        Args:
            proposed_step (array): The position np.array[x,y]
        Returns:
            vector array: np.array(N_walls,2)
        """
        walls_to_pos_vectors = shortest_vectors_from_points_to_lines(pos,self.walls)[0]
        return walls_to_pos_vectors

    def apply_boundary_conditions(self,
                                  pos):
        """Performs a boundary check. If pos is OUTside the environment and the boundary conditions are solid then it is returned 1cm within it. If pos is OUTside the environment but boundary conditions are periodic its position is looped to the other side
        Args:
            pos (np.array): 1 or 2 dimensional position
        returns new_pos
        """        
        if self.check_if_position_is_in_environment(pos) is False:
            if self.dimensionality =='1D':
                if self.boundary_conditions == 'periodic':
                    pos = pos % self.extent[1]
                if self.boundary_conditions == 'solid':
                    pos = min(max(pos,self.extent[0]+0.01),self.extent[1]-0.01)
                    pos = np.reshape(pos,(-1))
            if self.dimensionality == '2D':
                if self.boundary_conditions == 'periodic':
                    pos[0] = pos[0] % self.extent[1]
                    pos[1] = pos[1] % self.extent[3]
                if self.boundary_conditions == 'solid':
                    #in theory this wont be used as wall bouncing catches it earlier on
                    pos[0] = min(max(pos[0],self.extent[0]+0.01),self.extent[1]-0.01)
                    pos[1] = min(max(pos[1],self.extent[2]+0.01),self.extent[3]-0.01)
        return pos


class Agent:
    """This class defines an agent moving around the environment. Specifically this class handles the movement policy, and communicates with the environment class to ensure the agent's movement obeys boundaries and walls etc. Initialises with a params dictionary which must contain the Environment (class) in which the agent exists, and also other key parameters required for the motion model. Only has one key function update(dt) which moves the agent along in time by dt.

    A default parameters dictionary (with descriptions) can be fount in __init__()

    List of functions:
        • update()
        • plot_trajectory()
        • animate_trajectory()
        • plot_position_heatmap()

    List of key variables:
        • t
        • pos
        • velocity
        • history
    """

    def __init__(self, 
                 params={}):
        """Sets the parameters of agent (using default if not provided) and initialises everything. This includes: 
        Args:
            params (dict, optional): A dictionary of parameters which you want to differ from the default. Defaults to {}.
        """
        default_params = {
            #the Environment class
            "Environment": None,
            #speed params
            "speed_coherence_time": 3.0,
            "speed_mean": 0.2, 
            "speed_std":0.2, #meaningless in 2D  
            "rotational_velocity_coherence_time":1, 
            "rotational_velocity_std":np.pi/2, 
            # do (if so from how far) walls repel the agent 
            "walls_repel":True,
            "hug_walls" : True,
            "wall_repel_distance":0.1

        }

        update_class_params(self, default_params)
        update_class_params(self, params)

        # initialise history dataframes
        self.history = {}
        self.history["t"] = []
        self.history["pos"] = []
        self.history["vel"] = []
        self.history["rot_vel"] = []

        # time and runID
        self.t = 0
        self.dt=0.01 #defualt dt = 10 ms

        # initialise starting positions and velocity
        if self.Environment.dimensionality == "2D":
            self.pos = self.Environment.sample_positions(n=1,method='random')[0]
            direction = np.random.uniform(0, 2 * np.pi)
            self.velocity = self.speed_std * np.array(
                [np.cos(direction), np.sin(direction)]
            )
            self.rotational_velocity = 0
            

        if self.Environment.dimensionality == "1D":
            self.pos = self.Environment.sample_positions(n=1,method='random')[0]
            self.velocity = np.array([self.speed_mean])
        return

    def update(self, 
               dt=None,
               drift_velocity=None):
        """Movement policy update. 
            In principle this does a very simple thing: 
            • updates time by dt
            • updates velocity (speed and direction) according to a movement policy
            • updates position along the velocity direction 
            In reality it's a complex function as the policy requires checking for immediate or upcoming collisions with all walls at each step as well as handling boundary conditions.
            Specifically the full loop looks like this:
            1) Update time by dt
            2) Update velocity for the next time step. In 2D this is done by varying the agents heading direction and speed according to ornstein-uhlenbeck processes. In 1D, simply the velocity is varied according to ornstein-uhlenbeck. This includes, if turned on, being repelled by the walls.
            3) Propose a new position (x_new =? x_old + velocity.dt)
            3.1) Check if this step collides with any walls (and act accordingly)
            3.2) Check you distance and direction from walls and be repelled by them is necessary
            4) Check position is still within maze and handle boundary conditions appropriately 
            6) Store new position and time in history data frame
        """
        if dt == None: dt = self.dt
        self.dt = dt
        self.t += dt
        
        if self.Environment.dimensionality == '2D':
            # UPDATE VELOCITY there are a number of contributing factors 
            #1 Stochastically update the direction 
            direction = get_angle(self.velocity)
            self.rotational_velocity += ornstein_uhlenbeck(
                dt=dt,
                x=self.rotational_velocity,
                drift=0,
                noise_scale=self.rotational_velocity_std,
                coherence_time=self.rotational_velocity_coherence_time,
            )
            dtheta = self.rotational_velocity*dt
            self.velocity = rotate(self.velocity,dtheta)

            #2 Stochastically update the speed
            speed = np.linalg.norm(self.velocity)
            normal_variable = rayleigh_to_normal(speed,sigma=self.speed_mean)
            new_normal_variable = normal_variable + ornstein_uhlenbeck(
                dt=dt,
                x=normal_variable,
                drift=0,
                noise_scale=1,
                coherence_time=self.speed_coherence_time)
            speed_new = normal_to_rayleigh(new_normal_variable,sigma=self.speed_mean)
            self.velocity = (speed_new/speed)*self.velocity

            # Deterministically drift velocity towards the drift_velocity which has been passed into the update function
            if drift_velocity is not None:
                self.velocity += ornstein_uhlenbeck(
                    dt=dt,
                    x=self.velocity,
                    drift=drift_velocity,
                    noise_scale=0,
                    coherence_time=0.3) #<--- this control how "powerful" this signal is
            
            #Deterministically drift the velocity away from any nearby walls
            if self.walls_repel == True: 
                vectors_from_walls = self.Environment.vectors_from_walls(self.pos) #shape=(N_walls,2)
                if len(self.Environment.walls) > 0:
                    distance_to_walls = np.linalg.norm(vectors_from_walls,axis=-1)
                    normalised_vectors_from_walls = vectors_from_walls / np.expand_dims(distance_to_walls,axis=-1)
                    x, d, v = distance_to_walls, self.wall_repel_distance , self.speed_mean                  
                    if self.hug_walls == False:
                        # Spring acceletation model: in this case this is done by applying an acceleration whenever the agent is near to a wall. this acceleration matches that of a spring with spring constant 3x that of a spring which would, if the agent arrived head on at v = self.speed_mean, turn around exactly at the wall. this is solved by letting d2x/dt2 = -k.x where k = v**2/d**2 (v=seld.speed_mean, d = self.wall_repel_distance)
                        spring_constant = 3*v**2/d**2
                        wall_accelerations = np.piecewise(x=x,
                                condlist=[
                                            (x<=d),
                                            (x>d), ],
                                funclist=[
                                            lambda x: spring_constant*(d-x),
                                            lambda x: 0,])
                        wall_acceleration_vecs = np.expand_dims(wall_accelerations,axis=-1)*normalised_vectors_from_walls
                        wall_acceleration = wall_acceleration_vecs.sum(axis=0)
                        dv = wall_acceleration * dt
                        self.velocity += dv
                    elif self.hug_walls == True:
                        # Conveyor belt drift model. Instead of a spring model this is like a converyor belt model. when < wall_repel_distance from the wall the agents position is updated as though it were on a conveyor belt which moves at the speed of spring mass attached to the wall with starting velocity 5*self.speed_mean. This has a similar effect effect  as the spring model above in that the agent moves away from the wall BUT, crucially the update is made directly to the agents positin, not it's speed, so the next time step will not reflect this update. As a result the agent which is walking into the wall will continue to barge hopelessly into the wall causing it the "hug" close to the wall. 
                        wall_speeds = np.piecewise(x=x,
                                condlist=[
                                            (x<=d),
                                            (x>d), ],
                                funclist=[
                                            lambda x: 5*v*(1 - np.sqrt(1-(d-x)**2/d**2)),
                                            lambda x: 0,])
                        wall_speed_vecs = np.expand_dims(wall_speeds,axis=-1)*normalised_vectors_from_walls
                        wall_speed = wall_speed_vecs.sum(axis=0)
                        dx = wall_speed*dt
                        self.pos += dx


            #proposed position update
            proposed_new_pos = self.pos + self.velocity * dt
            proposed_step = np.array([self.pos, proposed_new_pos])
            wall_check = self.Environment.check_wall_collisions(proposed_step)
            walls = wall_check[0] #shape=(N_walls,2,2)
            wall_collisions = wall_check[1]  #shape=(N_walls,)

            if (wall_collisions is None) or (True not in wall_collisions):
                #it is safe to move to the new position
                self.pos = self.pos + self.velocity * dt
            
            #Bounce off walls you collide with
            elif True in wall_collisions:
                colliding_wall = walls[np.argwhere(wall_collisions==True)[0][0]]
                self.velocity = wall_bounce(self.velocity,colliding_wall)
                self.velocity = (0.5*self.speed_mean/(np.linalg.norm(self.velocity)))*self.velocity
                self.pos += self.velocity * dt

            #handles instances when agent leaves environmnet 
            if self.Environment.check_if_position_is_in_environment(self.pos) is False:
                self.pos = self.Environment.apply_boundary_conditions(self.pos)

            #calculate the velocity of the step that, after all that, was taken. 
            if len(self.history['vel'])>=1:
                last_pos = np.array(self.history["pos"][-1])
                shift = self.Environment.get_vectors_between___accounting_for_environment(pos1=self.pos,pos2=last_pos)
                save_velocity = shift.reshape(-1)/self.dt #accounts for periodic 
            else: save_velocity = self.velocity
            

        elif self.Environment.dimensionality == "1D":
            self.pos = self.pos + dt*self.velocity
            if self.Environment.check_if_position_is_in_environment(self.pos) is False:
                if self.Environment.boundary_conditions == 'solid':
                    self.velocity *= -1
                self.pos = self.Environment.apply_boundary_conditions(self.pos)
                
            self.velocity += ornstein_uhlenbeck(
                dt=dt,
                x=self.velocity,
                drift=self.speed_mean,
                noise_scale=self.speed_std,
                coherence_time=self.speed_coherence_time,
                )
            save_velocity = self.velocity

        # write to history
        self.history["t"].append(self.t)
        self.history["pos"].append(list(self.pos))
        self.history["vel"].append(list(save_velocity))
        if self.Environment.dimensionality == '2D': 
            self.history["rot_vel"].append(self.rotational_velocity)

        return self.pos
    
    def plot_trajectory(self, 
                        t_start=0, 
                        t_end=None, 
                        fig=None, 
                        ax=None,
                        decay_point_size=False,
                        xlim=None
                              ):

        """Plots the trajectory between t_start (seconds) and t_end (defaulting to the last time available)
        Args: 
            • t_start: start time in seconds 
            • t_end: end time in seconds (default = self.history["t"][-1])
            • fig, ax: the fig, ax to plot on top of, optional, if not provided used self.Environment.plot_Environment(). This can be used to plot trajectory on top of receptive fields etc. 
            • decay_point_size: decay trajectory point size over time (recent times = largest)
            xlim: In 1D, force the xlim to be a certain time (useful if animating this function)
        Returns:
            fig, ax
        """        
        dt = self.dt
        t, pos = np.array(self.history["t"]), np.array(self.history["pos"])
        approx_speed = max(self.speed_std,self.speed_mean)
        if t_end == None:
            t_end = t[-1]
        startid = np.argmin(np.abs(t - (t_start)))
        endid = np.argmin(np.abs(t - (t_end)))
        if self.Environment.dimensionality == '2D':
            scatter_distance = 0.02
            skiprate = max(1, int(scatter_distance / (approx_speed * dt)))
            trajectory = pos[startid:endid, :][::skiprate]
        if self.Environment.dimensionality == '1D':
            scatter_distance = 0.02
            skiprate = max(1, int(scatter_distance / (approx_speed * dt)))
            trajectory = pos[startid:endid][::skiprate]
        time = t[startid:endid][::skiprate]

        if self.Environment.dimensionality == "2D":
            # time = t[startid:endid][::skiprate]
            if fig is None and ax is None:
                fig, ax = self.Environment.plot_environment()
            s=15 * np.ones_like(time)
            if decay_point_size == True:
                s = 15*np.exp((time - time[-1])/10)
                s[(time[-1]-time)>15]*=0
            c = ["C0"]*len(time)
            s[-1]=40
            c[-1]='r'
            ax.scatter(
                trajectory[:, 0],
                trajectory[:, 1],
                s=s,
                alpha=0.7,
                zorder=2,
                c=c,
                linewidth=0,
            )
        if self.Environment.dimensionality == "1D":
            if fig is None and ax is None:
                fig, ax = plt.subplots(figsize=(6, 3))
            ax.scatter(time/60, trajectory,alpha=0.7)
            ax.spines["left"].set_position(("data", t[startid]))
            ax.set_xlabel("Time / min")
            ax.set_ylabel("Position / m")
            if xlim is not None: 
                ax.set_xlim(right=xlim)
            ax.set_ylim(bottom=0,top=self.Environment.extent[1])
            ax.spines["right"].set_color(None)
            ax.spines["top"].set_color(None)
            ax.set_xticks([t_start/60,t_end/60])
            ex = self.Environment.extent
            ax.set_yticks([ex[0],ex[1]])

        return fig, ax
    
    def animate_trajectory(self,
                           t_end=None,
                           speed_up=1):
        """Returns an animation (anim) of the trajectory, 20fps. 
        Should be saved using comand like 
        anim.save("./where_to_save/animations.gif",dpi=300)

        Args:
            t_end (_type_, optional): _description_. Defaults to None.
            speed_up: #times real speed animation should come out at 

        Returns:
            animation
        """        
        if t_end == None: 
            t_end = self.history['t'][-1]

        def animate(i,fig,ax,t_max,speed_up):
            t = self.history['t']
            t_start = t[0]
            t_end = t[0]+(i+1)*speed_up*50e-3
            ax.clear()
            if self.Environment.dimensionality == '2D':
                fig, ax = self.Environment.plot_environment(fig=fig,ax=ax)
                xlim=None
            if self.Environment.dimensionality == '1D':
                xlim=t_max
            fig, ax = self.plot_trajectory(t_start=t_start,t_end=t_end, fig=fig, ax=ax, decay_point_size=True,xlim=xlim)
            plt.close()
            return 

        fig,ax=self.plot_trajectory(0,10*self.dt)
        anim = matplotlib.animation.FuncAnimation(fig,animate,interval=50,frames=int(t_end/50e-3),blit=False,fargs=(fig,ax,t_end,speed_up))
        return anim

    def plot_position_heatmap(self,
                              dx=None,
                              weights=None,
                              fig=None,
                              ax=None):
        """Plots a heatmap of postions the agent has been in. vmin is always set to zero, so the darkest colormap color (if seen) represents locations which have never been visited 
        Args:
            dx (float, optional): The heatmap bin size. Defaults to 5cm in 2D or 1cm in 1D.
        """        
        if self.Environment.dimensionality == '1D':
            if dx is None: dx = 0.01
            pos = np.array(self.history["pos"])
            ex = self.Environment.extent
            if fig is None and ax is None: 
                fig, ax = self.Environment.plot_environment(height=1)
            heatmap, centres = bin_data_for_histogramming(data=pos,extent=ex,dx=dx)
            #maybe do smoothing? 
            ax.plot(centres,heatmap)
            ax.fill_between(centres,0,heatmap,alpha=0.3)
            ax.set_ylim(top=np.max(heatmap)*1.2)
            return fig, ax
        
        elif self.Environment.dimensionality == '2D':
            if dx is None: dx = 0.05
            pos = np.array(self.history["pos"])
            ex = self.Environment.extent
            heatmap=bin_data_for_histogramming(data=pos,extent=ex,dx=dx)
            if fig == None and ax == None: 
                fig, ax = self.Environment.plot_environment()
            vmin=0; vmax=np.max(heatmap)
            ax.imshow(heatmap, extent=ex, interpolation='bicubic',
            vmin=vmin,vmax=vmax,
            )
        return fig, ax

    def plot_histogram_of_speeds(self,
                                 fig = None,
                                 ax = None,
                                 color='C1'):
        """Plots a histogram of the observed speeds of the agent. 
        args:
            fig, ax: not required. the ax object to be drawn onto.  
            color: optional. the color.
        Returns:
            fig, ax: the figure
        """     
        velocities = np.array(self.history['vel'])
        speeds = np.linalg.norm(velocities,axis=1)
        if (fig is None) and (ax is None):
            fig, ax = plt.subplots()
        ax.hist(speeds,bins=50,color=color, alpha=0.8,density=True)
        return fig, ax 

    def plot_histogram_of_rotational_velocities(self,
                                 fig = None,
                                 ax = None,
                                 color='C1'):
        """Plots a histogram of the observed speeds of the agent. 
        args:
            fig, ax: not required. the ax object to be drawn onto.  
            color: optional. the color.
        Returns:
            fig, ax: the figure
        """        
        rot_vels = np.array(self.history['rot_vel'])
        if (fig is None) and (ax is None):
            fig, ax = plt.subplots()
        ax.hist(rot_vels,bins=50,color=color, alpha=0.8, density=True)
        return fig, ax 

class Neurons:
    """The Neuron class defines a population of Neurons. All Neurons have firing rates which depend explicity on (that is, they "encode" -- if I'm still allowed to say that) the state of the Agent. As the Agent moves the firing rate of the cells adjust accordingly. 

    The input dictionary 'params'  must contain the Agent class (who's position/velocity etc. determines how these neurons fire. Agent will itself contains the Environment - who's geometry and distance calculating abilities may be important for determining firing rates). 
   
    The next most key parameter is "cell_class". We current support four types of cells. Each of these has their own specific params (). 
        • 'cell_type':'place_cells'
                • 'width'
                    float, in meters 
                • 'description'
                    'gaussian'/'gaussian_threshold'/'diff_of_gaussians'/'top_hat'/'one_hot'
                • 'place_cell_centres'
                    Can be None. Or can be array or locations. 
                • 'wall_geometry'
                    'geodesic'/'euclidean'/'line_of_sight'
        • 'grid_cells'
                • 'gridscale':
                    float, meters 
                • 'random_orientations'
                    bool
                • 'random_gridscales'
                    bool
        • 'boundary_vector_cells'
        • 'velocity_cells'
        
    The key function here is "update_state()" which pulls the position, velocity etc. from the Agent and sets the firing rate of the cells accordingly. The neuron firing rates can also be queried for any arbitrary pos/velocity (i.e. not just the Agents current state) by passing these in directly to the function "get_state()".
    
    Importantly, a key and very useful ability of this class is that the firing rate of the neurons can be queried not just at a single position but in an array of positions. Doing so is a vectorised calculation so is relatively fast. It means (for position dependent cells, e.g. place_cells,  grid_cells, boundary_vector_cells) you can query their firing rate at all positions across the environment and see their receptive fields, by calling get_state(pos='all'). This first asks the Environment for an array of discretised positions and then returns the firing rate of neurons at all of them. (more generally  pos can be any array of positions shape=(N_pos,N_dimensionality)).

    Because geometry calculation come from the Environment class, Neurons obey the geometry and boundary conditions defined there. For example, if geometry is "line_of_sight" then a place cell will only fire if there is a direct line of sight between a position and its centre location (i.e. no obstructing walls). Another example, if boundary conditons are periodic, place cell, grid cell firing rates will respect this and loop over the edges. 


    List of key functions...
        ..that you're likely to use: 
            • update() updates firing rates according to the agent. Saves them
            • get_state(), gets firing rate, doesn't necessarily save them and positions/velocities don't have to come from the Agent
            • plot_rate_timeseries()
            • plot_rate_map()
            • plot_place_cell_locations() 
        ...that you probably won't directly use:
            • boundary_vector_preference_function()
    

    Finally, we intend for this to be a parent class. Feel free to write your own update() and get_state() functions (e.g. perhaps these neurons are the weighted sum of other neurons). As long as you save the firing rate on each step the plotting functions will still work. 
        
    """
    def __init__(self, 
                 params={}):
        """Initialises the Neurons class. Requires params dictionary. 
        If params is not provided/edits you, by default, will get 10 randomly distributed gaussian place cells of std 0.2m

        Args:
            params (dict, optional): _description_. Defaults to {}.
        """        
        default_params = {
            "Agent": None,
            "n":10,
            "cell_class":"place_cell",            
            #default place cell params
                'description':'gaussian',
                'widths':0.20,
                'place_cell_centres':None, #if given this will overwrite 'n',
                'wall_geometry':'geodesic',
            #default grid cell params
                "gridscale":0.45,
                "random_orientations":True,
                "random_gridscales":True,
            #default boundary vector cell params
            "color": None, #just for plotting
            "min_fr": 0,
            "max_fr": 1,
        }
        update_class_params(self, default_params)
        update_class_params(self, params)

        self.history = {}
        self.history["t"] = []
        self.history["firingrate"] = []
        self.history["spikes"] = []

        allowed_cell_classes = ['place_cell','grid_cell','boundary_vector_cell','velocity_cell']
        assert self.cell_class in allowed_cell_classes, f"cell class {self.cell_class} is not defined, must be in {allowed_cell_classes}"
        #Initialise place cells 
        if self.cell_class == 'place_cell':
            if self.place_cell_centres is None:
                self.place_cell_centres = self.Agent.Environment.sample_positions(n=self.n,method='uniform_jitter')
                np.random.shuffle(self.place_cell_centres)
            else: 
                self.n = self.place_cell_centres.shape[0]
            self.place_cell_widths = self.widths*np.ones(self.n)

        #Initialise grid cells 
        if self.cell_class == 'grid_cell':
            assert self.Agent.Environment.dimensionality == '2D', 'grid cells only available in 2D'
            self.phase_offsets = np.random.uniform(0,self.gridscale,size=(self.n,2))
            w = []
            for i in range(self.n):
                w1 = np.array([1,0]) 
                if self.random_orientations == True:
                    w1 = rotate(w1,np.random.uniform(0,2*np.pi))
                w2 = rotate(w1,np.pi/3)
                w3 = rotate(w1,2*np.pi/3)
                w.append(np.array([w1,w2,w3]))
            self.w = np.array(w)
            if self.random_gridscales == True: 
                self.gridscales = np.random.uniform(2*self.gridscale/3,1.5*self.gridscale,size=self.n)

        #Initialise velocity cells
        if self.cell_class == 'velocity_cell':
            if self.Agent.Environment.dimensionality=="2D":
                self.n = 4 #one up, one down, one left, one right
            if self.Agent.Environment.dimensionality=="1D":
                self.n = 2 #one left, one right
            self.one_sigma_speed = self.Agent.speed_mean + self.Agent.speed_std

        #Initialise boundary vector cells 
        if self.cell_class == 'boundary_vector_cell':
            assert self.Agent.Environment.dimensionality == '2D', "boundary cells only possible in 2D"
            assert self.Agent.Environment.boundary_conditions == 'solid', "boundary cells only possible with solid boundary conditions"
            xi = 0.08 #as in de cothi and barry 2020
            beta = 12
            test_direction = np.array([1,0])
            test_directions = [test_direction]
            test_angles = [0]
            self.n_test_angles = 360
            self.dtheta = 2*np.pi/self.n_test_angles
            for i in range(self.n_test_angles-1):
                test_direction_ = rotate(test_direction,2*np.pi*i/360)
                test_directions.append(test_direction_)
                test_angles.append(2*np.pi*i/360)
            self.test_directions = np.array(test_directions)
            self.test_angles = np.array(test_angles)
            self.sigma_angle = (11.25/360)*2*np.pi #11.25 degrees as in de Cothi and Barry 2020
            self.tuning_angles = np.random.uniform(0,2*np.pi,size=self.n)
            self.tuning_distances = np.maximum(0,np.random.normal(loc=0.15,
                                                             scale=0.1,
                                                             size=self.n))
            self.sigma_distances = self.tuning_distances/beta + xi

    def get_state(self,
                  pos="from_agent",
                  vel="from_agent"):
        """
        Returns the firing rate of the neurons. 
        pos can be
            • np.array(): an array of locations 
            • 'from_agent': uses Agent.pos 
            • 'all' : array of locations over entire environment (for rate map plotting) 
        vel can be:
            • np.array()
        Returns the firing rate of the neuron at all positions shape (N_cells,N_pos)
                """
        
        if self.cell_class == 'place_cell':
            if pos == "from_agent":
                pos = self.Agent.pos
            elif pos == "all":
                pos = self.Agent.Environment.flattened_discrete_coords

            #place cell fr's depend only on how far the agent is from cell centres (and their widths)
            dist = self.Agent.Environment.get_distances_between___accounting_for_environment(
                self.place_cell_centres,
                pos,wall_geometry=self.wall_geometry) #distances to place cell centres
            widths = np.expand_dims(self.place_cell_widths,axis=-1)

            if self.description == "gaussian":
                firingrate = np.exp(-(dist ** 2) / (2 * (widths ** 2)))
            if self.description == "gaussian_threshold":
                firingrate = np.maximum(
                    np.exp(-(dist ** 2) / (2 * (widths ** 2)))
                    - np.exp(-1 / 2),0,) / (1 - np.exp(-1 / 2))
            if self.description == "diff_of_gaussians":
                ratio = 1.5
                firingrate = np.exp(
                    -(dist ** 2) / (2 * (widths ** 2))) - (1 / ratio ** 2) * np.exp(
                    -(dist ** 2) / (2 * ((ratio * widths) ** 2)))
                firingrate *= ratio ** 2 / (ratio ** 2 - 1)
            if self.description == "one_hot":
                closest_centres = np.argmin(np.abs(dist), axis=0)
                firingrate = np.eye(self.n)[closest_centres].T
            if self.description == "top_hat":
                closest_centres = np.argmin(np.abs(dist), axis=0)
                firingrate = 1*(dist < 1)
        
        elif self.cell_class == 'grid_cell':
            """grid cells are modelled as the threwsholded sum of three cosines all at 60 degree offsets"""
            if pos == "from_agent":
                pos = self.Agent.pos
            elif pos == "all":
                pos = self.Agent.Environment.flattened_discrete_coords
            pos = pos.reshape(-1,pos.shape[-1])

            #vectors to grids cells "centred" at their (random) phase offsets 
            vecs = get_vectors_between(self.phase_offsets,pos) #shape = (N_cells,N_pos2)
            w1 = np.tile(np.expand_dims(self.w[:,0,:],axis=1),reps=(1,pos.shape[0],1))
            w2 = np.tile(np.expand_dims(self.w[:,1,:],axis=1),reps=(1,pos.shape[0],1))
            w3 = np.tile(np.expand_dims(self.w[:,2,:],axis=1),reps=(1,pos.shape[0],1))
            gridscales = np.tile(np.expand_dims(self.gridscales,axis=1),reps=(1,pos.shape[0]))
            phi_1 = ((2*np.pi)/gridscales)*(vecs*w1).sum(axis=-1)
            phi_2 = ((2*np.pi)/gridscales)*(vecs*w2).sum(axis=-1)
            phi_3 = ((2*np.pi)/gridscales)*(vecs*w3).sum(axis=-1)
            firingrate = 0.5*((np.cos(phi_1) + np.cos(phi_2) + np.cos(phi_3)))
            firingrate[firingrate<0] = 0

        elif self.cell_class == 'velocity_cell':
            """In 2D 4 velocity cells report, respectively, the thresholded leftward, rightward, upward and downwards velocity"""
            if vel == 'from_agent':
                vel = self.Agent.history['vel'][-1]
            if self.Agent.Environment.dimensionality == '1D':
                vleft_fr = max(0,vel[0])/self.one_sigma_speed
                vright_fr = max(0,-vel[0])/self.one_sigma_speed
                firingrate = np.array([vleft_fr,vright_fr])
            if self.Agent.Environment.dimensionality == '2D':
                vleft_fr = max(0,vel[0])/self.one_sigma_speed
                vright_fr = max(0,-vel[0])/self.one_sigma_speed                    
                vup_fr = max(0,vel[1])/self.one_sigma_speed
                vdown_fr = max(0,-vel[1])/self.one_sigma_speed
                firingrate = np.array([vleft_fr,vright_fr,vup_fr,vdown_fr])
        
        elif self.cell_class == 'boundary_vector_cell':
            """Here we implement the same type if boundary vector cells as de Cothi et al. (2020), who follow Barry & Burgess, (2007). See equations there. 
        
            The way I do this is a little complex. I will describe how it works from a single position (but remember this can be called in a vectorised manner from an arary of positons in parallel)
                1. An array of normalised "test vectors" span, in all directions at 1 degree increments, from the position
                2. These define an array of line segments stretching from [pos, pos+test vector]
                3. Where these line segments collide with all walls in the environment is established, this uses the function "vector_intercepts()"
                4. This pays attention to only consider the first (closest) wall forawrd along a line segment. Walls behind other walls are "shaded" by closer walls. Its a little complex to do this and requires the function "boundary_vector_preference_function()"
                5. Now that, for every test direction, the closest wall is established it is simple a process of finding the response of the neuron to that wall at that angle (multiple of two gaussians, see de Cothi (2020)) and then summing over all the test angles. 
        """
            if pos == "from_agent":
                pos = self.Agent.pos
            elif pos == "all":
                pos = self.Agent.Environment.flattened_discrete_coords
            N_cells = self.n
            pos = pos.reshape(-1,pos.shape[-1]) #(N_pos,2)
            N_pos = pos.shape[0]
            N_test = self.test_angles.shape[0] 
            pos_line_segments = np.tile(np.expand_dims(np.expand_dims(pos,axis=1),axis=1),reps=(1,N_test,2,1))#(N_pos,N_test,2,2) 
            test_directions_tiled = np.tile(np.expand_dims(self.test_directions,axis=0),reps=(N_pos,1,1)) #(N_pos,N_test,2)
            pos_line_segments[:,:,1,:] += test_directions_tiled #(N_pos,N_test,2,2)
            pos_line_segments = pos_line_segments.reshape(-1,2,2) #(N_pos x N_test,2,2)
            walls = self.Agent.Environment.walls #(N_walls,2,2)
            N_walls = walls.shape[0] 
            pos_lineseg_wall_intercepts = vector_intercepts(pos_line_segments,walls) #(N_pos x N_test,N_walls,2)
            pos_lineseg_wall_intercepts = pos_lineseg_wall_intercepts.reshape((N_pos,N_test,N_walls,2))#(N_pos,N_test,N_walls,2)
            dist_to_walls = pos_lineseg_wall_intercepts[:,:,:,0]#(N_pos,N_test,N_walls)
            first_wall_for_each_direction = self.boundary_vector_preference_function(pos_lineseg_wall_intercepts)#(N_pos,N_test,N_walls)
            first_wall_for_each_direction_id = np.expand_dims(np.argmax(first_wall_for_each_direction,axis=-1),axis=-1) #(N_pos,N_test,1)
            dist_to_first_wall = np.take_along_axis(dist_to_walls,first_wall_for_each_direction_id,axis=-1).reshape((N_pos,N_test)) #(N_pos,N_test)
            #reshape everything to have shape (N_cell,N_pos,N_test)

            test_angles = np.tile(np.expand_dims(np.expand_dims(self.test_angles,axis=0),axis=0),reps=(N_cells,N_pos,1))#(N_cell,N_pos,N_test)
            tuning_angles = np.tile(np.expand_dims(np.expand_dims(self.tuning_angles,axis=-1),axis=-1),reps=(1,N_pos,N_test))#(N_cell,N_pos,N_test)
            sigma_angle = np.tile(np.expand_dims(np.expand_dims(np.expand_dims(np.array(self.sigma_angle),axis=0),axis=0),axis=0),reps=(N_cells,N_pos,N_test))#(N_cell,N_pos,N_test)
            tuning_distances = np.tile(np.expand_dims(np.expand_dims(self.tuning_distances,axis=-1),axis=-1),reps=(1,N_pos,N_test))#(N_cell,N_pos,N_test)
            sigma_distances = np.tile(np.expand_dims(np.expand_dims(self.sigma_distances,axis=-1),axis=-1),reps=(1,N_pos,N_test))#(N_cell,N_pos,N_test)
            dist_to_first_wall = np.tile(np.expand_dims(dist_to_first_wall,axis=0),reps=(N_cells,1,1))#(N_cell,N_pos,N_test)

            g = gaussian(dist_to_first_wall,tuning_distances,sigma_distances) * gaussian(test_angles,tuning_angles,sigma_angle) #(N_cell,N_pos,N_test)

            firingrate = g.mean(axis=-1) #(N_cell,N_pos)
                
        firingrate = firingrate * (self.max_fr - self.min_fr) + self.min_fr #scales from being between [0,1] to [min_fr, max_fr]
        return firingrate

    def update(self):
        """Updates the state of these state neurons using the Agents current position. Stores the current firing rate in self.phi_U
        """
        firingrate = self.get_state(pos="from_agent")
        self.firingrate = firingrate.reshape(-1)
        cell_spikes = (np.random.uniform(0,1,size=(self.n,)) < (self.Agent.dt*self.firingrate))

        self.history["t"].append(self.Agent.t)
        self.history["firingrate"].append(list(self.firingrate))
        self.history["spikes"].append(list(cell_spikes))

    def plot_rate_timeseries(self,
                            t_start=0,
                            t_end=None,
                            chosen_neurons='10',
                            plot_spikes=True,
                            fig=None,
                            ax=None,
                            xlim=None):
        """Plots a timeseries of the firing rate of the neurons between t_start and t_end

        Args:
            t_start (int, optional): _description_. Defaults to 0.
            t_end (int, optional): _description_. Defaults to 60.
            chosen_neurons (str, optional): Which neurons to plot. string "10" will plot 10 of them, "all" will plot all of them, a list like [1,4,5] will plot cells indexed 1, 4 and 5. Defaults to "10".
            plot_spikes (bool, optional): If True, scatters exact spike times underneath each curve of firing rate. Defaults to True.
            the below params I just added for help with animations
            fig, ax: the figure, axis to plot on (can be None)
            xlim: fix xlim of plot irrespective of how much time you're plotting 
        Returns:
            fig, ax
        """        
        t = np.array(self.history["t"])
        #times to plot 
        if t_end is None:
            t_end = t[-1]
        startid = np.argmin(np.abs(t - (t_start)))
        endid = np.argmin(np.abs(t - (t_end)))
        rate_timeseries = np.array(self.history['firingrate'])
        spikes = np.array(self.history['spikes'])
        t = t[startid:endid]
        rate_timeseries = rate_timeseries[startid:endid]
        spikes = spikes[startid:endid]
        #neurons to plot
        if chosen_neurons == "all":
            chosen_neurons = np.arange(self.n)
        if type(chosen_neurons) is str:
            if chosen_neurons.isdigit():
                chosen_neurons = np.linspace(0,self.n-1, int(chosen_neurons)).astype(int)

        firingrates = rate_timeseries[:,chosen_neurons].T
        fig, ax = mountain_plot(
                X = t/60, 
                NbyX = firingrates, 
                color=self.color,
                xlabel="Time / min",
                ylabel="Neurons",
                xlim=None,
                fig=fig,
                ax=ax,
            )
        
        if plot_spikes == True: 
            for i in range(len(chosen_neurons)):
                time_when_spiked = t[spikes[:,chosen_neurons[i]]]/60
                h = (i+1-0.1)*np.ones_like(time_when_spiked)
                ax.scatter(time_when_spiked,h,color=self.color,alpha=0.5,s=1)

        ax.set_xticks([t_start/60,t_end/60])
        if xlim is not None: 
            ax.set_xlim(right=xlim/60)
            ax.set_xticks([0,xlim/60])
        
        return fig, ax

    def plot_rate_map(self, 
                      chosen_neurons="all", 
                      plot_spikes=True,
                      by_history=False,
                      fig=None,
                      ax=None):
        """Plots rate maps of neuronal firing rates across the environment
        Args:
            chosen_neurons (): Which neurons to plot. string "10" will plot 10 of them, "all" will plot all of them, a list like [1,4,5] will plot cells indexed 1, 4 and 5. Defaults to "10".
            
            plot_spikes: if True, also scatters points where the neuron spiked
            
            by_history: When True, instead of explicitly evaluating the firing rate of the neuron at all points on the environment this just uses the history of the past positions weighted by the firing rate to create an observed raster plot. This is a more robust way to plot the receptive field as it does not require the ability to analytically find firing rate at position x, rather it just needs historic data of [x0,x1,x2...] and [fr0,fr1,fr2...]

        Returns:
            fig, ax 
        """        
        rate_maps = self.get_state(pos="all")
        rate_timeseries = np.array(self.history['firingrate']).T
        spikes = np.array(self.history['spikes']).T

        if chosen_neurons == "all":
            chosen_neurons = np.arange(self.n)
        if type(chosen_neurons) is str:
            if chosen_neurons.isdigit():
                chosen_neurons = np.linspace(0,self.n-1, int(chosen_neurons)).astype(int)

        if self.Agent.Environment.dimensionality == "2D":
            if self.color is None: coloralpha = (1,1,1,0)
            else:
                color = list(matplotlib.colors.to_rgba(self.color))
                coloralpha = color; coloralpha[-1]=0.5 
            if fig is None and ax is None: 
                fig, ax = plt.subplots(
                    1,
                    len(chosen_neurons),
                    figsize=(3 * len(chosen_neurons), 3 * 1),
                    facecolor=coloralpha,
            )
            if not hasattr(ax, "__len__"):
                ax = [ax]
            for (i, ax_) in enumerate(ax):
                self.Agent.Environment.plot_environment(fig, ax_)
                if by_history == False: 
                    rate_map = rate_maps[chosen_neurons[i], :].reshape(self.
                    Agent.Environment.discrete_coords.shape[:2])
                    im = ax_.imshow(rate_map, extent=self.Agent.Environment.extent)
                elif by_history == True: 
                    ex = self.Agent.Environment.extent
                    pos = np.array(self.Agent.history['pos'])
                    rate_timeseries_ = rate_timeseries[chosen_neurons[i], :]
                    rate_map = bin_data_for_histogramming(
                        data=pos,
                        extent=ex,
                        dx=0.05,
                        weights=rate_timeseries_
                    )
                    im = ax_.imshow(rate_map, extent=ex,interpolation='bicubic')
                if plot_spikes == True: 
                    if len(spikes >= 1):
                        pos = np.array(self.Agent.history['pos'])
                        pos_where_spiked = pos[spikes[chosen_neurons[i], :]]
                        ax_.scatter(pos_where_spiked[:,0],pos_where_spiked[:,1])
                    else: 
                        pass 

            return fig, ax

        if self.Agent.Environment.dimensionality == "1D":
            if by_history == False: 
                rate_maps = rate_maps[chosen_neurons, :]
                x = self.Agent.Environment.flattened_discrete_coords[:, 0]
            elif by_history == True: 
                ex = self.Agent.Environment.extent
                pos = np.array(self.Agent.history['pos'])[:,0]
                rate_maps = []
                for neuron_id in chosen_neurons:
                    rate_map, x = (
                        bin_data_for_histogramming(
                            data=pos,
                            extent=ex,
                            dx=0.01,
                            weights=rate_timeseries[neuron_id,:])
                    )
                    x, rate_map = interpolate_and_smooth(x,rate_map,sigma=0.03)
                    rate_maps.append(rate_map)
                rate_maps = np.array(rate_maps)
            
            if fig is None and ax is None:
                fig, ax = self.Agent.Environment.plot_environment(height=len(chosen_neurons))
            fig, ax = mountain_plot(
                X = x, 
                NbyX = rate_maps, 
                color=self.color,
                xlabel="Position / m",
                ylabel="Neurons",
                fig=fig,
                ax=ax,
            )
            
            if plot_spikes == True: 
                if len(spikes >= 1):
                    for i in range(len(chosen_neurons)):
                        pos = np.array(self.Agent.history['pos'])[:,0]
                        pos_where_spiked = pos[spikes[chosen_neurons[i]]]
                        h = (i-0.1)*np.ones_like(pos_where_spiked)
                        ax.scatter(pos_where_spiked,h,color=self.color,alpha=0.5,s=1)
                else: 
                    pass 

        return fig, ax
        
    def plot_place_cell_locations(self):
        assert self.cell_class == 'place_cell', 'only place cells have well defined centres for plotting'
        fig, ax = self.Agent.Environment.plot_environment()
        place_cell_centres = self.place_cell_centres
        ax.scatter(
            place_cell_centres[:, 0], place_cell_centres[:, 1], c="C1", marker="x", s=15, zorder=2
        )
        return fig, ax
    
    def animate_rate_timeseries(self,
                                t_end=None,
                                chosen_neurons='all',
                                speed_up=1):
        """Returns an animation (anim) of the firing rates, 25fps. 
        Should be saved using comand like 
        anim.save("./where_to_save/animations.gif",dpi=300)

        Args:
            t_end (_type_, optional): _description_. Defaults to None.
            chosen_neurons: neurons to plot (as define in, e.g., plot_rate_map())
            speed_up: #times real speed animation should come out at. 

        Returns:
            animation
        """        

        if t_end == None: 
            t_end = self.history['t'][-1]

        def animate(i,fig,ax,chosen_neurons,t_max,speed_up):
            t = self.history['t']
            t_start = t[0]
            t_end = t[0]+(i+1)*speed_up*50e-3
            ax.clear()
            fig, ax = self.plot_rate_timeseries(t_start=t_start,t_end=t_end, chosen_neurons=chosen_neurons, plot_spikes=True, fig=fig, ax=ax,xlim=t_max)
            plt.close()
            return 

        fig, ax = self.plot_rate_timeseries(t_start=0,t_end=10*self.Agent.dt,chosen_neurons=chosen_neurons,xlim=t_end)
        anim = matplotlib.animation.FuncAnimation(fig,animate,interval=50,frames=int(t_end/50e-3),blit=False,fargs=(fig,ax,chosen_neurons,t_end,speed_up))
        return anim

    def boundary_vector_preference_function(self,x):
        """This is a random function needed to efficiently produce boundary vector cells. x is any array of final dimension shape shape[-1]=2. As I use it here x has the form of the output of vector_intercepts. I.e. each point gives shape[-1]=2 lambda values (lam1,lam2) for where a pair of line segments intercept. This function gives a preference for each pair. Preference is -1 if lam1<0 (the collision occurs behind the first point) and if lam2>1 or lam2<0 (the collision occurs ahead of the first point but not on the second line segment). If neither of these are true it's 1/x (i.e. it prefers collisions which are closest).

        Args:
            x (array): shape=(any_shape...,2)

        Returns:
            the preferece values: shape=(any_shape)
        """        
        assert x.shape[-1] == 2
        pref = np.piecewise(x=x,
                            condlist=(
                                x[...,0]>0,
                                x[...,0]<0,
                                x[...,1]<0,
                                x[...,1]>1,
                                ),
                            funclist=(
                                1/x[x[...,0]>0],
                                -1,
                                -1,
                                -1,
                            ))
        return pref[...,0]






"""
OTHER USEFUL FUNCTIONS
Split into the following catergories: 
• Geometry-assistance functions
• Stochasticic-assistance functions
• Plotting-assistance functions 
• Other
"""

"""Geometry functions"""
def get_perpendicular(a=None):
    """Given 2-vector, a, returns its perpendicular
    Args:
        a (array, optional): 2-vector direction. Defaults to None.
    Returns:
        array: perpendicular to a
    """
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def vector_intercepts(vector_list_a,
                      vector_list_b,
                      return_collisions=False):
    """
    Each element of vector_list_a gives a line segment of the form [[x_a_0,y_a_0],[x_a_1,y_a_1]], or, in vector notation [p_a_0,p_a_1] (same goes for vector vector_list_b). Thus
        vector_list_A.shape = (N_a,2,2)
        vector_list_B.shape = (N_b,2,2)
    where N_a is the number of vectors defined in vector_list_a

    Each line segments define an (infinite) line, parameterised by line_a = p_a_0 + l_a.(p_a_1-p_a_0). We want to find the intersection between these lines in terms of the parameters l_a and l_b. Iff l_a and l_b are BOTH between 0 and 1 then the line segments intersect. Thus the goal is to return an array, I,  of shape 
        I.shape = (N_a,N_b,2)
    where, if I[n_a,n_b][0] and I[n_a,n_b][1] are both between 0 and 1 then it means line segments vector_list_a[n_a] and vector_list_b[n_b] intersect. 

    To do this we consider solving the equation line_a = line_b. The solution to this is:
        l_a = dot((p_b_0 - p_a_0) , (p_b_1 - p_b_0)_p) / dot((p_a_1 - p_a_0) , (p_b_1 - p_b_0)_p)
        l_b = dot((p_a_0 - p_b_0) , (p_a_1 - p_a_0)_p) / dot((p_b_1 - p_b_0) , (p_a_1 - p_a_0)_p)
    where "_p" denotes the perpendicular (in two-D [x,y]_p = [-y,x]). Using notation
        l_a = dot(d0,sb_p) / dot(sa,sb_p)
        l_b = dot(-d0,sa_p) / dot(sb,sa_p)
    for 
        d0 = p_b_0 - p_a_0
        sa = p_a_1 - p_a_0 
        sb = p_b_1 - p_b_0
    We will calculate these first.

    If return_collisions == True, the list of intercepts is used to assess whether each pair of segments actually collide (True) or not (False) and this bollean array (shape = (N_a,N_b)) is returned instead.
    """
    assert (vector_list_a.shape[-2:] == (2,2)) and (vector_list_b.shape[-2:] == (2,2)), "vector_list_a and vector_list_b must be shape (_,2,2), _ is optional"
    vector_list_a = vector_list_a.reshape(-1,2,2)
    vector_list_b = vector_list_b.reshape(-1,2,2)
    vector_list_a = vector_list_a + np.random.normal(scale=1e-6,size=vector_list_a.shape)
    vector_list_b = vector_list_b + np.random.normal(scale=1e-6,size=vector_list_b.shape)

    N_a = vector_list_a.shape[0]
    N_b = vector_list_b.shape[0]

    d0 = np.expand_dims(vector_list_b[:,0,:],axis=0) - np.expand_dims(vector_list_a[:,0,:],axis=1) #d0.shape = (N_a,N_b,2)
    sa = vector_list_a[:,1,:] - vector_list_a[:,0,:] #sa.shape = (N_a,2)
    sb = vector_list_b[:,1,:] - vector_list_b[:,0,:]#sb.shape = (N_b,2)
    sa_p = np.flip(sa.copy(),axis=1)
    sa_p[:,0] = -sa_p[:,0] #sa_p.shape = (N_a,2)
    sb_p = np.flip(sb.copy(),axis=1)
    sb_p[:,0] = -sb_p[:,0] #sb.shape = (N_b,2)

    """Now we can go ahead and solve for the line segments
    since d0 has shape (N_a,N_b,2) in order to perform the dot product we must first reshape sa (etc.) by tiling to shape (N_a,N_b,2)
    """
    sa = np.tile(np.expand_dims(sa,axis=1),reps=(1,N_b,1)) #sa.shape = (N_a,N_b,2)
    sb = np.tile(np.expand_dims(sb,axis=0),reps=(N_a,1,1)) #sb.shape = (N_a,N_b,2)
    sa_p = np.tile(np.expand_dims(sa_p,axis=1),reps=(1,N_b,1)) #sa.shape = (N_a,N_b,2)
    sb_p = np.tile(np.expand_dims(sb_p,axis=0),reps=(N_a,1,1)) #sb.shape = (N_a,N_b,2)
    """The dot product can now be performed by broadcast multiplying the arraays then summing over the last axis"""
    l_a = (d0*sb_p).sum(axis=-1) / (sa*sb_p).sum(axis=-1) #la.shape=(N_a,N_b)
    l_b = (-d0*sa_p).sum(axis=-1) / (sb*sa_p).sum(axis=-1) #la.shape=(N_a,N_b)

    intercepts = np.stack((l_a,l_b),axis=-1)
    if return_collisions == True: 
        direct_collision = ((intercepts[:,:,0] > 0) *
                        (intercepts[:,:,0] < 1) *
                        (intercepts[:,:,1] > 0) *
                        (intercepts[:,:,1] < 1))
        return direct_collision
    else: 
        return intercepts

def shortest_vectors_from_points_to_lines(positions,
                                          vectors): 
    """
    Takes a list of positions and a list of vectors (line segments) and returns the pairwise  vectors of shortest distance FROM the vector segments TO the positions. 
    Suppose we have a list of N_p positions and a list of N_v line segments (or vectors). Each position is a point like [x_p,y_p], or p_p as a vector. Each vector is defined by two points [[x_v_0,y_v_0],[x_v_1,y_v_1]], or [p_v_0,p_v_1]. Thus 
        positions.shape = (N_p,2)
        vectors.shape = (N_v,2,2)
    
    Each vector defines an infinite line, parameterised by line_v = p_v_0 + l_v . (p_v_1 - p_v_0). We want to solve for the l_v defining the point on the line with the shortest distance to p_p. This is given by:
        l_v = dot((p_p-p_v_0),(p_v_1-p_v_0)/dot((p_v_1-p_v_0),(p_v_1-p_v_0)). 
    Or, using a diferrent notation
        l_v = dot(d,s)/dot(s,s)
    where 
        d = p_p-p_v_0 
        s = p_v_1-p_v_0"""
    assert (positions.shape[-1] == 2) and (vectors.shape[-2:] == (2,2)), "positions and vectors must have shapes (_,2) and (_,2,2) respectively. _ is optional"
    positions = positions.reshape(-1,2)
    vectors = vectors.reshape(-1,2,2)
    positions = positions + np.random.normal(scale=1e-6, size=positions.shape)
    vectors = vectors + np.random.normal(scale=1e-6, size=vectors.shape)

    N_p = positions.shape[0]
    N_v = vectors.shape[0]

    d = np.expand_dims(positions,axis=1) - np.expand_dims(vectors[:,0,:],axis=0) #d.shape = (N_p,N_v,2)
    s = vectors[:,1,:]-vectors[:,0,:] #vectors.shape = (N_v,2)


    """in order to do the dot product we must reshaope s to be d's shape."""
    s_ = np.tile(np.expand_dims(s.copy(),axis=0),reps=(N_p,1,1)) #s_.shape = (N_p,N_v,2)
    """now do the dot product by broadcast multiplying the arraays then summing over the last axis"""

    l_v = (d*s).sum(axis=-1)/(s*s).sum(axis=-1) #l_v.shape = (N_p,N_v)

    """
    Now we can actually find the vector of shortest distance from the line segments to the points which is given by the size of the perpendicular
        perp = p_p - (p_v_0 + l_v.s_)
    
    But notice that if l_v > 1 then the perpendicular drops onto a part of the line which doesn't exist. In fact the shortest distance is to the point on the line segment where l_v = 1. Likewise for l_v < 0. To fix this we should limit l_v to be between 1 and 0
    """
    l_v[l_v > 1] = 1
    l_v[l_v < 0] = 0

    """we must reshape p_p and p_v_0 to be shape (N_p,N_v,2), also reshape l_v to be shape (N_p, N_v,1) so we can broadcast multiply it wist s_"""
    p_p = np.tile(np.expand_dims(positions,axis=1),reps=(1,N_v,1)) #p_p.shape = (N_p,N_v,2)
    p_v_0 = np.tile(np.expand_dims(vectors[:,0,:],axis=0),reps=(N_p,1,1)) #p_v_0.shape = (N_p,N_v,2)
    l_v = np.expand_dims(l_v,axis=-1)

    perp = p_p - (p_v_0 + l_v*s_) #perp.shape = (N_p,N_v,2)
    
    return perp

def get_line_segments_between(pos1, 
                              pos2):
    """Takes two position arrays and returns the array of pair-wise line segments between positions in each array (from pos1 to pos2).
    Args:
        pos1 (array): (N x dimensionality) array of positions
        pos2 (array): (M x dimensionality) array of positions
    Returns:
        (N x M x 2 x dimensionality) array of vectors from pos1's to pos2's"""

    pos1_ = pos1.reshape(-1, 1, pos1.shape[-1])
    pos2_ = pos2.reshape(1, -1, pos2.shape[-1])
    pos1 = np.repeat(pos1_, pos2_.shape[1], axis=1)
    pos2 = np.repeat(pos2_, pos1_.shape[0], axis=0)
    lines = np.stack((pos1, pos2), axis=-2)
    return lines

def get_vectors_between(pos1=None,
                        pos2=None,
                        line_segments=None):
    """Takes two position arrays and returns the array of pair-wise vectors between positions in each array (from pos1 to pos2).
    Args:
        pos1 (array): (N x dimensionality) array of positions
        pos2 (array): (M x dimensionality) array of positions
        line_segments: if you already have th el line segments, just pass these 
    Returns:
            (N x M x dimensionality) array of vectors from pos1's to pos2's"""
    if line_segments is None:
        line_segments = get_line_segments_between(pos1,pos2)
    vectors = line_segments[..., 0, :] - line_segments[..., 1, :]
    return vectors

def get_distances_between(pos1=None,
                          pos2=None,
                          vectors=None):
    """Takes two position arrays and returns the array of pair-wise euclidean distances between positions in each array (from pos1 to pos2).
    Args:
        pos1 (array): (N x dimensionality) array of positions
        pos2 (array): (M x dimensionality) array of positions
        vectors: if you already have the pair-wise vectors between pos1 and pos2, just pass these 
    Returns:
            (N x M) array of distances from pos1's to pos2's"""
    if vectors is None:
        vectors = get_vectors_between(pos1,pos2)
    distances = np.linalg.norm(vectors,axis=-1)
    return distances

def get_angle(segment):
    """Given a 'segment' (either 2x2 start and end positions or 2x1 direction bearing) 
         returns the 'angle' of this segment modulo 2pi
    Args:
        segment (array): The segment, (2,2) or (2,) array 
    Returns:
        float: angle of segment
    """
    eps = 1e-6
    if segment.shape == (2,):
        return np.mod(np.arctan2(segment[1], (segment[0] + eps)), 2 * np.pi)
    elif segment.shape == (2, 2):
        return np.mod(
            np.arctan2(
                (segment[1][1] - segment[0][1]), (segment[1][0] - segment[0][0] + eps)
            ),
            2 * np.pi,
        )

def rotate(vector,
           theta):
    """rotates a vector shape (2,) by angle theta. 
    Args:
        vector (array): the 2d vector
        theta (flaot): the rotation angle
    """
    R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    vector_new = np.matmul(R,vector)
    return vector_new

def wall_bounce(current_velocity,
                wall):
    """Given current direction and wall returns a new direction which is the result of reflecting off that wall 
    Args:
        current_direction (array): the current direction vector
        wall (array): start and end coordinates of the wall
    Returns:
        array: new direction
    """
    wall_perp = get_perpendicular(wall[1] - wall[0])
    if np.dot(wall_perp, current_velocity) <= 0:
        wall_perp = (
            -wall_perp
        )  # it is now the get_perpendicular with smallest angle to dir
    wall_par = wall[1] - wall[0]
    if np.dot(wall_par, current_velocity) <= 0:
        wall_par = -wall_par  # it is now the parallel with smallest angle to dir
    wall_par, wall_perp = (
        wall_par / np.linalg.norm(wall_par),
        wall_perp / np.linalg.norm(wall_perp),
    )  # normalise
    new_velocity = wall_par * np.dot(
        current_velocity, wall_par
    ) - wall_perp * np.dot(current_velocity, wall_perp)

    return new_velocity

"""Stochastic-assistance functions"""
def ornstein_uhlenbeck(dt,
                       x, 
                       drift=0.0, 
                       noise_scale=0.2, 
                       coherence_time=5.0):
    """An ornstein uhlenbeck process in x.
    x can be multidimensional 
    Args:
        dt: update time step
        x: the stochastic variable being updated
        drift (float, or same type as x, optional): [description]. Defaults to 0.
        noise_scale (float, or same type as v, optional): Magnitude of deviations from drift. Defaults to 0.16 (16 cm s^-1).
        coherence_time (float, optional): Effectively over what time scale you expect x to change directions. Defaults to 5.

    Returns:
        dv (same type as v); the required update ot the velocity
    """
    x = np.array(x)
    drift = drift * np.ones_like(x)
    noise_scale = noise_scale * np.ones_like(x)
    coherence_time = coherence_time * np.ones_like(x)
    sigma = np.sqrt((2 * noise_scale ** 2) / (coherence_time * dt))
    theta = 1 / coherence_time
    dx = theta * (drift - x) * dt + sigma * np.random.normal(size=x.shape, scale=dt)
    return dx

def interpolate_and_smooth(x,
                           y,
                           sigma=0.03):
    """Interpolates with cublic spline x and y to 10x resolution then smooths these with a gaussian kernel of width sigma. Currently this only works for 1-dimensional x.
    Args:
        x 
        y 
        sigma 
    Returns (x_new,y_new)
    """ 
    from scipy.ndimage.filters import gaussian_filter1d
    from scipy.interpolate import interp1d

    y_cubic = interp1d(x, y, kind='cubic')
    x_new = np.arange(x[0],x[-1],(x[1]-x[0])/10)
    y_interpolated = y_cubic(x_new)
    y_smoothed = gaussian_filter1d(y_interpolated,
                        sigma=sigma/(x_new[1]-x_new[0]))
    return x_new,y_smoothed

def normal_to_rayleigh(x,sigma=1):
    """Converts a normally distributed variable (mean 0, var 1) to a rayleigh distributed variable (sigma)
    """    
    x = scipy.stats.norm.cdf(x) #norm to uniform)
    x = sigma * np.sqrt(-2*np.log(1-x)) #uniform to rayleigh
    return x

def rayleigh_to_normal(x,sigma=1):
    """Converts a rayleigh distributed variable (sigma) to a normally distributed variable (mean 0, var 1)
    """
    if x<=0: x = 1e-6
    if x >= 1: x = 1 - 1e-6  
    x = 1 - np.exp(-x**2/(2*sigma**2)) #rayleigh to uniform
    x = scipy.stats.norm.ppf(x) #uniform to normal
    return x 

"""Plotting functions"""    
def bin_data_for_histogramming(data,
                               extent,
                               dx,
                               weights=None):
    """Bins data ready for plotting. So for example if the data is 1D the extent is broken up into bins (leftmost edge = extent[0], rightmost edge = extent[1]) and then data is histogrammed into these bins. weights weights the histogramming process so the contribution of each data point to a bin count is the weight, not 1. 

    Args:
        data (array): (2,N) for 2D or (N,) for 1D)
        extent (_type_): _description_
        dx (_type_): _description_
        weights (_type_, optional): _description_. Defaults to None.

    Returns:
        (heatmap,bin_centres): if 1D
        (heatmap): if 2D
    """    
    if len(extent)==2: #dimensionality = "1D"
        bins = np.arange(extent[0],extent[1]+dx,dx)
        heatmap, xedges = np.histogram(data,bins=bins,weights=weights)
        centres = (xedges[1:]+xedges[:-1])/2
        return (heatmap,centres)
        
    elif len(extent)==4: #dimensionality = "2D"
        bins_x = np.arange(extent[0],extent[1]+dx,dx)
        bins_y = np.arange(extent[2],extent[3]+dx,dx)
        heatmap, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], bins=[bins_x,bins_y], weights=weights)
        heatmap = heatmap.T[::-1, :]
        return heatmap

def mountain_plot(X, 
                  NbyX, 
                  color="C0",
                  xlabel="",
                  ylabel="",
                  xlim=None,
                  fig=None,
                  ax=None,):
    """Make a mountain plot. NbyX is an N by X array of all the plots to display. The nth plot is shown at height n, line are scaled so the maximum value across all of them is 0.7, then they are all seperated by 1 (sot they don't overlap)

    Args:
        X: independent variable to go on X axis 
        NbyX: dependent variables to go on y axis
        color: plot color. Defaults to "C0".
        xlabel (str, optional): x axis label. Defaults to "".
        ylabel (str, optional): y axis label. Defaults to "".
        xlim (_type_, optional): fix xlim to this is desired. Defaults to None.
        fig (_type_, optional): fig to plot over if desired. Defaults to None.
        ax (_type_, optional): ax to plot on if desider. Defaults to None.

    Returns:
        fig, ax: _description_
    """  
    c = (color or 'C1')  
    c = np.array(matplotlib.colors.to_rgb(color))
    fc = 0.3*c + (1-0.3)*np.array([1,1,1]) #convert rgb+alpha to rgb

    NbyX = 0.7* NbyX / np.max(np.abs(NbyX))
    if fig is None and ax is None: 
        fig, ax = plt.subplots(figsize=(4, len(NbyX)*5.5/25)) #~6mm gap between lines
    for i in range(len(NbyX)):
        ax.plot(X, NbyX[i] + i + 1, c=color)
        ax.fill_between(X, NbyX[i] + i + 1, i + 1, facecolor=fc)
    ax.spines["left"].set_bounds(1, len(NbyX))
    ax.spines["bottom"].set_position(("outward", 1))
    ax.spines["left"].set_position(("outward", 1))
    ax.set_yticks([1, len(NbyX)])
    ax.set_ylim(
        1 - 0.5,
        len(NbyX) + 1
    )
    ax.set_xticks(np.arange(max(X + 0.1)))
    ax.spines["left"].set_color(None)
    ax.spines["right"].set_color(None)
    ax.spines["top"].set_color(None)
    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim()
    if xlim is not None:
        ax.set_xlim(right=xlim)

    return fig, ax

"""Other"""
def update_class_params(Class, 
                        params: dict):
    """Updates parameters from a dictionary. 
    All parameters found in params will be updated to new value
    Args:
        params (dict): dictionary of parameters to change
        initialise (bool, optional): [description]. Defaults to False.
    """
    for key, value in params.items():
        setattr(Class, key, value)

def gaussian(x,
             mu,
             sigma):
    """Gaussian function. x, mu and sigma can be any shape as long as they are all the same (or strictly, all broadcastabele) 
    Args:
        x
        mu 
        sigma
    Returns gaussian(x;mu,sigma)
    """    
    g = -(x-mu)**2
    g = g/(2*sigma**2)
    g = np.exp(g)
    g = g/np.sqrt(2*np.pi*sigma**2)
    return g 
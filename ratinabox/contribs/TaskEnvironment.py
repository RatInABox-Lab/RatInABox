# A goal-driven environement

from ratinabox.Environment import Environment

class TaskEnvironment(Environment):
    """
    Environment with task structure
    """
    def __init__(self, *pos, **kws):
        super().__init__(*pos, **kws)
        self.episode_history = [] # Written to upon completion of an episode

    def is_done(self):
        """
        Whether the current state is a terminal state
        """
        raise NotImplementedError("is_done() must be implemented")

    def reset(self):
        """
        How to reset the task when finisedh
        """
        raise NotImplementedError("reset() must be implemented")

    def write_episode(**kws):
        pass

    def episodes_to_df():
        pass

class SpatialGoalEnvironment(TaskEnvironment):
    """
    Creates a spatial goal-directed task
    """

    def __init__(self, *pos, possible_goal_locations=[], **kws):
        super().__init__(*pos, **kws)
        self.possible_goal_locations = possible_goal_locations



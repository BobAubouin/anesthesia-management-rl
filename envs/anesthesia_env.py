import numpy as np
import gym
from typing import Dict, Tuple
# from models.simple_pkpd import SimplePKPDModel
from models.pkpd import PKPDModel


class AnesthesiaEnv(gym.Env):
    """
    Simplified environment where agent controls the propofol infusion.
    """

    def __init__(self, config: Dict):
        """
        Initialise the anesthesia environment.

        Args:
            config(Dict): Configuration dictionary containing parameters
            for the environment.

                Expected keys:
                - 'ec50': Mean EC50 value for the patient population (default: 2.7)
                - 'ec50_std': Standard deviation of EC50 values for patient population (default: 0.3)
                - 'gamma': Sigmoid steepness parameter (default: 1.4)
                - 'ke0': Effect site equilibration rate (default:0.46)
                - 'max_surgery_length'

        """
        if config is None:
            config = {}
        self.config = config
        # Observation space: BIS value, effect site concentration
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0]),  # Min values for BIS, Ce
            high=np.array([100, 10]),  # Max values for BIS, Ce
            # ie BIS, effect site, vitals, cognitive load
            dtype=np.float32
        )
        # Action space: Propofol infusion rate (in mL/kg/min)
        self.action_space = gym.spaces.Box(
            low=np.array([0]),  # Min infusion rate
            high=np.array([10]),  # Max infusion rate
            dtype=np.float32
        )

        self.surgery_length = 0

        # self.pk_model = SimplePKPDModel(
        #     ec50=config.get('ec50', 2.7),
        #     gamma=config.get('gamma', 1.4),
        #     ke0=config.get('ke0', 0.46)
        # )

        self.pk_model = PKPDModel()

        self.previous_bis = 100
        self.previous_action = 3

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one step in the environment

        Args:
            action (float): The action taken by the agent (propofol infusion rate)

        Returns:
            Tuple[np.ndarray, float, bool, dict]:
            - observation: The current state of the environment
            - reward: the reward for the action taken
            - done: whether the episode has ended
            - info: additional info(empty in this case)
        """

        # simulate time passing (eg surgery length increases over the period)
        self.surgery_length += 1  # Increment the surgery length by 1 minute

        # update drug concentrations using the PK/PD model

        self.pk_model.update(action)

        # simulate anesthesiologists decision making
        bis = self.pk_model.calculate_bis()
        effect_site = self.pk_model.get_effect_site_concentration()

        obs = np.array([bis, action], dtype=np.float32)

        # calculate reward based on the current state
        reward = self.compute_reward(bis, action)
        self.previous_bis = bis
        self.previous_action = action

        # check if the episode has ended (surgery is over)
        done = self.surgery_length >= self.config.get('max_surgery_length', 120)

        # checks
        assert not np.isnan(bis), f"NaN in BIS! Effect_site: {effect_site}"
        assert not np.isnan(effect_site), "NaN in effect_site"
        assert not np.isnan(reward), f"NaN in reward! BIS: {bis}"

        return obs, reward, done, {}

    def reset(self) -> np.ndarray:
        """
    Reset the environment to its initial state

    Returns:
        np.ndarray: The initial observation
    """

        self.pk_model.reset()
        self.surgery_length = 0
        bis = self.pk_model.calculate_bis()
        effect_site = self.pk_model.get_effect_site_concentration()
        return np.array([bis, effect_site], dtype=np.float32)

    def compute_reward(self, bis, action):
        """
        Compute the reward based on the current BIS value; The reward function is from the paper
        "A Reinforcement Learning Approach to Weaning of Mechanical Ventilation in Intensive Care Units" by Prasad et al. in 2017.

        Args:
            bis (float): The bis value

        Returns:
            reward (float): The reward for the action taken
        """
        bis_min = 40
        bis_max = 60

        # reward = 10*(1/(1+np.exp(-(bis-bis_min))) - 1/(1+np.exp(-(bis-bis_max))) + 0.5)
        reward = -((bis - 50) ** 2) / 100 + 1
        reward += -0.3*max(0, np.abs(bis-self.previous_bis)/max(1, self.previous_bis) - 0.05)
        reward += -0.05*max(0, np.abs(action-self.previous_action)/max(1, self.previous_action) - 0.2)
        # if 40 <= bis <= 60:
        #     reward = 1.0
        # else:
        #     reward = -np.exp(0.2 * abs(bis -50))

        return reward

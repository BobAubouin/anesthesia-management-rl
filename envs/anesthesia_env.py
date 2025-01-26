import numpy as np
import gym
from collections import deque
import time
from typing import Dict, Tuple

class AnesthesiaEnv(gym.Env):
    """
    A custom environment simulating anesthesia management.

    This environment models the administration of propofol to maintain
    patient's consciousness level within a target range. It incorporates
    cognitive and environmental bounds such as fatigue, sensor noise and patient variability

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
                - 'obs_delay': Observation delay in steps (default: 2)
                - 'action_delay': UI interaction delay in seconds (default: 1.0)
                - 'shift': Current working shift ('day' or 'night', default: 'day')
        """

        #Observation space: BIS value, effect size concentration, vitals (focus :systolic blood pressure), cognitive load
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0]), #Min values for BIS, Ce, SBP, CL
            high=np.array([100, 10, 1, 1]), #Max values for BIS, Ce, SBP, CL
            #ie BIS, effect site, vitals, cognitive load
            dtype=np.float32
        

        )
        #Action space: Propofol infusion rate (in mL/kg/min)
        self.action_space = gym.spaces.Box(
            low=np.array([0]), #Min infusion rate
            high=np.array(10), #Max infusion rate
            dtype=np.float32
        )
        #Cognitive & Environmental bounds
        self.obs_delay = config.get('obs_delay', 2) #Number of steps to delay observations
        self.obs_buffer = deque(maxlen=self.obs_delay) #Buffer to store delayed observations
        self.last_action = 0.0 #Track the last action taken for decision fatigue modeling

        #Congitive limits
        self.surgeries_today = 0 #Number of surgeries performed today
        self.current_shift = config.get('shift', 'day')
        self.surgery_length = 0 #Length of the current surgery in minutes
        self.cognitive_load = 0.0 #Current cognitive load of the anesthesiologist ranging from 0 to 1

        #Initialise the patient model(Bayesian PK/PD model)
        self._setup_patient_model(config)

        
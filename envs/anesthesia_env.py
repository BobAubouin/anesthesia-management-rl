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
        self.config = config
        #Observation space: BIS value, effect size concentration, vitals (focus :systolic blood pressure), cognitive load
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0]), #Min values for BIS, Ce, SBP, CL, additional
            high=np.array([100, 10, 1, 1, 1]), #Max values for BIS, Ce, SBP, CL
            #ie BIS, effect site, vitals, cognitive load
            dtype=np.float32
        

        )
        #Action space: Propofol infusion rate (in mL/kg/min)
        self.action_space = gym.spaces.Box(
            low=np.array([0]), #Min infusion rate
            high=np.array([10]), #Max infusion rate
            dtype=np.float32
        )
        #Cognitive & Environmental bounds
        self.obs_delay = config.get('obs_delay', 2) #Number of steps to delay observations
        self.obs_buffer = deque(maxlen=self.obs_delay) #Buffer to store delayed observations
        self.last_action = 0.0 #Track the last action taken for decision fatigue modeling
        #--alarm count


        #Congitive limits
        self.surgeries_today = 0 #Number of surgeries performed today
        self.current_shift = config.get('shift', 'day')
        self.surgery_length = 0 #Length of the current surgery in minutes
        self.cognitive_load = 0.0 #Current cognitive load of the anesthesiologist ranging from 0 to 1

        #Initialise the patient model(Bayesian PK/PD model)
        self._setup_patient_model(config)

        #Initialise anesthesiologist model
        self._setup_anesthesiologist_model(config)

    def _setup_patient_model(self, config: Dict) -> None:
        """
        Initialise the bayesian PK/PD model for simulating patient drug response.

        Args:
            config(Dict): Configuration dictionary containing PK/PD parameters.
        """
        from models.pkpd_model import BayesianPKPDModel
        self.pk_model  = BayesianPKPDModel(
            ec50_mean=config.get('ec50', 2.7),
            ec50_std=config.get('ec50_std', 0.3),
            gamma=config.get('gamma', 1.4),
            ke0=config.get('ke0', 0.46)

        )

    def _setup_anesthesiologist_model(self, config: Dict) -> None:
        """
        Set up the anesthesiologist decision-making model

        Parameters:
        - config
        """
        from models.anesthesiologist_model import AnesthesiologistModel
        self.anesthesiologist = AnesthesiologistModel()
    
    def _get_vitals(self):
        """
        Simulate the patient's systolic blood pressure (SBP)

        SBP is influenced by:
        - the propofol infusion rate
        - random fluctuations which can be within normal limits
        """

        baseline_sbp = 110
        noise = np.random.normal(0,5) #random fluctuation in BP (+= 5mmHg)

        #infusion effect: high propofol infusion rate can decrease SBP
        propofol_effect = -self.last_action * 2 # every unit of propofol lowers sbp slightly
        sbp = baseline_sbp + noise + propofol_effect
        sbp = max(80, min(sbp, 160))
        return sbp

    def _compute_cognitive_load(self) -> float:
        """
        Compute the cognitive load based on:
        - Number of surgeries performed by the anesthetist that day
        - Current working shift (day vs night)
        - Length of the current surgery

        Returns:
            float: The cognitive load score ranging from 0-1
        """
        load = self.surgeries_today * 0.1 #Base load increases with surgeries

        if self.current_shift == "night":
            load += 0.2
        load += self.surgery_length * 0.01 #Longer surgeries  increase the load due to fatigue
        return min(load, 1.0) #Limit the load to 1
    
    def _check_termination(self) -> bool:
        """
        Determines whether the surgical episode is ended
        Assumes a surgery takes 120 minutes
        """
        max_surgery_length = self.config.get('max_surgery_length', 120)
        if self.surgery_length >=max_surgery_length:
            return True
        return False
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one step in the environment

        Args:
            action (np.ndarray): The action taken by the agent (propofol infusion rate)

        Returns:
            Tuple[np.ndarray, float, bool, dict]: 
            - observation: The current state of the environment
            - reward: the reward for the action taken
            - done: whether the episode has ended
            - info: additional info(empty in this case)        
        """

        #simulate time passing (eg surgery length increases over the period)
        self.surgery_length += 1 #Increment the surgery length by 1 minute

        #update cognitive load
        self.cognitive_load = self._compute_cognitive_load()

        #simulate UI interaction delay (eg time taken to adjust infusion pump)
        time.sleep(self.config.get('action_delay', 1.0))

        #update drug concentrations using the PK/PD model
        self.pk_model.update(action[0]) 

        #generate observation (noisy and delayed)
        obs = self._get_observation()

        #simulate anesthesiologists decision making
        bis = self.pk_model.calculate_bis()
        effect_site = self.pk_model.get_effect_site_concentration()
        vitals = self._get_vitals()
        infusion_rate = self.anesthesiologist.decide_infusion_rate(
            bis, effect_site, vitals, self.cognitive_load
        )

        #calculate reward with safety constraints
        reward = self._calculate_reward(infusion_rate)

        #track alarms for fatigue modeling
        
        if bis < 40 or bis > 60:
            self.alarm_count += 1
        
        #check if the episode has ended (surgery is over)
        done = self._check_termination()

        return obs, reward, done, {}

    def _get_observation(self) -> np.ndarray:
        """
        Generate a noisy and delayed observation of the environment with cognitive load

        Returns:
            np.ndarray: The current observation (BIS, Ce, SBP, cognitive load)
        """
        bis = self.pk_model.calculate_bis() 
        effect_site = self.pk_model.get_effect_site_concentration()
        vitals = self._get_vitals() #sbp
        additional = 4

        #add sensor noise (+-5% error) to simulate imperfect monitoring
        noise = np.random.normal(0, 0.05 * bis)
        obs = np.array([bis + noise, effect_site, vitals, self.cognitive_load, additional]) 

        #simulate observation delay by storing observations in a buffer
        self.obs_buffer.append(obs)
        return self.obs_buffer[0] if len(self.obs_buffer) > 0 else obs
    
    def _calculate_reward(self, action: np.array) -> float:
        """
        Calculate the reward for the action taken

        Args:
            action (np.ndarray): The action by the agent - propofol infusion rate

        Returns:
            float: the reward value
        """
        bis = self.pk_model.calculate_bis()
        reward = -abs(bis - 50) #primary reward based on BIS deviation from 50 (target 40-60)

        #safety penalty for critical ranges (BIS <30 or BIS >70)
        if bis < 30 or bis > 70:
            reward -= 10

        

        return reward
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to its initial state

        Returns:
            np.ndarray: The initial observation
        """
        self.pk_model.reset()
        self.obs_buffer.clear()
        self.alarm_count = 0
        self.surgery_length=0
        self.cognitive_load=0.0
        return self._get_observation()

    
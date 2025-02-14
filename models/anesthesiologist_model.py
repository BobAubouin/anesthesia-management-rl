import numpy as np

class AnesthesiologistModel:
    def __init__(self):
        """
        Initialise the anesthesiologist model
        """
        pass

    def decide_infusion_rate(self, bis: float, effect_site: float, vitals: float, cognitive_load: float) -> float:
        """
        To simulate anesthesiologist's decision making process

        Parameters:
        - bis: Current BIS value
        - effect_site: Current effect site concentration
        - vitals: current systolic blood pressure
        - cognitive_load: current cognitive load (0-1)

        Returns:
        - A float representing the infusion rate

        """

        #Target BIS is 50
        target_bis = 50
        bis_error = bis - target_bis

        #Base infusion rate adjustment (proportional control based on BIS error)
        base_rate = 5 #defaul infusion rate
        adjustment = bis_error * 0.1

        #Calculate final infusion rate
        infusion_rate = base_rate + adjustment

        #Ensure infusion rate is within safe limits
        return max(0, min(infusion_rate, 10))
import numpy as np

class BayesianPKPDModel:
    """
    Represents patient's response to propofol

    """
    def __init__(self, ec50_mean: float, ec50_std: float, gamma: float, ke0: float):
        self.ec50 = np.random.normal(ec50_mean, ec50_std) 
        self.gamma = gamma 
        self.ke0 = ke0
        self.Ce = 0.0 #initialising effect site concentration

    def update(self, infusion_rate: float):
        """
        Update the effect site concentration based on the infusion rate

        Args:
            infusion_rate: the rate propofol is infused
        """
        self.Ce += (infusion_rate - self.ke0 * self.Ce)

    def calculate_bis(self) -> float:
        """
        Calculate BIS using the sigmoid Emax model which describes the effectiveness of drug as a function of its concentration
=
        """
        if self.Ce <= 0:
            return 100.0
        Ce_gamma = self.Ce ** self.gamma
        ec50_gamma = self.ec50 ** self.gamma
        bis = 100* (1-Ce_gamma/ (Ce_gamma + ec50_gamma))
        return bis
    
    def get_effect_site_concentration(self) -> float:
        """
        Return effect site concentration
        """
        return self.Ce
    
    def reset(self) -> None:
        """Reset model to initial state"""
        self.Ce = 0.0

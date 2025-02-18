import numpy as np

class SimplePKPDModel:
    """
    Simplified PK/PD model for propofol effect on BIS
    """
    def __init__(self, ec50=2.7, gamma = 1.4, ke0=0.46):
        self.ec50 = ec50
        self.gamma = gamma
        self.ke0 = ke0
        self.effect_site = 0.0

    def update(self, infusion_rate: float) -> None:
        #ensure infusion_rate is non_negative
        infusion_rate = max(0.0, infusion_rate)

        self.effect_site += (infusion_rate - self.effect_site) * self.ke0

        #ensure effect site concentration is not negative
        self.effect_site = max(0.0, self.effect_site)

    def calculate_bis(self):
        """Calculate BIS using a sigmoid Emax model"""
        bis = 100 * (1- (self.effect_site ** self.gamma) / (self.effect_site ** self.gamma + self.ec50**self.gamma))
        return np.clip(bis, 0, 100)
    
    def get_effect_site_concentration(self) -> float:
        return self.effect_site
    
    def reset(self) -> None:
        self.effect_site = 0.0
    
import python_anesthesia_simulator as pas
import warnings

warnings.filterwarnings("ignore", module="python_anesthesia_simulator")


class PKPDModel:
    """
    Simplified PK/PD model for propofol effect on BIS
    """

    def __init__(self, age=28, height=170, weight=70, gender=0):
        self.age = age
        self.height = height
        self.weight = weight
        self.gender = gender

        self.patient_info = [age, height, weight, gender]
        self.patient = pas.Patient(
            self.patient_info,
            model_propo='Eleveld',
            random_PK=False,  # create a random model of propofol PK
            random_PD=False,  # create a random model of propofol PD
            ts=60,  # sampling time (s)
        )

        self.pk_model = self.patient.propo_pk
        self.effect_site = 0.0

    def update(self, infusion_rate: float) -> None:
        """use the PK model to update the effect site concentration

        Parameters
        ----------
        infusion_rate : float
            infusion rate of propofol in ml/kg/min
        """
        # ensure infusion_rate is non_negative
        infusion_rate = max(0.0, infusion_rate)
        # convert mL/kg/min to mg/s (usual concentration of 20 mg/mL)
        propo_rate = infusion_rate * self.weight / 60 / 20

        self.pk_model.one_step(u=propo_rate)

        self.effect_site = self.pk_model.x[3]  # effect site concentration

    def calculate_bis(self):
        """Calculate BIS using a sigmoid Emax model"""
        bis = self.patient.bis_pd.compute_bis(
            c_es_propo=self.effect_site,
            c_es_remi=0,
        )
        return bis

    def get_effect_site_concentration(self) -> float:
        return self.effect_site

    def get_plasma_concentration(self) -> float:
        return self.pk_model.x[0]

    def reset(self) -> None:
        self.patient = pas.Patient(
            self.patient_info,
            model_propo='Eleveld',
            random_PK=False,  # create a random model of propofol PK
            random_PD=False,  # create a random model of propofol PD
            ts=60,  # sampling time (s)
        )

        self.pk_model = self.patient.propo_pk
        self.effect_site = 0.0

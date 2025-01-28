import unittest
from envs.anesthesia_env import AnesthesiaEnv

class TestAnesthesiaEnv(unittest.TestCase):
    def test_cognitive_load(self):
        env = AnesthesiaEnv(config = {'shift': 'night'})
        env.surgeries_today =3
        env.surgery_length =120
        self.assertAlmostEqual(env._compute_cognitive_load(), 1.0)

import unittest
from src.schedules import GraySchedule, from_config

class TestSchedule(unittest.TestCase):
    def test_boundaries_and_equal_exposure(self):
        linear = GraySchedule('linear', 2, 5, .8)
        fixed = GraySchedule('fixed_expected_exposure', 2, 5, .8)
        off = GraySchedule('off', 2, 5, .8)
        self.assertEqual([linear(i) for i in (0, 1, 2, 6, 7)], [1., 1., .8, 0., 0.])
        self.assertEqual([off(i) for i in (1, 2, 7)], [1., 0., 0.])
        self.assertAlmostEqual(sum(linear(i) for i in range(2, 7)), sum(fixed(i) for i in range(2, 7)))
        self.assertTrue(all(0 <= linear(i) <= 1 for i in range(20)))
    def test_resume_and_config(self):
        cfg = {'smooth': {'mode':'linear','window_epochs':3,'initial_probability':1.}, 'training': {'stage1_epochs':6}}
        full = from_config(cfg, 4)
        resumed = from_config(cfg, 4)
        self.assertEqual([full(i) for i in range(29, 40)], [resumed(i) for i in range(29, 40)])
    def test_invalid(self):
        for args in [('bad', 0, 2, 1), ('linear', 0, 1, 1), ('linear', 0, 2, float('nan'))]:
            with self.assertRaises(ValueError): GraySchedule(*args)
        with self.assertRaises(ValueError): GraySchedule('linear', 0, 2)(-1)

if __name__ == '__main__': unittest.main()

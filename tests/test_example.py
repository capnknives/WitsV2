# tests/test_example.py
import unittest

class TestExample(unittest.TestCase):
    def test_example_assertion(self):
        self.assertEqual(1 + 1, 2, "Example test failed")

if __name__ == '__main__':
    unittest.main()

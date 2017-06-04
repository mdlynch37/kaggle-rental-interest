import unittest

class MyTest(unittest.TestCase):

    def testMethod(self):
        self.assertEqual(1 + 2, 3, "1 + 2 not equal to 3")

    def testMethod2(self):
        self.assertNotEqual(1 + 2, 3, "1 + 2 not equal to 3")


if __name__ == '__main__':
    unittest.main()

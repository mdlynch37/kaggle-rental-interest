import unittest

import numpy as np
import grid_explore as gx
import utilities as ut


import unittest


class Grades:

    cnt = 0

    def __init__(self, grade):
        self.grade = grade
        self.__class__.cnt += 1

    def is_true(self, x):
        return x

    def is_False(self, x):
        return not x

class MyClassTest(unittest.TestCase):


    ClassIsSetup = False

    def setUp(self):

        if not self.ClassIsSetup:
            print("Initializing testing environment")
            self.myclass = Grades(2)
            # remember that it was setup already
            self.__class__.ClassIsSetup = True  # not sure what's up
#             self.ClassIsSetup = True          # this should work but doesn't


    def test_pass(self):
        self.assertTrue(True)

    def test_fail(self):
        self.assertTrue(False)

test = unittest.TestLoader().loadTestsFromTestCase(MyClassTest)
x = unittest.TextTestRunner().run(test)

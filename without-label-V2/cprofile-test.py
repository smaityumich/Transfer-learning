import cProfile
from withoutLabelV2 import *

d = DataGenerator(5)
x_s, y_s, _ = d._getData(1000)
x_t, _, _ = d._getData(1000)
x_test, _, _ = d._getData(1000)

cl = WithoutLabelV2(x_s, y_s, x_t)
cProfile.run('t = cl._classify(x_test)')


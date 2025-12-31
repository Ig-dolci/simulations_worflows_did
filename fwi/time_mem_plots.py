import matplotlib.pyplot as plt

recompute_time = [43.98551416397095, 47.796653270721436, 44.00095224380493, 44.27761507034302, 46.3203558921814]
derivative_time = [162.21021723747253, 155.75003862380981, 151.44719171524048, 154.69695210456848, 164.2068681716919]
fwd_annotate = [45.40703892707825, 36.67819309234619, 36.94022989273071, 44.803985357284546, 39.763829708099365]
fwd = []

def average_time(time):
    return sum(time)/len(time)

average_time_recompute = average_time(recompute_time)
average_time_derivative = average_time(derivative_time)
average_time_fwd_annotate = average_time(fwd_annotate)

print("Average time recompute: ", average_time_recompute)
print("Average time derivative: ", average_time_derivative)
print("Average time fwd annotate: ", average_time_fwd_annotate)
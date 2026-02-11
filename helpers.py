import numpy as np


def vector_xy_to_rtheta(vector_xy):

    theta = np.arctan2(vector_xy[1], vector_xy[0])
    r = np.sqrt(vector_xy[0]**2 + vector_xy[1]**2)
    vector_rtheta = np.array([r,theta])
    
    return vector_rtheta

# rk4 function
def rk4(point, step: float, function: callable):

    # calc k's
    k1 = function(point)
    k2 = function(point + 0.5*step*k1) 
    k3 = function(point + 0.5*step*k2) 
    k4 = function(point + step*k3) 

    # integrate
    next_point = point + (step/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4) 

    return next_point












# rotation_matrix = np.array([[np.cos(theta), np.sin(theta)],
    #                             [-np.sin(theta), np.cos(theta)]])   
    
    # vector_rtheta = np.matmul(rotation_matrix, vector_xy)
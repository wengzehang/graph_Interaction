import math
import numpy as np

def R_2vect(vector_target, vector_base):
    # R
    R = np.zeros((3,3))

    # Convert the vectors to unit vectors.
    vector_target = vector_target / np.linalg.norm(vector_target)
    vector_base = vector_base / np.linalg.norm(vector_base)

    # The rotation axis (normalised).
    axis = np.cross(vector_target, vector_base)
    axis_len = np.linalg.norm(axis)
    if axis_len != 0.0:
        axis = axis / axis_len

    # Alias the axis coordinates.
    x = axis[0]
    y = axis[1]
    z = axis[2]

    # The rotation angle.
    angle = math.acos(np.dot(vector_target, vector_base))

    # Trig functions (only need to do this maths once!).
    ca = math.cos(angle)
    sa = math.sin(angle)

    # Calculate the rotation matrix elements.
    R[0,0] = 1.0 + (1.0 - ca)*(x**2 - 1.0)
    R[0,1] = -z*sa + (1.0 - ca)*x*y
    R[0,2] = y*sa + (1.0 - ca)*x*z
    R[1,0] = z*sa+(1.0 - ca)*x*y
    R[1,1] = 1.0 + (1.0 - ca)*(y**2 - 1.0)
    R[1,2] = -x*sa+(1.0 - ca)*y*z
    R[2,0] = -y*sa+(1.0 - ca)*x*z
    R[2,1] = x*sa+(1.0 - ca)*y*z
    R[2,2] = 1.0 + (1.0 - ca)*(z**2 - 1.0)
    return R

def Global2Local(startpoint, R, vector_ori):
    vector_ori = vector_ori - startpoint
    # vector_convert = R@vector_ori
    vector_convert = vector_ori@R
    return vector_convert

def Local2Global(startpoint, R, vector_convert):
    vector_ori = vector_convert@np.linalg.pinv(R)
    vector_ori = vector_ori + startpoint
    return vector_ori


if __name__ == "__main__":
    # start_pos = np.array([[1,0,0]])
    # a = np.array([[1,0,1]])
    # b = np.array([[1,0,0]])
    # R = R_2vect(a,b)

    start_pos = np.array([1,0,0])
    a = np.array([1,0,1])
    b = np.array([1,0,0])
    # R = R_2vect(a,b)
    R = R_2vect(b,a)

    print(R)

    c = np.array([[1.5,0,1],[1,0,0]])
    print(c.shape)
    c_transform = Global2Local(start_pos, R, c)
    print(c_transform)

    c = Local2Global(start_pos, R, c_transform)
    print(c)
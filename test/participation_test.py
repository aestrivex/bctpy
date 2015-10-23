# test suite for participation coefficient
# all results tested against MATLAB BCT

import bct
import numpy

def participation_test():
    W=numpy.eye(3)
    ci=numpy.array([1,1,2])

    assert numpy.allclose(bct.participation_coef(W,ci),[0,0,0])
    assert numpy.allclose(bct.participation_coef_sign(W,ci)[0],[0,0,0])

    W=numpy.ones((3,3))
    assert numpy.allclose(bct.participation_coef(W,ci),[ 0.44444444, 0.44444444, 0.44444444])
    assert numpy.allclose(bct.participation_coef_sign(W,ci)[0],[ 0.44444444, 0.44444444, 0.44444444])

    W=numpy.eye(3)
    W[0,1]=1
    W[0,2]=1
    assert numpy.allclose(bct.participation_coef(W,ci),[ 0.44444444, 0,0])
    assert numpy.allclose(bct.participation_coef_sign(W,ci)[0],[ 0.44444444, 0,0])

    W=numpy.eye(3)
    W[0,1]=-1
    W[0,2]=-1
    W[1,2]=1
    assert numpy.allclose(bct.participation_coef_sign(W,ci)[0],[ 0. ,  0.5,  0. ])

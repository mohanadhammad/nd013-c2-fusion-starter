# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        self.dim_state = params.dim_state # process model dimension
        self.dt = params.dt # sample time
        self.q = params.q # process noise value for Q covariance

    def F(self):
        ############
        # Step 1: implement and return system matrix F
        ############
        # F = np.identity((self.dim_state))
        # F = np.asmatrix(F) # convert from array to matrix instance
        # F[0, 3] = self.dt
        # F[1, 4] = self.dt
        # F[2, 5] = self.dt
        
        dt = params.dt
        F = np.matrix([[1,0,0,dt,0,0],
                       [0,1,0,0,dt,0],
                       [0,0,1,0,0,dt],
                       [0,0,0,1,0,0],
                       [0,0,0,0,1,0],
                       [0,0,0,0,0,1]])
        return F

    def Q(self):
        ############
        # Step 1: implement and return process noise covariance Q
        ############

        dt2 = self.dt ** 2
        dt3 = self.dt * dt2
        
        q_11 = dt3 * self.q / 3.0
        q_13 = dt2 * self.q / 2.0
        q_33 = self.dt * self.q

        return np.matrix([[q_11,  0.0,  0.0, q_13,  0.0,  0.0],
                         [ 0.0, q_11,  0.0,  0.0, q_13,  0.0],
                         [ 0.0,  0.0, q_11,  0.0,  0.0, q_13],
                         [q_13,  0.0,  0.0, q_33,  0.0,  0.0],
                         [ 0.0, q_13,  0.0,  0.0, q_33,  0.0],
                         [ 0.0,  0.0, q_13,  0.0,  0.0, q_33]])

    def predict(self, track):
        ############
        # Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############

        x_ = self.F() * track.x
        P_ = self.F() * track.P * self.F().T + self.Q()
        
        track.set_x(x_)
        track.set_P(P_)

    def update(self, track, meas):
        ############
        # Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############
        
        gamma = self.gamma(track, meas)                 # residual vector
        
        H = meas.sensor.get_H( track.x )
        S = self.S(track, meas, H)                      # residual covariance
        
        I = np.asmatrix(np.zeros((self.dim_state)))     # identity matrix
        
        K = track.P * H.T * S.I                         # Kalman gain
        x = track.x + K * gamma                         # state update
        P = (I - K*H) * track.P                         # covariance update
        
        track.set_x(x)                                  # track state assignment
        track.set_P(P)                                  # track covariance assignment
        
        track.update_attributes(meas)
    
    def gamma(self, track, meas):
        ############
        # Step 1: calculate and return residual gamma
        ############
        
        return (meas.z - meas.sensor.get_hx( track.x )) # innovation/residual

    def S(self, track, meas, H):
        ############
        # Step 1: calculate and return covariance of residual S
        ############
        
        return (H * track.P * H.T) + meas.R
    
#!/usr/bin/python
# -*- coding: utf-8 -*-

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import itertools as it

# import numpy
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
from matplotlib import tri, cm

from scipy.spatial import ConvexHull
import scipy.linalg as linalg

import cdd

import pprint
import time

from logging import getLogger, StreamHandler, DEBUG, INFO, WARNING, ERROR, CRITICAL
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

def h2v(A,b):
    # inmat = cdd.Matrix(np.hstack([b,-A]), number_type='float')
    inmat = cdd.Matrix(np.hstack([b.astype(np.float16),-A.astype(np.float16)]), number_type='float')
    inmat.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(inmat)
    retmat = poly.get_generators()
    return inmat, poly, retmat

def v2h(flags,vertices):
    # inmat = cdd.Matrix(np.hstack([flags, vertices]), number_type='float')
    inmat = cdd.Matrix(np.hstack([flags, vertices.astype(np.float16)]), number_type='float')
    inmat.rep_type = cdd.RepType.GENERATOR
    poly = cdd.Polyhedron(inmat)
    retmat = poly.get_inequalities()
    return inmat, poly, retmat

class PlotInterface():
    def __init__(self):
        self.fig = plt.figure(figsize=(12.0,12.0))
        self.ax = self.fig.gca(projection='3d')
        self.fig.subplots_adjust(left=0.02,right=0.98, bottom=0.02,top=0.98, wspace=0.1, hspace=1)

        self._text_pos = [-0.095,0.06]
        self.joint_angle_texts = [self.ax.text2D(self._text_pos[0], self._text_pos[1]+0.007*i,"", fontsize=25) for i in range(3)]
        self.max_moment_text = self.ax.text2D(self._text_pos[0], self._text_pos[1]+0.007*len(self.joint_angle_texts), "max moment =", fontsize=25)

        self.max_display_num = 1500
        self.ax.set_xlim3d(-self.max_display_num,self.max_display_num)
        self.ax.set_ylim3d(-self.max_display_num,self.max_display_num)
        self.ax.set_zlim3d(-self.max_display_num,self.max_display_num)

        self.prev_surf_list = []


    def plot_convex_hull(self, vertices):
        # ax.clear()
        # self.ax.set_xlim3d(-self.max_display_num,self.max_display_num)
        # self.ax.set_ylim3d(-self.max_display_num,self.max_display_num)
        # self.ax.set_zlim3d(-self.max_display_num,self.max_display_num)
        self.ax.set_xlabel("nx(roll) [Nm]")
        self.ax.set_ylabel("ny(pitch) [Nm]")
        self.ax.set_zlabel("nz(yaw) [Nm]")

        for surf in self.prev_surf_list: surf.remove()
        self.prev_surf_list = []

        # hull = ConvexHull(vertices, incremental=True)
        hull = ConvexHull(vertices, incremental=True, qhull_options="QJ")

        for  idx, face_indices in enumerate(hull.simplices): # faces -> hull.simplices
            # print idx, " face: ", face_indices
            x,y,z = vertices[face_indices,0], vertices[face_indices,1], vertices[face_indices,2]

            # print "x: ", x
            # print "y: ", y
            # print "z: ", z
            # print ""

            x = x + z*0.0001
            y = y + z*0.0001

            # S = ax.plot_trisurf(x,y,z, cmap=cm.jet)

            triang = tri.Triangulation(x, y)
            refiner = tri.UniformTriRefiner(triang)
            new, new_z = refiner.refine_field(z, subdiv=2)

            # norm = plt.Normalize(vmax=abs(y).max(), vmin=-abs(y).max())
            norm = plt.Normalize(vmax=1500, vmin=-1500)
            # kwargs = dict(triangles=new.triangles, cmap=cm.jet, norm=norm, linewidth=0.05, alpha = 0.3)
            kwargs = dict(triangles=new.triangles, cmap=cm.jet, norm=norm, linewidth=0.1, alpha = 0.3)

            self.prev_surf_list.append(self.ax.plot_trisurf(new.x, new.y, new_z, **kwargs))

        # plt.show()
        plt.pause(0.01)

def convert_to_skeleton_moment_vertices(A_theta, B_theta):
    # calc max_tau at current pose
    max_tau_theta = (max_tau/abs(A_theta.dot(A_theta.T))).min(axis=0) # tau_j = min_i(tau_i/|a_j.a_i|)
    logger.debug("max_tau_theta=" + str(max_tau_theta))
    # tau convex hull H->V
    A = np.vstack([np.identity(num_joints),-np.identity(num_joints)])
    b = np.vstack([max_tau_theta[:,np.newaxis],max_tau_theta[:,np.newaxis]]) # min_tau = - max_tau -> -min_tau = max_tau
    inmat, poly, retmat = h2v(A,b)

    logger.debug("max_tau")
    logger.debug(retmat)
    tau_vertices = np.array(retmat)[:,1:] # u_k^T

    # convert to tau_tilde V->H
    tau_tilde_vertices = tau_vertices.dot(B_theta.T) # u_k^~T
    b_tilde = np.ones(tau_vertices.shape[0])[:,np.newaxis] # only hull (no cone)
    inmat, poly, retmat = v2h(b_tilde, tau_tilde_vertices)
    logger.debug("tau_tilde")
    logger.debug(retmat)
    C = -np.array(retmat)[:,1:]
    d = np.array(retmat)[:,0:1]
    logger.debug("")

    # H->V
    # for max value
    A = np.vstack([C.dot(A_theta), np.identity(moment_dim), -np.identity(moment_dim)])
    b = np.vstack([d, max_value*np.ones(moment_dim)[:,np.newaxis], max_value*np.ones(moment_dim)[:,np.newaxis]])
    inmat, poly, retmat = h2v(A,b)
    logger.debug("final")
    logger.debug(retmat)

    n_vertices = np.array(retmat)[:,1:] # only hull (no cone)
    return n_vertices

def swipe_joint_range(division_num = None, dowait = None, tm = None, plot = None):
    if division_num is None:
        division_num = 5

    if dowait is None:
        dowait = True

    if tm is None:
        tm = 0.5

    if plot is None: plot = True

    max_moment_vec = float("-inf")*np.ones(moment_dim)
    min_moment_vec = float("inf")*np.ones(moment_dim)
    return swipe_joint_range_impl(joint_order, rot_list,  max_moment_vec, min_moment_vec, division_num = division_num, dowait = dowait ,tm = tm, plot = plot)


def swipe_joint_range_impl(child_joint_indices, rot_list, max_moment_vec, min_moment_vec, division_num = None, dowait = None, tm = None, escape = None, plot = None):
    logger.debug("swipe_joint_range_impl()")
    # print "child_joint_indices="
    # print child_joint_indices
    logger.debug("")

    if escape is None: escape = False

    if escape:
        return max_moment_vec, min_moment_vec, escape
    else:
        turn = A_theta.shape[0] - len(child_joint_indices)
        if len(child_joint_indices) > 1:
            child_joint_idx = child_joint_indices[1] # x/y/z = 0/1/2
            child_joint_range = joint_range_list[child_joint_idx]
            child_joint_axis = np.identity(3)[:,child_joint_idx]
            for child_joint_angle in np.linspace(child_joint_range[0], child_joint_range[1], division_num):
                logger.info(str(joint_name_list[child_joint_idx]) + " is " + str(child_joint_angle) + " [deg]")
                pi.joint_angle_texts[child_joint_idx].set_text(joint_name_list[child_joint_idx] + " = " + str(child_joint_angle) + " [deg]")
                rot_list[turn] = linalg.expm3( np.cross(np.identity(moment_dim), child_joint_axis*np.deg2rad(child_joint_angle) ) )
                max_moment_vec, min_moment_vec, escape = swipe_joint_range_impl(child_joint_indices[1:], rot_list, max_moment_vec ,min_moment_vec, dowait = dowait, division_num = division_num, tm = tm, escape = escape, plot = plot)

            return max_moment_vec, min_moment_vec, escape
        else:
            rot_list[-1] = np.identity(3) # turn = 3-1
            for i in range(A_theta.shape[0]):
                group_last_axis = [ joint_group for joint_group in joint_structure for joint_axis in joint_group if joint_axis == joint_order[i] ][0][-1]
                # A_theta[i] = reduce(lambda x,y: np.dot(x,y), rot_list[joint_order.tolist().index(group_last_axis):]).dot(local_axis_list[i][:,np.newaxis]).T[0]
                A_theta[i] = reduce(lambda x,y: np.dot(x,y), rot_list[i:]).dot(local_axis_list[i][:,np.newaxis]).T[0]
            B_theta = np.identity(moment_dim) - A_theta.dot(A_theta.T).dot(S)

            logger.debug("rot_list=")
            logger.debug(rot_list)
            logger.debug("A_theta=")
            logger.debug(A_theta)
            logger.debug("B_theta=")
            logger.debug(B_theta)
            logger.debug("")

            n_vertices = convert_to_skeleton_moment_vertices(A_theta,B_theta)

            max_moment_vec = np.vstack([n_vertices, max_moment_vec]).max(axis=0)
            min_moment_vec = np.vstack([n_vertices, min_moment_vec]).min(axis=0)
            max_moment_vec[np.ma.where(abs(max_moment_vec) < 10)] = 0
            min_moment_vec[np.ma.where(abs(min_moment_vec) < 10)] = 0
            max_moment_vec[np.ma.where(abs(max_moment_vec) >= max_value)] = np.inf
            min_moment_vec[np.ma.where(abs(min_moment_vec) >= max_value)] = -np.inf
            pi.max_moment_text.set_text("max moments = " + str(max_moment_vec) + " [Nm]")
            logger.info(" max: " + str(max_moment_vec))
            logger.info(" min: " + str(min_moment_vec))
            if plot: pi.plot_convex_hull(n_vertices)

            if dowait:
                logger.critical("RET to continue, q to escape")
                key = raw_input()
                if key == 'q': escape = True
            else:
                time.sleep(tm)

            return max_moment_vec, min_moment_vec, escape

def init_vals():
    global joint_order
    global num_joints
    global moment_dim
    global rot_list
    global local_axis_list
    global A_theta
    global S
    global max_tau

    joint_order = np.array([idx for l in joint_structure for idx in l])
    num_joints = len(joint_order)
    moment_dim = num_joints
    rot_list = np.array([np.identity(3) for x in range(num_joints)])
    # local_axis_list = [None for x in range(num_joints)]
    local_axis_list = np.identity(moment_dim)[joint_order] # each row is axis

    A_theta = np.zeros([num_joints,moment_dim])
    S = 0.99 * np.diag([1 if x in joint_structure[-1] else 0 for x in joint_order])

    assert len(joint_range_list) == num_joints

    max_tau = np.array([max_tau_list[joint_idx] for joint_idx in joint_order])[:,np.newaxis] # from root order
    assert len(max_tau) == num_joints

def set_joint_structure(_joint_structure):
    global joint_structure
    joint_structure = _joint_structure
    init_vals()

max_value = 10000
joint_name_list = ("hip-x", "hip-y", "hip-z")
joint_range_list = [(-30,60),(-120,55),(-90,90)] # roll, pitch, yaw
# joint_range_list = [(0,0),(0,0),(-90,90)] # roll, pitch, yaw
# joint_range_list = [(-30,30),(0,0),(0,0)] # roll, pitch, yaw
# joint_range_list = [(-90,90),(0,0),(0,0)] # roll, pitch, yaw
# joint_range_list = [(0,0),(90,90),(0,0)] # roll, pitch, yaw
# joint_range_list = [(0,0),(45,45),(0,0)] # roll, pitch=45, yaw
# joint_range_list = [(45,45),(45,45),(0,0)] # roll=45, pitch=45, yaw
# joint_range_list = np.array([(60,60),(45,45),(0,0)]) # roll, pitch, yaw
# joint_range_list = [(-30,30),(-45,45),(0,0)] # roll, pitch, yaw
# max_tau_list = np.array([330,700,120]) # roll, pitch, yaw
max_tau_list = np.array([300,700,120]) # roll, pitch, yaw
# max_tau_list = np.array([330,750,120]) # roll, pitch, yaw 426,750,607

pi = PlotInterface()

if __name__ == '__main__':
    set_joint_structure([[2],[0],[1],[]])
    swipe_joint_range(division_num = 0)

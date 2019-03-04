#!/usr/bin/python
# -*- coding: utf-8 -*-

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import itertools as it

# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
from matplotlib import tri, cm

from scipy.spatial import ConvexHull, qhull
import scipy.linalg as linalg

import cdd

import pprint, time, sys, os
from colorama import Fore, Back, Style
import pdb

import roslib

import cnoid.Body as Body

from logging import getLogger, StreamHandler, DEBUG, INFO, WARNING, ERROR, CRITICAL
logger = getLogger(__name__)
handler = StreamHandler()
# handler.setLevel(DEBUG)
# logger.setLevel(DEBUG)
# logger.setLevel(INFO)
logger.setLevel(ERROR)
logger.addHandler(handler)
logger.propagate = False

import jsk_choreonoid.util as jcu

def h2v(A,b):
    try:
        inmat = cdd.Matrix(np.hstack([b,-A]), number_type='float')
        inmat.rep_type = cdd.RepType.INEQUALITY
        poly = cdd.Polyhedron(inmat)
    except RuntimeError:
        inmat = cdd.Matrix(np.hstack([b.astype(np.float16),-A.astype(np.float16)]), number_type='float')
        inmat.rep_type = cdd.RepType.INEQUALITY
        poly = cdd.Polyhedron(inmat)
    retmat = poly.get_generators()
    return inmat, poly, retmat

def v2h(flags,vertices):
    try:
        inmat = cdd.Matrix(np.hstack([flags, vertices]), number_type='float')
        inmat.rep_type = cdd.RepType.GENERATOR
        poly = cdd.Polyhedron(inmat)
    except RuntimeError:
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
        self.max_moment_text = self.ax.text2D(self._text_pos[0], self._text_pos[1]+0.007*len(self.joint_angle_texts), "", fontsize=25)
        self._text_pos = [-0.095,0.08]
        self.joint_angle_text = self.ax.text2D(self._text_pos[0], self._text_pos[1]-0.007*0,"", fontsize=25)
        self.max_moment_text = self.ax.text2D(self._text_pos[0], self._text_pos[1]-0.007*1, "", fontsize=25)

        self.max_display_num = 1500
        self.ax.set_xlim3d(-self.max_display_num,self.max_display_num)
        self.ax.set_ylim3d(-self.max_display_num,self.max_display_num)
        self.ax.set_zlim3d(-self.max_display_num,self.max_display_num)

        self.prev_surf_list = []

    def reset_hull(self):
        self.vertices = None

    def plot_convex_hull(self, vertices, save_plot=None, fname="test.png", isInstant=None):

        if save_plot is None: save_plot=False
        if isInstant is None: isInstant=True

        # ax.clear()
        # self.ax.set_xlim3d(-self.max_display_num,self.max_display_num)
        # self.ax.set_ylim3d(-self.max_display_num,self.max_display_num)
        # self.ax.set_zlim3d(-self.max_display_num,self.max_display_num)
        self.ax.set_xlabel("nx(roll) [Nm]")
        self.ax.set_ylabel("ny(pitch) [Nm]")
        self.ax.set_zlabel("nz(yaw) [Nm]")

        for surf in self.prev_surf_list: surf.remove()
        self.prev_surf_list = []

        if isInstant or self.vertices is None: # reset vertices for instant skeleton load moment
            self.vertices = vertices
        else:
            # [ self.vertices.append(vertex) for vertices in vertices ]
            self.vertices = np.append(self.vertices, vertices, axis=0) # append vertices for total skeleton load moment
            # need to reduce vertices?

        self.vertices = np.round(self.vertices) # round to integer
        self.vertices = np.array(list(map(list, set(map(tuple, self.vertices.tolist()))))) # remove dupulicates

        try:
            # hull = ConvexHull(vertices, incremental=True)
            hull = ConvexHull(self.vertices, incremental=True, qhull_options="QJ")
            logger.debug("simplices")
            logger.debug(hull.simplices)
        except qhull.QhullError:
            logger.error(Fore.RED+'!QhullError!'+Style.RESET_ALL)
            return

        for idx, face_indices in enumerate(hull.simplices): # faces -> hull.simplices
            # logger.debug(str(idx)+" face: "+str(face_indices))
            x,y,z = self.vertices[face_indices,0], self.vertices[face_indices,1], self.vertices[face_indices,2]

            # plot by trisurf
            x = x + z*0.0001
            y = y + z*0.0001
            # logger.debug("x,y,z= "+str(x)+", "+str(y)+", "+str(z))
            try:
                triang = tri.Triangulation(x, y)
                refiner = tri.UniformTriRefiner(triang)
                new, new_z = refiner.refine_field(z, subdiv=2)

                # norm = plt.Normalize(vmax=abs(y).max(), vmin=-abs(y).max())
                norm = plt.Normalize(vmax=1500, vmin=-1500)
                # kwargs = dict(triangles=new.triangles, cmap=cm.jet, norm=norm, linewidth=0.05, alpha = 0.3)
                kwargs = dict(triangles=new.triangles, cmap=cm.jet, norm=norm, linewidth=0.1, alpha = 0.3, edgecolors='black')

                self.prev_surf_list.append(self.ax.plot_trisurf(new.x, new.y, new_z, **kwargs))
            except RuntimeError:
                logger.error(Fore.RED+'RuntimeError (provably "Error in qhull Delaunay triangulation calculation")'+Style.RESET_ALL)
                logger.debug(str(idx)+" face: "+str(face_indices))
                logger.debug("x,y,z= "+str(x)+", "+str(y)+", "+str(z))

            # # plot by surface
            # x,y,z = [np.append(vec,vec[2]).reshape((2,2)) for vec in [x,y,z]]
            # logger.debug("x,y,z=\n "+str(x)+",\n "+str(y)+",\n "+str(z))
            # self.ax.plot_surface(x,y,z,alpha = 0.3)

        # plt.show()
        plt.pause(0.01)

        if save_plot: plt.savefig(fname)

def convert_to_skeleton_moment_vertices(A_, B_):
    num_joints = A_.shape[1]
    load_dim = A_.shape[0]
    try:
        joint_order
        if load_dim == 3:
            max_tau_theta = max_tau[:load_dim].reshape(load_dim)
        else:
            max_tau_theta = max_tau
    except NameError:
        max_tau_theta = max_tau
    # load_dim = 6 # needless?
    # load_dim = 3 # needless?
    # max_tau_theta = (max_tau/abs(A_.dot(A_.T))).min(axis=0) # tau_j = min_i(tau_i/|a_j.a_i|)
    logger.debug("max_tau="+str(max_tau))
    logger.debug("max_tau_theta=" + str(max_tau_theta))
    # tau convex hull H->V
    A = np.vstack([np.identity(num_joints),-np.identity(num_joints)])
    b = np.vstack([max_tau_theta[:,np.newaxis],max_tau_theta[:,np.newaxis]]) # min_tau = - max_tau -> -min_tau = max_tau
    try:
        inmat, poly, retmat = h2v(A,b)
    except RuntimeError:
        logger.error(Fore.RED+'!!!!!RuntimeError (h2v())!!!!!'+Style.RESET_ALL)
        return np.array([range(6)])

    logger.debug("max_tau")
    logger.debug(retmat)
    tau_vertices = np.array(retmat)[:,1:] # u_k^T

    # convert to tau_tilde V->H
    tau_tilde_vertices = tau_vertices.dot(B_.T) # u_k^~T
    b_tilde = np.ones(tau_vertices.shape[0])[:,np.newaxis] # only hull (no cone)
    try:
        inmat, poly, retmat = v2h(b_tilde, tau_tilde_vertices)
    except RuntimeError:
        logger.error(Fore.RED+'!!!!!RuntimeError (v2h())!!!!!'+Style.RESET_ALL)
        return np.array([range(6)])
    logger.debug("tau_tilde")
    logger.debug(retmat)
    C = -np.array(retmat)[:,1:]
    d = np.array(retmat)[:,0:1]
    logger.debug("")

    # H->V
    # for max value
    A = np.vstack([C.dot(A_), np.identity(load_dim), -np.identity(load_dim)])
    b = np.vstack([d, max_value*np.ones(load_dim)[:,np.newaxis], max_value*np.ones(load_dim)[:,np.newaxis]])
    try:
        inmat, poly, retmat = h2v(A,b)
    except RuntimeError:
        logger.error(Fore.RED+'!!!!!RuntimeError (h2v())!!!!!'+Style.RESET_ALL)
        return np.array([range(6)])

    logger.debug("final")
    logger.debug(retmat)

    n_vertices = np.array(retmat)[:,1:] # only hull (no cone)
    return n_vertices

def sweep_joint_range(division_num=None, dowait=None, tm=None, plot=None, save_plot=None, fname=None, isInstant=None):
    logger.info("joint_structure:" + str(joint_structure))
    if division_num is None:
        division_num = 5

    if dowait is None:
        dowait = True

    if tm is None:
        tm = 0.5

    if plot is None: plot = True

    if save_plot is None: save_plot = False
    if fname is None: fname = ""

    if isInstant is None: isInstant = True
    # if not isInstant: pi.reset_hull() # error (popitem(): dictionary is empty)

    max_moment_vec = float("-inf")*np.ones(moment_dim)
    min_moment_vec = float("inf")*np.ones(moment_dim)
    return sweep_joint_range_impl(joint_order, rot_list,  max_moment_vec, min_moment_vec, division_num=division_num, dowait=dowait, tm=tm, plot=plot, save_plot=save_plot, fname=fname, isInstant=isInstant)


def sweep_joint_range_impl(child_joint_indices, rot_list, max_moment_vec, min_moment_vec, division_num, dowait, tm, plot, save_plot, fname, isInstant, escape=None):
    logger.debug("sweep_joint_range_impl()")
    logger.info("child_joint_indices="+str(child_joint_indices))
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
            logger.info("child_joint_range="+str(child_joint_range))
            for idx, child_joint_angle in enumerate(np.linspace(child_joint_range[0], child_joint_range[1], division_num)):
                logger.info(str(joint_name_list[child_joint_idx]) + " is " + str(child_joint_angle) + " [deg]")
                # pi.joint_angle_texts[child_joint_idx].set_text(joint_name_list[child_joint_idx] + " = " + str(child_joint_angle) + " [deg]")
                rot_list[turn] = linalg.expm3( np.cross(np.identity(moment_dim), child_joint_axis*np.deg2rad(child_joint_angle) ) )
                max_moment_vec, min_moment_vec, escape = sweep_joint_range_impl(child_joint_indices[1:], rot_list, max_moment_vec ,min_moment_vec, dowait=dowait, division_num=division_num, tm=tm, escape=escape, plot=plot,
                                                                                save_plot=save_plot, fname=fname.replace(".","-"+str(idx)+"."), isInstant=isInstant)

            return max_moment_vec, min_moment_vec, escape
        else:
            rot_list[-1] = np.identity(3) # turn = 3-1
            for i in range(A_theta.shape[0]):
                group_last_axis = [ joint_group for joint_group in joint_structure for joint_axis in joint_group if joint_axis == joint_order[i] ][0][-1]
                # A_theta[i] = reduce(lambda x,y: np.dot(x,y), rot_list[joint_order.tolist().index(group_last_axis):]).dot(local_axis_list[i][:,np.newaxis]).T[0]
                A_theta[i] = reduce(lambda x,y: np.dot(x,y), rot_list[i:]).dot(local_axis_list[i][:,np.newaxis]).T[0]
            B_theta = np.identity(moment_dim) - A_theta.dot(A_theta.T).dot(S) # E - A*A^T*S

            logger.debug("rot_list=")
            logger.debug(rot_list)
            logger.debug("A_theta=")
            logger.debug(A_theta)
            logger.debug("B_theta=")
            logger.debug(B_theta)
            logger.debug("")

            n_vertices = convert_to_skeleton_moment_vertices(A_theta,B_theta) # instance

            max_moment_vec = np.vstack([n_vertices, max_moment_vec]).max(axis=0)
            min_moment_vec = np.vstack([n_vertices, min_moment_vec]).min(axis=0)
            max_moment_vec[np.ma.where(abs(max_moment_vec) < 10)] = 0 # set |elements|<10 to 0
            min_moment_vec[np.ma.where(abs(min_moment_vec) < 10)] = 0
            max_moment_vec[np.ma.where(abs(max_moment_vec) >= max_value)] = np.inf # set |elements|>max_value to inf
            min_moment_vec[np.ma.where(abs(min_moment_vec) >= max_value)] = -np.inf
            pi.max_moment_text.set_text("max moments = " + str(max_moment_vec) + " [Nm]")
            logger.info(" max: " + str(max_moment_vec))
            logger.info(" min: " + str(min_moment_vec))
            if plot: pi.plot_convex_hull(n_vertices, save_plot=save_plot, fname=fname, isInstant=isInstant)

            if dowait:
                logger.critical(Fore.BLUE+"RET to continue, q to escape"+Style.RESET_ALL)
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

def skew(vec):
    return np.array([[0,-vec[2],vec[1]],
                     [vec[2],0,-vec[0]],
                     [-vec[1],vec[0],0]])

class JointLoadWrenchAnalyzer():
    def __init__(self, actuator_set_list_, joint_range_list=None, robot_item=None, robot_model_file=None, end_link_name=None):
        self.world = jcu.World()
        logger.info(" is_choreonoid:" + str(self.world.is_choreonoid))

        self.actuator_set_list = actuator_set_list_
        self.set_robot(robot_item, robot_model_file)
        if end_link_name is None: end_link_name = "LLEG_JOINT5"
        self.set_joint_path(end_link_name=end_link_name)

        self.axis_product_mat = reduce(lambda ret,vec: ret+np.array(vec).reshape(6,1)*np.array(vec), [np.zeros((6,6)),[1,1,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,1]])

        self.joint_index_offsets = [0,0,0,3,6,6]
        # set max_tau in order from root using jointAxis()
        max_tau_list = np.array([300,700,120, 0,700,0, 100,200,0]) # (hip-x,hip-y,hip-z,  nil,knee-y,nil, ankle-r,ankle-p)
        # max_tau_list = np.array([300,700,120, 0,700,0, 300,700,0]) # easy
        self.max_tau = np.array([ max_tau_list[offset_idx + list(self.joint_path.joint(joint_idx).jointAxis()).index(1)] for joint_idx,offset_idx in enumerate(self.joint_index_offsets) ])

        self.set_joint_range(joint_range_list)

    def set_robot(self, robot_item=None, robot_model_file=None):
        self.robot_item = robot_item
        self.robot_model_file = os.path.join(roslib.packages.get_pkg_dir("jsk_models"),"JAXON_RED/JAXON_REDmain.wrl") if robot_model_file is None else robot_model_file
        if self.robot_item is None:
            if self.world.is_choreonoid: # in choreonoid
                self.robot = self.world.set_robotItem(self.robot_model_file).body()
            else: # others
                self.robot = jcu.get_robot(self.robot_model_file)
        else:
            self.robot = self.robot_item.body()
        logger.info("robot model name: "+str(self.robot.modelName()))

    def set_joint_path(self, root_link_name=None,end_link_name=None):
        self.root_link = self.robot.rootLink() if root_link_name is None else self.robot.link(root_link_name)
        self.end_link = self.robot.link("LLEG_JOINT5") if end_link_name is None else self.robot.link(end_link_name)
        logger.info("end link: "+str(end_link_name))
        self.joint_path = Body.JointPath(self.root_link, self.end_link)

    def set_joint_range(self, joint_range_list=None):
        if joint_range_list is None: joint_range_list = [(-30,60),(-120,55),(-90,90), (0,0),(0,150),(0,0) ,(-60,60),(-120,120),(0,0)] # set full range to all joint
        self.joint_range_list = np.array([ joint_range_list[offset_idx + list(self.joint_path.joint(joint_idx).jointAxis()).index(1)] for joint_idx,offset_idx in enumerate(self.joint_index_offsets) ])

    # calc skeleton load wrench vertices at current pose
    def calc_current_load_wrench_vertices(self, target_link_name, root_link_name=None, end_link_name=None): # set joint name not joint index
        # self.robot_item.calcForwardKinematics()
        self.robot.calcForwardKinematics()
        if self.robot_item is not None: self.robot_item.notifyKinematicStateChange()

        root_link = self.root_link if root_link_name is None else self.robot.link(root_link_name)
        end_link = self.end_link if end_link_name is None else self.robot.link(end_link_name)
        target_link = self.robot.link(target_link_name)

        # calc selection matrix
        target_link_idx = self.joint_path.indexOf(target_link)
        num_adjacent_actuator_set = self.actuator_set_list[target_link_idx]
        diag_vec = np.array([i < target_link_idx + num_adjacent_actuator_set[1] + 1 for i in range(self.joint_path.numJoints())]) * np.array([i > target_link_idx - num_adjacent_actuator_set[0] for i in range(self.joint_path.numJoints())]).astype(np.int)
        self.S = 0.99 * np.diag(diag_vec)

        Jre = Body.JointPath(root_link, end_link).calcJacobian() # root->end
        Jri = Body.JointPath(root_link, target_link).calcJacobian() # root->i
        Jie = Body.JointPath(target_link, end_link).calcJacobian() # i->end

        R2i = np.r_[ np.c_[target_link.R,np.zeros((3,3))],
                     np.c_[np.zeros((3,3)), target_link.R] ]

        J6ei = np.r_[ np.c_[ np.eye(3),-skew(target_link.p-end_link.p) ], # [ [E,rx] , [0,E] ]
                      np.c_[ np.zeros((3,3)),np.eye(3) ] ]

        Ji_tilde = np.c_[ Jri, J6ei.dot(Jie) ] # [Jri J6ei*Jie]
        G = np.eye(self.joint_path.numJoints()) # tmp E
        A_theta = np.diag([0,0,0, 1,1,1]).dot(Jre)
        # A_t = np.diag([0,0,0, 1,1,1]).dot(Jre)

        axis_mat = A_theta[3:]
        global max_tau
        max_tau = (self.max_tau/abs(axis_mat.T.dot(axis_mat))).min(axis=1) # tau_j = min_i(tau_i/|a_j.a_i|)
        # max_tau = (self.max_tau/abs(self.axis_product_mat*axis_mat.T.dot(axis_mat))).min(axis=1) # tau_j = min_i(tau_i/|a_j.a_i|)

        self.Jre = Jre
        self.Jri = Jri
        self.J6ei = J6ei
        self.Ji_tilde = Ji_tilde
        self.R2i = R2i
        self.A_theta = A_theta

        return convert_to_skeleton_moment_vertices( Ji_tilde.transpose().dot(R2i), G.transpose()-Ji_tilde.transpose().dot(A_theta).dot(G.transpose()).dot(self.S) )

    def calc_whole_range_max_load_wrench(self, target_joint_name, joint_idx=None, do_plot=None, save_plot=None, fname=None, is_instant=None, do_wait=None, division_num=None, tm=None):
        if joint_idx is None:
            joint_idx = 0
            self.max_load_wrench = np.zeros(6)
            self.min_load_wrench = np.zeros(6)
        if do_plot is None: do_plot = True
        if save_plot is None: save_plot = False
        if fname is None: fname = ""
        if is_instant is None: is_instant = True
        if do_wait is None: do_wait = False
        if division_num is None: division_num = 1
        if tm is None: tm = 0.2

        joint_name = self.joint_path.joint(joint_idx).name()
        joint_range = self.joint_range_list[joint_idx]
        division_num_ = 1 if joint_range[0] == joint_range[1] else division_num

        if joint_idx < self.joint_path.numJoints() and logger.isEnabledFor(INFO): sys.stdout.write(Fore.GREEN+" "+"#"+joint_name+Style.RESET_ALL)
        if joint_idx+1 == self.joint_path.numJoints() and logger.isEnabledFor(INFO): print(" changed")
        for joint_angle in np.linspace(joint_range[0],joint_range[1],division_num_):
            self.robot.link(joint_name).q = np.deg2rad(joint_angle) # set joint angle [rad]
            if joint_idx+1 < self.joint_path.numJoints():
                self.calc_whole_range_max_load_wrench(target_joint_name,joint_idx+1,do_plot=do_plot,save_plot=save_plot,fname=fname,is_instant=is_instant,do_wait=do_wait,division_num=division_num,tm=tm)
            else:
                joint_angle_text = "joint angles: " + str([np.round(np.rad2deg(self.robot.link(self.joint_path.joint(idx).name()).q),1) for idx in range(self.joint_path.numJoints())]) + " [deg]" # round joint angles
                pi.joint_angle_text.set_text(joint_angle_text)
                logger.info(joint_angle_text)
                n_vertices = self.calc_current_load_wrench_vertices(target_joint_name)
                logger.debug("n_vertices=")
                logger.debug(n_vertices[:,3:])

                self.max_load_wrench = np.vstack([n_vertices, self.max_load_wrench]).max(axis=0)
                self.min_load_wrench = np.vstack([n_vertices, self.min_load_wrench]).min(axis=0)
                self.max_load_wrench[np.ma.where(abs(self.max_load_wrench) < 10)] = 0 # set |elements|<10 to 0
                self.min_load_wrench[np.ma.where(abs(self.min_load_wrench) < 10)] = 0
                self.max_load_wrench[np.ma.where(abs(self.max_load_wrench) >= max_value)] = np.inf # set |elements|>max_value to inf
                self.min_load_wrench[np.ma.where(abs(self.min_load_wrench) >= max_value)] = -np.inf

                pi.max_moment_text.set_text("max moments = " + str(self.max_load_wrench[3:]) + " [Nm]")
                logger.info(" max: " + str(self.max_load_wrench))
                logger.info(" min: " + str(self.min_load_wrench))
                if do_plot: pi.plot_convex_hull(n_vertices[:,3:], save_plot=save_plot, fname=fname, isInstant=is_instant)

                if do_wait:
                    logger.critical("RET to continue, q to escape"+Style.RESET_ALL)
                    key = raw_input()
                    if key == 'q': escape = True
                else:
                    time.sleep(tm)

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
# max_tau_list = np.array([300,700,120]) # roll, pitch, yaw
max_tau_list = np.array([300,700,120,700,100,100]) # roll, pitch, yaw
# max_tau_list = np.array([330,750,120]) # roll, pitch, yaw 426,750,607

pi = PlotInterface()

def export_snapshot():
    global joint_range_list
    max_display_num = 800
    pi.ax.set_xlim3d(-max_display_num,max_display_num)
    pi.ax.set_ylim3d(-max_display_num,max_display_num)
    pi.ax.set_zlim3d(-max_display_num,max_display_num)
    joint_range_list = [(-30,60),(0,80),(-90,90)]
    set_joint_structure([[2],[1],[0],[]])
    # pi.joint_angle_texts[joint_order[0]].set_text(joint_name_list[joint_order[0]] + " = "+ str(0.0) + " [deg]")
    sweep_joint_range(division_num=9, dowait=False, save_plot=True, fname="total-skeleton-load-moment-solid/total-skeleton-load-moment-solid.png", isInstant=False)
    sweep_joint_range(division_num=9, dowait=False, save_plot=True, fname="instant-skeleton-load-moment-solid/instant-skeleton-load-moment-solid.png")

if __name__ == '__main__':
    np.set_printoptions(precision=5)

    # set_joint_structure([[2],[1],[0],[]])
    # sweep_joint_range(division_num = 0)

    plt.rcParams["font.size"] = 25

    max_display_num = 800
    pi.ax.set_xlim3d(-max_display_num,max_display_num)
    pi.ax.set_ylim3d(-max_display_num,max_display_num)
    pi.ax.set_zlim3d(-max_display_num,max_display_num)

    # joint_range_list = [(0,0),(0,0),(0,0)]
    # set_joint_structure([[2],[0],[1],[]])
    # pi.joint_angle_texts[joint_order[0]].set_text(joint_name_list[joint_order[0]] + " = "+ str(0.0) + " [deg]")
    # sweep_joint_range(division_num = 1, dowait=False, save_plot=True, fname="initial-skeleton-load-moment-solid.png")

    # joint_range_list = [(35,35),(100,100),(20,20)]
    # pi.joint_angle_texts[joint_order[0]].set_text(joint_name_list[joint_order[0]] + " = "+ str(20.0) + " [deg]")
    # sweep_joint_range(division_num = 1, dowait=False, save_plot=True, fname="joint-structure-comparison-solid_zxy.png")

    # set_joint_structure([[2],[1],[0],[]])
    # pi.joint_angle_texts[joint_order[0]].set_text(joint_name_list[joint_order[0]] + " = "+ str(20.0) + " [deg]")
    # sweep_joint_range(division_num = 1, dowait=False, save_plot=True, fname="joint-structure-comparison-solid_zyx.png")


    # joint_range_list = [(20,20),(70,70),(0,0)]
    # max_display_num = 500
    # pi.ax.set_xlim3d(-max_display_num,max_display_num)
    # pi.ax.set_ylim3d(-max_display_num,max_display_num)
    # pi.ax.set_zlim3d(-max_display_num,max_display_num)

    # set_joint_structure([[2],[0],[1],[]])
    # pi.joint_angle_texts[joint_order[0]].set_text(joint_name_list[joint_order[0]] + " = "+ str(0.0) + " [deg]")
    # sweep_joint_range(division_num = 1, dowait=False, save_plot=True, fname="deflection-correction-comparison-solid_rotational.png")

    # set_joint_structure([[2],[0],[1]])
    # pi.joint_angle_texts[joint_order[0]].set_text(joint_name_list[joint_order[0]] + " = "+ str(0.0) + " [deg]")
    # sweep_joint_range(division_num = 1, dowait=False, save_plot=True, fname="deflection-correction-comparison-solid_tendon.png")

    # set_joint_structure([[2],[0,1]])
    # pi.joint_angle_texts[joint_order[0]].set_text(joint_name_list[joint_order[0]] + " = "+ str(0.0) + " [deg]")
    # sweep_joint_range(division_num = 1, dowait=False, save_plot=True, fname="deflection-correction-comparison-solid_linear.png")

    division_num=4; do_wait=False; tm=0;   do_plot=True
    # division_num=6, do_wait=False, tm=0.5, do_plot=False
    # division_num=9, do_wait=False, tm=0,   do_plot=False
    model_path = os.path.join(roslib.packages.get_pkg_dir("structure_analyzer"), "models")

    joint_range_list = [(-30,60),(-120,55),(-90,90), (0,0),(0,150),(0,0) ,(-60,60),(-80,75),(0,0)] # set full range to all joint
    # joint_range_list = [(0,60),(0,120),(0,90), (0,0),(90,90),(0,0) ,(0,0),(0,0),(0,0)] # hip only/half range
    # joint_range_list = [(60,60),(55,55),(-90,-90), (0,0),(90,90),(0,0) ,(0,0),(0,0),(0,0)] # hip only/fix

    # # JAXON_RED
    # analyzer = JointLoadWrenchAnalyzer([0,0,(0,0),0,0,0], robot_item=None)
    # analyzer.calc_whole_range_max_load_wrench('LLEG_JOINT2', division_num=division_num, do_wait=do_wait, tm=tm, do_plot=do_plot)
    # logger.critical(Fore.YELLOW+analyzer.robot.name()+" max wrench: "+str(analyzer.max_load_wrench)+Style.RESET_ALL)

    joint_structure_str="z-x-y_y_y-x"
    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], joint_range_list=joint_range_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_structure_str+".wrl"))
    analyzer.calc_whole_range_max_load_wrench('JOINT2', division_num=division_num, do_wait=do_wait, tm=tm, do_plot=do_plot)
    logger.critical(Fore.YELLOW+joint_structure_str+" max wrench: "+str(analyzer.max_load_wrench)+Style.RESET_ALL)

    joint_structure_str="z-x-y_y_x-y"
    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], joint_range_list=joint_range_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_structure_str+".wrl"))
    analyzer.calc_whole_range_max_load_wrench('JOINT2', division_num=division_num, do_wait=do_wait, tm=tm, do_plot=do_plot)
    logger.critical(Fore.YELLOW+joint_structure_str+" max wrench: "+str(analyzer.max_load_wrench)+Style.RESET_ALL)
    logger.critical("")


    joint_structure_str="z-y-x_y_y-x"
    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], joint_range_list=joint_range_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_structure_str+".wrl"))
    analyzer.calc_whole_range_max_load_wrench('JOINT2', division_num=division_num, do_wait=do_wait, tm=tm, do_plot=do_plot)
    logger.critical(Fore.YELLOW+joint_structure_str+" max wrench: "+str(analyzer.max_load_wrench)+Style.RESET_ALL)

    joint_structure_str="z-y-x_y_x-y"
    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], joint_range_list=joint_range_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_structure_str+".wrl"))
    analyzer.calc_whole_range_max_load_wrench('JOINT2', division_num=division_num, do_wait=do_wait, tm=tm, do_plot=do_plot)
    logger.critical(Fore.YELLOW+joint_structure_str+" max wrench: "+str(analyzer.max_load_wrench)+Style.RESET_ALL)
    logger.critical("")


    joint_structure_str="x-y-z_y_y-x"
    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], joint_range_list=joint_range_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_structure_str+".wrl"))
    analyzer.calc_whole_range_max_load_wrench('JOINT2', division_num=division_num, do_wait=do_wait, tm=tm, do_plot=do_plot)
    logger.critical(Fore.YELLOW+joint_structure_str+" max wrench: "+str(analyzer.max_load_wrench)+Style.RESET_ALL)

    joint_structure_str="x-y-z_y_x-y"
    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], joint_range_list=joint_range_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_structure_str+".wrl"))
    analyzer.calc_whole_range_max_load_wrench('JOINT2', division_num=division_num, do_wait=do_wait, tm=tm, do_plot=do_plot)
    logger.critical(Fore.YELLOW+joint_structure_str+" max wrench: "+str(analyzer.max_load_wrench)+Style.RESET_ALL)
    logger.critical("")


    joint_structure_str="y-x-z_y_y-x"
    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], joint_range_list=joint_range_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_structure_str+".wrl"))
    analyzer.calc_whole_range_max_load_wrench('JOINT2', division_num=division_num, do_wait=do_wait, tm=tm, do_plot=do_plot)
    logger.critical(Fore.YELLOW+joint_structure_str+" max wrench: "+str(analyzer.max_load_wrench)+Style.RESET_ALL)

    joint_structure_str="y-x-z_y_x-y"
    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], joint_range_list=joint_range_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_structure_str+".wrl"))
    analyzer.calc_whole_range_max_load_wrench('JOINT2', division_num=division_num, do_wait=do_wait, tm=tm, do_plot=do_plot)
    logger.critical(Fore.YELLOW+joint_structure_str+" max wrench: "+str(analyzer.max_load_wrench)+Style.RESET_ALL)
    logger.critical("")


    joint_structure_str="z-x-y_y_y-x"
    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(1,0),(0,0),(0,0),(0,0)], joint_range_list=joint_range_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_structure_str+".wrl"))
    analyzer.calc_whole_range_max_load_wrench('JOINT2', division_num=division_num, do_wait=do_wait, tm=tm, do_plot=do_plot)
    logger.critical(Fore.YELLOW+joint_structure_str+" max wrench: "+str(analyzer.max_load_wrench)+Style.RESET_ALL)

    joint_structure_str="z-y-x_y_y-x"
    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(1,0),(0,0),(0,0),(0,0)], joint_range_list=joint_range_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_structure_str+".wrl"))
    analyzer.calc_whole_range_max_load_wrench('JOINT2', division_num=division_num, do_wait=do_wait, tm=tm, do_plot=do_plot)
    logger.critical(Fore.YELLOW+joint_structure_str+" max wrench: "+str(analyzer.max_load_wrench)+Style.RESET_ALL)

    joint_structure_str="z-x-y_y_y-x"
    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(2,0),(0,0),(0,0),(0,0)], joint_range_list=joint_range_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_structure_str+".wrl"))
    analyzer.calc_whole_range_max_load_wrench('JOINT2', division_num=division_num, do_wait=do_wait, tm=tm, do_plot=do_plot)
    logger.critical(Fore.YELLOW+joint_structure_str+" max wrench: "+str(analyzer.max_load_wrench)+Style.RESET_ALL)

    joint_structure_str="z-y-x_y_y-x"
    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(2,0),(0,0),(0,0),(0,0)], joint_range_list=joint_range_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_structure_str+".wrl"))
    analyzer.calc_whole_range_max_load_wrench('JOINT2', division_num=division_num, do_wait=do_wait, tm=tm, do_plot=do_plot)
    logger.critical(Fore.YELLOW+joint_structure_str+" max wrench: "+str(analyzer.max_load_wrench)+Style.RESET_ALL)

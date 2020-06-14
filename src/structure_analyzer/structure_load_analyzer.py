#!/usr/bin/python
# -*- coding: utf-8 -*-

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib.ticker import LinearLocator, FormatStrFormatter, MaxNLocator
import itertools as it

# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
from matplotlib import tri, cm

from scipy.spatial import ConvexHull, qhull
import scipy.linalg as linalg

import cdd

from enum import Enum

import pprint, time, sys, os, re
from colorama import Fore, Back, Style
import functools

import roslib

from cnoid import Body, Base, UtilPlugin

from logger import *
import jsk_choreonoid.util as jcu
import jsk_choreonoid.body_util

if jcu.is_choreonoid():
    import pdb as ipdb
else:
    import ipdb

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

class MomentType(Enum):
    LINKCOORD = 0
    ACTUATORAXIS = 1

class PlotInterface():
    def __init__(self):
        self.fig = plt.figure()
        self.fig.set_size_inches((8.0,8.0))
        self.ax = self.fig.gca(projection='3d')
        self.fig.subplots_adjust(left=-0.05,right=0.95, bottom=0.02,top=1, wspace=0.1, hspace=1)
        self.fontsize = 35
        self.fontfamily = 'Times New Roman'

        font_row_height = self.fontsize/2500.0
        self._text_pos = [-0.08,0.075]
        self.joint_angle_text = self.ax.text2D(self._text_pos[0], self._text_pos[1]-font_row_height*0,"")
        self.joint_angle_text.set_fontsize(self.fontsize)
        self.joint_angle_text.set_family(self.fontfamily)
        self.max_moment_text = self.ax.text2D(self._text_pos[0], self._text_pos[1]-font_row_height*1, "")
        self.max_moment_text.set_fontsize(self.fontsize)
        self.max_moment_text.set_family(self.fontfamily)

        # label
        label_fontsize_rate = 1.1
        self.ax.set_xlabel('',fontsize=self.fontsize*label_fontsize_rate)
        self.ax.set_ylabel('',fontsize=self.fontsize*label_fontsize_rate)
        self.ax.set_zlabel('',fontsize=self.fontsize*label_fontsize_rate)

        # ticks
        tics_fontsize_rate = 0.8
        self.ax.tick_params(labelsize=self.fontsize*tics_fontsize_rate)

        # margin between tics and axis label
        labelpad_rate = 0.6
        self.ax.axes.xaxis.labelpad=self.fontsize*labelpad_rate
        self.ax.axes.yaxis.labelpad=self.fontsize*labelpad_rate
        self.ax.axes.zaxis.labelpad=self.fontsize*labelpad_rate

        # select tics position
        self.ax.axes.xaxis.tick_top()
        self.ax.axes.yaxis.tick_bottom()
        self.ax.axes.zaxis.tick_top()

        # set max tics num by Locator
        max_n_locator = 5
        self.ax.xaxis.set_major_locator(MaxNLocator(max_n_locator))
        self.ax.yaxis.set_major_locator(MaxNLocator(max_n_locator))
        self.ax.zaxis.set_major_locator(MaxNLocator(max_n_locator))

        self.prev_surf_list = []

        self.reset_hull()

    def set_max_display_num(self, max_display_num):
        self.ax.set_xlim3d(-max_display_num,max_display_num)
        self.ax.set_ylim3d(-max_display_num,max_display_num)
        self.ax.set_zlim3d(-max_display_num,max_display_num)

    def reset_hull(self):
        self.vertices = None

    def plot_convex_hull(self, vertices, save_plot=False, fname="test.png", isInstant=True):

        # ax.clear()
        # self.ax.set_xlim3d(-self.max_display_num,self.max_display_num)
        # self.ax.set_ylim3d(-self.max_display_num,self.max_display_num)
        # self.ax.set_zlim3d(-self.max_display_num,self.max_display_num)

        for surf in self.prev_surf_list: surf.remove()
        self.prev_surf_list = []
        self.ax.lines.clear() # clear edge plots

        if isInstant or self.vertices is None: # reset vertices for instant frame load moment
            self.vertices = vertices
        else:
            # [ self.vertices.append(vertex) for vertices in vertices ]
            self.vertices = np.append(self.vertices, vertices, axis=0) # append vertices for total frame load moment
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
                new, new_z = refiner.refine_field(z, triinterpolator=tri.LinearTriInterpolator(triang,z), subdiv=2)

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

        # plot edges
        kwargs = dict(linewidth=2.0, alpha = 0.9, color='black')
        face_pair_set = set([tuple(set((face_idx, neighbor_idx))) for face_idx, neighbor_indices in enumerate(hull.neighbors) for neighbor_idx in neighbor_indices])
        for face_idx0, face_idx1 in face_pair_set:
            face_vertices0 = hull.simplices[face_idx0]
            face_vertices1 = hull.simplices[face_idx1]
            edge_idx0, edge_idx1 = tuple( set(face_vertices0) & set(face_vertices1) )
            face0_normal = np.cross(hull.points[face_vertices0[1]] - hull.points[face_vertices0[0]], hull.points[face_vertices0[2]] - hull.points[face_vertices0[0]])
            face1_normal = np.cross(hull.points[face_vertices1[1]] - hull.points[face_vertices1[0]], hull.points[face_vertices1[2]] - hull.points[face_vertices1[0]])
            face0_normal /= np.linalg.norm(face0_normal)
            face1_normal /= np.linalg.norm(face1_normal)

            edge_vertices = hull.points[edge_idx0:edge_idx1+1:edge_idx1-edge_idx0,:]
            if abs(face0_normal.dot(face1_normal)) < 0.999:
                self.ax.plot(edge_vertices[:,0], edge_vertices[:,1], edge_vertices[:,2], **kwargs)

        # plt.show()
        plt.pause(0.01)

        if save_plot: plt.savefig(fname)

def skew(vec):
    return np.array([[0,-vec[2],vec[1]],
                     [vec[2],0,-vec[0]],
                     [-vec[1],vec[0],0]])

class JointLoadWrenchAnalyzer(object):
    def __init__(self, actuator_set_list_, joint_group_list=None, joint_range_list=None, max_tau_list=None,
                     robot_item=None, robot_model_file=None, end_link_name="LLEG_JOINT5", moment_colors=None,
                     step_angle_list=None, step_angle=10, saturation_vec = None):
        self.world = jcu.World()
        logger.info(" is_choreonoid:" + str(self.world.is_choreonoid))

        self.set_robot(robot_item, robot_model_file)
        self.set_joint_path(end_link_name=end_link_name)
        self.draw_interfaces = None
        self.set_moment_colors(moment_colors=moment_colors)

        self.actuator_set_list = actuator_set_list_
        if len(self.actuator_set_list) != self.joint_path.numJoints:
            raise RuntimeError('actuator_set_list {0} must be the same length with JointPath {1}'.format(len(self.actuator_set_list), self.joint_path.numJoints))

        self.apply_joint_group_list(joint_group_list)

        self.set_max_tau(max_tau_list)

        self.set_joint_range(joint_range_list)

        if step_angle_list is None: step_angle_list = [step_angle]*self.joint_path.numJoints
        self.step_angle_list = step_angle_list
        if len(self.step_angle_list) != self.joint_path.numJoints:
            raise RuntimeError('step_angle_list {0} must be the same length with JointPath {1}'.format(len(self.step_angle_list), self.joint_path.numJoints))

        self.load_dim = 6

        self.reset_max_min_wrench()

        # self.saturation_vec = np.full((self.load_dim,), 10000) if saturation_vec is None else saturation_vec
        self.saturation_vec = np.array([10000,10000,10000, 2000,2000,2000]) if saturation_vec is None else saturation_vec

        if self.world.is_choreonoid:
            self.tree_view = Base.ItemTreeView.instance
            self.message_view = Base.MessageView.instance
            self.scene_widget = Base.SceneView.instance.sceneWidget
            self.set_draw_interfaces()

    def set_robot(self, robot_item=None, robot_model_file=None):
        self.robot_item = robot_item
        self.robot_model_file = os.path.join(roslib.packages.get_pkg_dir("jsk_models"),"JAXON_RED/JAXON_REDmain.wrl") if robot_model_file is None else robot_model_file
        if self.robot_item is None:
            if self.world.is_choreonoid: # in choreonoid

                # not set RobotItem dupulicately
                childItem = self.world.worldItem.childItem
                while childItem:
                    if childItem.filePath == robot_model_file:
                        self.robot_item = childItem
                        break
                    childItem = childItem.nextItem
                if self.robot_item is None:
                    self.robot_item = self.world.set_robotItem(self.robot_model_file)

                self.robot = self.robot_item.body
            else: # others
                self.robot = jcu.get_robot(self.robot_model_file)
        else:
            self.robot = self.robot_item.body
        logger.info("robot model name: "+str(self.robot.modelName))

    def set_joint_path(self, root_link_name=None,end_link_name="LLEG_JOINT5"):
        self.root_link = self.robot.rootLink if root_link_name is None else self.robot.link(root_link_name)
        self.end_link = self.robot.link(end_link_name)
        logger.info("end link: "+str(end_link_name))
        self.joint_path = Body.JointPath.getCustomPath(self.robot, self.root_link, self.end_link)
        if self.joint_path.numJoints < 1: raise RuntimeError('JointPath length ({}->{}) is 0 or less. Please set the valid names of a root link and an end link'.format(root_link_name, end_link_name))

    def apply_joint_group_list(self, joint_group_list=None):
        joint_group_list = [3,1,2] if joint_group_list is None else joint_group_list

        if sum(joint_group_list) != self.joint_path.numJoints:
            raise RuntimeError('sum of joint_group_list {0} must be the same length with JointPath {1}'.format(sum(joint_group_list), self.joint_path.numJoints))

        self.axis_product_mat = np.zeros((self.joint_path.numJoints,self.joint_path.numJoints))
        for s,g in zip([sum(joint_group_list[:idx]) for idx in range(len(joint_group_list))], joint_group_list): self.axis_product_mat[s:s+g,s:s+g] = 1
        # eg) [[1,1,1,0,0,0],[1,1,1,0,0,0],[1,1,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,1],[0,0,0,0,1,1]] blocked diag matrix

        self.joint_index_offsets = sum([[3*idx]*group_num for idx, group_num in enumerate(joint_group_list)], []) # eg) 6 dof leg: [0,0,0,3,6,6] (each joint group is 3 dof)

    # set max_tau in order from root using jointAxis
    # max_tau_list is in order of coordinate axes 'x,y,z'
    def set_max_tau(self, max_tau_list=None):
        max_tau_list = np.array([300,700,120, 0,700,0, 100,200,0]) if max_tau_list is None else max_tau_list # (hip-x,hip-y,hip-z,  nil,knee-y,nil, ankle-r,ankle-p)
        # max_tau_list = np.array([300,700,120, 0,700,0, 300,700,0]) if max_tau_list is None else max_tau_list # easy
        self.max_tau = np.array([ max_tau_list[offset_idx + list(self.joint_path.joint(joint_idx).jointAxis).index(1)] for joint_idx,offset_idx in enumerate(self.joint_index_offsets) ])

    def set_joint_range(self, joint_range_list=None):
        if joint_range_list is None: joint_range_list = [(-30,60),(-120,55),(-90,90), (0,0),(0,150),(0,0) ,(-60,60),(-120,120),(0,0)] # set full range to all joint
        self.joint_range_list = np.array([ joint_range_list[offset_idx + list(self.joint_path.joint(joint_idx).jointAxis).index(1)] for joint_idx,offset_idx in enumerate(self.joint_index_offsets) ])

    def set_moment_colors(self, moment_colors=None):
        self.moment_colors = [[1,0,0],[0,1,0],[0,0.5,1]] if moment_colors is None else moment_colors
        if not self.draw_interfaces is None:
            [interface.setColor(color)  for interface,color in zip(self.draw_interfaces,self.moment_colors)]

    def reset_max_min_wrench(self):
        self.max_load_wrench = np.zeros(self.load_dim)
        self.min_load_wrench = np.zeros(self.load_dim)

    def set_draw_interfaces(self):
        self.draw_interfaces = [UtilPlugin.DrawInterface(moment_color) for moment_color in self.moment_colors]

    def array_str(self, array):
        return str(np.where(array == np.inf, np.inf, array.astype(np.int)))

    def draw_moment(self, moment_type=MomentType.ACTUATORAXIS, coord_link_name=None, axis_length=0.5, radius_size=0.0004):
        self.hide_moment()
        self.message_view.flush()
        E = np.eye(3)
        R = self.robot.link(self.joint_path.joint(2).name if coord_link_name is None else coord_link_name).R # default coord is joint(2)
        for idx,di in enumerate(self.draw_interfaces):
            link = self.robot.link(self.joint_path.joint(idx).name) # tmp

            if moment_type == MomentType.LINKCOORD :
                axis = R[:,idx]

                radius = R[:,(idx+1)%3]
            elif moment_type == MomentType.ACTUATORAXIS:
                # axis = link.R.dot(link.jointAxis)
                axis = link.R.dot(link.a)

                A = E - link.a
                radius = (A[np.where(np.where(A>0,True,False).sum(axis=0))[0]][:]+link.a).sum(axis=0)
                radius = link.R.dot(radius) # send in world frame
            else:
                Logger.error(Fore.RED+'!!! Not supported MomentType: {0}!!!'.format(moment_type)+Style.RESET_ALL)
                return

            moment = np.minimum( self.instant_max_load_wrench[3:], self.saturation_vec[3:] ) # saturation
            tau = axis.dot(R.dot(moment))

            radius = radius_size*radius*tau
            axis = axis_length*axis
            pos = link.p
            di.drawLineArcArrow(pos, radius, axis ,360, 0.1, 60, 0.5)
            di.show()

    def hide_moment(self):
        for di in self.draw_interfaces:
            di.hide()

    def __convert_to_frame_load_wrench_vertices(self, A_, B_):
        num_joints = self.joint_path.numJoints

        # tau convex hull H->V
        A = np.vstack([np.identity(num_joints),-np.identity(num_joints)])
        b = np.vstack([self.max_tau_theta[:,np.newaxis],self.max_tau_theta[:,np.newaxis]]) # min_tau = - max_tau -> -min_tau = max_tau
        try:
            inmat, poly, retmat = h2v(A,b)
        except RuntimeError:
            logger.error(Fore.RED+'!!!!!RuntimeError (h2v())!!!!!'+Style.RESET_ALL)
            return np.array([range(self.load_dim)])

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
            return np.array([range(self.load_dim)])
        logger.debug("tau_tilde")
        logger.debug(retmat)
        C = -np.array(retmat)[:,1:]
        d = np.array(retmat)[:,0:1]
        logger.debug("")

        # H->V
        # for max value
        A = np.vstack([C.dot(A_), np.identity(self.load_dim), -np.identity(self.load_dim)])
        b = np.vstack([d, self.saturation_vec[:,np.newaxis], self.saturation_vec[:,np.newaxis]])
        try:
            inmat, poly, retmat = h2v(A,b)
        except RuntimeError:
            logger.error(Fore.RED+'!!!!!RuntimeError (h2v())!!!!!'+Style.RESET_ALL)
            return np.array([range(self.load_dim)])

        logger.debug("final")
        logger.debug(retmat)

        try:
            return np.array(retmat)[:,1:] # only hull (no cone)
        except IndexError:
            logger.error(Fore.RED+'!!!!!IndexError of retmat!!!!!'+Style.RESET_ALL) # retmat is empty
            return np.zeros((1,self.load_dim))

    # calc frame load wrench vertices at current pose
    def calc_current_load_wrench_vertices(self, target_link_name, root_link_name=None, end_link_name=None, coord_link_name=None): # set joint name not joint index
        # self.robot_item.calcForwardKinematics()
        self.robot.calcForwardKinematics()
        if self.robot_item is not None: self.robot_item.notifyKinematicStateChange()

        root_link = self.root_link if root_link_name is None else self.robot.link(root_link_name)
        end_link = self.end_link if end_link_name is None else self.robot.link(end_link_name)
        target_link = self.robot.link(target_link_name)
        coord_link = target_link if coord_link_name is None else self.robot.link(coord_link_name)

        # calc selection matrix
        target_link_idx = self.joint_path.indexOf(target_link)
        num_adjacent_actuator_set = self.actuator_set_list[target_link_idx]
        diag_vec = np.array([i < target_link_idx + num_adjacent_actuator_set[1] + 1 for i in range(self.joint_path.numJoints)]) * np.array([i > target_link_idx - num_adjacent_actuator_set[0] for i in range(self.joint_path.numJoints)]).astype(np.int)
        self.S = 0.99 * np.diag(diag_vec)

        Jre = np.ndarray((self.load_dim,self.joint_path.numJoints)); Body.JointPath.getCustomPath(self.robot, root_link, end_link).calcJacobian(Jre) # root->end
        Jri = np.ndarray((self.load_dim,target_link_idx+1)); Body.JointPath.getCustomPath(self.robot, root_link, target_link).calcJacobian(Jri) # root->i
        Jie = np.ndarray((self.load_dim,self.joint_path.numJoints - Jri.shape[1])); Body.JointPath.getCustomPath(self.robot, target_link, end_link).calcJacobian(Jie) # i->end

        R2i = np.r_[ np.c_[coord_link.R,np.zeros((3,3))],
                     np.c_[np.zeros((3,3)), coord_link.R] ]

        J6ei = np.r_[ np.c_[ np.eye(3),-skew(target_link.p-end_link.p) ], # [ [E,rx] , [0,E] ]
                      np.c_[ np.zeros((3,3)),np.eye(3) ] ]

        Ji_tilde = np.c_[ Jri, J6ei.dot(Jie) ] # [Jri J6ei*Jie]
        G = np.eye(self.joint_path.numJoints) # tmp E
        A_theta = np.diag([0,0,0, 1,1,1]).dot(Jre)
        # A_t = np.diag([0,0,0, 1,1,1]).dot(Jre)

        axis_mat = A_theta[3:]
        # set tau_j to min_i(tau_i/|a_j.a_i|)
        # self.max_tau_theta = np.nanmin(self.max_tau/abs(axis_mat.T.dot(axis_mat)), axis=1) # with all joints (excluding nan)
        self.max_tau_theta = np.nanmin(self.max_tau/abs(self.axis_product_mat*axis_mat.T.dot(axis_mat)), axis=1) # with only intersecting joints (excluding nan)
        logger.debug("max_tau="+str(self.max_tau))
        logger.debug("max_tau_theta=" + str(self.max_tau_theta))

        self.Jre = Jre
        self.Jri = Jri
        self.J6ei = J6ei
        self.Ji_tilde = Ji_tilde
        self.R2i = R2i
        self.A_theta = A_theta

        return self.__convert_to_frame_load_wrench_vertices( Ji_tilde.transpose().dot(R2i), G.transpose()-Ji_tilde.transpose().dot(A_theta).dot(G.transpose()).dot(self.S) ) # Ji~^T*R2, G^T-Ji~^T*Atheta*Si

    def calc_instant_max_frame_load_wrench(self, target_joint_name, coord_link_name=None, do_plot=True, save_plot=False, fname="", save_model=False, do_wait=False, tm=0.2):
        return self.calc_max_frame_load_wrench(target_joint_name, coord_link_name=coord_link_name, do_plot=do_plot, save_plot=save_plot, fname="", is_instant=True, do_wait=False, tm=0.2)

    def calc_max_frame_load_wrench(self, target_joint_name, coord_link_name=None, do_plot=True, save_plot=False, fname="", is_instant=True,
                                       save_model=False, show_model=False, moment_type=None, do_wait=False, tm=0.2):
        show_model |= save_model

        joint_angle_text = r'$\theta$: ' + str(np.rad2deg(self.robot.angleVector()).astype(np.int)) + ' [deg]' # round joint angles
        pi.joint_angle_text.set_text(joint_angle_text)
        logger.info(joint_angle_text)
        n_vertices = self.calc_current_load_wrench_vertices(target_joint_name, coord_link_name=coord_link_name)
        logger.debug("n_vertices=")
        logger.debug(n_vertices[:,3:])

        self.instant_max_load_wrench = n_vertices.max(axis=0).astype(np.float64)
        self.instant_min_load_wrench = n_vertices.min(axis=0).astype(np.float64)
        self.instant_max_load_wrench[np.ma.where(abs(self.instant_max_load_wrench) < 10)] = 0 # set |elements|<10 to 0
        self.instant_min_load_wrench[np.ma.where(abs(self.instant_min_load_wrench) < 10)] = 0
        self.instant_max_load_wrench[np.ma.where(abs(self.instant_max_load_wrench) >= self.saturation_vec)] = np.inf # set |elements|>saturation value to inf
        self.instant_min_load_wrench[np.ma.where(abs(self.instant_min_load_wrench) >= self.saturation_vec)] = -np.inf

        tmp_max_load_moment = self.max_load_wrench[3:].copy()
        self.max_load_wrench = np.vstack([self.instant_max_load_wrench, self.max_load_wrench]).max(axis=0)
        self.min_load_wrench = np.vstack([self.instant_min_load_wrench, self.min_load_wrench]).min(axis=0)
        if np.any(tmp_max_load_moment != self.max_load_wrench[3:]): logger.error(" max moment changed to "+str(self.max_load_wrench[3:])+" ("+joint_angle_text+")")

        ret_max_load_wrench,ret_min_load_wrench = [self.instant_max_load_wrench,self.instant_min_load_wrench] if is_instant else [self.max_load_wrench, self.min_load_wrench]

        pi.max_moment_text.set_text('${}_{3}n_{\mathrm{ld}}$: ' + self.array_str(ret_max_load_wrench[3:]) + " [Nm]")
        logger.info(" max: " + str(ret_max_load_wrench))
        logger.info(" min: " + str(ret_min_load_wrench))
        if do_plot or save_plot : pi.plot_convex_hull(n_vertices[:,3:], save_plot=save_plot, fname=fname, isInstant=is_instant)

        if show_model and self.world.is_choreonoid:
            self.robot_item.notifyKinematicStateChange()
            self.tree_view.checkItem(self.robot_item, True)
            if moment_type: self.draw_moment(moment_type=moment_type, coord_link_name=coord_link_name)
            self.message_view.flush()
            head_fname = re.sub('[_0-9]*$',"",fname.replace(".png",""))
            if save_model: self.scene_widget.saveImage(str(fname.replace(head_fname,head_fname+"_pose")))

        if do_wait:
            logger.critical("RET to continue, q to escape"+Style.RESET_ALL)
            key = raw_input()
            if key == 'q': escape = True
        else:
            if do_plot: time.sleep(tm)

        return [ret_max_load_wrench, ret_min_load_wrench]

    def calc_whole_range_max_load_wrench(self, target_joint_name, coord_link_name=None, joint_idx=0, is_instant=True,
                                             do_plot=True, save_plot=False, save_model=False, show_model=False, fname="", do_wait=False, tm=0.2):
        if joint_idx == 0:
            self.reset_max_min_wrench()

        joint_name = self.joint_path.joint(joint_idx).name
        joint_range = self.joint_range_list[joint_idx]
        step_angle = self.step_angle_list[joint_idx]

        if joint_idx < self.joint_path.numJoints and logger.isEnabledFor(INFO): sys.stdout.write(Fore.GREEN+" "+"#"+joint_name+Style.RESET_ALL)
        if joint_idx+1 == self.joint_path.numJoints and logger.isEnabledFor(INFO): print(" changed")
        fname=fname.replace(".png","_0.png") # set dummy
        for division_idx,joint_angle in enumerate(np.arange(joint_range[0],joint_range[1]+1,step_angle)):
            fname=re.sub('_[0-9]*\.png',"_"+str(division_idx).zfill(int(1+int((joint_range[1]-joint_range[0])/step_angle)/10))+".png",fname)
            self.robot.link(joint_name).q = np.deg2rad(joint_angle) # set joint angle [rad]
            if joint_idx+1 < self.joint_path.numJoints:
                self.calc_whole_range_max_load_wrench(target_joint_name,coord_link_name,joint_idx+1,is_instant=is_instant,
                                                          do_plot=do_plot,save_plot=save_plot,fname=fname,save_model=save_model,show_model=show_model,
                                                          do_wait=do_wait,tm=tm)
            else:
                self.calc_max_frame_load_wrench(target_joint_name,coord_link_name,is_instant=is_instant,
                                                    do_plot=do_plot,save_plot=save_plot,fname=fname,save_model=save_model,show_model=show_model,
                                                    do_wait=do_wait,tm=tm)

        if joint_idx == 0:
            logger.critical(Fore.YELLOW
                                +re.search(r'([x-z][-_][x-z_-]+)', self.robot_model_file).group()
                                +" max wrench around "+target_joint_name+(" (in "+coord_link_name+")" if coord_link_name is not None else "")+": "
                                +self.array_str(self.max_load_wrench[3:])
                                +Style.RESET_ALL)
            if self.world.is_choreonoid:
                self.tree_view.checkItem(self.robot_item, False)
                self.message_view.flush()

def init_config():
    global package_path
    package_path = roslib.packages.get_pkg_dir("structure_analyzer")
    global model_path
    model_path = os.path.join(package_path, "models")

    # valid only when executing from python console
    os.system('cp {} {}'.format(os.path.join(package_path, 'config/Choreonoid.conf'), os.path.join(os.environ.get('HOME'),'.config/Choreonoid\ Inc.')))

def initialize_plot_interface():
    global pi
    pi = PlotInterface()

    np.set_printoptions(precision=5)

    # plt.rcParams["font.size"] = 25
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams.update({"pgf.preamble":["\\usepackage{bm}"]})

    pi.ax.view_init(30,-25) # rotate view

    pi.set_max_display_num(800)

    pi.ax.set_xlabel("${}_{3}n_{x}$")
    pi.ax.set_ylabel("${}_{3}n_{y}$")
    pi.ax.set_zlabel("  ${}_{3}n_{z}$")


def test_calculate_frame_load():
    logger.critical(Fore.BLUE+"test_calcuate_frame_load()"+Style.RESET_ALL)
    global analyzer

    # division_num=1; do_wait=False; tm=0;   do_plot=True
    # division_num=4; do_wait=False; tm=0;   do_plot=True
    # division_num=4; do_wait=False; tm=0;   do_plot=False
    # division_num=6; do_wait=False; tm=0; do_plot=False
    do_wait=False; tm=0;   do_plot=False
    package_path = roslib.packages.get_pkg_dir("structure_analyzer")
    model_path = os.path.join(package_path, "models")

    # step_angle_list = [20,20,20,20,20,20]
    step_angle_list = [20,20,20,20,360,360]
    # step_angle_list = [10,10,10,360,360,360]
    # joint_range_list = [(-30,60),(-120,55),(-90,90), (0,0),(0,150),(0,0) ,(-60,60),(-80,75),(0,0)] # set full range to all joint
    joint_range_list = [(0,60),(0,120),(0,90), (0,0),(0,90),(0,0) ,(0,60),(0,80),(0,0)] # set actual range to all joint
    # joint_range_list = [(0,60),(0,120),(0,90), (0,0),(0,90),(0,0) ,(0,0),(0,0),(0,0)] # hip only/half range
    # joint_range_list = [(0,60),(0,120),(0,90), (0,0),(90,90),(0,0) ,(0,0),(0,0),(0,0)] # hip only/half range
    # joint_range_list = [(60,60),(55,55),(-90,-90), (0,0),(90,90),(0,0) ,(0,0),(0,0),(0,0)] # hip only/fix

    constructor_args = {
        'joint_range_list': joint_range_list,
        'step_angle_list': step_angle_list,
        'end_link_name': 'JOINT5',
        }
    calc_args = {
        'do_wait': do_wait,
        'tm': tm,
        'do_plot': do_plot,
        }

    # # JAXON_RED
    # analyzer = JointLoadWrenchAnalyzer([0,0,(0,0),0,0,0], robot_item=None, step_angle_list=step_angle_list)
    # analyzer.calc_whole_range_max_load_wrench('LLEG_JOINT2', **calc_args)
    # logger.critical(Fore.YELLOW+analyzer.robot.name()+" max wrench: "+str(analyzer.max_load_wrench)+Style.RESET_ALL)

    # Serial rotary
    logger.critical(Fore.BLUE+"Serial rotary drive joint"+Style.RESET_ALL)
    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], robot_model_file=os.path.join(model_path,"universal-joint-robot_x-y-z_y_y-x.wrl"), **constructor_args)
    analyzer.calc_whole_range_max_load_wrench('JOINT2', **calc_args)

    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], robot_model_file=os.path.join(model_path,"universal-joint-robot_x-y-z_y_x-y.wrl"), **constructor_args)
    analyzer.calc_whole_range_max_load_wrench('JOINT2', **calc_args)
    logger.critical("")


    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], robot_model_file=os.path.join(model_path,"universal-joint-robot_y-x-z_y_y-x.wrl"), **constructor_args)
    analyzer.calc_whole_range_max_load_wrench('JOINT2', **calc_args)

    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], robot_model_file=os.path.join(model_path,"universal-joint-robot_y-x-z_y_x-y.wrl"), **constructor_args)
    analyzer.calc_whole_range_max_load_wrench('JOINT2', **calc_args)
    logger.critical("")


    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], robot_model_file=os.path.join(model_path,"universal-joint-robot_z-x-y_y_y-x.wrl"), **constructor_args)
    analyzer.calc_whole_range_max_load_wrench('JOINT2', **calc_args)

    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], robot_model_file=os.path.join(model_path,"universal-joint-robot_z-x-y_y_x-y.wrl"), **constructor_args)
    analyzer.calc_whole_range_max_load_wrench('JOINT2', **calc_args)
    logger.critical("")


    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], robot_model_file=os.path.join(model_path,"universal-joint-robot_z-y-x_y_y-x.wrl"), **constructor_args)
    analyzer.calc_whole_range_max_load_wrench('JOINT2', **calc_args)

    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], robot_model_file=os.path.join(model_path,"universal-joint-robot_z-y-x_y_x-y.wrl"), **constructor_args)
    analyzer.calc_whole_range_max_load_wrench('JOINT2', **calc_args)
    logger.critical("")


    # Serial linear
    logger.critical(Fore.BLUE+"Serial linear drive joint"+Style.RESET_ALL)
    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(1,0),(1,0),(0,0),(0,0)], robot_model_file=os.path.join(model_path,"universal-joint-robot_z-x-y_y_y-x.wrl"), **constructor_args)
    analyzer.calc_whole_range_max_load_wrench('JOINT2', **calc_args)

    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(1,0),(1,0),(0,0),(0,0)], robot_model_file=os.path.join(model_path,"universal-joint-robot_z-y-x_y_y-x.wrl"), **constructor_args)
    analyzer.calc_whole_range_max_load_wrench('JOINT2', **calc_args)

    # Parallel linear
    logger.critical(Fore.BLUE+"Parallel linear drive joint"+Style.RESET_ALL)
    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(2,0),(1,0),(0,0),(0,0)], robot_model_file=os.path.join(model_path,"universal-joint-robot_z-x-y_y_y-x.wrl"), **constructor_args)
    analyzer.calc_whole_range_max_load_wrench('JOINT2', **calc_args)

    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(2,0),(1,0),(0,0),(0,0)], robot_model_file=os.path.join(model_path,"universal-joint-robot_z-y-x_y_y-x.wrl"), **constructor_args)
    analyzer.calc_whole_range_max_load_wrench('JOINT2', **calc_args)

def test_7dof_calculate_frame_load(target_link_name="JOINT2", coord_link_name=None, do_wait=False, do_plot=False, tm=0):
    logger.critical(Fore.BLUE+"test_7dof_calcuate_frame_load()"+Style.RESET_ALL)
    global analyzer

    package_path = roslib.packages.get_pkg_dir("structure_analyzer")
    model_path = os.path.join(package_path, "models")

    global constructor_args
    constructor_args = {}

    constructor_args["joint_group_list"] = [3,1,3]

    constructor_args["joint_range_list"] = [(0, 90),(0,90), (0,90), (0,0),(0,90),(0,0) ,(0,80),(0,80),(0,90)] # full range

    constructor_args["max_tau_list"] = [200,190,100, 0,170,0, 45,45,100]

    # constructor_args["step_angle_list"] = [360,10,10, 10, 360,360,360]  # simplified test (The first joint has no effect)
    # constructor_args["step_angle_list"] = [360,10,10, 360, 360,360,360] # shoulder full range
    # constructor_args["step_angle_list"] = [360,10,10, 10, 360,360,360]  # excluding wrist
    # constructor_args["step_angle_list"] = [360,10,10, 10, 30,30,30]     # little fast full range
    # constructor_args["step_angle_list"] = [360,10,10, 10, 20,20,20]

    constructor_args["step_angle_list"] = [360,10,10, 10, 10,10,360]    # full range in effect

    # constructor_args["step_angle_list"] = [30,30,30, 30, 360,360,360]  # fast semi-full range
    # constructor_args["step_angle_list"] = [30,30,30, 30, 30,30,30]     # fast full range

    # constructor_args["saturation_vec"] = np.array([10000,10000,10000, 500,500,500])
    constructor_args["saturation_vec"] = np.array([10000,10000,10000, 10000,10000,10000])

    constructor_args["end_link_name"] ="JOINT6"

    global calc_args
    calc_args = {
        "do_wait": do_wait,
        "tm": tm,
        "do_plot": do_plot,
        }

    # Serial rotary
    logger.critical(Fore.BLUE+"Serial rotary drive joint"+Style.RESET_ALL)
    joint_conf_array = [
        ["x-y-z_y_x-y-z","x-y-z_y_y-x-z", "x-y-z_y_z-x-y","x-y-z_y_z-y-x"],
        ["y-x-z_y_x-y-z","y-x-z_y_y-x-z", "y-x-z_y_z-x-y","y-x-z_y_z-y-x"],
        ["z-x-y_y_x-y-z","z-x-y_y_y-x-z", "z-x-y_y_z-x-y","z-x-y_y_z-y-x"],
        ["z-y-x_y_x-y-z","z-y-x_y_y-x-z", "z-y-x_y_z-x-y","z-y-x_y_z-y-x"],
        ]
    for joint_conf_row in joint_conf_array:
        for joint_conf in joint_conf_row:
            analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],
                                                   robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_conf+".wrl"),
                                                   **constructor_args)
            analyzer.calc_whole_range_max_load_wrench(target_link_name, coord_link_name, **calc_args)
        logger.critical("")

    # Serial linear
    logger.critical(Fore.BLUE+"Serial linear drive joint"+Style.RESET_ALL)
    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(1,0),(1,0),(0,0),(0,0),(0,0)], robot_model_file=os.path.join(model_path,"universal-joint-robot_z-x-y_y_z-y-x.wrl"), **constructor_args)
    analyzer.calc_whole_range_max_load_wrench(target_link_name, coord_link_name, **calc_args)

    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(1,0),(1,0),(0,0),(0,0),(0,0)], robot_model_file=os.path.join(model_path,"universal-joint-robot_z-y-x_y_z-y-x.wrl"), **constructor_args)
    analyzer.calc_whole_range_max_load_wrench(target_link_name, coord_link_name, **calc_args)

    # Parallel linear
    logger.critical(Fore.BLUE+"Parallel linear drive joint"+Style.RESET_ALL)
    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(2,0),(1,0),(0,0),(0,0),(0,0)], robot_model_file=os.path.join(model_path,"universal-joint-robot_z-x-y_y_z-y-x.wrl"), **constructor_args)
    analyzer.calc_whole_range_max_load_wrench(target_link_name, coord_link_name, **calc_args)

    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(2,0),(1,0),(0,0),(0,0),(0,0)], robot_model_file=os.path.join(model_path,"universal-joint-robot_z-y-x_y_z-y-x.wrl"), **constructor_args)
    analyzer.calc_whole_range_max_load_wrench(target_link_name, coord_link_name, **calc_args)

def export_overall_frame_load_region():
    logger.critical(Fore.BLUE+"export_overall_frame_load_region()"+Style.RESET_ALL)

    package_path = roslib.packages.get_pkg_dir("structure_analyzer")
    model_path = os.path.join(package_path, "models")

    # joint_range_list = [(-30,60),(-120,55),(-90,90), (0,0),(0,150),(0,0) ,(-60,60),(-80,75),(0,0)] # set full range to all joint
    joint_range_list = [(0,60),(0,80),(0,0), (0,0),(0,90),(0,0) ,(0,0),(0,0),(0,0)]
    # joint_range_list = [(0,60),(0,120),(0,90), (0,0),(0,0),(0,0) ,(0,0),(0,0),(0,0)] # hip only/half range
    # joint_range_list = [(60,60),(55,55),(-90,-90), (0,0),(90,90),(0,0) ,(0,0),(0,0),(0,0)] # hip only/fix

    step_angle = 10

    global analyzer
    joint_configuration_str = "z-y-x_y_y-x"
    common_fname = os.path.join(package_path,"overall-frame-load-region","frame-load-region")
    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], joint_range_list=joint_range_list, step_angle=step_angle,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_configuration_str+".wrl"))
    analyzer.calc_whole_range_max_load_wrench('JOINT2', do_wait=False, tm=0, save_plot=True, fname=common_fname+"_overall.png", is_instant=False, save_model=False)
    analyzer.calc_whole_range_max_load_wrench('JOINT2', do_wait=False, tm=0, save_plot=True, fname=common_fname+"_instant.png", is_instant=True, save_model=True)
    logger.critical(Fore.YELLOW+joint_configuration_str+" max wrench: "+str(analyzer.max_load_wrench)+Style.RESET_ALL)

def export_joint_configuration_comparison():
    logger.critical(Fore.BLUE+"export_joint_configuration_comparison()"+Style.RESET_ALL)

    package_path = roslib.packages.get_pkg_dir("structure_analyzer")
    model_path = os.path.join(package_path, "models")

    joint_range_list = [(35,35),(100,100),(20,20), (0,0),(0,0),(0,0) ,(0,0),(0,0),(0,0)]

    global analyzer0
    joint_configuration_str0="z-x-y_y_y-x"
    analyzer0 = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], joint_range_list=joint_range_list,
                                        end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_configuration_str0+".wrl"))
    global analyzer1
    joint_configuration_str1="z-y-x_y_y-x"
    analyzer1 = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], joint_range_list=joint_range_list,
                                        end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_configuration_str1+".wrl"))
    target_end = analyzer0.joint_path.endLink

    common_fname=os.path.join(package_path,"joint-configuration-comparison","joint-configuration-comparison")

    if analyzer0.world.is_choreonoid:
        tree_view = Base.ItemTreeView.instance
        message_view = Base.MessageView.instance
        scene_widget = Base.SceneView.instance.sceneWidget

    analyzer0.robot.angleVector(np.deg2rad(np.array([0,0,0,0,0,0])))
    analyzer0.calc_max_frame_load_wrench('JOINT2', do_wait=False, tm=0, do_plot=True, save_plot=True, fname=common_fname+"_initial-pose_load-region.png")
    logger.critical(Fore.YELLOW+joint_configuration_str0+" max wrench: "+str(analyzer0.max_load_wrench)+Style.RESET_ALL)
    analyzer1.robot.angleVector(np.deg2rad(np.array([0,0,0,0,0,0])))
    if analyzer0.world.is_choreonoid:
            analyzer0.robot_item.notifyKinematicStateChange()
            tree_view.checkItem(analyzer0.robot_item, True)
            tree_view.checkItem(analyzer1.robot_item, False)
            message_view.flush()
            scene_widget.saveImage(str(common_fname+"_configuration0"+"_initial-pose.png"))
            tree_view.checkItem(analyzer0.robot_item, False)
            tree_view.checkItem(analyzer1.robot_item, True)
            message_view.flush()
            scene_widget.saveImage(str(common_fname+"_configuration1"+"_initial-pose.png"))

    l_angle = np.array([0,-70,0,20,0,0])
    u_angle = np.array([90,5,-80,120,0,0])
    analyzer1.robot.angleVector(np.deg2rad(l_angle[[0,2,1,3,4,5]]))
    analyzer1.robot.calcForwardKinematics()
    division_num = 20
    angle_vectors = np.vstack([np.linspace(l_angle,u_angle,division_num,endpoint=True), np.linspace(u_angle,l_angle,division_num,endpoint=True)])
    for idx, angle_vector in enumerate(angle_vectors):
        # index_str = "_"+'_'.join(angle_vector.astype(np.int).astype(np.str))
        index_str = "_"+str(idx).zfill(2)

        analyzer0.robot.angleVector(np.deg2rad(angle_vector))
        # FK is called in calc_max_frame_load_wrench
        analyzer0.calc_max_frame_load_wrench('JOINT2', do_wait=False, tm=0, do_plot=True, save_plot=True, fname=common_fname+"_configuration0"+"_load-region"+index_str+".png")
        # logger.info(Fore.YELLOW+joint_configuration_str0+" max wrench: "+str(analyzer0.max_load_wrench)+Style.RESET_ALL)

        H = np.vstack([ np.hstack([target_end.R, np.array([target_end.p]).T]), np.array([[0,0,0,1]]) ])
        logger.info("IK: " + str(analyzer1.joint_path.calcInverseKinematics(H)))
        analyzer1.calc_max_frame_load_wrench('JOINT2', do_wait=False, tm=0, do_plot=True, save_plot=True, fname=common_fname+"_configuration1"+"_load-region"+index_str+".png")
        # logger.info(Fore.YELLOW+joint_configuration_str1+" max wrench: "+str(analyzer1.max_load_wrench)+Style.RESET_ALL)

        if analyzer0.world.is_choreonoid:
            analyzer0.robot_item.notifyKinematicStateChange()
            tree_view.checkItem(analyzer0.robot_item, True)
            tree_view.checkItem(analyzer1.robot_item, False)
            analyzer0.draw_moment()
            analyzer1.hide_moment()
            message_view.flush()

            scene_widget.saveImage(str(common_fname+"_configuration0"+"_pose"+index_str+".png"))
            tree_view.checkItem(analyzer0.robot_item, False)
            tree_view.checkItem(analyzer1.robot_item, True)
            analyzer0.hide_moment()
            analyzer1.draw_moment()
            message_view.flush()
            scene_widget.saveImage(str(common_fname+"_configuration1"+"_pose"+index_str+".png"))

    if analyzer0.world.is_choreonoid:
        tree_view.checkItem(analyzer0.robot_item, False)
        tree_view.checkItem(analyzer1.robot_item, False)
        analyzer0.hide_moment()
        analyzer1.hide_moment()
        message_view.flush()

def export_drive_system_comparison():
    logger.critical(Fore.BLUE+"export_drive_system_comparison()"+Style.RESET_ALL)

    package_path = roslib.packages.get_pkg_dir("structure_analyzer")
    model_path = os.path.join(package_path, "models")

    pi.set_max_display_num(400)

    joint_range_list = [(35,35),(100,100),(20,20), (0,0),(0,0),(0,0) ,(0,0),(0,0),(0,0)]

    global constructor_args
    constructor_args = {}
    constructor_args['joint_range_list'] = [(35,35),(100,100),(20,20), (0,0),(0,0),(0,0) ,(0,0),(0,0),(0,0)]
    constructor_args['end_link_name'] = 'JOINT5'
    constructor_args['robot_model_file'] = os.path.join(model_path,"universal-joint-robot_z-x-y_y_y-x.wrl")

    global calc_args
    calc_args = {'do_wait':False, 'tm':0, 'do_plot':True, 'save_plot':True}

    global analyzer0
    analyzer0 = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], **constructor_args)
    global analyzer1
    analyzer1 = JointLoadWrenchAnalyzer([(0,0),(0,0),(1,0),(0,0),(0,0),(0,0)], **constructor_args)
    global analyzer2
    analyzer2 = JointLoadWrenchAnalyzer([(0,0),(0,0),(2,0),(0,0),(0,0),(0,0)], **constructor_args)

    if analyzer0.world.is_choreonoid:
        tree_view = Base.ItemTreeView.instance
        message_view = Base.MessageView.instance
        scene_widget = Base.SceneView.instance.sceneWidget
        tree_view.checkItem(analyzer0.robot_item, True)

    l_angle = np.array([30,-70,-40,50,0,0])
    u_angle = np.array([120,0,80,-90,0,0])
    common_fname=os.path.join(package_path,"drive-system-comparison","drive-system-comparison")
    division_num = 20
    angle_vectors = np.vstack([np.linspace(l_angle,u_angle,division_num,endpoint=True), np.linspace(u_angle,l_angle,division_num,endpoint=True)])
    for idx, angle_vector in enumerate(angle_vectors):
        # index_str = "_"+'_'.join(angle_vector.astype(np.int).astype(np.str))
        index_str = "_"+str(idx).zfill(2)

        analyzer0.robot.angleVector(np.deg2rad(angle_vector))
        analyzer0.calc_max_frame_load_wrench('JOINT2', fname=common_fname+"_system0"+"_load-region"+index_str+".png", **calc_args)

        analyzer1.robot.angleVector(np.deg2rad(angle_vector))
        analyzer1.calc_max_frame_load_wrench('JOINT2', fname=common_fname+"_system1"+"_load-region"+index_str+".png", **calc_args)

        analyzer2.robot.angleVector(np.deg2rad(angle_vector))
        analyzer2.calc_max_frame_load_wrench('JOINT2', fname=common_fname+"_system2"+"_load-region"+index_str+".png", **calc_args)

        if analyzer0.world.is_choreonoid:
            analyzer0.robot_item.notifyKinematicStateChange()
            analyzer0.draw_moment()
            analyzer1.hide_moment()
            analyzer2.hide_moment()
            message_view.flush()
            scene_widget.saveImage(str(common_fname+"_system0"+"_pose"+index_str+".png"))

            analyzer0.hide_moment()
            analyzer1.draw_moment()
            message_view.flush()
            scene_widget.saveImage(str(common_fname+"_system1"+"_pose"+index_str+".png"))

            analyzer1.hide_moment()
            analyzer2.draw_moment()
            message_view.flush()
            scene_widget.saveImage(str(common_fname+"_system2"+"_pose"+index_str+".png"))

    if analyzer0.world.is_choreonoid:
        tree_view.checkItem(analyzer0.robot_item, False)
        analyzer0.hide_moment()
        analyzer1.hide_moment()
        analyzer2.hide_moment()
        message_view.flush()

def export_arm_comparison(target_link_name='JOINT2', coord_link_name=None, do_wait=False, tm=0.1, do_plot=False, show_model=True):
    logger.critical(Fore.BLUE+"export_arm_comparison()"+Style.RESET_ALL)

    package_path = roslib.packages.get_pkg_dir("structure_analyzer")
    model_path = os.path.join(package_path, "models")

    # pi.set_max_display_num(400)
    pi.set_max_display_num(600)
    # pi.set_max_display_num(8000)

    global constructor_args
    constructor_args = {}

    constructor_args['joint_group_list'] = [3,1,3]

    # constructor_args['joint_range_list'] = [(0,180),(-180,120),(-90,90), (0,0),(0,90),(0,0) ,(0,80),(0,80),(0,0)] # actual range
    constructor_args['joint_range_list'] = [(0, 90),(0,90), (0,90), (0,0),(0,90),(0,0) ,(0,80),(0,80),(0,80)]

    # constructor_args['max_tau_list'] = [200,190,100, 0,170,0, 45,45,45] # almost same with 45,45,100
    constructor_args['max_tau_list'] = [200,190,100, 0,170,0, 45,45,100]

    # constructor_args['step_angle_list'] = [360,10,10, 360, 360,360,360]
    constructor_args['step_angle_list'] = [360,10,10, 10, 360,360,360] # simplified test (The first joint has no effect)
    # constructor_args['step_angle_list'] = [360,10,10, 10, 10,10,360] # full range in effect (same with fast)
    # constructor_args['step_angle_list'] = [360,10,10, 10, 20,20,360] # fast full range in effect

    # constructor_args['saturation_vec'] = np.array([10000,10000,10000, 500,500,500])
    constructor_args['saturation_vec'] = np.array([10000,10000,10000, 1000,1000,1000])
    # constructor_args['saturation_vec'] = np.array([10000,10000,10000, 10000,10000,10000])

    constructor_args['end_link_name'] ='JOINT6'

    global calc_args
    calc_args = {
        'do_wait': do_wait,
        'tm': tm,
        'do_plot': do_plot,
        'show_model': show_model,
        'moment_type': MomentType.LINKCOORD,
        'save_plot': True,
        }

    global analyzer0
    global analyzer1
    global analyzer2
    global analyzer3

    analyzer0 = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], robot_model_file=os.path.join(model_path,"universal-joint-robot_x-y-z_y_z-y-x.wrl"), **constructor_args)
    analyzer1 = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], robot_model_file=os.path.join(model_path,"universal-joint-robot_y-x-z_y_z-y-x.wrl"), **constructor_args)
    analyzer2 = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], robot_model_file=os.path.join(model_path,"universal-joint-robot_z-x-y_y_z-y-x.wrl"), **constructor_args)
    analyzer3 = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], robot_model_file=os.path.join(model_path,"universal-joint-robot_z-y-x_y_z-y-x.wrl"), **constructor_args)

    global analyzer4
    # analyzer4 = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], robot_model_file=os.path.join(model_path,"universal-joint-robot_y-x_z-y_z-x-y.wrl"), **constructor_args)
    analyzer4 = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], robot_model_file=os.path.join(model_path,"universal-joint-robot_y-x_z-y_z-y-x.wrl"), **constructor_args)
    # analyzer4 = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], robot_model_file=os.path.join(model_path,"universal-joint-robot_x-y_z-y_z-x-y.wrl"), **constructor_args)
    # analyzer4 = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], robot_model_file=os.path.join(model_path,"universal-joint-robot_x-y_z-y_z-y-x.wrl"), **constructor_args)
    # analyzer4.calc_whole_range_max_load_wrench('JOINT1', 'JOINT1', **calc_args)
    # analyzer4.calc_whole_range_max_load_wrench('JOINT2', 'JOINT1', **calc_args)

    # save image
    common_fname=os.path.join(package_path,"shoulder-configuration-comparison","shoulder-configuration-comparison")
    if analyzer1.world.is_choreonoid:
        tree_view = Base.ItemTreeView.instance
        message_view = Base.MessageView.instance
        scene_widget = Base.SceneView.instance.sceneWidget

        for analyzer in [analyzer0,analyzer1,analyzer2,analyzer3,analyzer4]: tree_view.checkItem(analyzer.robot_item, False) # hide all robots

        # angle_vector_list = [np.deg2rad([0,90, 0, 10, 0,0,0]), np.deg2rad([0,90,90, 10, 0,0,0])]
        angle_vector_list = [np.deg2rad([0,90, 0, 10, 0,0,0]), np.deg2rad([0,90,60, 10, 0,0,0])]
        for idx,angle_vector in enumerate(angle_vector_list):
            analyzer1.robot.angleVector(angle_vector)
            analyzer1.calc_max_frame_load_wrench('JOINT2','JOINT2', fname=common_fname+"_shoulder-yaw_load-region_{}.png".format(idx), **calc_args)
            # analyzer1.calc_max_frame_load_wrench('JOINT3','JOINT2')
            tree_view.checkItem(analyzer1.robot_item, True)
            tree_view.checkItem(analyzer4.robot_item, False)
            analyzer4.hide_moment()
            message_view.flush()
            scene_widget.saveImage(str(common_fname+"_shoulder-yaw"+"_pose_{}.png".format(idx)))


            analyzer4.robot.angleVector(angle_vector)
            analyzer4.calc_max_frame_load_wrench('JOINT1','JOINT1', fname=common_fname+"_elbow-yaw_load-region_{}.png".format(idx), **calc_args)
            # analyzer4.calc_max_frame_load_wrench('JOINT2','JOINT1')
            tree_view.checkItem(analyzer1.robot_item, False)
            tree_view.checkItem(analyzer4.robot_item, True)
            analyzer1.hide_moment()
            message_view.flush()
            scene_widget.saveImage(str(common_fname+"_elbow-yaw"+"_pose_{}.png".format(idx)))

    analyzer_list = []
    # analyzer_list = [analyzer1]
    # analyzer_list = [analyzer1, analyzer2, analyzer3]
    # analyzer_list = [analyzer0, analyzer1, analyzer2, analyzer3]
    for analyzer in analyzer_list:
        # normal
        logger.critical("normal")
        analyzer.calc_whole_range_max_load_wrench('JOINT2', 'JOINT2', **calc_args) # around proximal
        analyzer.calc_whole_range_max_load_wrench('JOINT3', 'JOINT2', **calc_args) # around distal

        # # yaw joint shift
        # logger.critical("yaw joint shift")
        # analyzer.calc_whole_range_max_load_wrench('JOINT2', 'JOINT1', **calc_args) # around proximal
        # analyzer.calc_whole_range_max_load_wrench('JOINT3', 'JOINT1', **calc_args) # around distal

        logger.critical("")

if __name__ == '__main__':
    init_config()

    initialize_plot_interface()

    # test
    test_calculate_frame_load()

    # make figures
    export_joint_configuration_comparison()

    export_drive_system_comparison()

    export_arm_comparison()

    # export_overall_frame_load_region()

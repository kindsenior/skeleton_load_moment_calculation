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

import pprint, time, sys, os, re
from colorama import Fore, Back, Style
import pdb

import roslib

import cnoid.Body as Body
import cnoid.Base as Base
import cnoid.BodyUtil as BodyUtil # tmp

from logger import *
import jsk_choreonoid.util as jcu
import jsk_choreonoid.body_util

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
        self.fig = plt.figure()
        self.fig.set_size_inches((8.0,8.0))
        self.ax = self.fig.gca(projection='3d')
        self.fig.subplots_adjust(left=-0.05,right=0.95, bottom=0.02,top=1, wspace=0.1, hspace=1)
        self.fontsize = 35

        font_row_height = self.fontsize/2500.0
        self._text_pos = [-0.08,0.075]
        self.joint_angle_text = self.ax.text2D(self._text_pos[0], self._text_pos[1]-font_row_height*0,"")
        self.joint_angle_text.set_fontsize(self.fontsize)
        self.max_moment_text = self.ax.text2D(self._text_pos[0], self._text_pos[1]-font_row_height*1, "")
        self.max_moment_text.set_fontsize(self.fontsize)

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

def convert_to_frame_load_wrench_vertices(A_, B_):
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

def skew(vec):
    return np.array([[0,-vec[2],vec[1]],
                     [vec[2],0,-vec[0]],
                     [-vec[1],vec[0],0]])

class JointLoadWrenchAnalyzer(object):
    def __init__(self, actuator_set_list_, joint_range_list=None, robot_item=None, robot_model_file=None, moment_colors=None, end_link_name="LLEG_JOINT5", step_angle_list=None, step_angle=10):
        self.world = jcu.World()
        logger.info(" is_choreonoid:" + str(self.world.is_choreonoid))

        self.actuator_set_list = actuator_set_list_
        self.set_robot(robot_item, robot_model_file)
        self.set_joint_path(end_link_name=end_link_name)
        self.draw_interfaces = None
        self.set_moment_colors(moment_colors=moment_colors)

        self.axis_product_mat = reduce(lambda ret,vec: ret+np.array(vec).reshape(6,1)*np.array(vec), [np.zeros((6,6)),[1,1,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,1]])

        self.joint_index_offsets = [0,0,0,3,6,6]
        # set max_tau in order from root using jointAxis()
        max_tau_list = np.array([300,700,120, 0,700,0, 100,200,0]) # (hip-x,hip-y,hip-z,  nil,knee-y,nil, ankle-r,ankle-p)
        # max_tau_list = np.array([300,700,120, 0,700,0, 300,700,0]) # easy
        self.max_tau = np.array([ max_tau_list[offset_idx + list(self.joint_path.joint(joint_idx).jointAxis()).index(1)] for joint_idx,offset_idx in enumerate(self.joint_index_offsets) ])

        self.set_joint_range(joint_range_list)

        if step_angle_list is None: step_angle_list = [step_angle]*self.joint_path.numJoints()
        self.step_angle_list = step_angle_list

        self.reset_max_min_wrench()

        if self.world.is_choreonoid:
            self.tree_view = Base.ItemTreeView.instance()
            self.message_view = Base.MessageView.instance()
            self.scene_widget = Base.SceneView.instance().sceneWidget()
            self.set_draw_interfaces()

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

    def set_joint_path(self, root_link_name=None,end_link_name="LLEG_JOINT5"):
        self.root_link = self.robot.rootLink() if root_link_name is None else self.robot.link(root_link_name)
        self.end_link = self.robot.link(end_link_name)
        logger.info("end link: "+str(end_link_name))
        self.joint_path = Body.JointPath(self.root_link, self.end_link)

    def set_joint_range(self, joint_range_list=None):
        if joint_range_list is None: joint_range_list = [(-30,60),(-120,55),(-90,90), (0,0),(0,150),(0,0) ,(-60,60),(-120,120),(0,0)] # set full range to all joint
        self.joint_range_list = np.array([ joint_range_list[offset_idx + list(self.joint_path.joint(joint_idx).jointAxis()).index(1)] for joint_idx,offset_idx in enumerate(self.joint_index_offsets) ])

    def set_moment_colors(self, moment_colors=None):
        self.moment_colors = [[1,0,0],[0,1,0],[0,0.5,1]] if moment_colors is None else moment_colors
        if not self.draw_interfaces is None:
            [interface.setColor(color)  for interface,color in zip(self.draw_interfaces,self.moment_colors)]

    def reset_max_min_wrench(self):
        self.max_load_wrench = np.zeros(6)
        self.min_load_wrench = np.zeros(6)

    def set_draw_interfaces(self):
        self.draw_interfaces = [BodyUtil.DrawInterface(moment_color) for moment_color in self.moment_colors]

    def draw_moment(self):
        E = np.eye(3)
        R = self.robot.link(self.joint_path.joint(2).name()).R
        for idx,di in enumerate(self.draw_interfaces):
            link = self.robot.link(self.joint_path.joint(idx).name())
            # axis = link.R.dot(link.jointAxis())
            axis = link.R.dot(link.a)
            moment = R.dot(self.instant_max_load_wrench[3:])
            tau = axis.dot(moment)
            A = E - link.a
            radius = (A[np.where(np.where(A>0,True,False).sum(axis=0))[0]][:]+link.a).sum(axis=0)
            radius = link.R.dot(radius) # send in world frame
            radius = 0.0004*radius*tau
            axis = 0.5*axis
            pos = link.p + np.array([0,0,0.1]) if idx is 0 else link.p # tmp
            di.drawLineArcArrow(pos, radius, axis ,360, 0.1,60)
            di.show()

    def hide_moment(self):
        for di in self.draw_interfaces:
            di.hide()

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
        diag_vec = np.array([i < target_link_idx + num_adjacent_actuator_set[1] + 1 for i in range(self.joint_path.numJoints())]) * np.array([i > target_link_idx - num_adjacent_actuator_set[0] for i in range(self.joint_path.numJoints())]).astype(np.int)
        self.S = 0.99 * np.diag(diag_vec)

        Jre = Body.JointPath(root_link, end_link).calcJacobian() # root->end
        Jri = Body.JointPath(root_link, target_link).calcJacobian() # root->i
        Jie = Body.JointPath(target_link, end_link).calcJacobian() # i->end

        R2i = np.r_[ np.c_[coord_link.R,np.zeros((3,3))],
                     np.c_[np.zeros((3,3)), coord_link.R] ]

        J6ei = np.r_[ np.c_[ np.eye(3),-skew(target_link.p-end_link.p) ], # [ [E,rx] , [0,E] ]
                      np.c_[ np.zeros((3,3)),np.eye(3) ] ]

        Ji_tilde = np.c_[ Jri, J6ei.dot(Jie) ] # [Jri J6ei*Jie]
        G = np.eye(self.joint_path.numJoints()) # tmp E
        A_theta = np.diag([0,0,0, 1,1,1]).dot(Jre)
        # A_t = np.diag([0,0,0, 1,1,1]).dot(Jre)

        axis_mat = A_theta[3:]
        global max_tau
        # set tau_j to min_i(tau_i/|a_j.a_i|)
        # max_tau = (self.max_tau/abs(axis_mat.T.dot(axis_mat))).min(axis=1) # with all joints
        max_tau = (self.max_tau/abs(self.axis_product_mat*axis_mat.T.dot(axis_mat))).min(axis=1) # with only intersecting joints

        self.Jre = Jre
        self.Jri = Jri
        self.J6ei = J6ei
        self.Ji_tilde = Ji_tilde
        self.R2i = R2i
        self.A_theta = A_theta

        return convert_to_frame_load_wrench_vertices( Ji_tilde.transpose().dot(R2i), G.transpose()-Ji_tilde.transpose().dot(A_theta).dot(G.transpose()).dot(self.S) ) # Ji~^T*R2, G^T-Ji~^T*Atheta*Si

    def calc_instant_max_frame_load_wrench(self, target_joint_name, coord_link_name=None, do_plot=True, save_plot=False, fname="", save_model=False, do_wait=False, tm=0.2):
        return self.calc_max_frame_load_wrench(target_joint_name, coord_link_name=coord_link_name, do_plot=do_plot, save_plot=save_plot, fname="", is_instant=True, do_wait=False, tm=0.2)

    def calc_max_frame_load_wrench(self, target_joint_name, coord_link_name=None, do_plot=True, save_plot=False, fname="", is_instant=True, save_model=False, do_wait=False, tm=0.2):
        joint_angle_text = r'$\theta$: ' + str(np.rad2deg(self.robot.angleVector()).astype(np.int)) + ' [deg]' # round joint angles
        pi.joint_angle_text.set_text(joint_angle_text)
        logger.info(joint_angle_text)
        n_vertices = self.calc_current_load_wrench_vertices(target_joint_name, coord_link_name=coord_link_name)
        logger.debug("n_vertices=")
        logger.debug(n_vertices[:,3:])

        self.instant_max_load_wrench = n_vertices.max(axis=0)
        self.instant_min_load_wrench = n_vertices.min(axis=0)
        self.instant_max_load_wrench[np.ma.where(abs(self.instant_max_load_wrench) < 10)] = 0 # set |elements|<10 to 0
        self.instant_min_load_wrench[np.ma.where(abs(self.instant_min_load_wrench) < 10)] = 0
        self.instant_max_load_wrench[np.ma.where(abs(self.instant_max_load_wrench) >= max_value)] = np.inf # set |elements|>max_value to inf
        self.instant_min_load_wrench[np.ma.where(abs(self.instant_min_load_wrench) >= max_value)] = -np.inf

        self.max_load_wrench = np.vstack([self.instant_max_load_wrench, self.max_load_wrench]).max(axis=0)
        self.min_load_wrench = np.vstack([self.instant_min_load_wrench, self.min_load_wrench]).min(axis=0)

        ret_max_load_wrench,ret_min_load_wrench = [self.instant_max_load_wrench,self.instant_min_load_wrench] if is_instant else [self.max_load_wrench, self.min_load_wrench]

        pi.max_moment_text.set_text('${}^{(3)}n_{\mathrm{frm}}$: ' + str(ret_max_load_wrench[3:].astype(np.int)) + " [Nm]")
        logger.info(" max: " + str(ret_max_load_wrench))
        logger.info(" min: " + str(ret_min_load_wrench))
        if do_plot: pi.plot_convex_hull(n_vertices[:,3:], save_plot=save_plot, fname=fname, isInstant=is_instant)

        if save_model and self.world.is_choreonoid:
            self.world.robotItem.notifyKinematicStateChange()
            self.tree_view.checkItem(analyzer.world.robotItem, True)
            self.message_view.flush()
            head_fname = re.sub('[_0-9]*$',"",fname.replace(".png",""))
            self.scene_widget.saveImage(str(fname.replace(head_fname,head_fname+"_pose")))

        if do_wait:
            logger.critical("RET to continue, q to escape"+Style.RESET_ALL)
            key = raw_input()
            if key == 'q': escape = True
        else:
            if do_plot: time.sleep(tm)

        return [ret_max_load_wrench, ret_min_load_wrench]

    def calc_whole_range_max_load_wrench(self, target_joint_name, joint_idx=0, do_plot=True, save_plot=False, fname="", is_instant=True, save_model=False, do_wait=False, tm=0.2):
        if joint_idx == 0:
            self.reset_max_min_wrench()

        joint_name = self.joint_path.joint(joint_idx).name()
        joint_range = self.joint_range_list[joint_idx]
        step_angle = self.step_angle_list[joint_idx]

        if joint_idx < self.joint_path.numJoints() and logger.isEnabledFor(INFO): sys.stdout.write(Fore.GREEN+" "+"#"+joint_name+Style.RESET_ALL)
        if joint_idx+1 == self.joint_path.numJoints() and logger.isEnabledFor(INFO): print(" changed")
        fname=fname.replace(".png","_0.png") # set dummy
        for division_idx,joint_angle in enumerate(np.arange(joint_range[0],joint_range[1]+1,step_angle)):
            fname=re.sub('_[0-9]*\.png',"_"+str(division_idx).zfill(1+int((joint_range[1]-joint_range[0])/step_angle)/10)+".png",fname)
            self.robot.link(joint_name).q = np.deg2rad(joint_angle) # set joint angle [rad]
            if joint_idx+1 < self.joint_path.numJoints():
                self.calc_whole_range_max_load_wrench(target_joint_name,joint_idx+1,do_plot=do_plot,save_plot=save_plot,fname=fname,is_instant=is_instant,save_model=save_model,do_wait=do_wait,tm=tm)
            else:
                self.calc_max_frame_load_wrench(target_joint_name,do_plot=do_plot,save_plot=save_plot,fname=fname,is_instant=is_instant,save_model=save_model,do_wait=do_wait,tm=tm)

max_value = 10000

def initialize_plot_interface():
    global pi
    pi = PlotInterface()

    np.set_printoptions(precision=5)

    # plt.rcParams["font.size"] = 25
    plt.rcParams.update({"pgf.preamble":["\\usepackage{bm}"]})

    pi.ax.view_init(30,-25) # rotate view

    pi.set_max_display_num(800)

    pi.ax.set_xlabel("${}^{(3)}n_x$")
    pi.ax.set_ylabel("${}^{(3)}n_y$")
    pi.ax.set_zlabel("  ${}^{(3)}n_z$")


def test_calcuate_frame_load():
    logger.critical(Fore.BLUE+"test_calcuate_frame_load()"+Style.RESET_ALL)

    # division_num=1; do_wait=False; tm=0;   do_plot=True
    # division_num=4; do_wait=False; tm=0;   do_plot=True
    # division_num=4; do_wait=False; tm=0;   do_plot=False
    # division_num=6; do_wait=False; tm=0; do_plot=False
    step_angle=10; do_wait=False; tm=0;   do_plot=False
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

    # # JAXON_RED
    # analyzer = JointLoadWrenchAnalyzer([0,0,(0,0),0,0,0], robot_item=None, step_angle_list=step_angle_list)
    # analyzer.calc_whole_range_max_load_wrench('LLEG_JOINT2', do_wait=do_wait, tm=tm, do_plot=do_plot)
    # logger.critical(Fore.YELLOW+analyzer.robot.name()+" max wrench: "+str(analyzer.max_load_wrench)+Style.RESET_ALL)

    # Serial rotary
    logger.critical(Fore.BLUE+"Serial rotary drive joint"+Style.RESET_ALL)
    joint_structure_str="x-y-z_y_y-x"
    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], joint_range_list=joint_range_list, step_angle_list=step_angle_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_structure_str+".wrl"))
    analyzer.calc_whole_range_max_load_wrench('JOINT2', do_wait=do_wait, tm=tm, do_plot=do_plot)
    logger.critical(Fore.YELLOW+joint_structure_str+" max wrench: "+str(analyzer.max_load_wrench[3:].astype(np.int))+Style.RESET_ALL)

    joint_structure_str="x-y-z_y_x-y"
    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], joint_range_list=joint_range_list, step_angle_list=step_angle_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_structure_str+".wrl"))
    analyzer.calc_whole_range_max_load_wrench('JOINT2', do_wait=do_wait, tm=tm, do_plot=do_plot)
    logger.critical(Fore.YELLOW+joint_structure_str+" max wrench: "+str(analyzer.max_load_wrench[3:].astype(np.int))+Style.RESET_ALL)
    logger.critical("")


    joint_structure_str="y-x-z_y_y-x"
    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], joint_range_list=joint_range_list, step_angle_list=step_angle_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_structure_str+".wrl"))
    analyzer.calc_whole_range_max_load_wrench('JOINT2', do_wait=do_wait, tm=tm, do_plot=do_plot)
    logger.critical(Fore.YELLOW+joint_structure_str+" max wrench: "+str(analyzer.max_load_wrench[3:].astype(np.int))+Style.RESET_ALL)

    joint_structure_str="y-x-z_y_x-y"
    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], joint_range_list=joint_range_list, step_angle_list=step_angle_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_structure_str+".wrl"))
    analyzer.calc_whole_range_max_load_wrench('JOINT2', do_wait=do_wait, tm=tm, do_plot=do_plot)
    logger.critical(Fore.YELLOW+joint_structure_str+" max wrench: "+str(analyzer.max_load_wrench[3:].astype(np.int))+Style.RESET_ALL)
    logger.critical("")


    joint_structure_str="z-x-y_y_y-x"
    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], joint_range_list=joint_range_list, step_angle_list=step_angle_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_structure_str+".wrl"))
    analyzer.calc_whole_range_max_load_wrench('JOINT2', do_wait=do_wait, tm=tm, do_plot=do_plot)
    logger.critical(Fore.YELLOW+joint_structure_str+" max wrench: "+str(analyzer.max_load_wrench[3:].astype(np.int))+Style.RESET_ALL)

    joint_structure_str="z-x-y_y_x-y"
    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], joint_range_list=joint_range_list, step_angle_list=step_angle_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_structure_str+".wrl"))
    analyzer.calc_whole_range_max_load_wrench('JOINT2', do_wait=do_wait, tm=tm, do_plot=do_plot)
    logger.critical(Fore.YELLOW+joint_structure_str+" max wrench: "+str(analyzer.max_load_wrench[3:].astype(np.int))+Style.RESET_ALL)
    logger.critical("")


    joint_structure_str="z-y-x_y_y-x"
    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], joint_range_list=joint_range_list, step_angle_list=step_angle_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_structure_str+".wrl"))
    analyzer.calc_whole_range_max_load_wrench('JOINT2', do_wait=do_wait, tm=tm, do_plot=do_plot)
    logger.critical(Fore.YELLOW+joint_structure_str+" max wrench: "+str(analyzer.max_load_wrench[3:].astype(np.int))+Style.RESET_ALL)

    joint_structure_str="z-y-x_y_x-y"
    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], joint_range_list=joint_range_list, step_angle_list=step_angle_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_structure_str+".wrl"))
    analyzer.calc_whole_range_max_load_wrench('JOINT2', do_wait=do_wait, tm=tm, do_plot=do_plot)
    logger.critical(Fore.YELLOW+joint_structure_str+" max wrench: "+str(analyzer.max_load_wrench[3:].astype(np.int))+Style.RESET_ALL)
    logger.critical("")


    # Serial linear
    logger.critical(Fore.BLUE+"Serial linear drive joint"+Style.RESET_ALL)
    joint_structure_str="z-x-y_y_y-x"
    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(1,0),(1,0),(0,0),(0,0)], joint_range_list=joint_range_list, step_angle_list=step_angle_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_structure_str+".wrl"))
    analyzer.calc_whole_range_max_load_wrench('JOINT2', do_wait=do_wait, tm=tm, do_plot=do_plot)
    logger.critical(Fore.YELLOW+joint_structure_str+" max wrench: "+str(analyzer.max_load_wrench[3:].astype(np.int))+Style.RESET_ALL)

    joint_structure_str="z-y-x_y_y-x"
    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(1,0),(1,0),(0,0),(0,0)], joint_range_list=joint_range_list, step_angle_list=step_angle_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_structure_str+".wrl"))
    analyzer.calc_whole_range_max_load_wrench('JOINT2', do_wait=do_wait, tm=tm, do_plot=do_plot)
    logger.critical(Fore.YELLOW+joint_structure_str+" max wrench: "+str(analyzer.max_load_wrench[3:].astype(np.int))+Style.RESET_ALL)

    # Parallel linear
    logger.critical(Fore.BLUE+"Parallel linear drive joint"+Style.RESET_ALL)
    joint_structure_str="z-x-y_y_y-x"
    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(2,0),(1,0),(0,0),(0,0)], joint_range_list=joint_range_list, step_angle_list=step_angle_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_structure_str+".wrl"))
    analyzer.calc_whole_range_max_load_wrench('JOINT2', do_wait=do_wait, tm=tm, do_plot=do_plot)
    logger.critical(Fore.YELLOW+joint_structure_str+" max wrench: "+str(analyzer.max_load_wrench[3:].astype(np.int))+Style.RESET_ALL)

    joint_structure_str="z-y-x_y_y-x"
    analyzer = JointLoadWrenchAnalyzer([(0,0),(0,0),(2,0),(1,0),(0,0),(0,0)], joint_range_list=joint_range_list, step_angle_list=step_angle_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_structure_str+".wrl"))
    analyzer.calc_whole_range_max_load_wrench('JOINT2', do_wait=do_wait, tm=tm, do_plot=do_plot)
    logger.critical(Fore.YELLOW+joint_structure_str+" max wrench: "+str(analyzer.max_load_wrench[3:].astype(np.int))+Style.RESET_ALL)

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
    target_end = analyzer0.joint_path.endLink()

    common_fname=os.path.join(package_path,"joint-configuration-comparison","joint-configuration-comparison")

    if analyzer0.world.is_choreonoid:
        tree_view = Base.ItemTreeView.instance()
        message_view = Base.MessageView.instance()
        scene_widget = Base.SceneView.instance().sceneWidget()

    analyzer0.robot.angleVector(np.deg2rad(np.array([0,0,0,0,0,0])))
    analyzer0.calc_max_frame_load_wrench('JOINT2', do_wait=False, tm=0, do_plot=True, save_plot=True, fname=common_fname+"_initial-pose_load-region.png")
    logger.critical(Fore.YELLOW+joint_configuration_str0+" max wrench: "+str(analyzer0.max_load_wrench)+Style.RESET_ALL)
    analyzer1.robot.angleVector(np.deg2rad(np.array([0,0,0,0,0,0])))
    if analyzer0.world.is_choreonoid:
            analyzer0.world.robotItem.notifyKinematicStateChange()
            tree_view.checkItem(analyzer0.world.robotItem, True)
            tree_view.checkItem(analyzer1.world.robotItem, False)
            message_view.flush()
            scene_widget.saveImage(str(common_fname+"_configuration0"+"_initial-pose.png"))
            tree_view.checkItem(analyzer0.world.robotItem, False)
            tree_view.checkItem(analyzer1.world.robotItem, True)
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

        logger.info("IK: " + str(analyzer1.joint_path.calcInverseKinematics(target_end.p, target_end.R)))
        analyzer1.calc_max_frame_load_wrench('JOINT2', do_wait=False, tm=0, do_plot=True, save_plot=True, fname=common_fname+"_configuration1"+"_load-region"+index_str+".png")
        # logger.info(Fore.YELLOW+joint_configuration_str1+" max wrench: "+str(analyzer1.max_load_wrench)+Style.RESET_ALL)

        if analyzer0.world.is_choreonoid:
            analyzer0.world.robotItem.notifyKinematicStateChange()
            tree_view.checkItem(analyzer0.world.robotItem, True)
            tree_view.checkItem(analyzer1.world.robotItem, False)
            analyzer0.draw_moment()
            analyzer1.hide_moment()
            message_view.flush()

            scene_widget.saveImage(str(common_fname+"_configuration0"+"_pose"+index_str+".png"))
            tree_view.checkItem(analyzer0.world.robotItem, False)
            tree_view.checkItem(analyzer1.world.robotItem, True)
            analyzer0.hide_moment()
            analyzer1.draw_moment()
            message_view.flush()
            scene_widget.saveImage(str(common_fname+"_configuration1"+"_pose"+index_str+".png"))

def export_drive_system_comparison():
    logger.critical(Fore.BLUE+"export_drive_system_comparison()"+Style.RESET_ALL)

    package_path = roslib.packages.get_pkg_dir("structure_analyzer")
    model_path = os.path.join(package_path, "models")

    pi.set_max_display_num(400)

    joint_range_list = [(35,35),(100,100),(20,20), (0,0),(0,0),(0,0) ,(0,0),(0,0),(0,0)]

    joint_configuration_str="z-x-y_y_y-x"
    global analyzer0
    analyzer0 = JointLoadWrenchAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], joint_range_list=joint_range_list,
                                        end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_configuration_str+".wrl"))
    global analyzer1
    analyzer1 = JointLoadWrenchAnalyzer([(0,0),(0,0),(1,0),(0,0),(0,0),(0,0)], joint_range_list=joint_range_list,
                                        end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_configuration_str+".wrl"))
    global analyzer2
    analyzer2 = JointLoadWrenchAnalyzer([(0,0),(0,0),(2,0),(0,0),(0,0),(0,0)], joint_range_list=joint_range_list,
                                        end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_configuration_str+".wrl"))

    if analyzer0.world.is_choreonoid:
        tree_view = Base.ItemTreeView.instance()
        message_view = Base.MessageView.instance()
        scene_widget = Base.SceneView.instance().sceneWidget()

    l_angle = np.array([30,-70,-40,50,0,0])
    u_angle = np.array([120,0,80,-90,0,0])
    common_fname=os.path.join(package_path,"drive-system-comparison","drive-system-comparison")
    division_num = 20
    angle_vectors = np.vstack([np.linspace(l_angle,u_angle,division_num,endpoint=True), np.linspace(u_angle,l_angle,division_num,endpoint=True)])
    for idx, angle_vector in enumerate(angle_vectors):
        # index_str = "_"+'_'.join(angle_vector.astype(np.int).astype(np.str))
        index_str = "_"+str(idx).zfill(2)

        analyzer0.robot.angleVector(np.deg2rad(angle_vector))
        analyzer0.calc_max_frame_load_wrench('JOINT2', do_wait=False, tm=0, do_plot=True, save_plot=True, fname=common_fname+"_system0"+"_load-region"+index_str+".png")
        # logger.info(Fore.YELLOW+joint_configuration_str+" max wrench: "+str(analyzer0.max_load_wrench)+Style.RESET_ALL)

        analyzer1.robot.angleVector(np.deg2rad(angle_vector))
        analyzer1.calc_max_frame_load_wrench('JOINT2', do_wait=False, tm=0, do_plot=True, save_plot=True, fname=common_fname+"_system1"+"_load-region"+index_str+".png")
        # logger.info(Fore.YELLOW+joint_configuration_str+" max wrench: "+str(analyzer1.max_load_wrench)+Style.RESET_ALL)

        analyzer2.robot.angleVector(np.deg2rad(angle_vector))
        analyzer2.calc_max_frame_load_wrench('JOINT2', do_wait=False, tm=0, do_plot=True, save_plot=True, fname=common_fname+"_system2"+"_load-region"+index_str+".png")
        # logger.info(Fore.YELLOW+joint_configuration_str+" max wrench: "+str(analyzer1.max_load_wrench)+Style.RESET_ALL)

        if analyzer0.world.is_choreonoid:
            analyzer0.world.robotItem.notifyKinematicStateChange()
            tree_view.checkItem(analyzer0.world.robotItem, True)
            tree_view.checkItem(analyzer1.world.robotItem, False)
            tree_view.checkItem(analyzer2.world.robotItem, False)
            analyzer0.draw_moment()
            analyzer1.hide_moment()
            analyzer2.hide_moment()
            message_view.flush()
            scene_widget.saveImage(str(common_fname+"_system0"+"_pose"+index_str+".png"))

            tree_view.checkItem(analyzer0.world.robotItem, False)
            tree_view.checkItem(analyzer1.world.robotItem, True)
            tree_view.checkItem(analyzer2.world.robotItem, False)
            analyzer0.hide_moment()
            analyzer1.draw_moment()
            message_view.flush()
            scene_widget.saveImage(str(common_fname+"_system1"+"_pose"+index_str+".png"))

            tree_view.checkItem(analyzer0.world.robotItem, False)
            tree_view.checkItem(analyzer1.world.robotItem, False)
            tree_view.checkItem(analyzer2.world.robotItem, True)
            analyzer1.hide_moment()
            analyzer2.draw_moment()
            message_view.flush()
            scene_widget.saveImage(str(common_fname+"_system2"+"_pose"+index_str+".png"))

if __name__ == '__main__':
    initialize_plot_interface()

    # test
    test_calcuate_frame_load()

    # make figures
    export_joint_configuration_comparison()

    export_drive_system_comparison()

    # export_overall_frame_load_region()

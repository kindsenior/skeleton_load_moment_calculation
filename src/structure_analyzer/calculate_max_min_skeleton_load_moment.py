#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import pprint, time, sys, os, re
from colorama import Fore, Back, Style
import pdb

import roslib

from logger import *

import structure_load_analyzer as sla
sla.logger.setLevel(sla.CRITICAL)

def solve_nth_degree_equation(vec,is_complex=False):
    logger.debug("solve: " + str(vec))
    dim =len(vec)
    if is_complex:
        A = np.zeros((dim,dim),dtype=complex)
    else:
        A = np.zeros((dim,dim))
    A[np.arange(dim-1),1+np.arange(dim-1)] =1
    A[-1,:] = -vec
    ans,vec = np.linalg.eig(A)
    return ans

def deco(func):
    import functools
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        return 0 if args[1] == 0 else func(*args,**kwargs)
    return wrapper

class Link(object):
    MOMENT_TYPE = (BENDING,TORSIONAL) = range(2)

    def __init__(self):
        self.E = 74*(10**9)
        self.G = 29*(10**9)
        self.v_thre = 0.001
        # self.phi_thre = 0.01
        self.phi_thre = 0.009
        self.l = 0.4
        self.t = 0.0025*np.ones(len(Link.MOMENT_TYPE))
        self.b = 0.12*np.ones(len(Link.MOMENT_TYPE))

        # self.E = 70*(10**9)
        # self.G = 25*(10**9)
        # self.v_thre = 0.0015
        # self.phi_thre = 0.0165
        # self.l = 0.35
        # self.t = 0.0025*np.ones(len(Link.MOMENT_TYPE))
        # self.b = 0.1*np.ones(len(Link.MOMENT_TYPE))

        # self.E = 70*(10**9)
        # self.G = 27*(10**9)
        # self.v_thre = 0.0015
        # self.phi_thre = 0.0135
        # self.l = 0.38
        # self.t = 0.0025*np.ones(len(Link.MOMENT_TYPE))
        # self.b = 0.1*np.ones(len(Link.MOMENT_TYPE))

        self.rho = 2.7*(10**6)

    def I_bending_thre(self, n_max):
        return n_max*self.l**2/(2*self.E*self.v_thre)

    def get_min_real(self, ans):
        ans = np.real_if_close(ans[np.isreal(ans)])
        ans = ans[ans>=0].min()
        return ans

    def calculate_min_mass(self, max_moment_vec):
        n_max = np.max(max_moment_vec[0:2])
        n_z_max = max_moment_vec[2]
        # width_vec = np.array([np.inf if np.isinf(n_max) else self.min_width_from_bending(n_max), np.inf if np.isinf(n_z_max) else self.min_width_from_torsional(n_z_max)])
        # logger.debug("width_vec: " + str(width_vec))
        # min_mass = self.rho*self.cross_section_area(width_vec)*self.l

        # self.b[Link.BENDING]   = np.inf if np.isinf(n_max)   else self.min_width_from_bending(n_max)
        # self.b[Link.TORSIONAL] = np.inf if np.isinf(n_z_max) else self.min_width_from_torsional(n_z_max)
        logger.debug("bending width: " + str(self.b[Link.BENDING]) + " torsional width: " + str(self.b[Link.TORSIONAL]))
        self.t[Link.BENDING]   = np.inf if np.isinf(n_max)   else self.min_thickness_from_bending(n_max)
        self.t[Link.TORSIONAL] = np.inf if np.isinf(n_z_max) else self.min_thickness_from_torsional(n_z_max)
        logger.debug("bending thickness: " + str(self.t[Link.BENDING]) + " torsional thickness: " + str(self.t[Link.TORSIONAL]))
        min_mass = self.rho*self.cross_section_area()*self.l # rho*S*l = rho*V
        logger.info("min mass of link: " + str(min_mass))
        return min_mass

class SquarePipeLink(Link):
    def __init__(self):
        super(SquarePipeLink, self).__init__()

    def I_torsional_thre(self, n_z_max):
        return n_z_max*self.l/(self.G*self.phi_thre)

    def min_width_from_bending(self, n_max):
        t = self.t[Link.BENDING]
        vec = (1.0/12)*np.array([-16*t**4, 32*t**3, -24*t**2, 8*t])
        vec[0] -= self.I_bending_thre(n_max)
        ans = solve_nth_degree_equation(vec)
        logger.debug("ans: " + str(ans))
        ans = self.get_min_real(ans)
        logger.debug("width: " + str(ans))
        return ans

    def min_width_from_torsional(self, n_z_max):
        t = self.t[Link.TORSIONAL]
        vec = (1.0/6)*np.array([-16*t**4, 32*t**3, -24*t**2, 8*t])
        vec[0] -= self.I_torsional_thre(n_z_max)
        ans = solve_nth_degree_equation(vec)
        logger.debug("ans: " + str(ans))
        ans = self.get_min_real(ans)
        logger.debug("width: " + str(ans))
        return ans

    @deco
    def min_thickness_from_bending(self, n_max):
        b = self.b[Link.BENDING]
        vec = (1.0/12)*np.array([0, 8*b**3, -24*b**2, 32*b, -16])
        vec[0] -= self.I_bending_thre(n_max)
        ans = solve_nth_degree_equation(vec)
        logger.debug("ans: " + str(ans))
        ans = self.get_min_real(ans)
        logger.debug("thickness: " + str(ans))
        return ans

    @deco
    def min_thickness_from_torsional(self, n_z_max):
        b = self.b[Link.TORSIONAL]
        # vec = (1.0/6)*np.array([0, 8*b**3, -24*b**2, 32*b, -16]) # circular section
        vec = np.array([0, b**3]) # thin-walled section
        vec[0] -= self.I_torsional_thre(n_z_max)
        ans = solve_nth_degree_equation(vec)
        logger.debug("ans: " + str(ans))
        ans = self.get_min_real(ans)
        logger.debug("thickness: " + str(ans))
        return ans

    # def cross_section_area(self, width):
    #     return width**2 - (width-2*self.t)**2

    def cross_section_area(self):
        return self.b**2 - (self.b-2*self.t)**2

class HSectionLink(Link):
    def __init__(self):
        super(HSectionLink, self).__init__()
        self.t2 = 0.002 # frange thickness
        # self.tc = 0.0005 # center thickness
        # self.tc = 0.0005 # center thickness
        self.tc = 0.001 # center thickness
        # self.tc = 0.0015 # center thickness

    def I_torsional_thre(self, n_z_max):
        return n_z_max*(self.l**3)/(3*self.E*self.phi_thre) # Do check

    def I_torsional_thre_simple(self, n_z_max):
        return n_z_max*self.l/(self.G*self.phi_thre)

    def min_width_from_bending(self, n_max):
        t = self.t[Link.BENDING]
        t2 = self.t2
        # vec = (1.0/12)*np.array([-8*t**4, 20*t**3, -18*t**2, 7*t])
        vec = (1.0/12)*np.array([-8*t*(t2**3), 8*(t2**3)+12*t*(t2**2), -(12*(t2**2)+6*t*t2), 6*t2+t])
        vec[0] -= self.I_bending_thre(n_max)
        ans = solve_nth_degree_equation(vec)
        logger.debug("ans: " + str(ans))
        ans = self.get_min_real(ans)
        logger.debug("width: " + str(ans))
        return ans

    def min_width_from_torsional(self, n_z_max):
        t = self.t[Link.BENDING]
        t2 = self.t2
        # vec = (1.0/24)*np.array([0,0,0, t**3, -2*t**2, t])
        vec = (1.0/24)*np.array([0,0,0, t2**3, -2*t2**2, t2])
        vec[0] -= self.I_torsional_thre(n_z_max)
        ans = solve_nth_degree_equation(vec)
        logger.debug("ans: " + str(ans))
        ans = self.get_min_real(ans)
        logger.debug("width: " + str(ans))
        return ans

    @deco
    def min_thickness_from_bending(self, n_max):
        b = self.b[Link.BENDING]
        tc = self.tc
        # vec = (1.0/12)*np.array([0, 7*b**3, -18*b**2, 20*b, -8]) # common t
        vec = (1.0/12)*np.array([tc*(b**3), 6*(b-tc)*(b**2), -12*(b-tc)*b, 8*(b-tc)]) # use tc
        vec[0] -= self.I_bending_thre(n_max)
        ans = solve_nth_degree_equation(vec)
        logger.debug("ans: " + str(ans))
        ans = self.get_min_real(ans)
        logger.debug("thickness: " + str(ans))
        return ans

    @deco
    def min_thickness_from_torsional(self, n_z_max):
        b = self.b[Link.TORSIONAL]
        tc = self.tc

        # # only warping torsion
        # vec = (1.0/24)*np.array([0, b**5, -2*(b**4), b**3])
        # vec[0] -= self.I_torsional_thre(n_z_max)

        # # only pure torsion (open cross-section)
        # vec = np.array([0, 0, 0, b]) # approximate
        # vec = np.array([0, 0, 0, b, -2.0/3])
        # vec[0] -= self.I_torsional_thre_simple(n_z_max)

        # only pure torsion (closed cross-section)
        # vec = (1.0/12)*np.array([0, 9*(b**3), -18*(b**2), 21*b, -10]) # common t
        vec = (1.0/12)*np.array([tc*(b**3)+(tc**3)*b, 2*(b-tc)*(4*(b**2)+b*tc+tc**2), -12*(b-tc)*b, 8*(b-tc)]) # use tc
        vec[0] -= self.I_torsional_thre_simple(n_z_max)

        ans = solve_nth_degree_equation(vec)
        logger.debug("ans: " + str(ans))
        ans = self.get_min_real(ans)
        logger.debug("thickness: " + str(ans))
        return ans

    # def cross_section_area(self, width):
    #     # return width**2 - (width-self.t)*(width-2*self.t)
    #     return width**2 - (width-self.t)*(width-2*self.t2)

    def cross_section_area(self):
        return self.b**2 - (self.b-self.t)*(self.b-2*self.t) # common t
        # return self.b**2 - (self.b-self.t)*(self.b-2*self.t2)
        # return self.b**2 - (self.b-self.tc)*(self.b-2*self.t) # use tc

# v_thre,n_xy_max -> I_thre -> b -> S -> m
# phi_thre,n_z_max -> I_s,omega_thre -> b -> S -> m

class LinkDeflectionAnalyzer(sla.JointLoadWrenchAnalyzer):
    def __init__(self, actuator_set_list_, link_shape_list, joint_range_list=None, robot_item=None, robot_model_file=None, end_link_name="LLEG_JOINT5", step_angle_list=None, step_angle=10):
        super(LinkDeflectionAnalyzer, self).__init__(actuator_set_list_, joint_range_list=joint_range_list,
                                                     robot_item=robot_item, robot_model_file=robot_model_file, end_link_name=end_link_name, step_angle_list=step_angle_list, step_angle=step_angle)

        self.set_link_shape_list(link_shape_list)

    def set_link_shape_list(self, link_shape_list):
        self.link_shape_list=link_shape_list

    def reset_min_mass_list(self):
        self.min_mass_list = np.array([[0]*len(Link.MOMENT_TYPE) for i in range(self.joint_path.numJoints())])

    def calc_min_section(self, target_joint_name, coord_link_name, do_plot, save_plot, fname, save_model, do_wait, tm):
        max_wrench,min_wrench = self.calc_instant_max_frame_load_wrench(target_joint_name,coord_link_name=target_link_name,do_plot=do_plot,save_plot=False,fname=fname,save_model=save_model,do_wait=do_wait,tm=tm)

    def calc_min_link_mass(self, joint_idx=0, do_plot=True, save_plot=False, fname="", save_model=False, do_wait=False, tm=0.2):
        if joint_idx == 0:
            self.reset_max_min_wrench()
            self.reset_min_mass_list()

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
                # set next joint angle
                # self.calc_whole_range_max_load_wrench(target_joint_name,joint_idx+1,do_plot=do_plot,save_plot=False,fname=fname,is_instant=is_instant,save_model=save_model,do_wait=do_wait,tm=tm)
                self.calc_min_link_mass(joint_idx+1,do_plot=do_plot,save_plot=False,fname=fname,save_model=save_model,do_wait=do_wait,tm=tm)
            else:
                # self.calc_max_frame_load_wrench(target_joint_name,do_plot=do_plot,save_plot=False,fname=fname,is_instant=True,save_model=save_model,do_wait=do_wait,tm=tm)
                for idx,link_shape in enumerate(self.link_shape_list):
                    if not link_shape is None:
                        target_joint_name = self.joint_path.joint(idx).name()
                        target_link_name = self.joint_path.joint(idx).name()
                        max_proximal_wrench,min_proximal_wrench = self.calc_instant_max_frame_load_wrench(target_joint_name,coord_link_name=target_link_name,do_plot=do_plot,save_plot=False,fname=fname,save_model=save_model,do_wait=do_wait,tm=tm)

                        target_link_name = self.joint_path.joint(idx+1).name()
                        max_distal_wrench,min_distal_wrench = self.calc_instant_max_frame_load_wrench(target_joint_name,coord_link_name=target_link_name,do_plot=do_plot,save_plot=False,fname=fname,save_model=save_model,do_wait=do_wait,tm=tm)

                        # logger.debug(max_proximal_wrench)
                        # logger.debug(max_distal_wrench)

                        self.min_mass_list[idx] = np.vstack([self.min_mass_list[idx], link_shape.calculate_min_mass(min_proximal_wrench, max_distal_wrench)]).max(axis=0)

        return self.min_mass_list


def calculate_max_skeleton_load_moment(_joint_structure):
    joint_structure = _joint_structure
    sla.set_joint_structure(joint_structure)
    max_moment_vec,_,_ = sla.sweep_joint_range(dowait=False, division_num=10 ,tm = 0, plot = False)
    logger.info("joint_structure=" + str(joint_structure) + " : max_moment=" + str(max_moment_vec))
    return max_moment_vec

def calculate_link_mass(_joint_structure):
    max_moment_vec = calculate_max_skeleton_load_moment(_joint_structure)
    sqlink.calculate_min_mass(max_moment_vec)
    hslink.calculate_min_mass(max_moment_vec)
    logger.info("")

sqlink = SquarePipeLink()
hslink = HSectionLink()

# roll=x=0, pitch=y=1, yaw=z=2
# # x-y-z
calculate_link_mass([[0],[1],[2],[]]) # 4733,2756,120 ok
calculate_link_mass([[0],[1],[2]])    # 4732,2756,0 ok
# calculate_link_mass([[0],[1,2]])      # inf,0,0 ok-
calculate_link_mass([[0,1],[2],[]])      # 770,771,0 ok
calculate_link_mass([[0,1],[2]])      # 770,771,0 ok
# x-z-y
calculate_link_mass([[0],[2],[1],[]]) # inf,700,2377 ok
calculate_link_mass([[0],[2],[1]])
# calculate_link_mass([[0],[2,1]])      # error
# calculate_link_mass([[0,2],[1]])      # 350,0,347 ok-
# y-x-z
calculate_link_mass([[1],[0],[2],[]]) # 787,795,120 ok
calculate_link_mass([[1],[0],[2]]) # 787,795,120 ok
# calculate_link_mass([[1],[0,2]])      # 0,inf,0 ok-
calculate_link_mass([[1,0],[2],[]])      # 770,771,0 ok
calculate_link_mass([[1,0],[2]])      # 770,771,0 ok
# y-z-x
calculate_link_mass([[1],[2],[0],[]]) # 330,inf,1933 ok
calculate_link_mass([[1],[2],[0]])
# calculate_link_mass([[1],[2,0]])      # 0,inf,0 ok-
# calculate_link_mass([[1,2],[0]])      # 0,715,666 ok-
# z-x-y
calculate_link_mass([[2],[0],[1],[]]) # JAXON1,2
calculate_link_mass([[2],[0],[1]])    # JAXON3
# calculate_link_mass([[2],[0,1],[]])   # 330,700,4317
calculate_link_mass([[2],[0,1]])      # 0,0,4317
# calculate_link_mass([[2,0],[1],[]]   # 350,700,347
# calculate_link_mass([[2,0],[1]])      # 350,0,347   x-
# z-y-x
calculate_link_mass([[2],[1],[0],[]]) # 330,3050,7456 ok
calculate_link_mass([[2],[1],[0]])    # 0,3050,7456 ok
sklms.max_tau_list = np.array([301,700,120]) # roll, pitch, yaw これがないとerror
calculate_link_mass([[2],[1,0]])      # 0,0,4660 ok
# calculate_link_mass([[2,1],[0]])     # 0,710,666 ok-

# calculate_link_mass([[2,0,1]])        # 0,0,0
# calculate_link_mass([[2,0,1],[]])     # 330,700,120

logger.setLevel(CRITICAL)
# moment_array,_ = np.mgrid[10:1000:10,0:3] # 10->1000
moment_array,_ = np.mgrid[140:1000:10,0:3] # 10->1000
# moment_array,_ = np.mgrid[120:1000:10,0:3] # 10->1000
# moment_array,_ = np.mgrid[400:800:10,0:3]
sqlink.t = 0.002*np.ones(2)
hslink.t = 0.002*np.ones(2)
sqlink_mass_array = np.array([sqlink.calculate_min_mass(moment_vec) for moment_vec in moment_array])
hslink_mass_array = np.array([hslink.calculate_min_mass(moment_vec) for moment_vec in moment_array])

fig, (bending_plt, torsional_plt) = plt.subplots(ncols=2, figsize=(15,4))
fig.subplots_adjust(left=0.05,right=0.98, bottom=0.15,top=0.9, wspace=0.1, hspace=1)

bending_plt.plot(moment_array[:,0], sqlink_mass_array[:,0], label="Square pipe link")
bending_plt.plot(moment_array[:,0], hslink_mass_array[:,0], label="H-section link")
bending_plt.set_title("Link mass calculated from bending thresholds")
bending_plt.set_xlabel("Skeleton load moment [Nm]")
bending_plt.set_ylabel("Mass [g]")
bending_plt.grid(True)
bending_plt.legend()

torsional_plt.plot(moment_array[:,0], sqlink_mass_array[:,1], label="Square pipe link")
torsional_plt.plot(moment_array[:,0], hslink_mass_array[:,1], label="H-section link")
torsional_plt.set_title("Link mass calculated from torsional thresholds")
torsional_plt.set_xlabel("Skeleton load oment [Nm]")
torsional_plt.grid(True)
torsional_plt.legend()

plt.rcParams["font.size"] = 15
plt.pause(0.1)
fig.savefig("skeleton-load-moment-link-mass-relation.png")

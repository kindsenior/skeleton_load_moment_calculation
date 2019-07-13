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
    # MOMENT_TYPE = (BENDING_Y,TORSIONAL) = range(2)
    MOMENT_TYPE = (BENDING_X,BENDING_Y,TORSIONAL) = range(3)

    def __init__(self):
        self.E = 74*(10**9)
        self.G = 29*(10**9)
        self.l = 0.4
        self.v_thre = 0.001
        # self.phi_thre = 0.01
        self.phi_thre = (self.v_thre/self.l)*2
        self.theta_thre = self.v_thre/self.l
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

    def I_bending_thre(self, min_proximal_wrench, max_distal_wrench):
        min_proximal_n = min_proximal_wrench[3:6]
        max_distal_n   = max_distal_wrench[3:6]
        Ix = max(((-1.0/3)*min_proximal_n[0] + (1.0/6)*max_distal_n[0])*(self.l**2)/(self.E*self.v_thre),
                 ((-1.0/2)*min_proximal_n[0] + (1.0/2)*max_distal_n[0])*self.l/(self.E*self.theta_thre))
        Iy = max(((-1.0/3)*min_proximal_n[1] + (1.0/6)*max_distal_n[1])*(self.l**2)/(self.E*self.v_thre),
                 ((-1.0/2)*min_proximal_n[1] + (1.0/2)*max_distal_n[1])*self.l/(self.E*self.theta_thre))

        return Ix,Iy

    def get_min_real(self, ans):
        ans = np.real_if_close(ans[np.isreal(ans)])
        try:
            return ans[ans>=0].min()
        except ValueError:
            logger.error(Fore.RED+"!!!no positive real in get_min_real()!!!"+Style.RESET_ALL)
            return 0

    def calculate_min_mass(self, min_proximal_wrench, max_distal_wrench):
        I_bending_thre  = self.I_bending_thre(min_proximal_wrench, max_distal_wrench)
        I_torsional_thre = self.I_torsional_thre(max_distal_wrench[5])

        # width_vec = np.array([np.inf if np.isinf(n_max) else self.min_width_from_bending(n_max), np.inf if np.isinf(n_z_max) else self.min_width_from_torsional(n_z_max)])
        # logger.debug("width_vec: " + str(width_vec))
        # min_mass = self.rho*self.cross_section_area(width_vec)*self.l

        # self.b[Link.BENDING]   = np.inf if np.isinf(n_max)   else self.min_width_from_bending(n_max)
        # self.b[Link.TORSIONAL] = np.inf if np.isinf(n_z_max) else self.min_width_from_torsional(n_z_max)
        logger.debug("bending width: " + str(self.b[Link.BENDING_Y]) + " torsional width: " + str(self.b[Link.TORSIONAL]))
        self.t[Link.BENDING_X] = np.inf if np.isinf(I_bending_thre[0]) else self.min_thickness_from_bending_x(I_bending_thre[0])
        self.t[Link.BENDING_Y] = np.inf if np.isinf(I_bending_thre[1]) else self.min_thickness_from_bending(I_bending_thre[1])
        self.t[Link.TORSIONAL] = np.inf if np.isinf(I_torsional_thre)  else self.min_thickness_from_torsional(I_torsional_thre)
        logger.debug("bending thickness: " + str(self.t[Link.BENDING_Y]) + " torsional thickness: " + str(self.t[Link.TORSIONAL]))
        min_mass = self.rho*self.cross_section_area()*self.l # rho*S*l = rho*V
        logger.info("min mass of link: " + str(min_mass))
        return min_mass # minimum mass list

    def cross_section_area(self):
        raise NotImplementedError("Must override cross_section_area()")

class SquarePipeLink(Link):
    def __init__(self):
        super(SquarePipeLink, self).__init__()

    def link_type(self): return "Square pipe"

    def I_torsional_thre(self, n_z_max):
        return n_z_max*self.l/(self.G*self.phi_thre)

    def min_width_from_bending(self, I_bending_thre):
        t = self.t[Link.BENDING_Y]
        vec = (1.0/12)*np.array([-16*t**4, 32*t**3, -24*t**2, 8*t])
        vec[0] -= I_bending_thre
        ans = solve_nth_degree_equation(vec)
        logger.debug("ans: " + str(ans))
        ans = self.get_min_real(ans)
        logger.debug("width: " + str(ans))
        return ans

    def min_width_from_torsional(self, I_torsional_thre):
        t = self.t[Link.TORSIONAL]
        vec = (1.0/6)*np.array([-16*t**4, 32*t**3, -24*t**2, 8*t])
        vec[0] -= I_torsional_thre
        ans = solve_nth_degree_equation(vec)
        logger.debug("ans: " + str(ans))
        ans = self.get_min_real(ans)
        logger.debug("width: " + str(ans))
        return ans

    @deco
    def min_thickness_from_bending_x(self, I_bending_thre):
        return self.min_thickness_from_bending(I_bending_thre)

    @deco
    def min_thickness_from_bending(self, I_bending_thre):
        b = self.b[Link.BENDING_Y]
        vec = (1.0/12)*np.array([0, 8*b**3, -24*b**2, 32*b, -16])
        vec[0] -= I_bending_thre
        ans = solve_nth_degree_equation(vec)
        logger.debug("ans: " + str(ans))
        ans = self.get_min_real(ans)
        logger.debug("thickness: " + str(ans))
        return ans

    @deco
    def min_thickness_from_torsional(self, I_torsional_thre):
        b = self.b[Link.TORSIONAL]
        # vec = (1.0/6)*np.array([0, 8*b**3, -24*b**2, 32*b, -16]) # circular section
        vec = np.array([0, b**3]) # thin-walled section
        vec[0] -= I_torsional_thre
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

    def link_type(self): return "HSection"

    # def I_torsional_thre(self, n_z_max):
    #     return n_z_max*(self.l**3)/(3*self.E*self.phi_thre) # Do check

    # def I_torsional_thre_simple(self, n_z_max):
    def I_torsional_thre(self, n_z_max):
        return n_z_max*self.l/(self.G*self.phi_thre)

    def min_width_from_bending(self, I_bending_thre):
        t = self.t[Link.BENDING_Y]
        t2 = self.t2
        # vec = (1.0/12)*np.array([-8*t**4, 20*t**3, -18*t**2, 7*t])
        vec = (1.0/12)*np.array([-8*t*(t2**3), 8*(t2**3)+12*t*(t2**2), -(12*(t2**2)+6*t*t2), 6*t2+t])
        vec[0] -= I_bending_thre
        ans = solve_nth_degree_equation(vec)
        logger.debug("ans: " + str(ans))
        ans = self.get_min_real(ans)
        logger.debug("width: " + str(ans))
        return ans

    def min_width_from_torsional(self, I_torsional_thre):
        t = self.t[Link.BENDING_Y]
        t2 = self.t2
        # vec = (1.0/24)*np.array([0,0,0, t**3, -2*t**2, t])
        vec = (1.0/24)*np.array([0,0,0, t2**3, -2*t2**2, t2])
        vec[0] -= I_torsional_thre
        ans = solve_nth_degree_equation(vec)
        logger.debug("ans: " + str(ans))
        ans = self.get_min_real(ans)
        logger.debug("width: " + str(ans))
        return ans

    @deco
    def min_thickness_from_bending_x(self, I_bending_thre):
        b = self.b[Link.BENDING_X]
        tc = self.tc
        # vec = (1.0/12)*np.array([0, 7*b**3, -18*b**2, 20*b, -8]) # common t
        # b(2t)^3 + tc(b-2t)^3
        vec = (1.0/12)*np.array([tc*(b**3), -6*tc*(b**2), 12*tc*b, 8*(b-tc)]) # use tc
        vec[0] -= I_bending_thre
        ans = solve_nth_degree_equation(vec)
        logger.debug("ans: " + str(ans))
        ans = self.get_min_real(ans)
        logger.debug("thickness: " + str(ans))
        return ans

    @deco
    def min_thickness_from_bending(self, I_bending_thre):
        b = self.b[Link.BENDING_Y]
        tc = self.tc
        # vec = (1.0/12)*np.array([0, 7*b**3, -18*b**2, 20*b, -8]) # common t
        # b^4 - (b-tc)(b-2t)^3
        vec = (1.0/12)*np.array([tc*(b**3), 6*(b-tc)*(b**2), -12*(b-tc)*b, 8*(b-tc)]) # use tc
        vec[0] -= I_bending_thre
        ans = solve_nth_degree_equation(vec)
        logger.debug("ans: " + str(ans))
        ans = self.get_min_real(ans)
        logger.debug("thickness: " + str(ans))
        return ans

    @deco
    def min_thickness_from_torsional(self, I_torsional_thre):
        b = self.b[Link.TORSIONAL]
        tc = self.tc

        # # only warping torsion
        # vec = (1.0/24)*np.array([0, b**5, -2*(b**4), b**3])
        # vec[0] -= I_torsional_thre

        # # only pure torsion (open cross-section)
        # vec = np.array([0, 0, 0, b]) # approximate
        # vec = np.array([0, 0, 0, b, -2.0/3])
        # vec[0] -= I_torsional_thre

        # only pure torsion (closed cross-section)
        # vec = (1.0/12)*np.array([0, 9*(b**3), -18*(b**2), 21*b, -10]) # common t
        vec = (1.0/12)*np.array([tc*(b**3)+(tc**3)*b, 2*(b-tc)*(4*(b**2)+b*tc+tc**2), -12*(b-tc)*b, 8*(b-tc)]) # use tc
        vec[0] -= I_torsional_thre

        ans = solve_nth_degree_equation(vec)
        logger.debug("ans: " + str(ans))
        ans = self.get_min_real(ans)
        logger.debug("thickness: " + str(ans))
        return ans

    # def cross_section_area(self, width):
    #     # return width**2 - (width-self.t)*(width-2*self.t)
    #     return width**2 - (width-self.t)*(width-2*self.t2)

    def cross_section_area(self):
        # return self.b**2 - (self.b-self.t)*(self.b-2*self.t) # common t
        # return self.b**2 - (self.b-self.t)*(self.b-2*self.t2)
        return self.b**2 - (self.b-self.tc)*(self.b-2*self.t) # use tc

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
        # self.min_mass_list = np.array([[0]*len(Link.MOMENT_TYPE) for i in range(self.joint_path.numJoints())])
        self.min_mass_list = np.array([ [[0]*len(Link.MOMENT_TYPE) for j in range(self.joint_path.numJoints())] for i in range(len(self.link_shape_list)) ])

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
                self.calc_min_link_mass(joint_idx+1,do_plot=do_plot,save_plot=False,fname=fname,save_model=save_model,do_wait=do_wait,tm=tm)
            else:
                for link_shape_idx,link_shape in enumerate(self.link_shape_list):
                    for link_idx in [2,3]:
                        target_link_name = self.joint_path.joint(link_idx).name()

                        target_joint_name = self.joint_path.joint(link_idx).name()
                        max_proximal_wrench,min_proximal_wrench = self.calc_instant_max_frame_load_wrench(target_joint_name,coord_link_name=target_link_name,do_plot=do_plot,save_plot=False,fname=fname,save_model=save_model,do_wait=do_wait,tm=tm)

                        target_joint_name = self.joint_path.joint(link_idx+1).name()
                        max_distal_wrench,min_distal_wrench = self.calc_instant_max_frame_load_wrench(target_joint_name,coord_link_name=target_link_name,do_plot=do_plot,save_plot=False,fname=fname,save_model=save_model,do_wait=do_wait,tm=tm)

                        # print link_idx
                        # print min_proximal_wrench
                        # print max_distal_wrench
                        # print self.min_mass_list[link_shape_idx]
                        # print link_shape.link_type()
                        self.min_mass_list[link_shape_idx][link_idx] = np.vstack([self.min_mass_list[link_shape_idx][link_idx], link_shape.calculate_min_mass(min_proximal_wrench, max_distal_wrench)]).max(axis=0)

        if joint_idx == 0:
            # logger.critical(Fore.YELLOW+" min mass:"+str(self.min_mass_list.tolist())+Style.RESET_ALL)
            for link_shape,mass_list in zip(self.link_shape_list, self.min_mass_list):
                logger.critical(Fore.YELLOW+link_shape.link_type()+"'s min mass:")
                logger.critical(str(mass_list.tolist()))
                logger.critical(str(mass_list.max(axis=1))+Style.RESET_ALL)
        return self.min_mass_list

def test_link_mass_graph_generation():
    logger.setLevel(CRITICAL)
    # moment_array,_ = np.mgrid[10:1000:10,0:3] # 10->1000
    moment_array,_ = np.mgrid[140:1000:10,0:3] # 10->1000
    # moment_array,_ = np.mgrid[120:1000:10,0:3] # 10->1000
    # moment_array,_ = np.mgrid[400:800:10,0:3]

    sqlink = SquarePipeLink()
    hslink = HSectionLink()

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

def test_min_link_mass():
    logger.critical(Fore.BLUE+"test_min_link_mass()"+Style.RESET_ALL)

    package_path = roslib.packages.get_pkg_dir("structure_analyzer")
    model_path = os.path.join(package_path, "models")

    # step_angle=10; do_wait=False; tm=0; do_plot=False
    step_angle=20; do_wait=False; tm=0; do_plot=False
    step_angle_list = [20,20,20,20,360,360]

    # joint_range_list = [(-30,60),(-120,55),(-90,90), (0,0),(0,150),(0,0) ,(-60,60),(-80,75),(0,0)] # set full range to all joint
    joint_range_list = [(0,60),(0,80),(0,0), (0,0),(0,90),(0,0) ,(0,0),(0,0),(0,0)]
    # joint_range_list = [(0,60),(0,120),(0,90), (0,0),(0,0),(0,0) ,(0,0),(0,0),(0,0)] # hip only/half range
    # joint_range_list = [(60,60),(55,55),(-90,-90), (0,0),(90,90),(0,0) ,(0,0),(0,0),(0,0)] # hip only/fix

    link_shape_list = [SquarePipeLink(), HSectionLink()]
    # link_shape_list = [SquarePipeLink()]
    # link_shape_list = [None,None,SquarePipeLink(), SquarePipeLink(),None]

    global analyzer

    logger.critical(Fore.BLUE+"Serial rotary drive joint"+Style.RESET_ALL)
    joint_configuration_str="x-y-z_y_y-x"
    logger.critical(Fore.RED+joint_configuration_str+Style.RESET_ALL)
    analyzer = LinkDeflectionAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], link_shape_list, joint_range_list=joint_range_list, step_angle_list=step_angle_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_configuration_str+".wrl"))
    analyzer.calc_min_link_mass(do_wait=do_wait, tm=tm, do_plot=do_plot)

    joint_configuration_str="x-y-z_y_x-y"
    # logger.critical(Fore.RED+joint_configuration_str+Style.RESET_ALL)
    # analyzer = LinkDeflectionAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], link_shape_list, joint_range_list=joint_range_list, step_angle_list=step_angle_list,
    #                                    end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_configuration_str+".wrl"))
    # analyzer.calc_min_link_mass(do_wait=do_wait, tm=tm, do_plot=do_plot)
    logger.critical("")


    # no model
    joint_configuration_str="x-z-y_y_y-x"
    # logger.critical(Fore.RED+joint_configuration_str+Style.RESET_ALL)
    # analyzer = LinkDeflectionAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], link_shape_list, joint_range_list=joint_range_list, step_angle_list=step_angle_list,
    #                                    end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_configuration_str+".wrl"))
    # analyzer.calc_min_link_mass(do_wait=do_wait, tm=tm, do_plot=do_plot)

    # no model
    joint_configuration_str="x-z-y_y_x-y"
    # logger.critical(Fore.RED+joint_configuration_str+Style.RESET_ALL)
    # analyzer = LinkDeflectionAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], link_shape_list, joint_range_list=joint_range_list, step_angle_list=step_angle_list,
    #                                    end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_configuration_str+".wrl"))
    # analyzer.calc_min_link_mass(do_wait=do_wait, tm=tm, do_plot=do_plot)
    logger.critical("")


    joint_configuration_str="y-x-z_y_y-x"
    logger.critical(Fore.RED+joint_configuration_str+Style.RESET_ALL)
    analyzer = LinkDeflectionAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], link_shape_list, joint_range_list=joint_range_list, step_angle_list=step_angle_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_configuration_str+".wrl"))
    analyzer.calc_min_link_mass(do_wait=do_wait, tm=tm, do_plot=do_plot)

    joint_configuration_str="y-x-z_y_x-y"
    # logger.critical(Fore.RED+joint_configuration_str+Style.RESET_ALL)
    # analyzer = LinkDeflectionAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], link_shape_list, joint_range_list=joint_range_list, step_angle_list=step_angle_list,
    #                                    end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_configuration_str+".wrl"))
    # analyzer.calc_min_link_mass(do_wait=do_wait, tm=tm, do_plot=do_plot)
    logger.critical("")


    # no model
    joint_configuration_str="y-z-x_y_y-x"
    # logger.critical(Fore.RED+joint_configuration_str+Style.RESET_ALL)
    # analyzer = LinkDeflectionAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], link_shape_list, joint_range_list=joint_range_list, step_angle_list=step_angle_list,
    #                                    end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_configuration_str+".wrl"))
    # analyzer.calc_min_link_mass(do_wait=do_wait, tm=tm, do_plot=do_plot)

    # no model
    joint_configuration_str="y-z-x_y_x-y"
    # logger.critical(Fore.RED+joint_configuration_str+Style.RESET_ALL)
    # analyzer = LinkDeflectionAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], link_shape_list, joint_range_list=joint_range_list, step_angle_list=step_angle_list,
    #                                    end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_configuration_str+".wrl"))
    # analyzer.calc_min_link_mass(do_wait=do_wait, tm=tm, do_plot=do_plot)
    logger.critical("")

    joint_configuration_str="z-x-y_y_y-x"
    logger.critical(Fore.RED+joint_configuration_str+Style.RESET_ALL)
    analyzer = LinkDeflectionAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], link_shape_list, joint_range_list=joint_range_list, step_angle_list=step_angle_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_configuration_str+".wrl"))
    analyzer.calc_min_link_mass(do_wait=do_wait, tm=tm, do_plot=do_plot)

    joint_configuration_str="z-x-y_y_x-y"
    # logger.critical(Fore.RED+joint_configuration_str+Style.RESET_ALL)
    # analyzer = LinkDeflectionAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], link_shape_list, joint_range_list=joint_range_list, step_angle_list=step_angle_list,
    #                                    end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_configuration_str+".wrl"))
    # analyzer.calc_min_link_mass(do_wait=do_wait, tm=tm, do_plot=do_plot)
    logger.critical("")


    joint_configuration_str="z-y-x_y_y-x"
    logger.critical(Fore.RED+joint_configuration_str+Style.RESET_ALL)
    analyzer = LinkDeflectionAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], link_shape_list, joint_range_list=joint_range_list, step_angle_list=step_angle_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_configuration_str+".wrl"))
    analyzer.calc_min_link_mass(do_wait=do_wait, tm=tm, do_plot=do_plot)

    # joint_configuration_str="z-y-x_y_x-y"
    # analyzer = LinkDeflectionAnalyzer([(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)], link_shape_list, joint_range_list=joint_range_list, step_angle_list=step_angle_list,
    #                                    end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_configuration_str+".wrl"))
    # analyzer.calc_min_link_mass(do_wait=do_wait, tm=tm, do_plot=do_plot)
    logger.critical("")


    # Serial linear
    logger.critical(Fore.BLUE+"Serial linear drive joint"+Style.RESET_ALL)
    joint_configuration_str="y-x-z_y_y-x"
    logger.critical(Fore.RED+joint_configuration_str+Style.RESET_ALL)
    analyzer = LinkDeflectionAnalyzer([(0,0),(0,0),(1,0),(0,0),(0,0),(0,0)], link_shape_list, joint_range_list=joint_range_list, step_angle_list=step_angle_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_configuration_str+".wrl"))
    analyzer.calc_min_link_mass(do_wait=do_wait, tm=tm, do_plot=do_plot)

    joint_configuration_str="z-x-y_y_y-x"
    logger.critical(Fore.RED+joint_configuration_str+Style.RESET_ALL)
    analyzer = LinkDeflectionAnalyzer([(0,0),(0,0),(1,0),(0,0),(0,0),(0,0)], link_shape_list, joint_range_list=joint_range_list, step_angle_list=step_angle_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_configuration_str+".wrl"))
    analyzer.calc_min_link_mass(do_wait=do_wait, tm=tm, do_plot=do_plot)

    joint_configuration_str="z-x-y_y_y-x"
    logger.critical(Fore.RED+joint_configuration_str+Style.RESET_ALL)
    analyzer = LinkDeflectionAnalyzer([(0,0),(0,0),(1,0),(1,0),(0,0),(0,0)], link_shape_list, joint_range_list=joint_range_list, step_angle_list=step_angle_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_configuration_str+".wrl"))
    analyzer.calc_min_link_mass(do_wait=do_wait, tm=tm, do_plot=do_plot)

    joint_configuration_str="z-y-x_y_y-x"
    logger.critical(Fore.RED+joint_configuration_str+Style.RESET_ALL)
    analyzer = LinkDeflectionAnalyzer([(0,0),(0,0),(1,0),(0,0),(0,0),(0,0)], link_shape_list, joint_range_list=joint_range_list, step_angle_list=step_angle_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_configuration_str+".wrl"))
    analyzer.calc_min_link_mass(do_wait=do_wait, tm=tm, do_plot=do_plot)

    # Parallel linear
    logger.critical(Fore.BLUE+"Parallel linear drive joint"+Style.RESET_ALL)
    joint_configuration_str="z-x-y_y_y-x"
    logger.critical(Fore.RED+joint_configuration_str+Style.RESET_ALL)
    analyzer = LinkDeflectionAnalyzer([(0,0),(0,0),(2,0),(0,0),(0,0),(0,0)], link_shape_list, joint_range_list=joint_range_list, step_angle_list=step_angle_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_configuration_str+".wrl"))
    analyzer.calc_min_link_mass(do_wait=do_wait, tm=tm, do_plot=do_plot)

    joint_configuration_str="z-y-x_y_y-x"
    logger.critical(Fore.RED+joint_configuration_str+Style.RESET_ALL)
    analyzer = LinkDeflectionAnalyzer([(0,0),(0,0),(2,0),(0,0),(0,0),(0,0)], link_shape_list, joint_range_list=joint_range_list, step_angle_list=step_angle_list,
                                       end_link_name="JOINT5", robot_model_file=os.path.join(model_path,"universal-joint-robot_"+joint_configuration_str+".wrl"))
    analyzer.calc_min_link_mass(do_wait=do_wait, tm=tm, do_plot=do_plot)

if __name__ == '__main__':
    sla.initialize_plot_interface()
    test_min_link_mass()

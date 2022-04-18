# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 20:49:05 2022

@author: jnguy
"""

from tracemalloc import start
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 

import seaborn as sns
import glob 
import pandas as pd
import os
import math
import re

import random
from sympy import S, symbols, printing
import numpy as np

def save_image(image_name, fig):
    """saves image"""
    image_format = 'svg' # e.g .png, .svg, etc.
    # image_name = 'myimage.svfg'
    
    fig.savefig('images/'+image_name+'.svg', format=image_format, dpi=1200)
    
def get_all_csv_files(path_directory):
    """get all csv files based on some path directory"""
    return glob.glob(path_directory + "/*.csv")

def get_csv_filename(filename_dir):
    """return file names of csv and removes the directory path"""
    return os.path.basename(os.path.normpath(filename_dir))

def compile_to_single_df(csv_files):
    overall_df = pd.DataFrame()
    for i,file_directory in enumerate(csv_files):
        csv_df = pd.read_csv(file_directory)
        
    return overall_df


def return_df_list(all_csv_files):
    """return df list and group by heuristics and number of drones"""
    df_list = []
    for i,file_directory in enumerate(all_csv_files):
        df = pd.read_csv(file_directory)
        #df = df.groupby(['min_max_weighted_heuristics', 'collision val'])
        df_list.append(df)

    return df_list

class Plot():
    def __init__(self, x_bounds, y_bounds, z_bounds):
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_bounds = z_bounds

    def set_color_gradient(self, color_num, flip=False):
        seaborn_gradients = ['mako', 'flare','crest', 'magma', 'rocket', 'viridis']
        return seaborn_gradients[color_num]

    def set_color_divergent(self,color_num, flip=False):
        
        if flip == True:
            seaborn_divergents = ['vlag_r', 'icefire_r', 'Spectral_r', 'coolwarm_r']
        else:
            seaborn_divergents = ['vlag', 'icefire', 'Spectral', 'coolwarm']    
        return seaborn_divergents[color_num]

    def set_3d_axis(self):
        """
        set 3d axis based on x_bounds and y_bounds based on list
        """
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.set_xlim(self.x_bounds)
        ax.set_ylim(self.y_bounds)
        ax.set_zlim(self.z_bounds)
        
        return fig,ax
                
    
    def set_2d_axis(self):
        """
        set 2d axis based on x_bounds and y_bounds based on list
        """
        fig = plt.figure()
        ax = plt.axes()
        ax.set_xlim(self.x_bounds)
        ax.set_ylim(self.y_bounds)
        
        return fig,ax
    
    def plot_multiple_values(
            self,x_list, y_list, x_label, y_label,plot_labels, gradient_color=True):
        """plot multiple values"""
        fig,ax = self.set_2d_axis()
    
        if gradient_color== True:
            color_pallet = self.set_color_divergent(2,flip=True)
            colors = sns.color_palette(color_pallet, n_colors=len(y_list))
    
        for x_vals, y_vals in zip(x_list, y_list):
            ax.plot(x_vals, y_vals)
        
        ax.tick_params(direction='in', right=True, left=True, labelsize = 10)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.legend()
        
    
    def plot_multiple_scatter_vals(
            self,x_list, y_list, x_label, y_label,plot_labels, gradient_color=True):
        """plot scatter values"""
        fig,ax = self.set_2d_axis()
        
        if gradient_color== True:
            color_pallet = self.set_color_gradient(2,flip=True)
            colors = sns.color_palette(color_pallet, n_colors=len(y_list))
            #print(colors)
            
        for i,(x_vals, y_vals, label) in enumerate(zip(x_list, y_list, plot_labels)):
            ax.scatter(x_vals, y_vals, color= colors[i], label=str(label))
    
        ax.tick_params(direction='in', right=True, left=True, labelsize = 10)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.legend()
            
    def plot_3d(
            self,x_list,y_list,z_list):
        fig,ax = self.set_3d_axis()
        
        ax.plot(x_list,y_list,z_list)
        
    def plot_multiple_lines(self,x_list, y_lists, line_labels, xlabel, ylabel):
        
        fig, ax = plt.subplots(constrained_layout=False)

        for i, y_vals in enumerate(y_lists):
            ax.plot(x_list, y_vals, label=line_labels[i])
            #plt.show()
        # secax = ax.secondary_xaxis('top', functions=())
        # secax.set_xlabel('Height (m)')
        ax.tick_params(direction='in', right=True, left=True, labelsize = 10)
        ax.autoscale(enable=True)    
        #ax.set_ylim([-25,20])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.legend()
            
def get_mean_std_radius(col_x_name, col_y_name):
    mean_r = []
    std_r = []
    
    """return a list of mean radius and standard deviations"""
    est_x_list = [df[col_x_name] for df in df_list]
    est_y_list = [df[col_y_name] for df in df_list]

    mean_x_list = [abs(df[col_x_name].mean()) for df in df_list]
    mean_y_list = [abs(df[col_y_name].mean()) for df in df_list]
    
    std_x_list = [df[col_x_name].std() for df in df_list]
    std_y_list = [df[col_y_name].std() for df in df_list]
    
    for x, y in zip(mean_x_list, mean_y_list):
        mean_r.append(math.sqrt(x**2+y**2))
    
    for x,y in zip(std_x_list, std_y_list):
        std_r.append(math.sqrt(x**2+y**2))
        
    return mean_r, std_r
 
def insert_mag_error(df_list, col_name1, col_name2, new_col):
    for df in df_list:
        df.dropna()
        true_x = df["true tag x"] - df['quad x'] 
        true_y = df["true tag y"] - df['quad y']
        df["true rel x"] = true_x
        df["true rel y"] = true_y
        error_x = df[col_name1] - true_x - 5.0
        error_y = df[col_name2] - true_y - 5.0
        vals = np.sqrt(error_x.to_numpy()**2 + error_y.to_numpy()**2)
        df[new_col]= vals
        
    return df_list

def compute_magnitude(df, col_name1, col_name2):
    true_x = df["true tag x"] - df['quad x'] 
    true_y = df["true tag y"] - df['quad x'] 
    vals = np.sqrt(true_x.to_numpy()**2 + true_y.to_numpy()**2)
    
    return vals

def get_kf_vals(kf_data):
    """parses out the kalman filter data out from the csv"""
    x_list = []
    y_list = []
    z_list = []
    for val in kf_data:
        patn = re.sub(r"[\([{})\]]", "", val)
        coords = [float(i) for i in patn.split(',')]
        x_list.append(coords[0])
        y_list.append(coords[1])
        z_list.append(coords[2])
        
    return x_list, y_list, z_list
    
if __name__== '__main__':
    
    folder_name = "apriltag_R"
    folder_name = "landing_test_data"
    folder_name = "tracking"
    path = os.getcwd() + "/"+folder_name
    all_csv_files = get_all_csv_files(path)
    df_list = return_df_list(all_csv_files)
    
    qe_names = []
    for i in range(1, 8):
        qe_name = "/qe"+str(i)+"_position"
        qe_names.append(qe_name)
    
    qe_error_covar = []
    for i in range(1, 8):
        qe_name = "/qe"+str(i)+"_error_covar"
        qe_error_covar.append(qe_name)
            
    kf_dict_list = []
    kf_p_error_list = []
    
    for df in df_list:
        df.dropna()
        kf_dict = {}
        for qe_name in qe_names:
            x_vals, y_vals, z_vals = get_kf_vals(df[qe_name])
            kf_dict[qe_name] = [x_vals, y_vals, z_vals]
        kf_dict_list.append(kf_dict)

    for df in df_list:
        df.dropna()
        p_error_dict = {}
        for qe_name in qe_error_covar:
            x_vals, y_vals, z_vals = get_kf_vals(df[qe_name])
            p_error_dict[qe_name] = [x_vals, y_vals, z_vals]
        kf_p_error_list.append(p_error_dict)
        
    df_list = insert_mag_error(df_list, 'tag x', 'tag y', 'mag error')
    df_list = insert_mag_error(df_list, 'kftag x', 'kftag y', 'kf mag error')
    
    # time_vec = [df['time'] for df in df_list]
    
    #fit line equations is y = 0.00232*n + 0.0295
    # line_labels = [5, 10, 15, 20, 25, 30, 35]
    # x_sample = np.arange(0,40)
    # p = np.polyfit(line_labels, std_r, 1)
    # slope = p[0]
    # intercept = p[1]
    # fit_eq = slope*x_sample + intercept
    
    #poly = sum(S("{:6.2f}".format(v))*x**i for i, v in enumerate(p[::-1]))
    #eq_latex = printing.latex(poly)
    #plt.close('all') 
    #%% Line fit process
    plt.close('all')
    line_labels = [5, 10,15,20,25, 30,35]
    # plot_stuff = Plot([0,40], [0,0.25], [0,0.25])
    
    # plot_stuff.plot_multiple_scatter_vals(
    #     line_labels, std_r, 'Height from Tag(m)', 'Standard Deviation(m)', line_labels)
    
    # plt.plot(x_sample, fit_eq, alpha=0.5, label='Linear Fit')
    
    #plt.plot(label="${}$".format(eq_latex))
    
    #x_bounds = [0,25]
    x_bounds = [-0.20, 0.15]
    y_bounds = [0, 0.15]
    
    #%%plotting kalman filter overall results
    # for df in df_list:
    #     plot_stuff = Plot(x_bounds, y_bounds, y_bounds)
    #     line_labels = ['true rel x', 'raw estimate', 'kf estimate']
    #     #y_lists = [df['mag error'], df['kf mag error']]
    #     y_lists = [df['true rel x'], df['tag x'] - 5.0, df['kftag x']- 5.0]
    #     xlabel = 'Time (s)'
    #     ylabel = 'Distance Error (m)'
    #     plot_stuff.plot_multiple_lines(df['time'],y_lists, line_labels, xlabel, ylabel)
    
    
    # plot performance of x
    test_num = 0
    kf_dict = kf_dict_list[test_num]
    df = df_list[test_num]
    for key, value in kf_dict.items():
        plot_stuff = Plot(x_bounds, y_bounds, y_bounds)
        line_labels = ['true rel x', 'raw estimate x', 'kf estimate']
        #y_lists = [df['mag error'], df['kf mag error']] 
        
        #subtracted 5.0 because quad is at 5.0 true position
        kf_x = np.array(value[0]) - 5.0
        y_lists = [df['true rel x'], df['tag x'] - 5.0, kf_x]
        
        # y_lists = [df['true rel y'], df['tag y'] - 3.0, np.array(value[1])- 3.0]
        xlabel = 'Time (s)'
        ylabel = 'Distance (m)'
        plot_stuff.plot_multiple_lines(df['time'],y_lists, line_labels, xlabel, ylabel)

    y_lists = []
    for key, value in kf_dict.items():
        kf_x = np.array(value[0]) - 5.0
        y_lists.append(kf_x)
        
    plot_stuff = Plot(x_bounds, y_bounds, y_bounds)
    xlabel = 'Time (s)'
    ylabel = 'Distance (m)'
    plot_stuff.plot_multiple_lines(df['time'],y_lists, qe_names, xlabel, ylabel)
        
    #%% plotting P Error covariance matrix
    y_lists = []
    for key, value in p_error_dict.items():
        
        y_lists.append(value[1][0:250])
    
    plot_stuff = Plot(x_bounds, y_bounds, y_bounds)
    xlabel = 'Time (s)'
    ylabel = 'Error'
    plot_stuff.plot_multiple_lines(df['time'][0:250],y_lists, qe_names, xlabel, ylabel)
        
    # plt.plot(df['time'], df['mag error'], color='blue')
    # plt.show()
    # plt.plot(df['time'], df['kf mag error'], color='red')
    
    # plot_stuff.plot_multiple_scatter_vals(
    #     est_x_list, est_y_list, 'X Position (m)','Y Position (m)', line_labels,
    #     gradient_color=True)
    
    #%% Q analysis and design for Kalman Filter
    
    
    #%% Plot and compare landing
    # x_bounds = [-0.2, 0.2]
    # y_bounds = [-0.2,0.2]
    # z_bounds = [0,40]
    # plot_stuff = Plot(x_bounds, y_bounds, z_bounds)
    # # x_label = 'X Position (m)'
    # # y_label = 'Y Position (m)'
    # # plot_stuff.plot_multiple_scatter_vals(est_x_list, est_y_list, x_label, y_label, line_labels)
    # plot_stuff.plot_3d(x_list=df['kftag x'], y_list=df['kftag y'], z_list=df['quad z'])
    #plt.scatter(df["true tag x"], df["true tag y"])
    
    
    
    
    
    
    
    
    
    



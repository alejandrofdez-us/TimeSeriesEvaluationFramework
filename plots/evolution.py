from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt
import pandas

def generate_evolution_figures(args):
    plot_array = []
    
    plot_array.append(generate_figure_from_df(args["ori_data_sample"], args["generated_data_sample_df"]))
    for index, column in enumerate(args["header"]):
        plot_array.append(generate_figures_by_column(index, column, args["ori_data_sample"], args["generated_data_sample"]))
        
    return plot_array

def generate_figure_from_df(ori_data_sample, generated_data_sample_df):
    ori_data_sample_df = pandas.DataFrame(ori_data_sample, columns=[f'{column_name}_original' for column_name in
                                                                    generated_data_sample_df.columns])
    plt.rcParams["figure.figsize"] = (18, 3)
    fig, ax = plt.subplots(1)

    ori_data_sample_df.plot(ax=ax, style='--', color='darkgrey')
    plt.xlim([0, generated_data_sample_df.shape[0]])

    plt.title("original columns")
    plt.legend(loc='best')
    plt.xlabel('time')
    plt.ylabel('y_label')

    plot_tuple = (fig, ax)

    return plot_tuple

def generate_figures_by_column(column_number, column_name, ori_data_sample, generated_data_sample):


    return create_figure(ori_column_values_array=[ori_data_sample[:, column_number]],
                  generated_column_values=generated_data_sample[:, column_number], column_name=column_name)
    

def create_figure(ori_column_values_array, generated_column_values, column_name, axis=None):
    plt.rcParams["figure.figsize"] = (18, 3)
    fig, ax = plt.subplots(1)
    i = 1
    cycol = cycle('grcmk')

    for ori_column_values in ori_column_values_array:
        plt.plot(ori_column_values, c=next(cycol), label=f'Original_{i}', linewidth=1)
        i += 1

    plt.plot(generated_column_values, c="blue", label="Synthetic", linewidth=2)
    if axis is not None:
        plt.axis(axis)
    else:
        plt.xlim([0, len(ori_column_values_array[0])])

    plt.title(f'{column_name} original vs synthetic')
    plt.xlabel('time')
    plt.ylabel(column_name)
    ax.legend()

    plot_tuple = (fig, ax)

    return plot_tuple

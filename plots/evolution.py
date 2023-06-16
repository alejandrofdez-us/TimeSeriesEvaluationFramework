from itertools import cycle

import matplotlib.pyplot as plt
import pandas

from plots.plot import Plot

class Evolution(Plot):
    def generate_figures(self, args):
        plot_array = self.__generate_evolution_figures(args)
        return plot_array

    def __generate_evolution_figures(self, args):
        plot_array = []
        
        plot_array.append(self.__generate_figure_from_df(args["ori_data_sample"], args["generated_data_sample_df"]))
        for index, column in enumerate(args["header"]):
            plot_array.append(self.__generate_figures_by_column(index, column, args["ori_data_sample"], args["generated_data_sample"]))
            
        return plot_array

    def __generate_figure_from_df(self, ori_data_sample, generated_data_sample_df):
        ori_data_sample_df = pandas.DataFrame(ori_data_sample, columns=[f'{column_name}_TS_1' for column_name in
                                                                        generated_data_sample_df.columns])
        plt.rcParams["figure.figsize"] = (18, 3)
        fig, ax = plt.subplots(1)

        ori_data_sample_df.plot(ax=ax, color='darkgrey')

        new_column_names = {col: col + "_TS_2" for col in generated_data_sample_df.columns}
        generated_data_sample_df = generated_data_sample_df.rename(columns=new_column_names)
        generated_data_sample_df.plot(ax=ax, style='--')

        plt.xlim([0, generated_data_sample_df.shape[0]])

        plt.title("all_columns")
        plt.legend(loc='best')
        plt.xlabel('time')
        plt.ylabel('y_label')

        plot_tuple = (fig, ax)

        return plot_tuple

    def __generate_figures_by_column(self, column_number, column_name, ori_data_sample, generated_data_sample):

        return self.__create_figure(ori_column_values_array=[ori_data_sample[:, column_number]],
                    generated_column_values=generated_data_sample[:, column_number], column_name=column_name)
        

    def __create_figure(self, ori_column_values_array, generated_column_values, column_name, axis=None):
        plt.rcParams["figure.figsize"] = (18, 3)
        fig, ax = plt.subplots(1)
        i = 1
        cycol = cycle('grcmk')

        for ori_column_values in ori_column_values_array:
            plt.plot(ori_column_values, c=next(cycol), label=f'TS_1', linewidth=1)
            i += 1

        plt.plot(generated_column_values, c="blue", label="TS_2", linewidth=2)
        if axis is not None:
            plt.axis(axis)
        else:
            plt.xlim([0, len(ori_column_values_array[0])])

        plt.title(f'{column_name}_TS_1_vs_TS_2')
        plt.xlabel('time')
        plt.ylabel(column_name)
        ax.legend()

        plot_tuple = (fig, ax)

        return plot_tuple

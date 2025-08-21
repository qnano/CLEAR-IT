import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

def survival_kpplot(classifier_file, **kwargs):
    """
    Plot Kaplan-Meier survival curves for high and low classifier groups.

    Parameters:
    classifier_file (str): Path to the Excel file containing the classifier data.
    **kwargs: Keyword arguments for plot customization. Common options include:
              - figsize (tuple): Size of the plot.
              - fontsize (int): Font size for text elements.
              - title (str): Title of the plot.
              - xlabel (str): Label for the x-axis.
              - ylabel (str): Label for the y-axis.
              - xlim (tuple): Limits for the x-axis.
              - ylim (tuple): Limits for the y-axis.
              - legend_loc (str): Location of the legend.

    Returns:
    matplotlib.pyplot: Matplotlib pyplot object with the created survival plot.
    """
    # Plot configuration
    figsize = kwargs.get('figsize', (10, 6))
    fontsize = kwargs.get('fontsize', 10)
    title = kwargs.get('title', '')
    xlabel = kwargs.get('xlabel', 'Years')
    ylabel = kwargs.get('ylabel', 'Overall survival rate')
    xlim = kwargs.get('xlim', (-0.2, 14.2))
    xticks = kwargs.get('xticks',[])
    ylim = kwargs.get('ylim', (0, 1.04))
    yticks = kwargs.get('yticks',[])
    legend_loc = kwargs.get('legend_loc', 'best')
    time_unit = kwargs.get('time_unit','days')
    
    # Load and preprocess the data
    classifier_frame = pd.read_excel(classifier_file)
    classifier_frame = classifier_frame.set_index(classifier_frame.keys()[0])

    # Separate the data into high and low classifier groups
    T1 = classifier_frame['os'][classifier_frame['classifier'] == True]
    E1 = classifier_frame['osi'][classifier_frame['classifier'] == True]

    T2 = classifier_frame['os'][classifier_frame['classifier'] == False]
    E2 = classifier_frame['osi'][classifier_frame['classifier'] == False]

    plt.figure(figsize=figsize)
    
    # Plot the Kaplan-Meier curves
    kmf = KaplanMeierFitter()

    # Set correction factor for units of time
    if time_unit == 'days':
        cf = 365
    elif time_unit == 'months':
        cf = 12
    else:
        cf = 1
    
    # Fit and plot for high classifier group
    kmf.fit(T1/cf, event_observed=E1, label='classifier high, n=' + str(len(T1)))
    kmf.plot(ci_show=False, show_censors=True, color='tab:blue', censor_styles={"ms":8})

    # Perform log-rank test and fit/plot for low classifier group
    p_value = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2).p_value
    kmf.fit(T2/cf, event_observed=E2, label=f'classifier low, n={len(T2)}, p = {p_value:.4f}')
    kmf.plot(ci_show=False, show_censors=True, color='tab:orange', censor_styles={"ms":8})
    
    # Customize the plot
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.xlim(xlim)
    if xticks:plt.xticks(xticks)
    plt.xticks(fontsize=fontsize)
    plt.ylim(ylim)
    if yticks:plt.yticks(yticks)
    plt.yticks(fontsize=fontsize)
    plt.legend(loc=legend_loc, fontsize=fontsize)
    
    plt.tight_layout()
    
    return plt

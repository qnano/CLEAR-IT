import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


def survival_kpplot(classifier_file, **kwargs):
    """
    Plot Kaplan-Meier survival curves for high and low classifier groups
    and return tables containing the plotted data.

    Returns
    -------
    plt : matplotlib.pyplot object
    tables : dict[str, pd.DataFrame]
        - 'classifier high': KM step function for the high group
        - 'classifier low' : KM step function for the low group
          Each of these has columns:
              ['time', 'survival', 'n_risk', 'n_events', 'n_censored']
        - 'summary'        : per-group summary with columns:
              ['group', 'n', 'n_events', 'median_survival', 'logrank_p']
    """
    # Plot configuration
    figsize    = kwargs.get('figsize', (10, 6))
    fontsize   = kwargs.get('fontsize', 10)
    title      = kwargs.get('title', '')
    xlabel     = kwargs.get('xlabel', 'Years')
    ylabel     = kwargs.get('ylabel', 'Overall survival rate')
    xlim       = kwargs.get('xlim', (-0.2, 14.2))
    xticks     = kwargs.get('xticks', [])
    ylim       = kwargs.get('ylim', (0, 1.04))
    yticks     = kwargs.get('yticks', [])
    legend_loc = kwargs.get('legend_loc', 'best')
    time_unit  = kwargs.get('time_unit', 'days')

    # Load and preprocess the data
    classifier_frame = pd.read_excel(classifier_file)
    classifier_frame = classifier_frame.set_index(classifier_frame.columns[0])

    # Separate the data into high and low classifier groups
    T1 = classifier_frame['os'][classifier_frame['classifier'] == True]
    E1 = classifier_frame['osi'][classifier_frame['classifier'] == True]

    T2 = classifier_frame['os'][classifier_frame['classifier'] == False]
    E2 = classifier_frame['osi'][classifier_frame['classifier'] == False]

    # Set correction factor for units of time (to match x-axis)
    if time_unit == 'days':
        cf = 365
    elif time_unit == 'months':
        cf = 12
    else:
        cf = 1

    # Prepare plot
    plt.figure(figsize=figsize)

    tables = {}

    # High classifier group
    kmf_high = KaplanMeierFitter()
    label_high = f'classifier high, n={len(T1)}'
    kmf_high.fit(T1 / cf, event_observed=E1, label=label_high)
    kmf_high.plot(ci_show=False, show_censors=True,
                  color='tab:blue', censor_styles={"ms": 8})

    # Build KM table for high group
    et_high = kmf_high.event_table.reset_index()   # column 'event_at'
    sf_high = kmf_high.survival_function_.reset_index()  # column 'timeline' and label_high
    merged_high = et_high.merge(sf_high,
                                left_on='event_at',
                                right_on='timeline',
                                how='left')

    high_df = pd.DataFrame({
        'time':       merged_high['timeline'],      # already in scaled units (T/cf)
        'survival':   merged_high[label_high],
        'n_risk':     merged_high['at_risk'],
        'n_events':   merged_high['observed'],
        'n_censored': merged_high['censored'],
    })
    tables['classifier high'] = high_df

    # Log-rank test
    p_value = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2).p_value

    # Low classifier group
    kmf_low = KaplanMeierFitter()
    label_low = f'classifier low, n={len(T2)}, p = {p_value:.4f}'
    kmf_low.fit(T2 / cf, event_observed=E2, label=label_low)
    kmf_low.plot(ci_show=False, show_censors=True,
                 color='tab:orange', censor_styles={"ms": 8})

    # Build KM table for low group
    et_low = kmf_low.event_table.reset_index()
    sf_low = kmf_low.survival_function_.reset_index()
    merged_low = et_low.merge(sf_low,
                              left_on='event_at',
                              right_on='timeline',
                              how='left')

    low_df = pd.DataFrame({
        'time':       merged_low['timeline'],
        'survival':   merged_low[label_low],
        'n_risk':     merged_low['at_risk'],
        'n_events':   merged_low['observed'],
        'n_censored': merged_low['censored'],
    })
    tables['classifier low'] = low_df

    # Summary sheet
    summary = pd.DataFrame([
        {
            'group': 'classifier high',
            'n': len(T1),
            'n_events': int(E1.sum()),
            'median_survival': kmf_high.median_survival_time_,
            'logrank_p': p_value,
        },
        {
            'group': 'classifier low',
            'n': len(T2),
            'n_events': int(E2.sum()),
            'median_survival': kmf_low.median_survival_time_,
            'logrank_p': p_value,
        },
    ])
    tables['summary'] = summary

    # Customize the plot
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.xlim(xlim)
    if xticks:
        plt.xticks(xticks)
    plt.xticks(fontsize=fontsize)
    plt.ylim(ylim)
    if yticks:
        plt.yticks(yticks)
    plt.yticks(fontsize=fontsize)
    plt.legend(loc=legend_loc, fontsize=fontsize)

    plt.tight_layout()

    return plt, tables

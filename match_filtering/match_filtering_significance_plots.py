import numpy as np
import pickle
import pylab
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt

import numpy
from pycbc.waveform import get_td_waveform, get_fd_waveform
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.filter import matched_filter, resample_to_delta_t, highpass, highpass_fir, sigma
from pycbc.vetoes import power_chisq
from pycbc.events.ranking import newsnr
from pycbc.detector import Detector
from pycbc.frame import read_frame
from pycbc import frame

import pickle
import os

import argparse

script_dir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description="Load results pkl ")
parser.add_argument('--p', type=str, required=True, help="Path to .pkl")
args = parser.parse_args()


results = {}
with open(args.p, "rb") as f:
    results = pickle.load(f)

p_tmp=args.p.replace(".pkl", "")

detectors = list(results.keys())      

print(detectors)
print(results)

########################################  Signal Consistency and Significance power_chisq ##################


for det in detectors:
    if det in results:
        for chunk in results[det]:
            chisq = results[det][chunk]['chisq']
            peak_time = results[det][chunk]['peak_time']
            
            
            fig_chisq = go.Figure()
            
            # Add chi-squared plot for each detector
            fig_chisq.add_trace(go.Scatter(
                x=chisq.sample_times,
                y=chisq.numpy(),
                mode='lines',
                name=f'{det} χ²_r',
            ))

            # Layout customizations
            fig_chisq.update_layout(
                title="Signal Consistency and Significance (χ²_r)",
                xaxis_title="Time (s)",
                yaxis_title="χ²_r",
                hovermode="x unified",
                legend_title="Detectors",
                xaxis=dict(range=[peak_time - 0.15, peak_time + 0.15]),  # Zoom in around peak_time
                yaxis=dict(range=[0, 5]),
                showlegend=True,
            )

            # Save to HTML
            safe_chunk = str(chunk).replace("/", "_").replace("\\", "_")
            output_path = os.path.join(script_dir, f"./"+args.j+"_plots/html_significance_plots/power_chisq_plots/{det}_{safe_chunk}_chisq_r_interactive_plot.html")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig_chisq.write_html(output_path)


############### Re-weighted Signal-to-noise #############


# Calculate nsnr for each detector and chunk
for det in detectors:
    if det in results:
        for chunk in results[det]:
            results[det][chunk]['nsnr'] = newsnr(abs(results[det][chunk]['snr']), results[det][chunk]['chisq'])

# Now create interactive Plotly plots for different views
for w, title in [(7, 'Wide View'), (.15, 'Close to GW170814')]:

    for det in detectors:
        if det in results:
            for chunk in results[det]:

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=results[det][chunk]['snr'].sample_times,
                    y=results[det][chunk]['nsnr'],
                    mode='lines',
                    name=f'{det}'
                ))

                # Layout customization for the figure
                fig.update_layout(
                    title=title,
                    xaxis_title='Time (s)',
                    yaxis_title='Re-weighted Signal-to-Noise',
                    xaxis_range=[results[det][chunk]['peak_time'] - w, results[det][chunk]['peak_time'] + w],
                    yaxis_range=[0, 15],
                    hovermode='x unified',
                    legend_title='Detectors'
                )

                # Save to HTML
                safe_chunk = str(chunk).replace("/", "_").replace("\\", "_")
                output_path = os.path.join(script_dir, f"./{p_tmp}_plots/html_significance_plots/rsnr_plots/{det}_{safe_chunk}_reweighted_snr.html")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                fig.write_html(output_path)
            
                

############################# Calculating the background and significance
"""
# Calculate the time of flight between the Virgo detectors and each LIGO observatory
d = Detector("V1")

tof = {}
for det in ['E1', 'E2']:
    if det in results:
        #tof[det] = d.light_travel_time_to_detector(Detector(det))
        if(det == 'E1'):
            tof[det] = d.light_travel_time_to_detector(Detector('H1'))
        elif(det == 'E2'):
            tof[det] = d.light_travel_time_to_detector(Detector('L1'))
        

print("tof ", tof)
# Record the time of the peak in the LIGO observatories [ptime]

# Initialize Plotly figure

# Calculate time of flight and plot the data
for det in detectors:
    if det in results:
        # Skip Virgo detector (E3)
        if det != 'E3':
            for chunk in results[det]:
                # Calculate the peak time
                peak_time = results[det][chunk]['snr'].sample_times[results[det][chunk]['nsnr'].argmax()]
                results[det][chunk]['ptime'] = peak_time
                
                fig = go.Figure()

                # Add shaded region to indicate potential peak times in Virgo
                fig.add_shape(
                    type="rect",
                    x0=peak_time - tof[det],
                    x1=peak_time + tof[det],
                    y0=0,
                    y1=15,
                    fillcolor="rgba(0, 0, 255, 0.2)",
                    line=dict(width=0),
                    layer="below"
                )

                # Add SNR trace
                fig.add_trace(go.Scatter(
                    x=results[det][chunk]['snr'].sample_times,
                    y=results[det][chunk]['nsnr'],
                    mode='lines',
                    name=f'{det} Re-weighted SNR'
                ))

                # Update layout with labels, axis titles, and interactivity
                fig.update_layout(
                    title="Re-weighted Signal-to-Noise Ratio with Time-of-Flight Shading",
                    xaxis_title="Time (s)",
                    yaxis_title="Re-weighted Signal-to-Noise Ratio",
                    hovermode='x unified',
                    xaxis=dict(range=[results[det][chunk]['ptime'] - 0.05, results[det][chunk]['ptime'] + 0.10]),
                    yaxis=dict(range=[0, 15]),
                    legend_title="Detectors"
                )

                safe_chunk = str(chunk).replace("/", "_").replace("\\", "_")
                output_path = os.path.join(script_dir, f"./{p_tmp}_plots/html_significance_plots/tof_plots/{det}_{safe_chunk}_interactive_snr_plot_with_tof.html")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                fig.write_html(output_path)
                #fig.show()

# Calculate the span of time that a Virgo peak could in principle happen in from time of flight
# considerations.

# Initialize list for start and end times based on time-of-flight
start = []
end = []

if 'E1' in tof and 'E2' in tof:
    for det in detectors:
        if det in results:
            if det != 'E3':
                for chunk in results[det]:
                    if det == 'E1':
                        start.append(results['E1'][chunk]['ptime'] - tof['E1'])
                for chunk in results[det]:
                    if det == 'E2':
                        end.append(results['E2'][chunk]['ptime'] + tof['E2'])

    ic = 0
    for det in detectors:
        if det in results:
            for chunk in results[det]:
                if det == 'E3':
                    # Calculate window size for Virgo detector
                    # convert the times to indices along with how large the region is in number of samples

                    window_size = int((end[ic] - start[ic]) * results['E3'][chunk]['snr'].sample_rate)
                    sidx = int((start[ic] - results['E3'][chunk]['snr'].start_time) * results['E3'][chunk]['snr'].sample_rate)
                    ic += 1
                    eidx = sidx + window_size
                    si, ei = sorted([sidx, eidx])  # ensures start <= end

                    # Calculate the on-source peak re-weighted (newsnr) statistic value
                    onsource = 0

                    nsnr_slice = results['E3'][chunk]['nsnr'][si:ei]

                    if nsnr_slice.size > 0:  # Check if the array is not empty
                        onsource = nsnr_slice.max()

                    print('Virgo Peak has a statistic value of {}'.format(onsource))
                    window_size = abs(window_size)


                    ###################################


                    # Now that we've calculate the onsource peak, we should calculate the background peak values.
                    # We do this by chopping up the time series into chunks that are the same size as our
                    # onsource and repeating the same peak finding (max) procedure.

                    # Walk through the data in chunks and calculate the peak statistic value in each.

                    peaks = []
                    i = 0
                    while i + window_size < len(results['E3'][chunk]['nsnr']):
                        p = results['E3'][chunk]['nsnr'][i:i + window_size].max()
                        peaks.append(p)
                        i += window_size

                        # Skip past the onsource time
                        if abs(i - si) < window_size:
                            i += window_size * 2

                    peaks = np.array(peaks)

                    # Calculate the p-value
                    # The p-value is just the number of samples observed in the background with a
                    # value equal or higher than the onsource divided by the number of samples.
                    # We can make the mapping between statistic value and p-value using our background
                    # samples.
                    
                    pcurve = np.arange(1, len(peaks) + 1)[::-1] / float(len(peaks))
                    peaks.sort()

                    pvalue = (peaks > onsource).sum() / float(len(peaks))

                    # Create Plotly figure for p-value curve
                    fig = go.Figure()

                    # Plot off-source (Noise Background) peaks
                    fig.add_trace(go.Scatter(
                        x=peaks, y=pcurve, mode='markers', name='Off-source (Noise Background)', marker=dict(color='black')
                    ))

                    # Highlight the on-source peak with a red line
                    fig.add_trace(go.Scatter(
                        x=[onsource, onsource], y=[0, pvalue], mode='lines', name='On-source', line=dict(color='red', dash='dash')
                    ))

                    if peaks:
                        # Add horizontal line for p-value
                        fig.add_trace(go.Scatter(
                            x=[min(peaks), max(peaks)], y=[pvalue, pvalue], mode='lines', name='p-value', line=dict(color='red', dash='dash')
                        ))

                    # Update layout for better visualization
                    fig.update_layout(
                        title="p-value Curve for Signal Detection",
                        xaxis_title="Re-weighted Signal-to-noise",
                        yaxis_title="p-value",
                        yaxis=dict(type='log', range=[1e-3, 1e0]),
                        xaxis=dict(range=[2, 5]),
                        hovermode='closest',
                        legend_title="Detectors"
                    )


                    print(f"The p-value associated with the peak is {pvalue}")

                    safe_chunk = str(chunk).replace("/", "_").replace("\\", "_")
                    output_path = os.path.join(script_dir, f"./{p_tmp}_plots/html_significance_plots/pvalue_plots/{det}_{safe_chunk}_pvalue_plot.html")
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    fig.write_html(output_path)
                    #fig.show()
"""
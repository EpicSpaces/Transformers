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
########### Raw strain Data


for det in detectors:
    if det in results:
        for chunk in results[det]:
            strain = results[det][chunk]['strain']

            print(results[det][chunk]['template_name'], )

            fig = go.Figure()

            fig.add_trace(go.Scatter(x=strain.sample_times,
                                     y=strain.numpy(),
                                     mode='lines',
                                     name=f'{det}'))

            # Layout settings
            fig.update_layout(
                title='Interactive Strain Plot',
                xaxis_title='Time (s)',
                yaxis_title='Strain',
                hovermode='x unified'
            )

            
            # Save the plot to an HTML file
            
            safe_chunk = str(chunk).replace("/", "_").replace("\\", "_")
            
            output_path = os.path.join(script_dir, f"./{p_tmp}_plots/html_plots/strain_plots/{det}_{safe_chunk}_strain_plot.html")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.write_html(output_path)

#################################### PSD ########

for det in detectors:
    if det in results:
        for chunk in results[det]:
            psd = results[det][chunk]['psd']

            fig_psd = go.Figure()

            fig_psd.add_trace(go.Scatter(
                x=psd.sample_frequencies,
                y=psd.numpy(),
                mode='lines',
                name=f'{det} PSD'
            ))

            fig_psd.update_layout(
                title='Power Spectral Densities',
                xaxis_title='Frequency (Hz)',
                yaxis_title='Strain² / Hz',
                xaxis_type='log',
                yaxis_type='log',
                xaxis_range=[np.log10(30), np.log10(1024)],
                hovermode='x unified'
            )

            safe_chunk = str(chunk).replace("/", "_").replace("\\", "_")
            output_path = os.path.join(script_dir, f"./{p_tmp}_plots/html_plots/psd_plots/{det}_{safe_chunk}_interactive_psd_plot.html")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.write_html(output_path)

            
################## template plot

conditioned = results[det][chunk]['conditioned']
m = 36  # Solar masses

# Generate the template waveform
hp, hc = get_td_waveform(approximant=results[det][chunk]['approximant_td'],
                                  mass1=m, mass2=m,
                                  delta_t=conditioned.delta_t,
                                  f_lower=20)
hp.resize(len(conditioned))
template = hp.cyclic_time_shift(hp.start_time)

fig = go.Figure()
fig.add_trace(go.Scatter(x=template.sample_times,
                          y=template.numpy(),
                          mode='lines',
                          name='template'))

# Layout settings
fig.update_layout(
    title='Interactive template Plot',
    xaxis_title='Time (s)',
    yaxis_title='Strain',
    hovermode='x unified'
)

# Show inline (in notebook) or export to HTML
output_path = os.path.join(script_dir, f"./{p_tmp}_plots/html_plots/interactive_template_plot.html")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
fig.write_html(output_path)

################################################# Matched filter SNRs


for det in detectors:
    if det in results:
        for chunk in results[det]:
            snr = results[det][chunk]['snr']

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=snr.sample_times,
                y=abs(snr.numpy()),
                mode='lines',
                name=f'{det} SNR'
            ))

            fig.update_layout(
                title='Matched Filter SNR (Interactive)',
                xaxis_title='Time (s)',
                yaxis_title='Signal-to-Noise Ratio',
                hovermode='x unified',
                legend_title='Detectors'
            )

            safe_chunk = str(chunk).replace("/", "_").replace("\\", "_")
            output_path = os.path.join(script_dir, f"./{p_tmp}_plots/html_plots/snr_plots/{det}_{safe_chunk}_matched_filter_snr.html")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.write_html(output_path)
            


################################################## Whitened data + templates


for det in detectors:
    if det in results:
        for chunk in results[det]:
            # Plot whitened data
            whitened_data = results[det][chunk]['white_data']

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=whitened_data.sample_times,
                y=whitened_data.numpy(),
                mode='lines',
                name=f'{det} Whitened Data'
            ))

            # Plot template
            white_template = results[det][chunk]['white_template']
            fig.add_trace(go.Scatter(
                x=white_template.sample_times,
                y=white_template.numpy(),
                mode='lines',
                name=f'{det} Template',
                line=dict(dash='dash')  # Dashed line for templates
            ))

            # Choose ~10 evenly spaced tick positions
            #peak_time = results[det][chunk]['peak_time']
            #x_vals = whitened_data.sample_times
            #tickvals = np.linspace(x_vals[0], x_vals[-1], 10)
            #ticktext = [f"{x - peak_time:.3f}" for x in tickvals]

            #fig.update_xaxes(
            #    title_text='Time (s) relative to peak',
            #    tickvals=tickvals,
            #    ticktext=ticktext
            #)

            # Layout customization
            fig.update_layout(
                title='Whitened Data and Templates',
                xaxis_title='Time (s)',
                yaxis_title='Amplitude',
                hovermode='x unified',
                legend_title='Detectors'
            )

            safe_chunk = str(chunk).replace("/", "_").replace("\\", "_")
            output_path = os.path.join(script_dir, f"./{p_tmp}_plots/html_plots/whitened_plots/{det}_{safe_chunk}_whitened_data_and_templates.html")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.write_html(output_path)

 ############################################# Q-Transform Plots (separate)

for det in detectors:
    if det in results:
        for chunk in results[det]:
            peak_time = results[det][chunk]['peak_time']
            titles = ['Original', 'Signal Subtracted']
            data_variants = [
                results[det][chunk]['conditioned'],
                results[det][chunk]['conditioned'] - results[det][chunk]['aligned']
            ]

            fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=[f"{det} {t}" for t in titles])

            for idx, (data, title_suffix) in enumerate(zip(data_variants, titles)):
                data = data.time_slice(peak_time - 4, peak_time + 4)
                t, f, p = data.whiten(4, 4).qtransform(0.001, logfsteps=100, qrange=(8, 8), frange=(20, 512))

                # Plotly Heatmap expects 2D z, with x and y axes
                fig.add_trace(
                    go.Heatmap(
                        x=t,
                        y=f,
                        z=p**0.5,
                        colorscale='Viridis',
                        zmin=1,
                        zmax=6,
                        colorbar=dict(title="Power^0.5") if idx == 1 else None
                    ),
                    row=1,
                    col=idx + 1
                )

                fig.update_xaxes(title_text="Time (s)", range=[peak_time - 2, peak_time + 1], row=1, col=idx + 1)
                if idx == 0:
                    fig.update_yaxes(title_text="Frequency (Hz)", type='log', row=1, col=1)
                else:
                    fig.update_yaxes(type='log', row=1, col=2)

            fig.update_layout(
                title_text=f"Q-Transform: {det} Chunk {chunk}",
                height=400,
                width=1000
            )

            # Save or show
            safe_chunk = str(chunk).replace("/", "_").replace("\\", "_")
            output_path = os.path.join(script_dir, f"./{p_tmp}_plots/html_plots/q_transform_plots/qtransforms_{det}_{safe_chunk}.html")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.write_html(output_path)

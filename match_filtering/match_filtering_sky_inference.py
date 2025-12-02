import pylab
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy
from pycbc.waveform import get_td_waveform, get_fd_waveform
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.filter import matched_filter, resample_to_delta_t, highpass, highpass_fir, sigma
from pycbc.vetoes import power_chisq
from pycbc.events.ranking import newsnr
from pycbc.detector import Detector
from pycbc.frame import read_frame
from pycbc import frame

from pycbc.inference import models, sampler

from pycbc.distributions import Uniform, JointDistribution, SinAngle, UniformSky

import numpy as np
np.float = float    
np.int = int   #module 'numpy' has no attribute 'int'
np.object = object    #module 'numpy' has no attribute 'object'
np.bool = bool    #module 'numpy' has no attribute 'bool'
        

import copy

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

chunks = []

for det in detectors:
    if det in results:
        for idx, chunk in enumerate(results[det]):
            chunks.append({
                'det': det,
                'chunk': chunk
            })

############################################

static = {'mass1':1.3757,
          'mass2':1.3757,
          'f_lower':25.0,
          'approximant':"TaylorF2",
          'polarization':0,
         }

variable = ('distance',
            'inclination',
            'tc',
            'ra',
            'dec',
            )

psds = {}
data = {}
# Now loop over only those complete chunks
for chunk in chunks:
    for det in detectors:
        if det == chunk['det']:
            psds[det] = results[det][chunk['chunk']]['psd']

            # Load and condition strain
            channel_name = det+":STRAIN"

            strain = read_frame(chunk['chunk'], channel_name, results[det][chunk['chunk']]['peak_time'] -260, results[det][chunk['chunk']]['peak_time'] + 40)
            strain = resample_to_delta_t(highpass_fir(strain, 15.0, 512), 1.0/2048)
            #strain = resample_to_delta_t(highpass(strain, 15.0) , 1.0/2048)
            #strain = strain.time_slice(results[det][chunk['chunk']]['peak_time']-112, results[det][chunk['chunk']]['peak_time'] + 16) # Limit to times around the signal
            data[det] = strain.to_frequencyseries()

            # Estimate the power spectral density of the data
            #psd = interpolate(strain.psd(4), strain.delta_f)
            psd = interpolate(strain.psd(len(strain) / strain.sample_rate), strain.delta_f)

            psd = inverse_spectrum_truncation(psd, int(4 * psd.sample_rate),
                                              trunc_method='hann',
                                              low_frequency_cutoff=20.0)
            psds[det] = psd

            print("det: ", det, "chunk: ", chunk['chunk'])



for chunk in chunks:
    for det in detectors:
        if det == chunk['det']:
            prior = JointDistribution(variable,
                                    SinAngle(inclination=None),
                                    Uniform(
                                        inclination=(2, numpy.pi),
                                        distance=(20, 50),
                                        tc=(results[det][chunk['chunk']]['peak_time']+0.02, results[det][chunk['chunk']]['peak_time']+0.05),
                                        ),
                                        UniformSky(),   # This is a custom distribution which
                                                        # expects ra / dec and creates a isotropic distribution
                                    )

            model = models.SingleTemplate(variable, copy.deepcopy(data),
                                        low_frequency_cutoff={'E1':25, 'E2':25, 'E3':25},
                                        psds = psds,
                                        static_params = static,
                                        prior = prior,
                                        sample_rate = 8192,
                                    )

            smpl = sampler.EmceePTSampler(model, 3, 200, nprocesses=8)
            _ = smpl.set_p0() # If we don't set p0, it will use the models prior to draw initial points!


            # Note it may take ~1-3 minutes for this to run
            smpl.run_mcmc(200)

            lik = smpl.model_stats['loglikelihood']
            s = smpl.samples

            # Note how we have to access the arrays differently that before since there is an additional dimension.
            # The zeroth element of that dimension represents the 'coldest' and is the one we want for our results.
            # The other temperatures represent a modified form of the likelihood that allows walkers to traverse
            # the space more freely.
            
            # Extract data for plotting
            dist = s['ra'][0, :, -1]
            incl = s['dec'][0, :, -1]
            loglik = lik[0, :, -1]
            dist_samples = s['distance'][0,:,-1].flatten()

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Distance vs Inclination", "Posterior Distribution of Distance")
            )

            # Scatter plot
            scatter = go.Scatter(
                x=dist, y=incl, mode='markers',
                marker=dict(color=loglik, colorscale='Viridis',
                            colorbar=dict(title="Loglikelihood")),
                name="Scatter"
            )
            fig.add_trace(scatter, row=1, col=1)

            # Histogram
            hist = go.Histogram(
                x=dist_samples,
                nbinsx=30,
                marker_color='indianred',
                name="Histogram"
            )
            fig.add_trace(hist, row=1, col=2)

            # Update layout
            fig.update_layout(
                height=500, width=1000,
                title_text=f"MCMC Inference for Chunk {chunk['chunk']} Detector {det}",
                showlegend=False
            )

            # Save HTML
            safe_chunk = str(chunk['chunk']).replace("/", "_").replace("\\", "_")
            output_path = os.path.join(script_dir, f"./{p_tmp}_plots/html_inference_plots/sky_plots/sky_plots_{safe_chunk}_{det}.html")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.write_html(output_path)
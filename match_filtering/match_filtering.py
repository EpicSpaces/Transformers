import pylab
import numpy
from pycbc.waveform import get_td_waveform, get_fd_waveform
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.filter import matched_filter, resample_to_delta_t, highpass, highpass_fir, lowpass_fir, sigma
from pycbc.vetoes import power_chisq
from pycbc.events.ranking import newsnr
from pycbc.detector import Detector
from pycbc.frame import read_frame
from pycbc import frame
import pickle

import os
import glob

import argparse
import json

base_dir = ''


parser = argparse.ArgumentParser(description="Load base_dir / Load filter config from JSON.")
parser.add_argument('--b', type=str, required=True, help="Path to gwfs")
parser.add_argument('--j', type=str, required=True, help="Path to config.json")
args = parser.parse_args()

base_dir = args.b

# Detectors
detectors = ['E1']#, 'E2', 'E3'] # E1  E2 E3

# Masses for the template
m = 36  # Solar masses


filter = {}

# Load filter configuration from JSON
with open("confs/"+args.j, "r") as f:
    filter = json.load(f)

# Now you can access it like a regular dict
print("filter : ", filter)

# Dictionary to store results
results = {}

# Initialize the chunks dictionary
chunks = {}

# templates

banks = [{'name': 'template1',"m1":1.4, "m2":1.4, "f_lower":20.0, "distance": 8295, "coa_phase":6.18, "inclination" : 0.44, "spin1z":0.05, "spin2z":0.05}
         #{'name': 'template2', "m1":50, "m2":50, "f_lower":20.0},
         #{'name': 'template3', "m1":75, "m2":75, "f_lower":20.0},
         #{'name': 'template4', "m1":100, "m2":100, "f_lower":20.0},
         #{'name': 'template5', "m1":120, "m2":120, "f_lower":20.0},
         #{'name': 'template6', "m1":140, "m2":140, "f_lower":20.0},
         #{'name': 'template7', "m1":160, "m2":160, "f_lower":20.0},
         #{'name': 'template8', "m1":180, "m2":180, "f_lower":20.0},
         #{'name': 'template9', "m1":200, "m2":200, "f_lower":20.0},
         #{'name': 'template10', "m1":180, "m2":110, "f_lower":20.0},

         #{'name': 'template11', "m1": 36,  "m2": 50, "f_lower":20.0},
         #{'name': 'template12', "m1": 50,  "m2": 75, "f_lower":20.0},
         #{'name': 'template13', "m1": 75,  "m2": 100, "f_lower":20.0},
         #{'name': 'template14', "m1": 100, "m2": 150, "f_lower":20.0},
         #{'name': 'template15', "m1": 120, "m2": 60, "f_lower":20.0},
         #{'name': 'template16', "m1": 90,  "m2": 30, "f_lower":20.0},
         #{'name': 'template17', "m1": 140, "m2": 100, "f_lower":20.0},
         #{'name': 'template18', "m1": 160, "m2": 120, "f_lower":20.0},
         #{'name': 'template19', "m1": 200, "m2": 150, "f_lower":20.0},
         #{'name': 'template20', "m1": 250, "m2": 250, "f_lower":20.0},

         #{'name': 'template21', "m1": 50,  "m2": 36, "f_lower":20.0},
         #{'name': 'template22', "m1": 100,  "m2": 75, "f_lower":20.0},
         ]


# Loop over each detector folder
for det in detectors:
    folder_path = os.path.join(base_dir, det)
    # Find all .gwf files in the folder
    gwf_files = glob.glob(os.path.join(folder_path, '*.gwf'))
    # Assign to chunks dictionary
    chunks[det] = gwf_files

#chunks = {
#    'E1': ["./gwfs/E1/E-E1_STRAIN_DATA-1000000000-2048.gwf"],
#    'E2': ["./gwfs/E2/E-E2_STRAIN_DATA-1000000000-2048.gwf"],
#    'E3': ["./gwfs/E3/E-E3_STRAIN_DATA-1000000000-2048.gwf"],
#}



# Loop over detectors
for det in detectors:
    print(f"\nProcessing {det} detector...\n")

    for chunk in chunks[det]:

        if det not in results:
            results[det] = {}
        
        # Load and condition strain
        channel_name = det+":STRAIN"

        strain = read_frame(chunk, channel_name)

        #strain = read_frame(chunk, channel_name, 1187008882.4 - 260, 1187008882.4 + 40)
        strain = resample_to_delta_t(highpass_fir(strain, filter['strain_high_pass'], 512), 1.0/filter['resample_to_delta_t'])
        strain = resample_to_delta_t(lowpass_fir(strain, filter['strain_low_pass'], 512), 1.0/filter['resample_to_delta_t'])

        conditioned = strain.crop(2, 2)

        # Estimate PSD
        psd = conditioned.psd(filter['psd_segment_length'])
        psd = interpolate(psd, conditioned.delta_f)
        psd = inverse_spectrum_truncation(psd, int(filter['psd_segment_length'] * conditioned.sample_rate),
                                          #trunc_method='hann',
                                         low_frequency_cutoff=filter['psd_low_frequency_cutoff'])
        best_filter = filter.copy()
        
        for template_ in banks:
            # Generate the template waveform
            hp, hc = get_td_waveform(approximant=filter["approximant_td"],
                                    mass1=template_['m1'], mass2=template_['m2'],
                                    delta_t=conditioned.delta_t,
                                    f_lower=template_['f_lower'])
            hp.resize(len(conditioned))
            template = hp.cyclic_time_shift(hp.start_time)

            # Matched filtering
            snr = matched_filter(template, conditioned, psd=psd, low_frequency_cutoff=filter['match_filter_low_frequency_cutoff'])
            
            snr = snr.crop(5, 4)

            peak = abs(snr).numpy().argmax()
            snrp = snr[peak]
            time = snr.sample_times[peak]


            if abs(snrp) > best_filter['snr_threshold']:
                #print(f"{det} - Signal found at {time} s with SNR {abs(snrp)}")

                # Align and whiten
                # The time, amplitude, and phase of the SNR peak tell us how to align
                # our proposed signal with the data.

                # Shift the template to the peak time
                dt = time - conditioned.start_time
                aligned = template.cyclic_time_shift(dt)
                
                # scale the template so that it would have SNR 1 in this data
                aligned /= sigma(aligned, psd=psd, low_frequency_cutoff=filter['sigma_low_frequency_cutoff'])

                # Scale the template amplitude and phase to the peak value
                aligned = (aligned.to_frequencyseries() * snrp).to_timeseries()
                aligned.start_time = conditioned.start_time

                # We do it this way so that we can whiten both the template and the data
                white_data = (conditioned.to_frequencyseries() / psd**0.5).to_timeseries()

                # apply a smoothing of the turnon of the template to avoid a transient
                # from the sharp turn on in the waveform.
                tapered = aligned.highpass_fir(filter['aligned_high_pass'], 512, remove_corrupted=False)
                white_template = (tapered.to_frequencyseries() / psd**0.5).to_timeseries()

                # Select the time around the merger
                white_data = white_data.highpass_fir(filter['white_data_high_pass'], 512).lowpass_fir(filter['white_data_low_pass'], 512)
                white_template = white_template.highpass_fir(filter['white_template_high_pass'], 512).lowpass_fir(filter['white_template_low_pass'], 512)

                # Cut around merger time
                white_data = white_data.time_slice(time - 0.2, time + 0.1)
                white_template = white_template.time_slice(time - 0.2, time + 0.1)

                ########################################  Signal Consistency and Significance ##################

                # The number of bins to use. In principle, this choice is arbitrary. In practice,
                # this is empirically tuned.
                nbins = filter['chisq_bins']
                
                hp2, _ = get_fd_waveform(approximant=filter["approximant_fd"],
                                    mass1=template_['m1'], mass2=template_['m2'],
                                    f_lower=template_['f_lower'], delta_f=conditioned.delta_f)
                hp2.resize(len(psd))

                chisq = power_chisq(hp2, conditioned, nbins, psd, low_frequency_cutoff=filter['power_chisq_low_frequency_cutoff'])
                chisq = chisq.crop(5, 4)

                dof = nbins * 2 - 2
                chisq /= dof

                ########################################  Results ##################

                # Save everything for later plotting
                 
                best_filter[ 'snr_threshold'] = abs(snrp)
                best_filter['max_snr_time'] = time
                best_filter['max_snr_chunk'] = chunk       
                
                results[det][chunk] = { 
                    'strain': strain,
                    'psd': psd,
                    'snr': snr,
                    'snrp': abs(snrp),
                    'peak_time': time,
                    'white_data': white_data,
                    'white_template': white_template,
                    'conditioned': conditioned,
                    'aligned': aligned,
                    'chisq': chisq,
                    'template_name': template_['name'],
                    'approximant_td': filter['approximant_td'],
                    'approximant_fd': filter['approximant_fd']
                }


j_tmp=args.j.replace(".json", "")
with open(j_tmp+"_results.pkl", "wb") as f:
    pickle.dump(results, f)

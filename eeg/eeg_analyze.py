import mne
import matplotlib.pyplot as plt
from mne.io.eeglab.eeglab import RawEEGLAB
import numpy as np

FILE_PATH = "eeg/4EB_01-100.set"  # Replace with the path to your .set file


def interpolate_bad_channels(raw: RawEEGLAB, show=False):
    if show:
        # plot the data to get an idea of what it looks like and to determine bad channels
        raw.plot()
        plt.show()

    # after determining bad channels, add them to the list of bad channels and interpolate them
    raw.info["bads"] = ["PO5", "PO6", "T7"]
    raw.interpolate_bads()

    if show:
        raw.plot()
        plt.show()


def run_ica_and_remove_eye_movement(raw: RawEEGLAB, show=False):
    # run ICA on the data
    ica = mne.preprocessing.ICA(n_components=15, random_state=97)

    # As ICA is sensitive to low-frequency drifts, it requires the data to be high-pass filtered prior to fitting.
    # Typically, a cutoff frequency of 1 Hz is recommended.
    filtered = raw.copy().filter(
        l_freq=1, h_freq=None, verbose=True
    )  # high-pass filtering
    # Fit the ICA model to the data
    ica.fit(filtered)

    # plot the components to determine which ones are eye movements
    if show:
        ica.plot_components(show=False)
        ica.plot_sources(raw)
        plt.show()

    # remove the eye movement components
    ica.exclude = [0, 1]
    ica.apply(raw)

    if show:
        ica.plot_components(show=False)
        ica.plot_sources(raw)
        plt.show()


def plot_psd(raw: RawEEGLAB, show=False):
    events = mne.events_from_annotations(raw)[0]
    # Extract 120s following event IDs 43 and 32
    epochs = mne.Epochs(
        raw, events, event_id=[43, 32], tmin=0, tmax=120, preload=True, baseline=(0, 0)
    )

    # psd_welch doesn't exist in my version so doing a workaround with psd_array_welch
    # I compute the PSD between 8Hz and 12Hz
    psd, freqs = mne.time_frequency.psd_array_welch(
        epochs.get_data(), raw.info["sfreq"], average="mean", fmin=1, fmax=30
    )
    # The variable psd will contain the amplitudes and freqs will contain the frequencies.
    # It is conventional to plot PSD using a logarithmic scale.
    psd = 10 * np.log10(psd)

    # find POZ index so we can plot it later
    poz_index = epochs.ch_names.index("POZ")

    # find the relevant events to plot
    psd_poz_1 = psd[0, poz_index, :]
    psd_poz_2 = psd[1, poz_index, :]

    if not show:
        return

    # plot the PSD
    plt.plot(freqs, psd_poz_1, label="43")
    plt.plot(freqs, psd_poz_2, label="32")
    # add labels and title
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (dB/Hz)")
    plt.title("PSD of POZ Channel")
    plt.grid(True)
    plt.legend()

    plt.show()

    # Calculate the average alpha power across all channels for each epoch
    alpha_power = np.mean(psd[:, :, (freqs >= 8) & (freqs <= 12)], axis=2)

    # Compute the difference in average alpha power between the two epochs
    alpha_diff = alpha_power[0, :] - alpha_power[1, :]

    # Plot the topographic maps for each epoch's average alpha power
    # Epoch 1
    # scale around the mean before plotting it
    scaled_event_43 = alpha_power[0, :] - alpha_power[0, :].mean()
    mne.viz.plot_topomap(scaled_event_43, raw.info)

    # Epoch 2
    # scale around the mean before plotting it
    scaled_event_32 = alpha_power[1, :] - alpha_power[1, :].mean()
    mne.viz.plot_topomap(scaled_event_32, raw.info)

    # Plot the topographic map of the difference between the epochs
    mne.viz.plot_topomap(
        alpha_diff, raw.info, names=epochs.info["ch_names"], show_names=True
    )
    plt.show()


# Load the data
raw_eeg_data: RawEEGLAB = mne.io.read_raw_eeglab(FILE_PATH, preload=True)

# step 1
hash_before_interpolation = hash(raw_eeg_data)
interpolate_bad_channels(raw_eeg_data, show=False)
# making sure that the data has changed
hash_after_interpolation = hash(raw_eeg_data)
if hash_before_interpolation == hash_after_interpolation:
    print("Interpolation did not change the data")
    exit(1)

# step 2
hash_before_ica = hash(raw_eeg_data)
run_ica_and_remove_eye_movement(raw_eeg_data, show=True)
# making sure that the data has changed
hash_after_ica = hash(raw_eeg_data)
if hash_before_ica == hash_after_ica:
    print("ICA did not change the data")
    exit(1)

# step 3
plot_psd(raw_eeg_data, show=True)

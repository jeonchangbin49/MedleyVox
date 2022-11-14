import random

import numpy as np
import parselmouth
from parselmouth.praat import call


def change_pitch(sound, factor):
    manipulation = call(sound, "To Manipulation", 0.01, 75, 600)

    pitch_tier = call(manipulation, "Extract pitch tier")

    # Arguments : Time range (s), Time range (s), Frequency shift, Unit
    call(pitch_tier, "Shift frequencies", 0, 1000, factor, "semitones")

    call([pitch_tier, manipulation], "Replace pitch tier")
    return call(manipulation, "Get resynthesis (overlap-add)")


def change_formant(
    sound, formant_shift_ratio, pitch_shift_ratio, pitch_range_factor, duration_factor
):
    # https://www.fon.hum.uva.nl/praat/manual/Sound__Change_gender___.html
    # Arguments : Minimum pitch(Hz), Maximum pitch(Hz), Formant shift ratio, New pitch median(Hz), Pitch range factor, Duration factor
    # formant_shift_ratio should be around 0.8-1.2 (N(1.,0.2))
    # A ratio of 1.1 will change a male voice to a voice with approximate female formant characteristics.
    # A ratio of 1/1.1 will change a female voice to a voice with approximate male formant characteristics.

    # Pitch range factor should be in between 0.8-1.2 (N(1.,0.2))

    # duration_factor should be in between 0.9-1.1 N(1.,0.1))

    return call(
        sound,
        "Change gender",
        75,
        600,
        formant_shift_ratio,
        pitch_shift_ratio,
        pitch_range_factor,
        duration_factor,
    )


def change_pitch_and_formant_random(audio, sample_rate):
    sound = parselmouth.Sound(audio, sampling_frequency=sample_rate)
    original_size = sound.values.shape[1]  # shape of sound.values is [1, audio_length]
    pitch_shift_ratio = random.uniform(-0.2, 0.2)  # -15 to +15 cents
    pitch_shift_ratio = random.choice([-12, 0, 12]) + pitch_shift_ratio

    sound = change_pitch(sound, pitch_shift_ratio)  # -15 to +15 cents

    formant_shift_ratio = random.uniform(1, 1.4)
    formant_shift_ratio = random.choice([formant_shift_ratio, 1 / formant_shift_ratio])

    sound = change_formant(
        sound, formant_shift_ratio, 0.0, 1, max(0.7, random.normalvariate(1.0, 0.05))
    )

    output = sound.values[0]  # shape of sound.values is [1, audio_length]
    if output.shape[0] >= original_size:
        output = output[:original_size]
    else:
        output = np.pad(output, (0, original_size - output.shape[0]))

    return output


def change_pitch_and_formant(
    audio,
    sample_rate,
    pitch_shift_ratio,
    formant_shift_ratio,
    pitch_range_ratio,
    time_stretch_ratio,
):
    sound = parselmouth.Sound(audio, sampling_frequency=sample_rate)
    original_size = sound.values.shape[1]  # shape of sound.values is [1, audio_length]

    # sound = change_pitch(sound, random.uniform(-0.15, 0.15)) # -15 to +15 cents
    sound = change_pitch(sound, pitch_shift_ratio)  # -15 to +15 cents
    sound = change_formant(
        sound, formant_shift_ratio, 0.0, pitch_range_ratio, time_stretch_ratio
    )

    output = sound.values[0]  # shape of sound.values is [1, audio_length]
    if output.shape[0] >= original_size:
        output = output[:original_size]
    else:
        output = np.pad(output, (0, original_size - output.shape[0]))

    return output

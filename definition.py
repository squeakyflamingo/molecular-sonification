import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors, Descriptors
import selfies
from music21.note import Note
from music21.chord import Chord
from music21.stream import Stream
from music21 import key as kyy
import matplotlib.pyplot as plt
# import magenta.music as mm
import wave
import os
# from magenta.models.music_vae import configs
# from magenta.models.music_vae.trained_model import TrainedModel

# import tensorflow.compat.v1 as tf
#
# tf.disable_v2_behavior()

suppl = Chem.SDMolSupplier('structures.sdf')
selfs = {}
all_selfies = []
for k in suppl:
    try:
        Chem.SanitizeMol(k)
        sel = selfies.encoder(Chem.MolToSmiles(k))
    except:
        continue

    if sel is None:
        continue

    all_selfies.append(sel)
    for j in list(selfies.split_selfies(sel)):
        if j not in selfs:
            selfs[j] = 0
        selfs[j] = selfs[j] + 1
print(len(all_selfies))
ranked_selfies_tokens = (sorted(selfs.items(), key=lambda item: item[1], reverse=True))


def getAll(x): return [getMW(x), getLogP(x), getHBD(x), getHBA(x), getPSA(x), getROTB(x)]


# return [getLogP(x),getMW(x),getHBD(x),getHBA(x),getPSA(x),getROTB(x),getFSP3(x),getFC(x),getQED(x)]

def getLogP(x): return Chem.rdMolDescriptors.CalcCrippenDescriptors(x)[0]


def getMW(x): return Chem.Descriptors.MolWt(x)


def getHBD(x): return Chem.rdMolDescriptors.CalcNumHBD(x)


def getHBA(x): return Chem.rdMolDescriptors.CalcNumHBA(x)


def getPSA(x): return Chem.rdMolDescriptors.CalcTPSA(x)


def getROTB(x): return Chem.rdMolDescriptors.CalcNumRotatableBonds(x)


def getAROM(x): return Chem.rdMolDescriptors.CalcNumAromaticRings(x)


def getFSP3(x): return Chem.rdMolDescriptors.CalcFractionCSP3(x)


def getFC(x): return Chem.rdmolops.GetFormalCharge(x)


def getQED(x): return Chem.QED.qed(x)


print("mapping tokens to major scale")
note_to_shift = {}
shift_to_note = {}
# 2 2 1 2 2 2 1
major_scale = [0, 2, 4, 5, 7, 9, 11,
               12, 14, 16, 17, 19, 21, 23,
               24, 26, 28, 29, 31, 33, 35,
               36, 38, 40, 41, 43, 45, 47,
               48, 50, 52, 53, 55, 57, 59]
for i, k in enumerate(ranked_selfies_tokens[0:32]):
    note_to_shift[k[0]] = major_scale[i]
    shift_to_note[major_scale[i]] = k[0]

print("generating valid selfies from dataset")
valid_selfies_for_model = []
for i, k in enumerate(all_selfies):
    tpl = list(selfies.split_selfies(k))
    not_good = False
    for tok in tpl:
        if tok not in note_to_shift:
            not_good = True
            break
    if not_good:
        continue
    else:
        valid_selfies_for_model.append(k)


# print(len(valid_selfies_for_model))

def get_phys_prop_array(selfie):
    return getAll(Chem.MolFromSmiles(selfies.decoder(selfie)))


def hash_selfie(selfie):
    props = get_phys_prop_array(selfie)
    bit = 0
    for i, k in enumerate(props): bit = bit + k
    return bit


keys = ["G", "C", "D", "A", "Db", "F", "Ab", "E", "Bb", "F#", "B", "Eb"]


def hash_to_key(hash, hashMin, hashMax):
    y = (((hash - hashMin) / (hashMax - hashMin)) * (len(keys) - 1)) + 0
    return keys[int(y)]


# print("calculating max and min hash values")
hashes = []
for k in valid_selfies_for_model:
    hashes.append(hash_selfie(k))
minHash = np.min(hashes)
maxHash = np.max(hashes)


def create_wav_spectra(filename):
    signal_wave = wave.open('wav/' + str(filename) + '.wav', 'r')
    sample_rate = -1
    sig = np.frombuffer(signal_wave.readframes(sample_rate), dtype=np.int16)
    sig = sig[:]

    left, right = sig[0::2], sig[1::2]

    plt.figure(1)

    plot_a = plt.subplot(111)
    plot_a.plot(sig)
    plot_a.set_xlabel('sample rate * time')
    plot_a.set_ylabel('amplitude')
    xt = [plot_a.get_xticks()[1], (plot_a.get_xticks()[1] + plot_a.get_xticks()[-2]) / 2, plot_a.get_xticks()[-2]]
    plot_a.set_xticks(xt)
    plot_a.set_yticks([plot_a.get_yticks()[0], 0, plot_a.get_yticks()[-1]])
    # plot_b = plt.subplot(212)
    # plot_b.specgram(sig, NFFT=1024, Fs=sample_rate, noverlap=900)
    # plot_b.set_xlabel('Time')
    # plot_b.set_ylabel('Frequency')

    # plt.show()
    plt.tight_layout()
    # plt.show()
    plt.savefig(str(filename) + ".png", dpi=300)
    # plt.close()


key_to_midi = {"C": 48, "Db": 49, "D": 50, "Eb": 51, "E": 52, "F": 53, "F#": 54, "G": 55, "Ab": 56, "A": 57, "Bb": 58,
               "B": 59}


def create_midi_file(smiles, filename, directory="test", chords=True):
    ml = Chem.MolFromSmiles(smiles)
    Chem.SanitizeMol(ml)
    selfie = selfies.encoder(Chem.MolToSmiles(ml))
    print(selfie)

    key = hash_to_key(hash_selfie(selfie), np.min(hashes), np.max(hashes))
    note_pattern = []
    for tkn in selfies.split_selfies(selfie):
        note_pattern.append(note_to_shift[tkn])
    print(key)
    stream = Stream()
    stream.keySignature = kyy.Key(key, 'major')
    pitches = []
    for i, k in enumerate(note_pattern):
        pitch = key_to_midi[key] + k
        pitches.append(pitch)
        if chords:
            if i % 4 == 0:
                chord = []
                idx = major_scale.index(k)
                chord.append(pitch)
                chord.append(key_to_midi[key] + major_scale[idx + 2])
                chord.append(key_to_midi[key] + major_scale[idx + 4])
                stream.append(Chord(chord, type='eighth'))
            else:
                stream.append(Note(pitch, type='eighth'))
        else:
            stream.append(Note(pitch, type='eighth'))
    stream.write('midi', directory + '/' + str(filename) + ".mid")


# smiles = "CC[C@]1(C=CC2)CC(C(OC)=O)=C([C@]34[C@@]1([H])N2CC3)NC5=C4C=CC=C5"  # @param
smiles = "CCCCCCc1cc(sc1C)C"
filename = "P3HT"  # @param

create_midi_file(smiles, filename)
os.system("fluidsynth -ni FluidR3_GM.sf2 test/" + filename + ".mid -F wav/" + filename + ".wav -r 44100")
# create_wav_spectra(filename)

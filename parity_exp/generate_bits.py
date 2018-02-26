import numpy as np
import itertools
from itertools import chain


def parity(bits):
    return len([b for b in bits if b]) % 2 == 0


def task1(bits):
    return bits[0] or parity(bits[1:6])


def generate_bits():
    bits = list(itertools.product([True, False], repeat=8))
    results = [task1(b) for b in bits]

    return (bits, results)


def generate_bits_old():
    bits = []
    results = []

    for bits in itertools.product([True, False], repeat=8):
        bits.append(
            list("1" if b else "0" for b in chain(bits, [task1(bits)]))
        )

    return (bits, results)

def bits_prepared():
    bits, results = generate_bits()
    return np.array(bits), np.array(results)


def correct_bit_percentage(labels, results):
    prediction = np.round(results)
    correct = sum(prediction == labels)
    return correct / results.shape

def load_bits():
    with open("bits.txt", "r") as f:
        return f.readlines()

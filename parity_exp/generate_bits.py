import numpy as np
import itertools
from itertools import chain


def parity(bits):
    return len([b for b in bits if b]) % 2 == 0


def task1(bits):
    return bits[0] or parity(bits[1:6])


def task2(bits):
    return (not bits[0]) or parity(bits[1:6])


def task3(bits):
    return bits[0] and parity(bits[1:6])


def task4(bits):
    return (not bits[0]) and parity(bits[1:6])


def generate_bits():
    bits = list(itertools.product([True, False], repeat=8))
    results1 = [task1(b) for b in bits]
    results2 = [task2(b) for b in bits]
    results3 = [task3(b) for b in bits]
    results4 = [task4(b) for b in bits]

    return (bits, results1, results2, results3, results4)


def generate_bits_old():
    bits = []
    results = []

    for bits in itertools.product([True, False], repeat=8):
        bits.append(
            list("1" if b else "0" for b in chain(bits, [task1(bits)]))
        )

    return (bits, results)

def bits_prepared():
    return (np.array(a, dtype=float) for a in generate_bits())


def correct_bit_percentage(labels, results):
    prediction = np.round(results)
    correct = sum(prediction == labels)
    return correct / results.shape

def load_bits():
    with open("bits.txt", "r") as f:
        return f.readlines()

import string, os
import numpy as np
import pandas as pd
from scipy.stats import levy_stable
from scipy.special import gamma

def clean_text(txt):
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    txt = txt.encode("utf8").decode("ascii",'ignore')
    return txt 

def encrypt(text, s=23):
    result = ""
    for i in range(len(text)):
        char = text[i]
        if char.isupper():
            result += chr((ord(char) + s - 65) % 26 + 65)
        elif char.islower():
            result += chr((ord(char) + s - 97) % 26 + 97)
    return result

def encode(x, samples=10):
    rv = levy_stable(2.0, 0.0)
    proj_matrix = rv.rvs(size=samples)

    return proj_matrix * x

def geometric_mean(alpha, sketch_size, x):
    return np.prod(np.power(np.abs(x), alpha/sketch_size))/np.power(2*gamma(alpha/sketch_size)*gamma(1-1/sketch_size)*np.sin(np.pi*alpha/2/sketch_size)/np.pi, sketch_size)

# http://python.algorithmexamples.com/web/ciphers/decrypt_caesar_with_chi_squared.html
def decrypt_caesar_with_chi_squared(
    ciphertext: str,
    cipher_alphabet=None,
    frequencies_dict=None,
    case_sensetive: bool = False,
    samples = 10,
) -> tuple:
 
    alphabet_letters = cipher_alphabet or [chr(i) for i in range(97, 123)]
    frequencies_dict = frequencies_dict or {}
 
    if frequencies_dict == {}:
        # Frequencies of letters in the english language (how much they show up)
        frequencies = {
            "a": 0.08497,
            "b": 0.01492,
            "c": 0.02202,
            "d": 0.04253,
            "e": 0.11162,
            "f": 0.02228,
            "g": 0.02015,
            "h": 0.06094,
            "i": 0.07546,
            "j": 0.00153,
            "k": 0.01292,
            "l": 0.04025,
            "m": 0.02406,
            "n": 0.06749,
            "o": 0.07507,
            "p": 0.01929,
            "q": 0.00095,
            "r": 0.07587,
            "s": 0.06327,
            "t": 0.09356,
            "u": 0.02758,
            "v": 0.00978,
            "w": 0.02560,
            "x": 0.00150,
            "y": 0.01994,
            "z": 0.00077,
        }
    else:
        # Custom frequencies dictionary
        frequencies = frequencies_dict
 
    if not case_sensetive:
        ciphertext = ciphertext.lower()
 
    # Chi squared statistic values
    chi_squared_statistic_values = {}
 
    # cycle through all of the shifts
    for shift in range(len(alphabet_letters)):
        # print(shift)
        decrypted_with_shift = ""
 
        # decrypt the message with the shift
        for letter in ciphertext:
            # print(letter)
            try:
                # Try to index the letter in the alphabet
                new_key = (alphabet_letters.index(letter) - shift) % len(
                    alphabet_letters
                )
                decrypted_with_shift += alphabet_letters[new_key]
            except ValueError:
                # Append the character if it isn't in the alphabet
                decrypted_with_shift += letter
 
        chi_squared_statistic = np.zeros(samples)
 
        # Loop through each letter in the decoded message with the shift
        for letter in decrypted_with_shift:
            if case_sensetive:
                if letter in frequencies:
                    # Get the amount of times the letter occurs in the message
                    occurrences = decrypted_with_shift.count(letter)
 
                    # Get the excepcted amount of times the letter should appear based
                    # on letter frequencies
                    expected = frequencies[letter] * occurrences
 
                    # Complete the chi squared statistic formula
                    chi_letter_value = np.sqrt((occurrences - expected) / expected)
 
                    # Add the margin of error to the total chi squared statistic
                    chi_squared_statistic += encode(chi_letter_value, samples)
            else:
                if letter.lower() in frequencies:
                    # Get the amount of times the letter occurs in the message
                    occurrences = decrypted_with_shift.count(letter)
 
                    # Get the excepcted amount of times the letter should appear based
                    # on letter frequencies
                    expected = frequencies[letter] * occurrences
 
                    # Complete the chi squared statistic formula
                    chi_letter_value = np.sqrt((occurrences - expected) / expected)
 
                    # Add the margin of error to the total chi squared statistic
                    chi_squared_statistic += encode(chi_letter_value, samples)
 
        # Add the data to the chi_squared_statistic_values dictionary
        chi_squared_statistic_values[shift] = [
            geometric_mean(2.0, samples, chi_squared_statistic),
            decrypted_with_shift,
            shift,
        ]
 
    # Get the most likely cipher by finding the cipher with the smallest chi squared
    # statistic
    most_likely_cipher = min(
        chi_squared_statistic_values, key=chi_squared_statistic_values.get
    )
 
    # Get all the data from the most likely cipher (key, decoded message)
    most_likely_cipher_chi_squared_value = chi_squared_statistic_values[
        most_likely_cipher
    ][0]
    decoded_most_likely_cipher = chi_squared_statistic_values[most_likely_cipher][1]
 
    # Return the data on the most likely shift
    return (
        most_likely_cipher,
        most_likely_cipher_chi_squared_value,
        decoded_most_likely_cipher,
        chi_squared_statistic_values[most_likely_cipher][2],
    )

if __name__ == '__main__':

    iters = 100

    play_df = pd.read_csv('../dataset/Shakespeare_data.csv')
    all_lines = [h for h in play_df.PlayerLine]
    corpus = [clean_text(x) for x in all_lines]
    cipher = ""
    for text in corpus:
        cipher = cipher + encrypt(text)

    for i in range(1, 6):
        correct = 0
        for j in range(iters):
            _, _, decrypted, shift = decrypt_caesar_with_chi_squared(cipher[1000*j:1000*(j+1)], samples=10*i)
            if shift == 23:
                correct += 1
            else:
                print(shift)
        print(i*10, ", ", correct/iters)

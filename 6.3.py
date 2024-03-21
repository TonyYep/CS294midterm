#3. Compression:
'''(a)
Create a long random string using a Python program, and use a lossless compression algorithm of your choice to compress the string. Note the compression ratio.'''

#Lossless Compression Algorithm (String -> emoji with count!)
import string

import zlib
import sys

a = "test"

def compress(string):
    return zlib.compress(string.encode())

def decompress(string):
    return zlib.decompress(string).decode()

def size(x):
    return sys.getsizeof(x)

#Random string generation
import random

random.seed(10)
def random_string(str_length):
    chars = string.ascii_letters + string.digits + string.punctuation + ' '
    new = ''
    for i in range(str_length):
        new = new + random.choice(chars)
    return new

#Compression and Ratio
rand_str = random_string(50)
compressed_str = compress(rand_str)
uncompressed = decompress(compressed_str)
print("Compression Ratio test 1: (8 bits * len(Decompressed String)) / (8 bits * len(Compressed String)) = " + str(size(uncompressed) / size(compressed_str)) + " bits/bit")
rand_str = 'aaaaaabbbbbcccccccccccccddddddddddddddddeeeeeeeeeeeeeeeeeeffffffffffff1341241'
compressed_str = compress(rand_str)
uncompressed = decompress(compressed_str)
print("Compression Ratio test 2: (8 bits * len(Decompressed String)) / (8 bits * len(Compressed String)) = " + str(size(uncompressed) / size(compressed_str)) + " bits/bit")

'''
**************
****ANSWER****
**************

We see that for a completely random string, the compression ratio is ~1.088 bits / bit
We see for a more generic string with characters occuring repeatedly, the compression ratio is ~1.969 bits / bit
'''


'''(b) What is the expected compression ratio in (a)? Explain why?'''

'''
**************
****ANSWER****
**************

For a compression algorithm to work, we expect a ratio > 1 for any compression to occur. Using the compression algorithm from the zlib library (using LZ77 and huffman encoding jointly)
we achieved a compression ration > 1 as expected.

NOTE: I initially tried the common Run-Length Encoding compression initially, but recognized that one of it's weaknesses was strings with nonreoccuring characters. For those strings, the
compression ratio was less than 1 and the algorithm increased the bit count. Hence, I chose to search for a more optimized general algorithm and found zlib. The zlib compression algorithm
is extremely complicated and out of scope for expected compression ratio calculations    D:
'''
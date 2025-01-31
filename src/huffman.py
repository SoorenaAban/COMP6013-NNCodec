# huffman.py

import heapq
from collections import defaultdict

class HuffmanNode:
    def __init__(self, symbol=None, freq=0):
        self.left = None
        self.right = None
        self.symbol = symbol  # Should be an instance of Symbol
        self.freq = freq

    def __lt__(self, other):
        return self.freq < other.freq


def generate_codes_from_symbols(symbols):
    # Build frequency dictionary
    freq_dict = defaultdict(int)
    for sym in symbols:
        freq_dict[sym] += 1

    # Build Huffman tree
    heap = [HuffmanNode(symbol=sym, freq=freq) for sym, freq in freq_dict.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = HuffmanNode(freq=node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)
    if heap:
        huffman_tree = heap[0]
    else:
        huffman_tree = None

    # Build Huffman codes
    codes = {}
    def build_codes(node, code=''):
        if node.symbol is not None:
            codes[node.symbol] = code
        else:
            build_codes(node.left, code + '0')
            build_codes(node.right, code + '1')
    if huffman_tree:
        build_codes(huffman_tree)
    return codes


def huffman_encode(symbol_collection):
    # Encode the symbol collection using the codes assigned to symbols
    encoded_bits = ''.join([sym.code for sym in symbol_collection])
    return encoded_bits


def huffman_decode(encoded_bits, codes):
    # Build reverse mapping
    code_to_symbol = {code: sym for sym, code in codes.items()}
    decoded_symbols = []
    current_code = ''
    for bit in encoded_bits:
        current_code += bit
        if current_code in code_to_symbol:
            decoded_symbols.append(code_to_symbol[current_code])
            current_code = ''
    return decoded_symbols


def calculate_dict_size(codes):
    # Calculate the size needed to store the Huffman codes
    size = 0
    for symbol, code in codes.items():
        symbol_size = len(symbol.data) * 8  # Size in bits
        code_size = len(code)  # Code size in bits
        size += symbol_size + code_size
    return size

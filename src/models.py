class Symbol:
    def __init__(self, data, is_wild=False):
        self.data = data  # Binary representation of the data
        self.code = None  # Huffman code assigned to the symbol

    def __eq__(self, other):
        return (
            isinstance(other, Symbol)
            and self.data == other.data
        )

class Dictionary:
    def __init__(self, name="unknown"):
        self.symbols = set()  # Set of Symbol instances
        self.name = name

    def add(self, symbol):
        self.symbols.add(symbol)
        # Generate Huffman codes when a new symbol is added
        self.generate_huffman_codes()

    def add_multiple(self, symbols):
        self.symbols.update(symbols)
        # Generate Huffman codes after adding multiple symbols
        self.generate_huffman_codes()

    def generate_huffman_codes(self):
        # Generate Huffman codes for all symbols in the dictionary
        from huffman import generate_codes_from_symbols

        codes = generate_codes_from_symbols(self.symbols)
        for symbol, code in codes.items():
            symbol.code = code

    def contains(self, symbol):
        return symbol in self.symbols

    def __str__(self):
        symbols_list = ", ".join([str(sym) for sym in self.symbols])
        return f"Dictionary Name: {self.name}, Symbols: [{symbols_list}]"

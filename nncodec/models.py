"""
models.py

The shared objects used in the nncodec.

"""


from typing import Optional, Iterable, List, Dict

class Symbol:
    """
    Represents a single symbol in the data.
    """
    def __init__(self, data: bytes) -> None:
        if not isinstance(data, bytes):
            raise ValueError("Data must be of type bytes")
        self.data: bytes = data

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Symbol):
            return self.data == other.data
        return False

    def __str__(self) -> str:
        try:
            return str(self.data)
        except Exception:
            return f"[Symbol:{self.__hash__()}]"

    def __repr__(self) -> str:
        return str(self.data)

    def __hash__(self) -> int:
        return hash(self.data)


class SymbolFrequency:
    """
    Represents a symbol together with its frequency.
    """
    def __init__(self, symbol: Symbol, frequency: int) -> None:
        self.symbol: Symbol = symbol
        self.frequency: int = frequency
        
    def __str__(self) -> str:
        return f"[{self.symbol}, {self.frequency}]"

    def __repr__(self) -> str:
        return f"[{self.symbol}, {self.frequency}]"


class Dictionary:
    """
    Represents a dictionary of unique symbols found in the data.
    """
    def __init__(self) -> None:
        self.symbols: set[Symbol] = set()
        self._sorted_symbols: Optional[List[Symbol]] = None
        self._symbol_to_index: Optional[Dict[Symbol, int]] = None

    def add(self, symbol: Symbol) -> bool:
        """
        Add a symbol to the dictionary.

        Returns:
            bool: True if the symbol was already present; False if added.
        """
        if symbol in self.symbols:
            return True
        self.symbols.add(symbol)
        # Invalidate cached sorted list and mapping.
        self._sorted_symbols = None
        self._symbol_to_index = None
        return False

    def add_multiple(self, symbols: Iterable[Symbol]) -> int:
        """
        Add multiple symbols to the dictionary.

        Args:
            symbols (Iterable[Symbol]): Iterable of symbols to add.

        Returns:
            int: Count of symbols that were already present.
        """
        count = 0
        for symbol in symbols:
            if self.add(symbol):
                count += 1
        return count

    def get_size(self) -> int:
        """
        Get the number of unique symbols in the dictionary.

        Returns:
            int: Number of symbols.
        """
        return len(self.symbols)

    def contains(self, symbol: Symbol) -> bool:
        """
        Check if the symbol is in the dictionary.

        Args:
            symbol (Symbol): The symbol to check.

        Returns:
            bool: True if present, False otherwise.
        """
        return symbol in self.symbols

    def contains_data(self, data: bytes) -> bool:
        """
        Check if any symbol in the dictionary contains the given data.

        Args:
            data (bytes): Data to check.

        Returns:
            bool: True if a symbol with the data exists, False otherwise.
        """
        for symbol in self.symbols:
            if symbol.data == data:
                return True
        return False

    def contains_code(self, code: int) -> bool:
        """
        Check if any symbol in the dictionary has a matching 'code' attribute.

        Args:
            code (int): The code to check.

        Returns:
            bool: True if found, False otherwise.
        """
        for symbol in self.symbols:
            if hasattr(symbol, "code") and symbol.code == code:
                return True
        return False

    def _build_index(self) -> None:
        """
        Sort symbols and build a mapping from each symbol to its index.
        """
        self._sorted_symbols = sorted(self.symbols, key=lambda s: s.data)
        self._symbol_to_index = {s: i for i, s in enumerate(self._sorted_symbols)}

    def get_index(self, symbol: Symbol) -> int:
        """
        Get the index of the given symbol from the sorted list.

        Args:
            symbol (Symbol): The symbol to look up.

        Returns:
            int: The index of the symbol.
        """
        if self._symbol_to_index is None:
            self._build_index()
        return self._symbol_to_index[symbol]

    def get_sorted_symbols(self) -> List[Symbol]:
        """
        Get the list of symbols sorted by their data.

        Returns:
            List[Symbol]: The sorted symbols.
        """
        if self._sorted_symbols is None:
            self._build_index()
        return self._sorted_symbols

    def get_symbol_by_index(self, index: int) -> Symbol:
        """
        Get the symbol at the specified index from the sorted list.

        Args:
            index (int): Index of the symbol.

        Returns:
            Symbol: The symbol at the given index.
        """
        return self.get_sorted_symbols()[index]

    def __eq__(self, other: object) -> bool:
        """
        Compare this dictionary with another for equality.

        Args:
            other (object): Another Dictionary instance.

        Returns:
            bool: True if dictionaries are equal, False otherwise.
        """
        if not isinstance(other, Dictionary):
            return False
        if len(self.symbols) != len(other.symbols):
            return False
        for symbol in self.symbols:
            if not other.contains(symbol):
                return False
        return True

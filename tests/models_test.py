import unittest
from nncodec.models import Symbol, SymbolFrequency, Dictionary

class TestSymbol(unittest.TestCase):
    def test_equality_and_hash(self):
        s1 = Symbol(b'a')
        s2 = Symbol(b'a')
        s3 = Symbol(b'b')
        self.assertEqual(s1, s2)
        self.assertNotEqual(s1, s3)
        self.assertEqual(hash(s1), hash(s2))

    def test_str_and_repr(self):
        s = Symbol(b'c')
        self.assertIn("b'c'", str(s))
        self.assertIn("b'c'", repr(s))

class TestSymbolFrequency(unittest.TestCase):
    def test_str_and_repr(self):
        s = Symbol(b'x')
        sf = SymbolFrequency(s, 10)
        expected = f"[{s}, 10]"
        self.assertEqual(str(sf), expected)
        self.assertEqual(repr(sf), expected)

class TestDictionary(unittest.TestCase):
    def test_add_and_contains(self):
        d = Dictionary()
        s = Symbol(b'a')
        self.assertFalse(d.add(s))
        self.assertTrue(d.add(s))
        self.assertTrue(d.contains(s))
    
    def test_add_multiple_and_size(self):
        d = Dictionary()
        s1 = Symbol(b'a')
        s2 = Symbol(b'b')
        s3 = Symbol(b'c')
        count = d.add_multiple([s1, s2, s1, s3, s2])
        self.assertEqual(count, 2)
        self.assertEqual(d.get_size(), 3)
    
    def test_contains_data(self):
        d = Dictionary()
        s = Symbol(b'a')
        d.add(s)
        self.assertTrue(d.contains_data(b'a'))
        self.assertFalse(d.contains_data(b'b'))
    
    def test_contains_code(self):
        d = Dictionary()
        s = Symbol(b'a')
        s.code = 123
        d.add(s)
        self.assertTrue(d.contains_code(123))
        self.assertFalse(d.contains_code(999))
    
    def test_sorted_symbols_and_index(self):
        d = Dictionary()
        s1 = Symbol(b'b')
        s2 = Symbol(b'a')
        s3 = Symbol(b'c')
        d.add(s1)
        d.add(s2)
        d.add(s3)
        sorted_syms = d.get_sorted_symbols()
        self.assertEqual(sorted_syms, [s2, s1, s3])
        self.assertEqual(d.get_index(s2), 0)
        self.assertEqual(d.get_index(s1), 1)
        self.assertEqual(d.get_index(s3), 2)
    
    def test_get_symbol_by_index(self):
        d = Dictionary()
        s1 = Symbol(b'a')
        s2 = Symbol(b'b')
        d.add(s2)
        d.add(s1)
        sorted_syms = d.get_sorted_symbols()
        self.assertEqual(d.get_symbol_by_index(0), sorted_syms[0])
        self.assertEqual(d.get_symbol_by_index(1), sorted_syms[1])
    
    def test_get_index_nonexistent(self):
        d = Dictionary()
        s = Symbol(b'a')
        with self.assertRaises(KeyError):
            d.get_index(s)

if __name__ == '__main__':
    unittest.main()

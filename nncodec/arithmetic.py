#arithmetic.py

# Provides the state and behaviors that arithmetic coding encoders and decoders share.
class ArithmeticCoderBase(object):

	def __init__(self, numbits):
		if numbits < 1:
			raise ValueError("State size out of range")

		self.num_state_bits = numbits
		self.full_range = 1 << self.num_state_bits
		self.half_range = self.full_range >> 1
		self.quarter_range = self.half_range >> 1
		self.minimum_range = self.quarter_range + 2
		self.maximum_total = self.minimum_range
		self.state_mask = self.full_range - 1

		self.low = 0
		self.high = self.state_mask
	

	def update(self, freqs, symbol):
		low = self.low
		high = self.high
		# if low >= high or (low & self.state_mask) != low or (high & self.state_mask) != high:
		# 	raise AssertionError("Low or high out of range")
		range = high - low + 1
		# if not (self.minimum_range <= range <= self.full_range):
		# 	raise AssertionError("Range out of range")

		total = int(freqs[-1])
		symlow = int(freqs[symbol-1]) if symbol > 0 else 0
		symhigh = int(freqs[symbol])
		#total = freqs.get_total()
		#symlow = freqs.get_low(symbol)
		#symhigh = freqs.get_high(symbol)
		# if symlow == symhigh:
		# 	raise ValueError("Symbol has zero frequency")
		# if total > self.maximum_total:
		# 	raise ValueError("Cannot code symbol because total is too large")

		newlow  = low + symlow  * range // total
		newhigh = low + symhigh * range // total - 1
		self.low = newlow
		self.high = newhigh

		while ((self.low ^ self.high) & self.half_range) == 0:
			self.shift()
			self.low  = ((self.low  << 1) & self.state_mask)
			self.high = ((self.high << 1) & self.state_mask) | 1
		while (self.low & ~self.high & self.quarter_range) != 0:
			self.underflow()
			self.low = (self.low << 1) ^ self.half_range
			self.high = ((self.high ^ self.half_range) << 1) | self.half_range | 1

	def shift(self):
		raise NotImplementedError()
	

	def underflow(self):
		raise NotImplementedError()


class ArithmeticEncoder(ArithmeticCoderBase):

	def __init__(self, numbits, bitout):
		super(ArithmeticEncoder, self).__init__(numbits)
		self.output = bitout
		self.num_underflow = 0

	def write(self, freqs, symbol):
		self.update(freqs, symbol)

	def finish(self):
		self.output.write(1)
	
	
	def shift(self):
		bit = self.low >> (self.num_state_bits - 1)
		self.output.write(bit)

		for _ in range(self.num_underflow):
			self.output.write(bit ^ 1)
		self.num_underflow = 0
	
	
	def underflow(self):
		self.num_underflow += 1

class ArithmeticDecoder(ArithmeticCoderBase):

	def __init__(self, numbits, bitin):
		super(ArithmeticDecoder, self).__init__(numbits)
		self.input = bitin
		self.code = 0
		for _ in range(self.num_state_bits):
			self.code = self.code << 1 | self.read_code_bit()
	

	def read(self, freqs):
		#if not isinstance(freqs, CheckedFrequencyTable):
		#	freqs = CheckedFrequencyTable(freqs)

		total = int(freqs[-1])
		#total = freqs.get_total()
		#if total > self.maximum_total:
		#	raise ValueError("Cannot decode symbol because total is too large")
		range = self.high - self.low + 1
		offset = self.code - self.low
		value = ((offset + 1) * total - 1) // range
		#assert value * range // total <= offset
		#assert 0 <= value < total

		start = 0
		end = len(freqs)
		#end = freqs.get_symbol_limit()
		while end - start > 1:
			middle = (start + end) >> 1
			low = int(freqs[middle-1]) if middle > 0 else 0
			#if freqs.get_low(middle) > value:
			if low > value:
				end = middle
			else:
				start = middle
		#assert start + 1 == end
		
		symbol = start
		self.update(freqs, symbol)
		return symbol
	
	
	def shift(self):
		self.code = ((self.code << 1) & self.state_mask) | self.read_code_bit()
	
	
	def underflow(self):
		self.code = (self.code & self.half_range) | ((self.code << 1) & (self.state_mask >> 1)) | self.read_code_bit()

	def read_code_bit(self):
		temp = self.input.read()
		if temp == -1:
			temp = 0
		return temp

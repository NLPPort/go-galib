package ga

type Sketch struct {
	sum, squared float64
	n            uint64
}

func (s *Sketch) Add(x float64) {
	s.sum += x
	s.squared += x * x
	s.n++
}

func (s *Sketch) Average() float64 {
	return s.sum / float64(s.n)
}

func (s *Sketch) Variance() float64 {
	average := s.Average()
	return s.squared/float64(s.n) - average*average
}

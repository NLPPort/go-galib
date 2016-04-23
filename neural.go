package ga

import (
	"math/rand"

	"github.com/pointlander/gobrain"
)

type GANeural interface {
	Train(genomes GAGenomes, selector GASelector)
	Morph(genome GAGenome) GAGenome
	String() string
}

type GAFeedForwardNeural struct {
	ff         [8]*gobrain.FeedForward32
	mse        [8]float32
	Dropout    float32
	Regression bool
	Noise      float32
}

const (
	SELECT = 5
)

func (n *GAFeedForwardNeural) Train(genomes GAGenomes, selector GASelector) {
	done, width := make([]chan bool, len(n.ff)), genomes[0].Len()
	for f := range n.ff {
		done[f] = make(chan bool, 1)
		selected := make(GAGenomes, SELECT)
		for i := range selected {
			selected[i] = selector.SelectOne(genomes)
		}
		go func(done chan bool, i int, selected GAGenomes) {
			patterns := make([][][]float32, SELECT)
			for i, g := range selected {
				pattern, source := make([]float32, width), g.(*GAFloatGenome)
				for j := range pattern {
					pattern[j] = float32(source.Gene[j])
				}
				patterns[i] = [][]float32{pattern, pattern}
			}
			if n.ff[i] == nil {
				n.ff[i] = &gobrain.FeedForward32{}
				n.ff[i].Init(width, width/2, width)
				n.ff[i].Dropout = n.Dropout
				n.ff[i].Regression = n.Regression
			}
			mse := n.ff[i].Train(patterns, 1, 0.6, 0.4, false)
			n.mse[i] = mse[0]

			done <- true
		}(done[f], f, selected)
	}
	for i := range done {
		<-done[i]
	}
}

func (n *GAFeedForwardNeural) Morph(genome GAGenome) GAGenome {
	width := genome.Len()
	morph, source := make([]float32, width), genome.(*GAFloatGenome)
	for i := range morph {
		morph[i] = float32(source.Gene[i])
	}
	noise := make([]float32, width+width/2+width)
	_noise := [][]float32{noise[:width], noise[width : width+width/2], noise[width+width/2:]}
	ff := rand.Intn(len(n.ff))
	for i := range _noise[0] {
		n := n.Noise * float32(rand.NormFloat64()) / n.mse[ff]
		_noise[0][i] = n
		_noise[2][i] = n
	}
	morphed := n.ff[ff].UpdateWithNoise(morph, _noise)

	cp := source.Copy().(*GAFloatGenome)
	for i := range cp.Gene {
		cp.Gene[i] = float64(morphed[i])
	}
	cp.Reset()
	return cp
}

func (n *GAFeedForwardNeural) String() string {
	return "GAFeedForwardNeural"
}

package ga

import (
	"math/rand"

	"github.com/pointlander/gobrain"
)

// See: http://www.nextbigfuture.com/2016/03/sander-olson-interviewed-dr-stephen.html

type GANeural interface {
	Train(genomes GAGenomes, selector GASelector)
	Morph(genome GAGenome) GAGenome
	String() string
}

type GAExpert struct {
	gobrain.FeedForward32
	mse      float32
	selected [SELECT]GAGenome
}

type GAFeedForwardNeural struct {
	Experts []GAExpert
	Noise   float32
	Single  bool
}

const (
	SELECT = 5
)

func NewGAFeedForwardNeural(noise float32, experts, width int, dropout float32, regression bool) *GAFeedForwardNeural {
	_experts := make([]GAExpert, experts)
	for e := range _experts {
		_experts[e].Init(width, width/2, width)
		_experts[e].Dropout = dropout
		_experts[e].Regression = regression
	}
	return &GAFeedForwardNeural{Experts: _experts, Noise: noise}
}

func (n *GAFeedForwardNeural) Train(genomes GAGenomes, selector GASelector) {
	done, width := make([]chan bool, len(n.Experts)), genomes[0].Len()
	for f := range n.Experts {
		done[f] = make(chan bool, 1)
		for i := range n.Experts[f].selected {
			n.Experts[f].selected[i] = selector.SelectOne(genomes)
		}
		go func(done chan bool, i int) {
			patterns := make([][][]float32, SELECT)
			for i, g := range n.Experts[i].selected {
				pattern, source := make([]float32, width), g.(*GAFloat32Genome)
				for j := range pattern {
					pattern[j] = source.Gene[j]
				}
				patterns[i] = [][]float32{pattern, pattern}
			}
			mse := n.Experts[i].Train(patterns, 1, 0.6, 0.4, false)
			n.Experts[i].mse = mse[0]

			done <- true
		}(done[f], f)
	}
	for i := range done {
		<-done[i]
	}
}

func (n *GAFeedForwardNeural) Morph(genome GAGenome) GAGenome {
	width := genome.Len()
	morph, source := make([]float32, width), genome.(*GAFloat32Genome)
	for i := range morph {
		morph[i] = source.Gene[i]
	}
	noise := make([]float32, width+width/2+width)
	_noise := [][]float32{noise[:width], noise[width : width+width/2], noise[width+width/2:]}
	ff := rand.Intn(len(n.Experts))
	for i := range _noise[0] {
		n := n.Noise * float32(rand.NormFloat64()) / n.Experts[ff].mse
		_noise[0][i] = n
		_noise[2][i] = n
	}
	morphed := n.Experts[ff].UpdateWithNoise(morph, _noise)

	cp := source.Copy().(*GAFloat32Genome)
	if n.Single {
		mutations := int(rand.NormFloat64()) + 1
		for m := 0; m < mutations; m++ {
			i := rand.Intn(len(cp.Gene))
			cp.Gene[i] = morphed[i]
		}
	} else {
		for i := range cp.Gene {
			cp.Gene[i] = morphed[i]
		}
	}
	cp.Reset()
	return cp
}

func (n *GAFeedForwardNeural) String() string {
	return "GAFeedForwardNeural"
}

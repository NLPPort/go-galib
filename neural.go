package ga

import (
	//"fmt"
	"math/rand"

	"github.com/goml/gobrain"
)

type GANeural interface {
	Train(genomes GAGenomes)
	Morph(genome GAGenome) GAGenome
	String() string
}

type GAFeedForwardNeural struct {
	ff *gobrain.FeedForward
}

func (n *GAFeedForwardNeural) Train(genomes GAGenomes) {
	patterns, width := [][][]float64{}, genomes[0].Len()
	for _, g := range genomes {
		if rand.Float64() > .5 {
			pattern, source := make([]float64, width), g.(*GAFixedBitstringGenome)
			for j := range pattern {
				if source.Gene[j] {
					pattern[j] = 1
				}
			}
			patterns = append(patterns, [][]float64{pattern, pattern})
		}
	}
	n.ff = &gobrain.FeedForward{}
	n.ff.Init(width, width-1, width)
	n.ff.Train(patterns, 1, 0.6, 0.4, true)
}

func (n *GAFeedForwardNeural) Morph(genome GAGenome) GAGenome {
	width := genome.Len()
	morph, source := make([]float64, width), genome.(*GAFixedBitstringGenome)
	for i := range morph {
		if source.Gene[i] {
			morph[i] = 1
		}
	}
	p := rand.Intn(source.Len())
	morph[p] += rand.Float64()*2 - 1
	if morph[p] > 1 {
		morph[p] = 1
	} else if morph[p] < 0 {
		morph[p] = 0
	}
	morphed := n.ff.Update(morph)
	cp := source.Copy().(*GAFixedBitstringGenome)
	for i := range cp.Gene {
		cp.Gene[i] = morphed[i] > .5
	}
	cp.Reset()
	//fmt.Println(cp.Gene)
	return cp
}

func (n *GAFeedForwardNeural) String() string {
	return "GAFeedForwardNeural"
}

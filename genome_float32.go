/*
Copyright 2010 Thomas Jager <mail@jager.no> All rights reserved.

Use of this source code is governed by a BSD-style
license that can be found in the LICENSE file.

Floating point genome. For solving functions for example.
*/

package ga

import (
	"fmt"
	"math/rand"
	"sync"
)

type GAFloat32Genome struct {
	Gene     []float32
	score    float32
	Max      float32
	Min      float32
	hasscore bool
	sfunc    func(ga *GAFloat32Genome) float32
	sync.RWMutex
}

func NewFloat32Genome(i []float32, sfunc func(ga *GAFloat32Genome) float32, max float32, min float32) *GAFloat32Genome {
	g := new(GAFloat32Genome)
	g.Gene = i
	g.sfunc = sfunc
	g.Max = max
	g.Min = min
	return g
}

// Partially mapped crossover.
func (a *GAFloat32Genome) Crossover(bi GAGenome, p1, p2 int) (GAGenome, GAGenome) {
	ca := a.Copy().(*GAFloat32Genome)
	b := bi.(*GAFloat32Genome)
	cb := b.Copy().(*GAFloat32Genome)
	copy(ca.Gene[p1:p2+1], b.Gene[p1:p2+1])
	copy(cb.Gene[p1:p2+1], a.Gene[p1:p2+1])
	ca.Reset()
	cb.Reset()
	return ca, cb
}

func (a *GAFloat32Genome) Splice(bi GAGenome, from, to, length int) {
	b := bi.(*GAFloat32Genome)
	copy(a.Gene[to:length+to], b.Gene[from:length+from])
	a.Reset()
}

func (g *GAFloat32Genome) Valid() bool {
	//TODO: Make this
	return true
}

func (g *GAFloat32Genome) Switch(x, y int) {
	g.Gene[x], g.Gene[y] = g.Gene[y], g.Gene[x]
	g.Reset()
}

func (g *GAFloat32Genome) Randomize() {
	l := len(g.Gene)
	for i := 0; i < l; i++ {
		g.Gene[i] = rand.Float32()*(g.Max-g.Min) + g.Min
	}
	g.Reset()
}

func (g *GAFloat32Genome) Copy() GAGenome {
	n := new(GAFloat32Genome)
	n.Gene = make([]float32, len(g.Gene))
	copy(n.Gene, g.Gene)
	n.sfunc = g.sfunc
	n.score = g.score
	n.Max = g.Max
	n.Min = g.Min
	n.hasscore = g.hasscore
	return n
}

func (g *GAFloat32Genome) Len() int { return len(g.Gene) }

func (g *GAFloat32Genome) Score() float64 {
	g.RLock()
	hasscore, score := g.hasscore, g.score
	g.RUnlock()

	if !hasscore {
		g.Lock()
		if !g.hasscore {
			score = g.sfunc(g)
			g.score = score
			g.hasscore = true
		}
		g.Unlock()
	}

	return float64(score)
}

func (g *GAFloat32Genome) Reset() { g.hasscore = false }

func (g *GAFloat32Genome) String() string { return fmt.Sprintf("%v", g.Gene) }

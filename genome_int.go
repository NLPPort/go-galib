/*
Copyright 2009 Thomas Jager <mail@jager.no> All rights reserved.
Use of this source code is governed by a BSD-style
license that can be found in the LICENSE file.
*/

package ga

import (
	"fmt"
	"math/rand"
)

type GAIntGenome struct {
	Gene     []int
	min, max int
	score    float64
	hasscore bool
	sfunc    func(ga *GAIntGenome) float64
}

func NewIntGenome(i []int, sfunc func(ga *GAIntGenome) float64, min, max int) *GAIntGenome {
	g := new(GAIntGenome)
	g.Gene = i
	g.min = min
	g.max = max
	g.sfunc = sfunc
	return g
}

func (a *GAIntGenome) Crossover(bi GAGenome, p1, p2 int) (GAGenome, GAGenome) {
	ca := a.Copy().(*GAIntGenome)
	b := bi.(*GAIntGenome)
	cb := b.Copy().(*GAIntGenome)
	copy(ca.Gene[p1:p2+1], b.Gene[p1:p2+1])
	copy(cb.Gene[p1:p2+1], a.Gene[p1:p2+1])
	ca.Reset()
	cb.Reset()
	return ca, cb
}

func (a *GAIntGenome) Splice(bi GAGenome, from, to, length int) {
	b := bi.(*GAIntGenome)
	copy(a.Gene[to:length+to], b.Gene[from:length+from])
	a.Reset()
}

func (g *GAIntGenome) Valid() bool {
	return true
}

func (g *GAIntGenome) Switch(x, y int) {
	g.Gene[x], g.Gene[y] = g.Gene[y], g.Gene[x]
	g.Reset()
}

func (g *GAIntGenome) Randomize() {
	min, max := g.min, g.max
	r := max - min + 1
	for i := range g.Gene {
		g.Gene[i] = rand.Intn(r) + min
	}
	g.Reset()
}

func (g *GAIntGenome) Copy() GAGenome {
	n := new(GAIntGenome)
	n.Gene = make([]int, len(g.Gene))
	copy(n.Gene, g.Gene)
	n.min = g.min
	n.max = g.max
	n.sfunc = g.sfunc
	n.score = g.score
	n.hasscore = g.hasscore
	return n
}

func (g *GAIntGenome) Len() int { return len(g.Gene) }

func (g *GAIntGenome) Score() float64 {
	if !g.hasscore {
		g.score = g.sfunc(g)
		g.hasscore = true
	}
	return g.score
}

func (g *GAIntGenome) Reset() { g.hasscore = false }

func (g *GAIntGenome) String() string { return fmt.Sprintf("%v", g.Gene) }

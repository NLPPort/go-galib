/*
Copyright 2009 Thomas Jager <mail@jager.no> All rights reserved.
Use of this source code is governed by a BSD-style
license that can be found in the LICENSE file.

go-galib beeders
*/

package ga

import (
	"math/rand"
)

type GABreeder interface {
	// Breeds two parent GAGenomes and returns two children
	Breed(a, b GAGenome) (ca, cb GAGenome)
	// String name of breeder
	String() string
}

//Combines genomes by selecting 2 points to exchange
// Example: Parent 1 = 111111, Parent 2 = 000000, Child = 111001
type GA2PointBreeder struct{}

func (breeder *GA2PointBreeder) Breed(a, b GAGenome) (ca, cb GAGenome) {
	if a.Len() != b.Len() {
		panic("Length mismatch in pmx")
	}
	p1 := rand.Intn(a.Len())
	p2 := rand.Intn(b.Len())
	if p1 > p2 {
		p1, p2 = p2, p1
	}
	ca, cb = a.Crossover(b, p1, p2)
	return
}

func (b *GA2PointBreeder) String() string { return "GA2PointBreeder" }

type GAUniformBreeder struct{}

func (u *GAUniformBreeder) Breed(a, b GAGenome) (ca, cb GAGenome) {
	if a.Len() != b.Len() {
		panic("Length mismatch in pmx")
	}
	ca, cb = a.Copy(), b.Copy()
	length := a.Len()
	for i := 0; i < length; i++ {
		if rand.Intn(2) == 0 {
			ca.Splice(a, i, i, 1)
			cb.Splice(b, i, i, 1)
		} else {
			ca.Splice(b, i, i, 1)
			cb.Splice(a, i, i, 1)
		}
	}
	return
}

func (u *GAUniformBreeder) String() string { return "GAUniformBreeder" }

//Totally useless breeader. Copies input and shuffles it.
type GARandomBreeder struct{}

func (breeder *GARandomBreeder) Breed(a, b GAGenome) (ca, cb GAGenome) {
	ca = a.Copy()
	ca.Randomize()
	cb = b.Copy()
	cb.Randomize()
	return
}

func (b *GARandomBreeder) String() string { return "GARandomBreeder" }

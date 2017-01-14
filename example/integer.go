/*
Copyright 2009 Thomas Jager <mail@jager.no> All rights reserved.
Use of this source code is governed by a BSD-style
license that can be found in the LICENSE file.

Example of uing the ordered int genome and mutators
*/
package main

import (
	"fmt"
	"math/rand"
	"time"

	ga "github.com/pointlander/go-galib"
)

var scores int

// Boring fitness/score function.
func score(g *ga.GAIntGenome) float64 {
	var total int
	for _, c := range g.Gene {
		total += c
	}
	scores++
	if total < 0 {
		total = -total
	}
	return float64(total)
}

func main() {
	rand.Seed(time.Now().UTC().UnixNano())

	m := ga.NewMultiMutator()
	msh := new(ga.GAShiftMutator)
	msw := new(ga.GASwitchMutator)
	m.Add(msh)
	m.Add(msw)

	param := ga.GAParameter{
		Initializer: new(ga.GARandomInitializer),
		Selector:    ga.NewGATournamentSelector(0.7, 5),
		Breeder:     new(ga.GA2PointBreeder),
		Mutator:     m,
		PMutate:     0.1,
		PBreed:      0.7}

	gao := ga.NewGA(param)

	genome := ga.NewIntGenome(make([]int, 100), score, -10, 10)

	gao.Init(10, genome) //Total population
	gao.OptimizeUntil(func(best ga.GAGenome) bool {
		return best.Score() == 0
	})
	gao.PrintTop(10)

	fmt.Printf("Calls to score = %d\n", scores)
	fmt.Printf("%s\n", m.Stats())
}

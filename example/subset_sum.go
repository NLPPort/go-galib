/*
Copyright 2009 Thomas Jager <mail@jager.no> All rights reserved.
Use of this source code is governed by a BSD-style
license that can be found in the LICENSE file.

subset sum solver
*/
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"runtime/pprof"
	"time"

	"github.com/pointlander/go-galib"
)

var theset = [200]int{332, 401, -178, 60, -436, -135, -275, 192, -223, -150, -67, 401, -140, -71, 152, 478, 210, -485, -465, -492, -355, -474, -420, 213, -63, 33, 366, -94, -469, 429, -307, -291, 176, 465, 180, 28, 408, -245, -318, -66, -158, -202, -191, 47, -71, -320, 142, 305, 429, -449, -58, -115, 153, -47, 95, 215, 82, 452, 390, 331, -419, -68, -416, 331, -35, -102, 270, -72, -81, 133, 159, -417, 455, -99, -137, -477, -99, 312, -409, -401, -468, -453, -165, 163, 415, -85, 304, 307, -38, -439, 162, 310, 13, 320, 362, 336, -461, 435, 378, 194, -430, -322, 307, -159, -325, -290, -339, 485, -464, 315, -205, 385, 98, 439, -82, 374, -288, -407, -225, -463, 302, -442, 237, -427, -40, -156, -117, -53, -386, -133, -5, -287, -403, -487, -134, -273, 481, -405, 459, 108, 454, -106, 76, 116, -390, 90, -52, -120, -213, -62, -481, -417, 115, -33, 484, -8, 243, 439, -491, -299, -289, 191, 394, -161, 109, -468, -289, 355, 293, -54, -499, -374, -99, -142, 238, 115, -46, -182, 398, -32, 186, -91, -479, -108, -399, -231, -212, -233, 178, 82, -126, 304, -140, -364, 14, -307, 280, -334, 2, -444}

var scores int

// Fitness function for subset sum
func score(g *ga.GAFloatGenome) float64 {
	scores++
	total := 0
	for i, c := range g.Gene {
		if c > .5 {
			total += theset[i]
		}
	}
	if total < 0 {
		return float64(-total)
	}
	return float64(total)
}

func optimize(neural bool) (int, int) {
	m := ga.NewMultiMutator()
	msh := new(ga.GAShiftMutator)
	msw := new(ga.GASwitchMutator)
	m.Add(msh)
	m.Add(msw)

	param := ga.GAParameter{
		Initializer: new(ga.GARandomInitializer),
		Selector:    ga.NewGATournamentSelector(0.7, 5),
		PMutate:     0.2,
		PBreed:      0.2}
	if neural {
		param.Neural = &ga.GAFeedForwardNeural{
			Dropout:   true,
			Mutations: 16,
		}
		param.PMutate = 1
	} else {
		param.Breeder = new(ga.GA2PointBreeder)
		param.Mutator = m
	}

	gao := ga.NewGA(param)

	genome := ga.NewFloatGenome(make([]float64, len(theset)), score, 1, 0)

	gao.Init(128, genome) //Total population

	generations := 0
	scores = 0
	for {
		gao.Optimize(1) // Run genetic algorithm for 20 generations.
		generations++
		best := gao.Best().(*ga.GAFloatGenome)
		//fmt.Printf("best=%v\n", best.Score())
		sum := 0
		if best.Score() == 0 {
			for n, value := range best.Gene {
				if value > .5 {
					fmt.Printf("%d,", theset[n])
					sum += theset[n]
				}
			}
			fmt.Printf(" = %d / %f\n", sum, best.Score())
			break
		}
	}
	fmt.Printf("Calls to score = %d\n", scores)
	return generations, scores
}

const (
	SAMPLES = 128
)

var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")

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

func main() {
	flag.Parse()
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	rand.Seed(time.Now().UTC().UnixNano())
	generations, scores := Sketch{}, Sketch{}
	for i := 0; i < SAMPLES; i++ {
		g, s := optimize(false)
		generations.Add(float64(g))
		scores.Add(float64(s))
	}
	ngenerations, nscores := Sketch{}, Sketch{}
	for i := 0; i < SAMPLES; i++ {
		g, s := optimize(true)
		ngenerations.Add(float64(g))
		nscores.Add(float64(s))
	}
	fmt.Printf("average generations = %v\n", generations.Average())
	fmt.Printf("variance generations = %v\n", generations.Variance())
	fmt.Printf("average scores = %v\n", scores.Average())
	fmt.Printf("variance scores = %v\n", scores.Variance())
	fmt.Printf("average neural generations = %v\n", ngenerations.Average())
	fmt.Printf("variance neural generations = %v\n", ngenerations.Variance())
	fmt.Printf("average neural scores = %v\n", nscores.Average())
	fmt.Printf("variance neural scores = %v\n", nscores.Variance())
}

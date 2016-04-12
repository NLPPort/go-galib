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
	"math"
	"math/rand"
	"os"
	"runtime/pprof"
	"time"

	"github.com/pointlander/go-galib"
)

var scores int

func ackley(g *ga.GAFloatGenome) float64 {
	scores++
	var sum1 float64 = 0.0
	for _, c := range g.Gene {
		sum1 += float64(c * c)
	}
	t1 := math.Exp(-0.2 * (math.Sqrt((1.0 / 5.0) * sum1)))
	sum1 = 0.0
	for _, c := range g.Gene {
		sum1 += math.Cos(float64(2.0 * math.Pi * c))
	}
	t2 := math.Exp((1.0 / 5.0) * sum1)
	return (20 + math.Exp(1) - 20*t1 - t2)
}

func rosenbrock(g *ga.GAFloatGenome) float64 {
	scores++
	var sum float64
	for i := 1; i < len(g.Gene); i++ {
		sum += 100.0*math.Pow(g.Gene[i]-math.Pow(g.Gene[i-1], 2), 2) + math.Pow(1-g.Gene[i-1], 2)
	}

	if math.IsNaN(sum) {
		sum = math.MaxFloat64
	}
	return sum
}

func optimize(neural bool) (int, int) {
	param := ga.GAParameter{
		Initializer: new(ga.GARandomInitializer),
		Selector:    ga.NewGATournamentSelector(0.2, 5),
		PMutate:     0.5,
		PBreed:      0.2}
	if neural {
		param.Neural = &ga.GAFeedForwardNeural{
			Regression: true,
			Mutations:  4,
		}
	} else {
		param.Breeder = new(ga.GA2PointBreeder)
		param.Mutator = ga.NewGAGaussianMutator(0.4, 0)
	}

	gao := ga.NewGA(param)

	genome := ga.NewFloatGenome(make([]float64, 2), rosenbrock, 1, -1)

	gao.Init(100, genome) //Total population

	generations := 0
	scores = 0
	gao.OptimizeUntil(func(best ga.GAGenome) bool {
		//fmt.Printf("best = %v\n", best.Score())
		generations++
		return best.Score() < 1e-3
	})
	best := gao.Best().(*ga.GAFloatGenome)
	fmt.Printf("%s = %f\n", best, best.Score())
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

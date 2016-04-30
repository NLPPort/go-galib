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

type Experiment struct {
	name         string
	set          string
	configurator func() *ga.GA
	generations  ga.Sketch
	scores       ga.Sketch
}

var scores int

func ackley(g *ga.GAFloatGenome) float64 {
	scores++
	var sum1 float64
	d := float64(len(g.Gene))
	for _, c := range g.Gene {
		sum1 += float64(c * c)
	}
	t1 := math.Exp(-0.2 * (math.Sqrt((1.0 / d) * sum1)))
	sum1 = 0.0
	for _, c := range g.Gene {
		sum1 += math.Cos(float64(2.0 * math.Pi * c))
	}
	t2 := math.Exp((1.0 / d) * sum1)
	x := (20 + math.Exp(1) - 20*t1 - t2)

	if math.IsNaN(x) {
		x = math.MaxFloat64
	}
	return x
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

func rosenbrockNormal() *ga.GA {
	width := 2
	param := ga.GAParameter{
		Initializer: new(ga.GARandomInitializer),
		Selector:    ga.NewGATournamentSelector(0.2, 5),
		PMutate:     0.5,
		PBreed:      0.2,
		Breeder:     new(ga.GA2PointBreeder),
		Mutator:     ga.NewGAGaussianMutator(0.4, 0),
	}
	gao := ga.NewGA(param)
	gao.Init(100, ga.NewFloatGenome(make([]float64, width), rosenbrock, 1, -1))
	return gao
}

func rosenbrockNeural() *ga.GA {
	width := 2
	param := ga.GAParameter{
		Initializer: new(ga.GARandomInitializer),
		Selector:    ga.NewGATournamentSelector(0.2, 5),
		PMutate:     0.5,
		PBreed:      0.2,
		Neural:      ga.NewGAFeedForwardNeural(.001, 8, width, 0, true),
	}
	gao := ga.NewGA(param)
	gao.Init(100, ga.NewFloatGenome(make([]float64, width), rosenbrock, 1, -1))
	return gao
}

func ackleyNormal() *ga.GA {
	width := 4
	param := ga.GAParameter{
		Initializer: new(ga.GARandomInitializer),
		Selector:    ga.NewGATournamentSelector(0.2, 5),
		PMutate:     0.5,
		PBreed:      0.2,
		Breeder:     new(ga.GA2PointBreeder),
		Mutator:     ga.NewGAGaussianMutator(0.4, 0),
	}
	gao := ga.NewGA(param)
	gao.Init(100, ga.NewFloatGenome(make([]float64, width), ackley, 32, -32))
	return gao
}

func ackleyNeural() *ga.GA {
	width := 4
	param := ga.GAParameter{
		Initializer: new(ga.GARandomInitializer),
		Selector:    ga.NewGATournamentSelector(0.2, 5),
		PMutate:     0.5,
		PBreed:      0.2,
		Neural:      ga.NewGAFeedForwardNeural(1E-9, 64, width, 0, true),
	}
	gao := ga.NewGA(param)
	gao.Init(100, ga.NewFloatGenome(make([]float64, width), ackley, 32, -32))
	return gao
}

var Experiments = [...]Experiment{
	{
		name:         "Rosenbrock Normal",
		set:          "rosenbrock",
		configurator: rosenbrockNormal,
	},
	{
		name:         "Rosenbrok Neural",
		set:          "rosenbrock",
		configurator: rosenbrockNeural,
	},
	{
		name:         "Ackley Normal",
		set:          "ackley",
		configurator: ackleyNormal,
	},
	{
		name:         "Ackley Neural",
		set:          "ackley",
		configurator: ackleyNeural,
	},
}

const (
	SAMPLES = 128
)

func (e *Experiment) Run() {
	for i := 0; i < SAMPLES; i++ {
		gao := e.configurator()

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
		e.generations.Add(float64(generations))
		e.scores.Add(float64(scores))
	}
}

func (e *Experiment) Results() {
	fmt.Printf("%v generations = %v+-%v\n", e.name, e.generations.Average(), e.generations.Variance())
	fmt.Printf("%v scores = %v+-%v\n", e.name, e.scores.Average(), e.scores.Variance())
}

var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")
var set = flag.String("set", "all", "experiment set to run")

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
	experiments := []*Experiment{}
	for e := range Experiments {
		if *set == "all" || *set == Experiments[e].set {
			experiments = append(experiments, &Experiments[e])
		}
	}
	for e := range experiments {
		experiments[e].Run()
	}
	for e := range experiments {
		experiments[e].Results()
	}
}

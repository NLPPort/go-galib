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
	"github.com/pointlander/gobrain"
)

type Experiment struct {
	name         string
	set          string
	configurator func() *ga.GA
	generations  ga.Sketch
	scores       ga.Sketch
}

var scores int

func ackley(g *ga.GAFloat32Genome) float32 {
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
	return float32(x)
}

func rosenbrock(g *ga.GAFloat32Genome) float32 {
	scores++
	var sum float64
	for i := 1; i < len(g.Gene); i++ {
		sum += 100.0*math.Pow(float64(g.Gene[i])-math.Pow(float64(g.Gene[i-1]), 2), 2) + math.Pow(1-float64(g.Gene[i-1]), 2)
	}

	if math.IsNaN(sum) {
		sum = math.MaxFloat64
	}
	return float32(sum)
}

const (
	NeuralInputWidth  = 2
	NeuralMiddleWidth = 2
	NeuralOutputWidth = 1
	NeuralWidth       = (NeuralMiddleWidth+1)*(NeuralInputWidth+1) + NeuralOutputWidth*(NeuralMiddleWidth+1)
)

var Patterns = [][][]float32{
	{{0, 0}, {0}},
	{{0, 1}, {1}},
	{{1, 0}, {1}},
	{{1, 1}, {0}},
}

func neuralNetwork(g *ga.GAFloat32Genome) float32 {
	scores++
	ff := &gobrain.FeedForward32{}
	ff.Init(2, 2, 1)
	ff.SetWeights(g.Gene)
	var e float32
	for p := range Patterns {
		out := ff.Update(Patterns[p][0])
		for o := range out {
			d := out[o] - Patterns[p][1][o]
			if d < 0 {
				d = -d
			}
			e += d
		}
	}

	if math.IsNaN(float64(e)) {
		e = math.MaxFloat32
	}
	return e
}

const (
	RNNInputWidth  = 1
	RNNMiddleWidth = 8
	RNNOutputWidth = RNNInputWidth //+ 1
	RNNStateWidth  = RNNInputWidth - 1
	RNNWidth       = (RNNMiddleWidth+1)*(RNNInputWidth+1+RNNMiddleWidth) + RNNOutputWidth*(RNNMiddleWidth+1)
)

var States = []float32{
	1, 0, 0, 1,
	0, 1, 0, 0,
	1, 1, 1, 0,
	0, 0, 1, 0,

	0, 1, 0, 0,
	0, 0, 1, 1,
	0, 1, 0, 1,
	0, 1, 0, 1,
}

func init() {
	for s := range States {
		if States[s] == 0 {
			States[s] = -1
		}
	}
}

func rnn(g *ga.GAFloat32Genome) float32 {
	scores++
	ff := &gobrain.RNN32{}
	ff.Init(RNNInputWidth, RNNMiddleWidth, RNNOutputWidth)
	ff.Reset()
	ff.SetWeights(g.Gene)
	var e float32
	input := make([]float32, RNNInputWidth)
	input[0] = States[0]
	for _, s := range States[1:] {
		output := ff.Update(input)
		d := output[0] - s
		e += d * d
		copy(input, output)
	}

	if math.IsNaN(float64(e)) {
		e = math.MaxFloat32
	}
	return e
}

type State struct {
	state float32
	set   bool
}

func rnnb(g *ga.GAFloatGenome) float64 {
	scores++
	ff := &gobrain.RNN32{}
	ff.Init(RNNInputWidth, RNNMiddleWidth, RNNOutputWidth)
	ff.Reset()
	weights := make([]float32, len(g.Gene))
	for w := range weights {
		weights[w] = float32(g.Gene[w])
	}
	ff.SetWeights(weights)
	var e float64
	states := make([]State, len(States))
	input := make([]float32, RNNInputWidth)
	input[0] = States[0]
	states[0].state = States[0]
	states[0].set = true
	i, length := 0, len(states)
	for range States[1:] {
		output := ff.Update(input)
		offset := int(math.Floor(float64(output[1] * float32(length))))
		i = (i + offset) % length
		if i < 0 {
			i += length
		}
		for states[i].set {
			i = (i + 1) % length
		}
		states[i].state = output[0]
		states[i].set = true
		d := output[0] - States[i]
		e += float64(d * d)
		input[0] = output[0]
	}

	if math.IsNaN(e) {
		e = math.MaxFloat64
	}
	return e
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
	gao.Init(100, ga.NewFloat32Genome(make([]float32, width), rosenbrock, 1, -1))
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
	gao.Init(100, ga.NewFloat32Genome(make([]float32, width), rosenbrock, 1, -1))
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
	gao.Init(100, ga.NewFloat32Genome(make([]float32, width), ackley, 32, -32))
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
	gao.Init(100, ga.NewFloat32Genome(make([]float32, width), ackley, 32, -32))
	return gao
}

func neuralNetworkNormal() *ga.GA {
	param := ga.GAParameter{
		Initializer: new(ga.GARandomInitializer),
		Selector:    ga.NewGATournamentSelector(0.2, 5),
		PMutate:     0.5,
		PBreed:      0.2,
		Breeder:     new(ga.GA2PointBreeder),
		Mutator:     ga.NewGAGaussianMutator(.4, 0),
	}
	gao := ga.NewGA(param)
	gao.Init(1000, ga.NewFloat32Genome(make([]float32, NeuralWidth), neuralNetwork, 1, -1))
	return gao
}

func neuralNetworkNeural() *ga.GA {
	param := ga.GAParameter{
		Initializer: new(ga.GARandomInitializer),
		Selector:    ga.NewGATournamentSelector(0.2, 5),
		PMutate:     0.5,
		PBreed:      0.2,
		Neural:      ga.NewGAFeedForwardNeural(.001, 128, NeuralWidth, 0, true),
	}
	gao := ga.NewGA(param)
	gao.Init(1000, ga.NewFloat32Genome(make([]float32, NeuralWidth), neuralNetwork, 1, -1))
	return gao
}

func rnnNeural() *ga.GA {
	param := ga.GAParameter{
		Initializer: new(ga.GARandomInitializer),
		Selector:    ga.NewGATournamentSelector(0.2, 5),
		PMutate:     0.5,
		PBreed:      0.2,
		//Neural:      ga.NewGAFeedForwardNeural(.4, 8, RNNWidth, 0, true),
		Breeder: new(ga.GA2PointBreeder),
		Mutator: ga.NewGAGaussianMutator(.5, 0),
	}
	gao := ga.NewGA(param)
	gao.Init(1000, ga.NewFloat32Genome(make([]float32, RNNWidth), rnn, 1, -1))
	return gao
}

func rnnNeuralU() *ga.GA {
	param := ga.GAParameter{
		Initializer: new(ga.GARandomInitializer),
		Selector:    ga.NewGATournamentSelector(0.2, 5),
		PMutate:     0.5,
		PBreed:      0.2,
		Breeder:     new(ga.GAUniformBreeder),
		Mutator:     ga.NewGAGaussianMutator(.5, 0),
	}
	gao := ga.NewGA(param)
	gao.Init(1000, ga.NewFloat32Genome(make([]float32, RNNWidth), rnn, 1, -1))
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
	{
		name:         "Neural Network Normal",
		set:          "nn",
		configurator: neuralNetworkNormal,
	},
	{
		name:         "Neural Network Neural",
		set:          "nnn",
		configurator: neuralNetworkNeural,
	},
	{
		name:         "Recurrent Neural Network Neural",
		set:          "rnn",
		configurator: rnnNeural,
	},
	{
		name:         "Recurrent Neural Network Neural Uniform",
		set:          "rnnu",
		configurator: rnnNeuralU,
	},
}

const (
	SAMPLES = 1
)

func (e *Experiment) Run() {
	for i := 0; i < SAMPLES; i++ {
		gao := e.configurator()

		generations := 0
		scores = 0
		gao.OptimizeUntil(func(best ga.GAGenome) bool {
			fmt.Printf("best = %v\n", best.Score())
			generations++
			if e.set == "rnn" || e.set == "rnnu" {
				g := best.(*ga.GAFloat32Genome)
				ff := &gobrain.RNN32{}
				ff.Init(RNNInputWidth, RNNMiddleWidth, RNNOutputWidth)
				ff.Reset()
				weights := make([]float32, len(g.Gene))
				for w := range weights {
					weights[w] = float32(g.Gene[w])
				}
				ff.SetWeights(weights)
				input := make([]float32, RNNInputWidth)
				input[0] = States[0]
				if input[0] > 0 {
					fmt.Printf("1")
				} else {
					fmt.Printf("0")
				}
				correct := true
				for _, s := range States[1:] {
					output := ff.Update(input)
					if output[0] >= 0 {
						fmt.Printf("1")
						if s < 0 {
							correct = false
						}
					} else {
						fmt.Printf("0")
						if s >= 0 {
							correct = false
						}
					}
					input[0] = output[0]
				}
				fmt.Printf("\n")
				if correct {
					return correct
				}
			}
			return best.Score() < 1e-3
		})
		best := gao.Best().(*ga.GAFloat32Genome)
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

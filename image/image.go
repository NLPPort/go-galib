package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"math"
	"os"

	"github.com/nfnt/resize"
	"github.com/pointlander/go-galib"
	"github.com/pointlander/gobrain"
)

func Gray(input image.Image) *image.Gray {
	bounds := input.Bounds()
	output := image.NewGray(bounds)
	width, height := bounds.Max.X, bounds.Max.Y
	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			r, g, b, _ := input.At(x, y).RGBA()
			output.SetGray(x, y, color.Gray{uint8((r + g + b) / 768)})
		}
	}
	return output
}

type Image struct {
	image  *image.Gray
	scores int
}

func NewImage() *Image {
	file, err := os.Open("lenna.png")
	if err != nil {
		log.Fatal(err)
	}
	input, _, err := image.Decode(file)
	if err != nil {
		log.Fatal(err)
	}
	file.Close()

	width, height, scale := input.Bounds().Max.X, input.Bounds().Max.Y, 8
	width, height = width/scale, height/scale
	fmt.Printf("width=%v height=%v\n", width, height)
	input = resize.Resize(uint(width), uint(height), input, resize.NearestNeighbor)

	gray := Gray(input)

	file, err = os.Create("lenna_small.png")
	if err != nil {
		log.Fatal(err)
	}

	err = png.Encode(file, gray)
	if err != nil {
		log.Fatal(err)
	}
	file.Close()

	return &Image{
		image: gray,
	}
}

const (
	RNNInputWidth  = 1
	RNNHiddenWidth = 8
	RNNOutputWidth = RNNInputWidth
	RNNWidth       = (RNNHiddenWidth + RNNOutputWidth) * (RNNInputWidth + RNNHiddenWidth + 1)
)

func (i *Image) Fitness(g *ga.GAFloat32Genome) float32 {
	i.scores++
	ff := &gobrain.RNN32{}
	ff.Init(RNNInputWidth, RNNHiddenWidth, RNNOutputWidth)
	ff.Reset()
	ff.SetWeights(g.Gene)
	var e float32
	input := make([]float32, RNNInputWidth)

	bounds := i.image.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			output := ff.Update(input)
			pixel := float32(i.image.GrayAt(x, y).Y) / 255
			d := output[0] - pixel
			e += d * d
			copy(input, output)
		}
	}

	if math.IsNaN(float64(e)) {
		e = math.MaxFloat32
	}
	return e
}

func main() {
	img := NewImage()
	param := ga.GAParameter{
		Initializer: new(ga.GARandomInitializer),
		Selector:    ga.NewGATournamentSelector(0.2, 5),
		PMutate:     0.5,
		PBreed:      0.2,
		Breeder:     new(ga.GAUniformBreeder),
		Mutator:     ga.NewGAGaussianMutator(10, 0),
	}
	gao := ga.NewGA(param)
	gao.Init(1000, ga.NewFloat32Genome(make([]float32, RNNWidth), img.Fitness, 1, -1))
	gao.OptimizeUntil(func(best ga.GAGenome) bool {
		fmt.Printf("best = %v\n", best.Score())
		return best.Score() < 1e-3
	})
}

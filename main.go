package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/xwi88/gp"
)

func main() {
	// register model
	modelName := "ggp"
	tags := []string{"serve"}
	// exportDir := "testdata/saved_model_half_plus_two_cpu/000001"
	// gp.RegisterTFModelWithParamName(modelName, exportDir, tags, "x", "y")

	// exportDir := "testdata/saved_model_half_plus_three/000001"
	// gp.RegisterTFModelWithParamName(modelName, exportDir, tags, "x", "y")

	// gp.RegisterTFModel(modelName, exportDir, tags)

	exportDir := "testdata/test_fm_model"
	// gp.RegisterTFModelWithParamName(modelName, exportDir, tags, "input", "output")
	gp.RegisterTFModelWithParamName(modelName, exportDir, tags, "input", "output/Identity")
	// gp.RegisterTFModel(modelName, exportDir, tags)

	// get model
	// gp.GetModel(modelName)
	fmt.Println("load model success")

	// size := 2_000
	size := 0_3

	timeStart := time.Now()
	for i := 0; i < size; i++ {
		rand.Seed(time.Now().UnixNano())
		// inputS := generateSliceFloat2(int(rand.Int31n(20)) + 1)
		// inputS := [][]float32{{1.0, 2.0}}
		inputS := [][]float64{
			{
				0.000000000000000000e+00, 1.079680441762320697e-04, 1.525902189314365387e-05, 0.000000000000000000e+00, 1.790296286344528198e-02, 3.008136712014675140e-02, 1.206272630952298641e-03, 0.000000000000000000e+00, 6.816632812842726707e-04, 0.000000000000000000e+00, 1.265822816640138626e-02, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00, 8.900000000000000000e+01, 2.534000000000000000e+03, 2.462000000000000000e+03, 1.200000000000000000e+01, 6.000000000000000000e+00, 1.449000000000000000e+03, 4.000000000000000000e+00, 2.000000000000000000e+00, 2.196000000000000000e+03, 9.950000000000000000e+02, 5.179000000000000000e+03, 7.670000000000000000e+02, 1.500000000000000000e+01, 2.820000000000000000e+02, 3.675000000000000000e+03, 0.000000000000000000e+00, 4.020000000000000000e+02, 6.400000000000000000e+01, 1.000000000000000000e+00, 3.034000000000000000e+03, 0.000000000000000000e+00, 1.000000000000000000e+01, 4.590000000000000000e+02, 2.000000000000000000e+00, 7.530000000000000000e+02,
			},
		}
		// predict with the special model
		output, err := gp.Predict(modelName, inputS)
		if err != nil {
			fmt.Printf("Predict err: %v", err)
		}
		_ = output
		fmt.Printf("output: %v\n", output)
	}
	cost := time.Now().Sub(timeStart)
	fmt.Printf("size:%v, cost:%v\n", size, cost)
}

func generateSliceFloat2(size int) (s []float32) {
	rand.Seed(time.Now().UnixNano())
	s = make([]float32, size)
	for i := 0; i < size; i++ {
		s[i] = float32(rand.Int63n(200))
	}
	return s
}
